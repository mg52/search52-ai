package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strings"
	"time"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/store"
	"github.com/mg52/search52-ai/internal/vec"
)

type Tagger struct {
	llm       *ai.LLMClient
	embedding *ai.EmbeddingClient
	prompts   *ai.Prompts
	store     store.Store
	maxTags   int
	threshold float64
	embedder  *Embedder
}

func NewTagger(llm *ai.LLMClient, embedding *ai.EmbeddingClient, prompts *ai.Prompts, st store.Store, maxTags int, threshold float64, embedder *Embedder) *Tagger {
	return &Tagger{
		llm:       llm,
		embedding: embedding,
		prompts:   prompts,
		store:     st,
		maxTags:   maxTags,
		threshold: threshold,
		embedder:  embedder,
	}
}

type taggingResponse struct {
	Tags []string `json:"tags"`
}

// TagByLLM sends existing tags + document content to the LLM, which decides
// whether to reuse existing tags or create new ones.
func (t *Tagger) TagByLLM(ctx context.Context, doc store.Document) (store.Document, error) {
	systemPrompt, err := t.prompts.RenderTaggingSystem(ai.TaggingData{MaxTags: t.maxTags})
	if err != nil {
		return doc, fmt.Errorf("rendering system prompt: %w", err)
	}

	userPrompt, err := t.prompts.RenderTaggingUser(ai.TaggingData{
		ExistingTags: strings.Join(t.store.GetTagNames(), ", "),
		MaxTags:      t.maxTags,
		Content:      doc.Content,
	})
	if err != nil {
		return doc, fmt.Errorf("rendering user prompt: %w", err)
	}

	var tags []string
	const maxAttempts = 3
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		content, err := t.llm.Complete(ctx, systemPrompt, userPrompt)
		if err != nil {
			return doc, fmt.Errorf("LLM call: %w", err)
		}

		var resp taggingResponse
		if jsonErr := json.Unmarshal([]byte(extractJSON(content)), &resp); jsonErr != nil {
			if attempt == maxAttempts {
				log.Printf("tagger/llm: JSON parse error for doc %s: %v (raw: %q)", doc.ID, jsonErr, content)
				return doc, fmt.Errorf("parsing LLM response: %w", jsonErr)
			}
			log.Printf("tagger/llm: JSON parse error for doc %s (attempt %d/%d), retrying: %v", doc.ID, attempt, maxAttempts, jsonErr)
			continue
		}

		// Drop blank/whitespace tags and duplicates the model may emit (e.g. [""]).
		if tags = cleanTags(resp.Tags); len(tags) > 0 {
			break
		}
		if attempt == maxAttempts {
			log.Printf("tagger/llm: no usable tags for doc %s (raw: %q)", doc.ID, content)
			return doc, fmt.Errorf("LLM returned no usable tags after %d attempts", maxAttempts)
		}
		log.Printf("tagger/llm: no usable tags for doc %s (attempt %d/%d), retrying", doc.ID, attempt, maxAttempts)
	}

	doc.Tags = tags

	var newTags []string
	for _, name := range tags {
		if _, exists := t.store.GetTag(name); !exists {
			if err := t.store.SaveTag(store.Tag{Name: name, CreatedAt: time.Now()}); err != nil {
				log.Printf("tagger/llm: saving tag %q: %v", name, err)
				continue
			}
			newTags = append(newTags, name)
		}
	}

	if err := t.store.SaveDocument(doc); err != nil {
		return doc, fmt.Errorf("saving document: %w", err)
	}

	if len(newTags) > 0 {
		go t.embedder.ProcessTags(context.Background(), newTags)
	}

	return doc, nil
}

// EmbedResult is returned by TagByEmbedding and includes per-tag similarity scores.
type EmbedResult struct {
	Document  store.Document     `json:"document"`
	TagScores map[string]float64 `json:"tag_scores"`
}

// TagByEmbedding embeds the document content and assigns existing tags whose
// vectors are above the similarity threshold. No new tags are created.
func (t *Tagger) TagByEmbedding(ctx context.Context, doc store.Document) (EmbedResult, error) {
	docVec, err := t.embedding.Embed(ctx, doc.Content)
	if err != nil {
		return EmbedResult{}, fmt.Errorf("embedding document: %w", err)
	}

	docNorm := vec.Norm(docVec)
	if docNorm == 0 {
		return EmbedResult{}, fmt.Errorf("document embedding is a zero vector")
	}

	type scored struct {
		name  string
		score float64
	}

	allTags := t.store.ListTags()
	var candidates []scored
	for _, tag := range allTags {
		if tag.Norm == 0 {
			continue // tag has no embedding yet
		}
		if sim := vec.Cosine(docVec, tag.Vector, docNorm, tag.Norm); sim >= t.threshold {
			candidates = append(candidates, scored{tag.Name, sim})
		}
	}

	if len(candidates) == 0 {
		return EmbedResult{}, fmt.Errorf("no tags above similarity threshold %.2f — ensure tags have embeddings via GET /tags", t.threshold)
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})
	if len(candidates) > t.maxTags {
		candidates = candidates[:t.maxTags]
	}

	tagNames := make([]string, len(candidates))
	tagScores := make(map[string]float64, len(candidates))
	for i, c := range candidates {
		tagNames[i] = c.name
		tagScores[c.name] = c.score
	}

	doc.Tags = tagNames

	if err := t.store.SaveDocument(doc); err != nil {
		return EmbedResult{}, fmt.Errorf("saving document: %w", err)
	}

	return EmbedResult{Document: doc, TagScores: tagScores}, nil
}
