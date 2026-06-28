package pipeline

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/store"
	"github.com/mg52/search52-ai/internal/vec"
)

type Embedder struct {
	llm       *ai.LLMClient
	embedding *ai.EmbeddingClient
	prompts   *ai.Prompts
	store     store.Store
}

func NewEmbedder(llm *ai.LLMClient, embedding *ai.EmbeddingClient, prompts *ai.Prompts, st store.Store) *Embedder {
	return &Embedder{
		llm:       llm,
		embedding: embedding,
		prompts:   prompts,
		store:     st,
	}
}

type descriptionResponse struct {
	Description string `json:"description"`
}

func (e *Embedder) ProcessTags(ctx context.Context, tagNames []string) {
	for _, name := range tagNames {
		if err := e.processTag(ctx, name); err != nil {
			log.Printf("embedder: processing tag %q: %v", name, err)
		}
	}
}

func (e *Embedder) processTag(ctx context.Context, tagName string) error {
	tag, exists := e.store.GetTag(tagName)
	if !exists {
		return fmt.Errorf("tag not found: %s", tagName)
	}

	docs := e.store.GetDocumentsByTag(tagName)
	if len(docs) > 10 {
		docs = docs[:10]
	}

	examples := make([]string, len(docs))
	lines := make([]string, len(docs))
	for i, doc := range docs {
		examples[i] = doc.Content
		lines[i] = fmt.Sprintf("%d. %s", i+1, doc.Content)
	}

	systemPrompt, err := e.prompts.RenderDescriptionSystem(ai.DescriptionData{TagName: tagName})
	if err != nil {
		return fmt.Errorf("rendering system prompt: %w", err)
	}
	userPrompt, err := e.prompts.RenderDescriptionUser(ai.DescriptionData{
		TagName:  tagName,
		Examples: strings.Join(lines, "\n"),
	})
	if err != nil {
		return fmt.Errorf("rendering user prompt: %w", err)
	}

	var resp descriptionResponse
	const maxAttempts = 3
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		var content string
		content, err = e.llm.Complete(ctx, systemPrompt, userPrompt)
		if err != nil {
			return fmt.Errorf("LLM call: %w", err)
		}
		if jsonErr := json.Unmarshal([]byte(extractJSON(content)), &resp); jsonErr == nil {
			break
		} else if attempt == maxAttempts {
			// Last resort: model returned plain prose — use it directly.
			if !strings.Contains(content, "{") {
				resp.Description = strings.TrimSpace(content)
				break
			}
			return fmt.Errorf("parsing description response %q: %w", content, jsonErr)
		}
		log.Printf("embedder: tag %q JSON parse failed (attempt %d/%d), retrying", tagName, attempt, maxAttempts)
	}

	vector, err := e.embedding.Embed(ctx, fmt.Sprintf("%s: %s", tagName, resp.Description))
	if err != nil {
		return fmt.Errorf("embedding: %w", err)
	}

	tag.Description = resp.Description
	tag.Vector = vector
	tag.Norm = vec.Norm(vector)
	tag.Examples = examples

	if err := e.store.UpdateTag(tag); err != nil {
		return fmt.Errorf("updating tag: %w", err)
	}

	log.Printf("embedder: tag %q ready (dims=%d)", tagName, len(vector))
	return nil
}
