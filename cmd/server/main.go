package main

import (
	"log"
	"net/http"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/config"
	"github.com/mg52/search52-ai/internal/handler"
	"github.com/mg52/search52-ai/internal/pipeline"
	"github.com/mg52/search52-ai/internal/search"
	"github.com/mg52/search52-ai/internal/store"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	prompts, err := ai.LoadPrompts(cfg.PromptsDir)
	if err != nil {
		log.Fatalf("prompts: %v", err)
	}

	st := store.NewMemoryStore()
	llm := ai.NewLLMClient(cfg.LLMBaseURL, cfg.LLMAPIKey, cfg.LLMModel)
	emb := ai.NewEmbeddingClient(cfg.EmbeddingBaseURL, cfg.EmbeddingAPIKey, cfg.EmbeddingModel)

	embedder := pipeline.NewEmbedder(llm, emb, prompts, st)
	tagger := pipeline.NewTagger(llm, emb, prompts, st, cfg.MaxTagsPerDoc, cfg.TagMatchThreshold, embedder)
	searcher := search.NewEngine(emb, st, cfg.TagMatchThreshold)

	h := handler.New(st, tagger, searcher)

	addr := ":" + cfg.Port
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, h.Routes()); err != nil {
		log.Fatalf("server: %v", err)
	}
}
