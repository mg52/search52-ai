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

	st := store.NewMemoryStore()
	emb := ai.NewEmbeddingClient(cfg.EmbeddingBaseURL, cfg.EmbeddingAPIKey, cfg.EmbeddingModel)

	categorizer := pipeline.NewCategorizer(emb, st, cfg.CategoryThreshold, cfg.MaxCategoriesPerDoc, cfg.MaxCategories)
	searcher := search.NewEngine(emb, st, cfg.TopNCategories)

	h := handler.New(st, categorizer, searcher)

	addr := ":" + cfg.Port
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, h.Routes()); err != nil {
		log.Fatalf("server: %v", err)
	}
}
