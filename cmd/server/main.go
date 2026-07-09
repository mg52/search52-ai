package main

import (
	"log"
	"net/http"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/config"
	"github.com/mg52/search52-ai/internal/engine"
	"github.com/mg52/search52-ai/internal/handler"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	emb := ai.NewEmbeddingClient(cfg.EmbeddingBaseURL, cfg.EmbeddingAPIKey, cfg.EmbeddingModel)

	mgr := handler.NewManager(cfg.DataDir, emb, engine.Config{
		CategoryThreshold:   cfg.CategoryThreshold,
		MaxCategoriesPerDoc: cfg.MaxCategoriesPerDoc,
		MaxCategories:       cfg.MaxCategories,
		TopNCategories:      cfg.TopNCategories,
		VarianceThreshold:   cfg.VarianceThreshold,
		VarianceMinCount:    cfg.VarianceMinCount,
	})
	if err := mgr.LoadExisting(); err != nil {
		log.Fatalf("load indexes: %v", err)
	}

	addr := ":" + cfg.Port
	log.Printf("listening on %s (data dir %s)", addr, cfg.DataDir)
	if err := http.ListenAndServe(addr, mgr.Routes()); err != nil {
		log.Fatalf("server: %v", err)
	}
}
