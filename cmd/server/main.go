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

	deps := engine.Deps{Embedder: emb}
	if cfg.LLMBaseURL != "" {
		deps.LLMClient = ai.NewLLMClient(cfg.LLMBaseURL, cfg.LLMAPIKey, cfg.LLMModel)
		prompts, err := ai.LoadPrompts(cfg.PromptsDir)
		if err != nil {
			log.Fatalf("loading prompts from %s: %v", cfg.PromptsDir, err)
		}
		deps.Prompts = prompts
	} else {
		log.Printf("LLM_BASE_URL not set: the \"llm\" index kind is disabled")
	}

	mgr := handler.NewManager(cfg.DataDir, deps, engine.Config{
		CategoryThreshold:   cfg.CategoryThreshold,
		MaxCategoriesPerDoc: cfg.MaxCategoriesPerDoc,
		MaxCategories:       cfg.MaxCategories,
		TopNCategories:      cfg.TopNCategories,
		VarianceThreshold:   cfg.VarianceThreshold,
		VarianceMinCount:    cfg.VarianceMinCount,
		DisableSplit:        cfg.DisableSplit,
		MaxDocsPerCategory:  cfg.MaxDocsPerCategory,
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
