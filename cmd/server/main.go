package main

import (
	"log"
	"net/http"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/config"
	"github.com/mg52/search52-ai/internal/handler"
	"github.com/mg52/search52-ai/internal/index"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	idx := index.New(ai.NewLLMClient(cfg.LLMBaseURL, cfg.LLMAPIKey, cfg.LLMModel), index.Options{
		MaxPromptLabels: cfg.MaxPromptLabels,
		MinLabels:       cfg.MinLabels,
		MaxLabels:       cfg.MaxLabels,
	})

	addr := ":" + cfg.Port
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, handler.New(idx).Routes()); err != nil {
		log.Fatalf("server: %v", err)
	}
}
