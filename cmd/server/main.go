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

	llm := ai.NewLLMClient(cfg.LLMBaseURL, cfg.LLMAPIKey, cfg.LLMModel)
	llm.Timeout, llm.MaxAttempts, llm.ReasoningEffort, llm.Temperature, llm.MaxTokens = cfg.LLMTimeout, cfg.LLMMaxAttempts, cfg.LLMReasoningEffort, cfg.LLMTemperature, cfg.LLMMaxTokens
	idx := index.New(llm, index.Options{
		MaxLabels:          cfg.MaxLabels,
		MaxLabelWords:      cfg.MaxLabelWords,
		MinLabelDocs:       cfg.MinLabelDocs,
		MaxResults:         cfg.MaxResults,
		MinRelativeScore:   cfg.MinRelativeScore,
		SendExistingLabels: cfg.SendExistingLabels,
	})

	addr := ":" + cfg.Port
	log.Printf("listening on %s", addr)
	if err := http.ListenAndServe(addr, handler.New(idx).Routes()); err != nil {
		log.Fatalf("server: %v", err)
	}
}
