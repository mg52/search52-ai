package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Port       string
	LLMBaseURL string
	LLMAPIKey  string // optional (e.g. local Ollama)
	LLMModel   string
	// MaxPromptLabels caps how many existing labels are shown to the LLM per call.
	MaxPromptLabels int
	// MinLabels and MaxLabels are how many labels the LLM is asked to return
	// per document or query; labels beyond MaxLabels are dropped.
	MinLabels, MaxLabels int
}

func Load() (*Config, error) {
	c := &Config{
		Port:            "8080",
		MaxPromptLabels: 1000,
		MinLabels:       5,
		MaxLabels:       10,
		LLMBaseURL:      os.Getenv("LLM_BASE_URL"),
		LLMAPIKey:       os.Getenv("LLM_API_KEY"),
		LLMModel:        os.Getenv("LLM_MODEL"),
	}
	if v := os.Getenv("PORT"); v != "" {
		c.Port = v
	}
	for name, dst := range map[string]*int{
		"MAX_PROMPT_LABELS": &c.MaxPromptLabels,
		"MIN_LABELS":        &c.MinLabels,
		"MAX_LABELS":        &c.MaxLabels,
	} {
		if v := os.Getenv(name); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("%s must be a positive integer, got %q", name, v)
			}
			*dst = n
		}
	}
	if c.MinLabels > c.MaxLabels {
		return nil, fmt.Errorf("MIN_LABELS (%d) must not exceed MAX_LABELS (%d)", c.MinLabels, c.MaxLabels)
	}
	if c.LLMBaseURL == "" || c.LLMModel == "" {
		return nil, fmt.Errorf("LLM_BASE_URL and LLM_MODEL are required")
	}
	return c, nil
}
