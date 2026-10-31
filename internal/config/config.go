package config

import (
	"fmt"
	"os"
)

type Config struct {
	Port       string
	LLMBaseURL string
	LLMAPIKey  string // optional (e.g. local Ollama)
	LLMModel   string
}

func Load() (*Config, error) {
	c := &Config{
		Port:       "8080",
		LLMBaseURL: os.Getenv("LLM_BASE_URL"),
		LLMAPIKey:  os.Getenv("LLM_API_KEY"),
		LLMModel:   os.Getenv("LLM_MODEL"),
	}
	if v := os.Getenv("PORT"); v != "" {
		c.Port = v
	}
	if c.LLMBaseURL == "" || c.LLMModel == "" {
		return nil, fmt.Errorf("LLM_BASE_URL and LLM_MODEL are required")
	}
	return c, nil
}
