package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	LLMBaseURL        string
	LLMAPIKey         string
	LLMModel          string
	EmbeddingBaseURL  string
	EmbeddingAPIKey   string
	EmbeddingModel    string
	Port              string
	MaxTagsPerDoc     int
	TagMatchThreshold float64
	PromptsDir        string
}

func Load() (*Config, error) {
	c := &Config{}

	var missing []string
	getRequired := func(key string) string {
		v := os.Getenv(key)
		if v == "" {
			missing = append(missing, key)
		}
		return v
	}

	c.LLMBaseURL = getRequired("LLM_BASE_URL")
	c.LLMAPIKey = getRequired("LLM_API_KEY")
	c.LLMModel = getRequired("LLM_MODEL")
	c.EmbeddingBaseURL = getRequired("EMBEDDING_BASE_URL")
	c.EmbeddingAPIKey = getRequired("EMBEDDING_API_KEY")
	c.EmbeddingModel = getRequired("EMBEDDING_MODEL")

	if len(missing) > 0 {
		return nil, fmt.Errorf("missing required environment variables: %v", missing)
	}

	c.Port = getEnvDefault("PORT", "8080")
	c.PromptsDir = getEnvDefault("PROMPTS_DIR", "./prompts")

	var err error
	c.MaxTagsPerDoc, err = strconv.Atoi(getEnvDefault("MAX_TAGS_PER_DOC", "8"))
	if err != nil {
		return nil, fmt.Errorf("invalid MAX_TAGS_PER_DOC: %w", err)
	}

	c.TagMatchThreshold, err = strconv.ParseFloat(getEnvDefault("TAG_MATCH_THRESHOLD", "0.60"), 64)
	if err != nil {
		return nil, fmt.Errorf("invalid TAG_MATCH_THRESHOLD: %w", err)
	}

	return c, nil
}

func getEnvDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
