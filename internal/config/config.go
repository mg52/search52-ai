package config

import (
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	EmbeddingBaseURL string
	EmbeddingAPIKey  string
	EmbeddingModel   string
	Port             string
	DataDir          string // where per-index snapshots live

	// LLM index support. Optional — only required if an index of kind "llm"
	// is actually created; LLMBaseURL == "" means no LLM client is wired up.
	LLMBaseURL string
	LLMAPIKey  string
	LLMModel   string
	PromptsDir string // fixed tagging/description prompt templates for the LLM index

	// Default categorization / search tuning for new indexes.
	CategoryThreshold   float64 // min cosine to join an existing category (embedding index)
	MaxCategoriesPerDoc int     // cap on categories one document may join
	MaxCategories       int     // cap on total categories
	TopNCategories      int     // nearest categories scanned per query
	VarianceThreshold   float64 // Welford variance above which a category's ShouldSplit flips true (embedding index)
	VarianceMinCount    int     // min category member count before ShouldSplit can fire (embedding index)
	DisableSplit        bool    // turns off automatic category splitting for new indexes (embedding index)
	MaxDocsPerCategory  int     // docs returned per matched category in Search (llm index)
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

	c.EmbeddingBaseURL = getRequired("EMBEDDING_BASE_URL")
	c.EmbeddingAPIKey = os.Getenv("EMBEDDING_API_KEY") // optional (e.g. local Ollama)
	c.EmbeddingModel = getRequired("EMBEDDING_MODEL")

	if len(missing) > 0 {
		return nil, fmt.Errorf("missing required environment variables: %v", missing)
	}

	c.Port = getEnvDefault("PORT", "8080")
	c.DataDir = getEnvDefault("DATA_DIR", "./data")

	c.LLMBaseURL = os.Getenv("LLM_BASE_URL") // optional: unset disables the "llm" index kind
	c.LLMAPIKey = os.Getenv("LLM_API_KEY")
	c.LLMModel = os.Getenv("LLM_MODEL")
	c.PromptsDir = getEnvDefault("PROMPTS_DIR", "./prompts")

	var err error
	if c.CategoryThreshold, err = parseFloat("CATEGORY_THRESHOLD", "0.60"); err != nil {
		return nil, err
	}
	if c.MaxCategoriesPerDoc, err = parseInt("MAX_CATEGORIES_PER_DOC", "3"); err != nil {
		return nil, err
	}
	if c.MaxCategories, err = parseInt("MAX_CATEGORIES", "100"); err != nil {
		return nil, err
	}
	if c.TopNCategories, err = parseInt("TOP_N_CATEGORIES", "3"); err != nil {
		return nil, err
	}
	if c.VarianceThreshold, err = parseFloat("VARIANCE_THRESHOLD", "0.02"); err != nil {
		return nil, err
	}
	if c.VarianceMinCount, err = parseInt("VARIANCE_MIN_COUNT", "100"); err != nil {
		return nil, err
	}
	if c.DisableSplit, err = parseBool("DISABLE_SPLIT", "false"); err != nil {
		return nil, err
	}
	if c.MaxDocsPerCategory, err = parseInt("MAX_DOCS_PER_CATEGORY", "50"); err != nil {
		return nil, err
	}

	return c, nil
}

func getEnvDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func parseInt(key, def string) (int, error) {
	n, err := strconv.Atoi(getEnvDefault(key, def))
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", key, err)
	}
	return n, nil
}

func parseFloat(key, def string) (float64, error) {
	f, err := strconv.ParseFloat(getEnvDefault(key, def), 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", key, err)
	}
	return f, nil
}

func parseBool(key, def string) (bool, error) {
	b, err := strconv.ParseBool(getEnvDefault(key, def))
	if err != nil {
		return false, fmt.Errorf("invalid %s: %w", key, err)
	}
	return b, nil
}
