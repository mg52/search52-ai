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

	// Default categorization / search tuning for new indexes.
	CategoryThreshold   float64 // min cosine to join an existing category
	MaxCategoriesPerDoc int     // cap on categories one document may join
	MaxCategories       int     // cap on total categories
	TopNCategories      int     // nearest categories scanned per query
	VarianceThreshold   float64 // Welford variance above which a category's ShouldSplit flips true
	VarianceMinCount    int     // min category member count before ShouldSplit can fire
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
