package config

import "testing"

func setRequired(t *testing.T) {
	t.Helper()
	t.Setenv("EMBEDDING_BASE_URL", "http://emb")
	t.Setenv("EMBEDDING_API_KEY", "k")
	t.Setenv("EMBEDDING_MODEL", "m")
}

func TestLoadMissingRequired(t *testing.T) {
	for _, k := range []string{"EMBEDDING_BASE_URL", "EMBEDDING_MODEL"} {
		t.Setenv(k, "")
	}
	if _, err := Load(); err == nil {
		t.Fatal("expected error for missing required vars")
	}
}

func TestLoadAPIKeyOptional(t *testing.T) {
	t.Setenv("EMBEDDING_BASE_URL", "http://emb")
	t.Setenv("EMBEDDING_MODEL", "m")
	t.Setenv("EMBEDDING_API_KEY", "") // optional (local Ollama)
	if _, err := Load(); err != nil {
		t.Fatalf("API key should be optional: %v", err)
	}
}

func TestLoadDefaults(t *testing.T) {
	setRequired(t)
	for _, k := range []string{"PORT", "CATEGORY_THRESHOLD", "MAX_CATEGORIES_PER_DOC", "MAX_CATEGORIES", "TOP_N_CATEGORIES"} {
		t.Setenv(k, "")
	}
	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Port != "8080" {
		t.Errorf("Port = %q, want 8080", c.Port)
	}
	if c.CategoryThreshold != 0.60 {
		t.Errorf("CategoryThreshold = %v, want 0.60", c.CategoryThreshold)
	}
	if c.MaxCategoriesPerDoc != 3 || c.MaxCategories != 100 || c.TopNCategories != 3 {
		t.Errorf("category defaults wrong: %+v", c)
	}
}

func TestLoadOverrides(t *testing.T) {
	setRequired(t)
	t.Setenv("PORT", "9090")
	t.Setenv("CATEGORY_THRESHOLD", "0.42")
	t.Setenv("MAX_CATEGORIES_PER_DOC", "5")
	t.Setenv("MAX_CATEGORIES", "10")
	t.Setenv("TOP_N_CATEGORIES", "7")

	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Port != "9090" || c.CategoryThreshold != 0.42 || c.MaxCategoriesPerDoc != 5 || c.MaxCategories != 10 || c.TopNCategories != 7 {
		t.Errorf("overrides not applied: %+v", c)
	}
}

func TestLoadInvalidThreshold(t *testing.T) {
	setRequired(t)
	t.Setenv("CATEGORY_THRESHOLD", "not-a-float")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for invalid CATEGORY_THRESHOLD")
	}
}

func TestLoadInvalidMaxCategories(t *testing.T) {
	setRequired(t)
	t.Setenv("MAX_CATEGORIES", "not-a-number")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for invalid MAX_CATEGORIES")
	}
}
