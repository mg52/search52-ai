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
	if c.MaxDocsPerCategory != 50 {
		t.Errorf("MaxDocsPerCategory = %d, want 50", c.MaxDocsPerCategory)
	}
	if c.PromptsDir != "./prompts" {
		t.Errorf("PromptsDir = %q, want ./prompts", c.PromptsDir)
	}
}

func TestLoadLLMConfigOptional(t *testing.T) {
	setRequired(t)
	for _, k := range []string{"LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"} {
		t.Setenv(k, "")
	}
	c, err := Load()
	if err != nil {
		t.Fatalf("LLM env vars should be optional: %v", err)
	}
	if c.LLMBaseURL != "" || c.LLMAPIKey != "" || c.LLMModel != "" {
		t.Errorf("expected empty LLM config, got %+v", c)
	}
}

func TestLoadLLMConfigOverrides(t *testing.T) {
	setRequired(t)
	t.Setenv("LLM_BASE_URL", "http://llm")
	t.Setenv("LLM_API_KEY", "k")
	t.Setenv("LLM_MODEL", "gpt")
	t.Setenv("PROMPTS_DIR", "/tmp/prompts")
	t.Setenv("MAX_DOCS_PER_CATEGORY", "25")

	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.LLMBaseURL != "http://llm" || c.LLMAPIKey != "k" || c.LLMModel != "gpt" {
		t.Errorf("LLM overrides not applied: %+v", c)
	}
	if c.PromptsDir != "/tmp/prompts" || c.MaxDocsPerCategory != 25 {
		t.Errorf("overrides not applied: %+v", c)
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
