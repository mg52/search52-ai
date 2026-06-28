package config

import "testing"

func setRequired(t *testing.T) {
	t.Helper()
	t.Setenv("LLM_BASE_URL", "http://llm")
	t.Setenv("LLM_API_KEY", "k")
	t.Setenv("LLM_MODEL", "m")
	t.Setenv("EMBEDDING_BASE_URL", "http://emb")
	t.Setenv("EMBEDDING_API_KEY", "k")
	t.Setenv("EMBEDDING_MODEL", "m")
}

func TestLoadMissingRequired(t *testing.T) {
	for _, k := range []string{
		"LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
		"EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL",
	} {
		t.Setenv(k, "")
	}
	if _, err := Load(); err == nil {
		t.Fatal("expected error for missing required vars")
	}
}

func TestLoadDefaults(t *testing.T) {
	setRequired(t)
	// Force optional vars to empty so defaults apply deterministically.
	t.Setenv("PORT", "")
	t.Setenv("PROMPTS_DIR", "")
	t.Setenv("MAX_TAGS_PER_DOC", "")
	t.Setenv("TAG_MATCH_THRESHOLD", "")

	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Port != "8080" {
		t.Errorf("Port = %q, want 8080", c.Port)
	}
	if c.PromptsDir != "./prompts" {
		t.Errorf("PromptsDir = %q, want ./prompts", c.PromptsDir)
	}
	if c.MaxTagsPerDoc != 8 {
		t.Errorf("MaxTagsPerDoc = %d, want 8", c.MaxTagsPerDoc)
	}
	if c.TagMatchThreshold != 0.60 {
		t.Errorf("TagMatchThreshold = %v, want 0.60", c.TagMatchThreshold)
	}
}

func TestLoadOverrides(t *testing.T) {
	setRequired(t)
	t.Setenv("PORT", "9090")
	t.Setenv("PROMPTS_DIR", "/tmp/p")
	t.Setenv("MAX_TAGS_PER_DOC", "3")
	t.Setenv("TAG_MATCH_THRESHOLD", "0.42")

	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Port != "9090" || c.PromptsDir != "/tmp/p" || c.MaxTagsPerDoc != 3 || c.TagMatchThreshold != 0.42 {
		t.Errorf("overrides not applied: %+v", c)
	}
}

func TestLoadInvalidMaxTags(t *testing.T) {
	setRequired(t)
	t.Setenv("MAX_TAGS_PER_DOC", "not-a-number")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for invalid MAX_TAGS_PER_DOC")
	}
}

func TestLoadInvalidThreshold(t *testing.T) {
	setRequired(t)
	t.Setenv("TAG_MATCH_THRESHOLD", "not-a-float")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for invalid TAG_MATCH_THRESHOLD")
	}
}
