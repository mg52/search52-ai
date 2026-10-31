package config

import "testing"

func TestLoadMissingRequired(t *testing.T) {
	t.Setenv("LLM_BASE_URL", "")
	t.Setenv("LLM_MODEL", "m")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for missing LLM_BASE_URL")
	}
}

func TestLoadDefaultsAndOptionalKey(t *testing.T) {
	t.Setenv("LLM_BASE_URL", "http://llm")
	t.Setenv("LLM_MODEL", "m")
	t.Setenv("LLM_API_KEY", "")
	t.Setenv("PORT", "")
	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Port != "8080" {
		t.Errorf("Port = %q, want 8080", c.Port)
	}
}

func TestLoadPortOverride(t *testing.T) {
	t.Setenv("LLM_BASE_URL", "http://llm")
	t.Setenv("LLM_MODEL", "m")
	t.Setenv("PORT", "9000")
	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.Port != "9000" {
		t.Errorf("Port = %q, want 9000", c.Port)
	}
}
