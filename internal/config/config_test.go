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
	if c.MaxPromptLabels != 1000 || c.MinLabels != 5 || c.MaxLabels != 10 {
		t.Errorf("defaults = %d/%d/%d, want 1000/5/10", c.MaxPromptLabels, c.MinLabels, c.MaxLabels)
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

func TestLoadMaxPromptLabels(t *testing.T) {
	t.Setenv("LLM_BASE_URL", "http://llm")
	t.Setenv("LLM_MODEL", "m")
	t.Setenv("MAX_PROMPT_LABELS", "50")
	if c, err := Load(); err != nil || c.MaxPromptLabels != 50 {
		t.Fatalf("Load = %+v, %v; want MaxPromptLabels 50", c, err)
	}
	for _, bad := range []string{"0", "-1", "abc"} {
		t.Setenv("MAX_PROMPT_LABELS", bad)
		if _, err := Load(); err == nil {
			t.Errorf("MAX_PROMPT_LABELS=%q: expected error", bad)
		}
	}
}

func TestLoadLabelRange(t *testing.T) {
	t.Setenv("LLM_BASE_URL", "http://llm")
	t.Setenv("LLM_MODEL", "m")
	t.Setenv("MIN_LABELS", "2")
	t.Setenv("MAX_LABELS", "4")
	if c, err := Load(); err != nil || c.MinLabels != 2 || c.MaxLabels != 4 {
		t.Fatalf("Load = %+v, %v; want 2..4", c, err)
	}
	for _, kv := range [][2]string{{"MIN_LABELS", "0"}, {"MAX_LABELS", "x"}, {"MIN_LABELS", "11"}} {
		t.Setenv(kv[0], kv[1])
		if _, err := Load(); err == nil {
			t.Errorf("%s=%s: expected error", kv[0], kv[1])
		}
	}
}
