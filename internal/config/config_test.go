package config

import (
	"os"
	"testing"
	"time"
)

// setRequired sets the two required variables; every other variable is cleared
// so a value from the developer's shell cannot leak into a test.
func setRequired(t *testing.T) {
	t.Helper()
	for _, name := range []string{"PORT", "LLM_API_KEY", "LLM_TIMEOUT", "LLM_MAX_ATTEMPTS", "LLM_TEMPERATURE", "LLM_MAX_TOKENS", "SEND_EXISTING_LABELS", "MAX_LABELS", "MAX_LABEL_WORDS", "MIN_LABEL_DOCS", "MIN_RELATIVE_SCORE", "MAX_RESULTS"} {
		t.Setenv(name, "")
	}
	t.Setenv("LLM_BASE_URL", "http://llm")
	t.Setenv("LLM_MODEL", "m")
	t.Setenv("LLM_REASONING_EFFORT", "") // registers the restore; "unset" and "empty" differ, so unset it
	os.Unsetenv("LLM_REASONING_EFFORT")
}

func TestLoadMissingRequired(t *testing.T) {
	setRequired(t)
	for _, name := range []string{"LLM_BASE_URL", "LLM_MODEL"} {
		t.Run(name, func(t *testing.T) {
			setRequired(t)
			t.Setenv(name, "")
			if _, err := Load(); err == nil {
				t.Fatalf("expected error for missing %s", name)
			}
		})
	}
}

func TestLoadDefaults(t *testing.T) {
	setRequired(t)
	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want := Config{
		Port: "8080", LLMBaseURL: "http://llm", LLMModel: "m",
		LLMTimeout: 10 * time.Minute, LLMMaxAttempts: 3, LLMReasoningEffort: "none", LLMTemperature: 0, LLMMaxTokens: 300, SendExistingLabels: false,
		MaxLabels: 50, MaxLabelWords: 1, MinLabelDocs: 5, MinRelativeScore: 0.2, MaxResults: 100,
	}
	if *c != want {
		t.Errorf("defaults = %+v\nwant       %+v", *c, want)
	}
}

func TestLoadOverrides(t *testing.T) {
	setRequired(t)
	for name, value := range map[string]string{
		"PORT": "9000", "LLM_API_KEY": "k", "LLM_TIMEOUT": "90s", "LLM_MAX_ATTEMPTS": "5",
		"LLM_REASONING_EFFORT": "low", "LLM_TEMPERATURE": "0.3", "LLM_MAX_TOKENS": "500", "SEND_EXISTING_LABELS": "true", "MAX_LABELS": "30", "MAX_LABEL_WORDS": "2",
		"MIN_LABEL_DOCS": "20", "MIN_RELATIVE_SCORE": "0.35", "MAX_RESULTS": "10",
	} {
		t.Setenv(name, value)
	}
	c, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want := Config{
		Port: "9000", LLMBaseURL: "http://llm", LLMModel: "m", LLMAPIKey: "k",
		LLMTimeout: 90 * time.Second, LLMMaxAttempts: 5, LLMReasoningEffort: "low", LLMTemperature: 0.3, LLMMaxTokens: 500, SendExistingLabels: true,
		MaxLabels: 30, MaxLabelWords: 2, MinLabelDocs: 20, MinRelativeScore: 0.35, MaxResults: 10,
	}
	if *c != want {
		t.Errorf("config = %+v\nwant     %+v", *c, want)
	}
}

// LLM_REASONING_EFFORT set to an empty string is kept, so the field is omitted.
func TestLoadEmptyReasoningEffort(t *testing.T) {
	setRequired(t)
	t.Setenv("LLM_REASONING_EFFORT", "")
	c, err := Load()
	if err != nil || c.LLMReasoningEffort != "" {
		t.Fatalf("Load = %+v, %v; want empty LLMReasoningEffort", c, err)
	}
}

func TestLoadInvalid(t *testing.T) {
	for name, bad := range map[string][]string{
		"LLM_MAX_ATTEMPTS":     {"0", "-1", "abc"},
		"MAX_LABELS":           {"0", "-1", "abc"},
		"LLM_MAX_TOKENS":       {"0", "-1", "abc"},
		"MAX_LABEL_WORDS":      {"0", "abc"},
		"MIN_LABEL_DOCS":       {"0", "abc"},
		"MAX_RESULTS":          {"0", "abc"},
		"LLM_TIMEOUT":          {"0s", "-5s", "10", "abc"},
		"MIN_RELATIVE_SCORE":   {"-0.1", "1.5", "abc"},
		"LLM_TEMPERATURE":      {"-0.1", "2.5", "abc"},
		"SEND_EXISTING_LABELS": {"maybe", "2", "yes please"},
	} {
		for _, v := range bad {
			setRequired(t)
			t.Setenv(name, v)
			if _, err := Load(); err == nil {
				t.Errorf("%s=%q: expected error", name, v)
			}
		}
	}
}

func TestMinRelativeScoreBounds(t *testing.T) {
	for _, v := range []string{"0", "1"} {
		setRequired(t)
		t.Setenv("MIN_RELATIVE_SCORE", v)
		if _, err := Load(); err != nil {
			t.Errorf("MIN_RELATIVE_SCORE=%s: %v", v, err)
		}
	}
}
