package ai

import (
	"strings"
	"testing"
)

func TestPrompts(t *testing.T) {
	for name, p := range map[string]func([]string, int, int) string{"index": IndexPrompt, "search": SearchPrompt} {
		got := p([]string{"audio", "earphones"}, 5, 10)
		if !strings.Contains(got, "audio, earphones") || strings.Contains(got, "{{") {
			t.Errorf("%s prompt did not render existing labels:\n%s", name, got)
		}
		if !strings.Contains(p(nil, 5, 10), "(none yet)") {
			t.Errorf("%s prompt should say the index is empty", name)
		}
		if !strings.Contains(got, "between 5 and 10 labels") {
			t.Errorf("%s prompt lacks the label count range:\n%s", name, got)
		}
		if strings.Contains(got, "JSON object") {
			t.Errorf("%s prompt still asks for JSON", name)
		}
	}
	if IndexPrompt(nil, 5, 10) == SearchPrompt(nil, 5, 10) {
		t.Error("index and search prompts must differ")
	}
}
