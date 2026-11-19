package ai

import (
	"strings"
	"testing"
)

func TestPrompt(t *testing.T) {
	got := Prompt(5, 1)
	if strings.Contains(got, "{{") {
		t.Errorf("prompt not rendered:\n%s", got)
	}
	for _, rule := range []string{"Return from 1 to 5 labels", "do not have to reach 5", "exactly one word", "simple", "in English"} {
		if !strings.Contains(got, rule) {
			t.Errorf("prompt lacks %q:\n%s", rule, got)
		}
	}
}

// The LLM is asked for 1 to the configured number, is told it need not reach it
// and is never nudged to give more, and the example answer has 5 labels.
func TestPromptLabelCount(t *testing.T) {
	got := Prompt(50, 1)
	if !strings.Contains(got, "Return from 1 to 50 labels") {
		t.Errorf("prompt lacks the label count:\n%s", got)
	}
	for _, bad := range []string{"as many", "More labels", "more labels is better", "at most"} {
		if strings.Contains(got, bad) {
			t.Errorf("prompt must not say %q:\n%s", bad, got)
		}
	}
	_, example, ok := strings.Cut(got, "Example output: ")
	if !ok || len(strings.Split(example, ",")) != 5 {
		t.Errorf("example output should have 5 labels, got %q", example)
	}
}

// Broad categories come before the specific labels.
func TestPromptOrder(t *testing.T) {
	got := Prompt(5, 1)
	broad, specific := strings.Index(got, "First the broad categories"), strings.Index(got, "Then the specific and detailed")
	if broad < 0 || specific < broad {
		t.Errorf("prompt must put broad categories before specific labels (%d, %d):\n%s", broad, specific, got)
	}
}

// The word rule follows the configured limit.
func TestPromptWordRule(t *testing.T) {
	for maxWords, want := range map[int]string{2: "1 to 2 words", 0: "short word or phrase"} {
		got := Prompt(5, maxWords)
		if !strings.Contains(got, want) || strings.Contains(got, "exactly one word") || strings.Contains(got, "{{") {
			t.Errorf("Prompt(5, %d) lacks %q:\n%s", maxWords, want, got)
		}
	}
}

// PromptWithLabels is Prompt plus the existing labels, in the order given, and
// Prompt itself never mentions them.
func TestPromptWithLabels(t *testing.T) {
	base := Prompt(5, 1)
	if strings.Contains(base, "already used") {
		t.Errorf("Prompt must not mention existing labels:\n%s", base)
	}

	got := PromptWithLabels(5, 1, []string{"audio", "kitchen", "tea"})
	if !strings.HasPrefix(got, base) {
		t.Errorf("PromptWithLabels should start with Prompt:\n%s", got)
	}
	if !strings.Contains(got, "most used first:\naudio, kitchen, tea\n") || strings.Contains(got, "{{") {
		t.Errorf("existing labels not rendered:\n%s", got)
	}
	if empty := PromptWithLabels(5, 1, nil); !strings.Contains(empty, "(none yet)") {
		t.Errorf("an empty index should say so:\n%s", empty)
	}
}
