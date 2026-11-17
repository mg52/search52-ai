package index

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// fakeLLM answers with the labels registered for the exact input text.
type fakeLLM map[string]string

func (f fakeLLM) Complete(_ context.Context, _, user string) (string, error) {
	out, ok := f[user]
	if !ok {
		return "", fmt.Errorf("unexpected input %q", user)
	}
	return out, nil
}

var testOpts = Options{MaxPromptLabels: 1000, MinLabels: 5, MaxLabels: 10}

func newTestIndex() *Index {
	return New(fakeLLM{
		"bose earbuds":    "```\nAudio, earphones, audio, , \"Noise Cancelling.\"\n```",
		"sony headphones": "audio, headphones",
		"coffee beans":    "coffee",
		"earbuds query":   "earphones, audio",
		"junk":            " , ",
	}, testOpts)
}

func TestAddSearch(t *testing.T) {
	idx := newTestIndex()
	ctx := context.Background()
	for id, content := range map[string]string{"a": "bose earbuds", "b": "sony headphones", "c": "coffee beans"} {
		if _, err := idx.Add(ctx, id, content); err != nil {
			t.Fatalf("Add %s: %v", id, err)
		}
	}

	docs, labels, err := idx.Search(ctx, "earbuds query")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if !reflect.DeepEqual(labels, []string{"earphones", "audio"}) {
		t.Errorf("labels = %v", labels)
	}
	// "a" shares two labels, "b" one, "c" none.
	if len(docs) != 2 || docs[0].ID != "a" || docs[1].ID != "b" || docs[0].Content != "bose earbuds" ||
		docs[0].Score != 2 || docs[1].Score != 1 {
		t.Errorf("docs = %+v", docs)
	}
	if !reflect.DeepEqual(docs[0].Labels, []string{"audio", "earphones", "noise cancelling"}) {
		t.Errorf("stored labels = %v, want cleaned [audio earphones noise cancelling]", docs[0].Labels)
	}
}

func TestAddReplacesOldLabels(t *testing.T) {
	idx := newTestIndex()
	ctx := context.Background()
	idx.Add(ctx, "a", "bose earbuds")
	idx.Add(ctx, "a", "coffee beans")

	if len(idx.byLabel["earphones"]) != 0 || len(idx.byLabel["coffee"]) != 1 {
		t.Errorf("byLabel = %v", idx.byLabel)
	}
}

func TestRemove(t *testing.T) {
	idx := newTestIndex()
	ctx := context.Background()
	idx.Add(ctx, "a", "bose earbuds")

	if !idx.Remove("a") || idx.Remove("a") {
		t.Fatal("Remove should report true once, then false")
	}
	if len(idx.docs) != 0 || len(idx.byLabel) != 0 {
		t.Errorf("index not empty: %v %v", idx.docs, idx.byLabel)
	}
}

func TestAddLLMFailures(t *testing.T) {
	idx := newTestIndex()
	ctx := context.Background()
	for _, content := range []string{"junk", "unknown"} {
		if _, err := idx.Add(ctx, "x", content); err == nil {
			t.Errorf("Add(%q): expected error", content)
		}
	}
	if len(idx.docs) != 0 {
		t.Error("failed Add must not store a document")
	}
}

func TestParseLabels(t *testing.T) {
	tests := []struct {
		raw  string
		want []string
	}{
		{"label1, label2", []string{"label1", "label2"}},
		{"  Audio ,SOUND,audio,, ", []string{"audio", "sound"}},
		{"audio\nsound\n", []string{"audio", "sound"}},
		{"```text\naudio, sound\n```", []string{"audio", "sound"}},
		{`"audio", 'sound', ` + "`earphones`.", []string{"audio", "sound", "earphones"}},
		{"- audio\n* sound\n• earphones", []string{"audio", "sound", "earphones"}},
		{"noise cancelling earphones", []string{"noise cancelling earphones"}},
		{"", []string{}},
		{" , \n", []string{}},
	}
	for _, tc := range tests {
		if got := parseLabels(tc.raw); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("parseLabels(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

// promptLLM records the system prompt of every call and always answers "audio".
type promptLLM struct{ systems []string }

func (p *promptLLM) Complete(_ context.Context, system, _ string) (string, error) {
	p.systems = append(p.systems, system)
	return "audio, earphones", nil
}

// Both prompts must carry the labels already in the index, and differ per task.
func TestPromptsIncludeExistingLabels(t *testing.T) {
	llm := &promptLLM{}
	idx := New(llm, testOpts)
	ctx := context.Background()

	idx.Add(ctx, "a", "first")
	idx.Add(ctx, "b", "second")
	idx.Search(ctx, "query")

	if !strings.Contains(llm.systems[0], "(none yet)") {
		t.Errorf("first index prompt should say the index is empty:\n%s", llm.systems[0])
	}
	for i, name := range map[int]string{1: "index", 2: "search"} {
		if !strings.Contains(llm.systems[i], "audio, earphones") {
			t.Errorf("%s prompt lacks existing labels:\n%s", name, llm.systems[i])
		}
	}
	if llm.systems[1] == llm.systems[2] {
		t.Error("index and search must use different prompts")
	}
}

// A document having all of the query's labels outranks ones having fewer,
// whatever their ids.
func TestSearchScore(t *testing.T) {
	llm := fakeLLM{"d1": "x", "d2": "x, y", "d150": "x, y, z", "q": "x, y, z"}
	idx := New(llm, testOpts)
	ctx := context.Background()
	for _, id := range []string{"d150", "d2", "d1"} {
		idx.Add(ctx, id, id)
	}
	res, _, err := idx.Search(ctx, "q")
	if err != nil {
		t.Fatal(err)
	}
	got := fmt.Sprintf("%s %d %s %d %s %d", res[0].ID, res[0].Score, res[1].ID, res[1].Score, res[2].ID, res[2].Score)
	if got != "d150 3 d2 2 d1 1" {
		t.Errorf("got %s", got)
	}
}

// The prompt shows the most used labels first and at most maxPromptLabels.
func TestExistingLabelsCapped(t *testing.T) {
	llm := fakeLLM{"a": "rare, common", "b": "common, other"}
	idx := New(llm, Options{MaxPromptLabels: 2})
	ctx := context.Background()
	idx.Add(ctx, "a", "a")
	idx.Add(ctx, "b", "b")
	if got := idx.existingLabels(); !reflect.DeepEqual(got, []string{"common", "other"}) {
		t.Errorf("existingLabels = %v, want [common other]", got)
	}
	idx.opts.MaxPromptLabels = 0
	if got := idx.existingLabels(); len(got) != 3 {
		t.Errorf("no cap: got %v, want 3 labels", got)
	}
}

// The LLM is asked for MinLabels to MaxLabels labels and extras are dropped.
func TestLabelCountRange(t *testing.T) {
	llm := &promptLLM{}
	idx := New(llm, Options{MinLabels: 2, MaxLabels: 3})
	doc, err := idx.Add(context.Background(), "a", "x")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(llm.systems[0], "between 2 and 3 labels") {
		t.Errorf("prompt lacks range:\n%s", llm.systems[0])
	}
	if len(doc.Labels) != 2 { // promptLLM answers "audio, earphones"; cap is 3
		t.Errorf("labels = %v", doc.Labels)
	}

	idx = New(fakeLLM{"x": "a, b, c, d, e"}, Options{MaxLabels: 3})
	if doc, _ := idx.Add(context.Background(), "a", "x"); !reflect.DeepEqual(doc.Labels, []string{"a", "b", "c"}) {
		t.Errorf("labels = %v, want first 3", doc.Labels)
	}
}
