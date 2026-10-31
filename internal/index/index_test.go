package index

import (
	"context"
	"fmt"
	"reflect"
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

func newTestIndex() *Index {
	return New(fakeLLM{
		"bose earbuds":    "```json\n{\"labels\": [\"Audio\", \"earphones\", \"audio\", \" \"]}\n```",
		"sony headphones": `{"labels": ["audio", "headphones"]}`,
		"coffee beans":    `{"labels": ["coffee"]}`,
		"earbuds query":   `{"labels": ["earphones", "audio"]}`,
		"junk":            `{"labels": []}`,
	})
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
	if len(docs) != 2 || docs[0].ID != "a" || docs[1].ID != "b" || docs[0].Content != "bose earbuds" {
		t.Errorf("docs = %+v", docs)
	}
	if !reflect.DeepEqual(docs[0].Labels, []string{"audio", "earphones"}) {
		t.Errorf("stored labels = %v, want cleaned [audio earphones]", docs[0].Labels)
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
