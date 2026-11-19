package index

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/mg52/search52-ai/internal/ai"
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

var testOpts = Options{MaxLabels: 50, MaxLabelWords: 1, MinLabelDocs: 5, MaxResults: 100, MinRelativeScore: 0.2}

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
	if !reflect.DeepEqual(labels, []string{"earphone", "audio"}) {
		t.Errorf("labels = %v", labels)
	}
	// "a" shares two labels, "b" one, "c" none.
	// audio has 2 docs, earphone 1: total 3; both count as 5 docs, so each is worth 0.6.
	if len(docs) != 2 || docs[0].ID != "a" || docs[1].ID != "b" || docs[0].Content != "bose earbuds" ||
		docs[0].Score != 1.2 || docs[1].Score != 0.6 {
		t.Errorf("docs = %+v", docs)
	}
	if want := []string{"audio", "earphone"}; !reflect.DeepEqual(idx.byID["a"], want) {
		t.Errorf("stored labels = %v, want %v", idx.byID["a"], want)
	}
}

func TestAddReplacesOldLabels(t *testing.T) {
	idx := newTestIndex()
	ctx := context.Background()
	idx.Add(ctx, "a", "bose earbuds")
	idx.Add(ctx, "a", "coffee beans")

	if len(idx.byLabel["earphone"]) != 0 || len(idx.byLabel["coffee"]) != 1 {
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
	if len(idx.docs) != 0 || len(idx.byID) != 0 || len(idx.byLabel) != 0 {
		t.Errorf("index not empty: %v %v %v", idx.docs, idx.byID, idx.byLabel)
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
		{`"audio", 'sound', ` + "`earphones`.", []string{"audio", "sound", "earphone"}},
		{"- audio\n* sound\n• earphones", []string{"audio", "sound", "earphone"}},
		// more than one word (split by spaces, underscores or hyphens): ignored
		{"Noise Cancelling, noise_cancelling, noise-cancelling, usb-c, kettle", []string{"kettle"}},
		// plural endings are dropped
		{"Waffles, waffle, batteries, wireless speakers", []string{"waffle", "battery"}},
		{"glass, bus, analysis, status", []string{"glass", "bus", "analysis", "status"}},
		{"", []string{}},
		{" , \n", []string{}},
	}
	for _, tc := range tests {
		if got := parseLabels(tc.raw, 1); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("parseLabels(%q) = %q, want %q", tc.raw, got, tc.want)
		}
	}
}

// promptLLM records the system prompt of every call and always answers "audio, earphones".
type promptLLM struct{ systems []string }

func (p *promptLLM) Complete(_ context.Context, system, _ string) (string, error) {
	p.systems = append(p.systems, system)
	return "audio, earphones", nil
}

// Rare labels weigh more. label1 has 100 docs, label2 200, label4 150 and the
// query also has label8, which is not in the index and is ignored: total is
// 450. Doc 42 (label2+label4) scores 450/200 + 450/150 = 5.25, doc 43
// (label1 only) 450/100 = 4.5.
func TestSearchScore(t *testing.T) {
	llm := fakeLLM{"q": "label2, label4, label1, label8", "42": "label2, label4", "43": "label1"}
	for i := 0; i < 99; i++ {
		llm[fmt.Sprint("a", i)] = "label1"
	}
	for i := 0; i < 199; i++ {
		llm[fmt.Sprint("b", i)] = "label2"
	}
	for i := 0; i < 149; i++ {
		llm[fmt.Sprint("c", i)] = "label4"
	}
	for i := 0; i < 50; i++ {
		llm[fmt.Sprint("d", i)] = "label3" // not in the query
	}
	idx := New(llm, testOpts)
	ctx := context.Background()
	for id := range llm {
		if id != "q" {
			idx.Add(ctx, id, id)
		}
	}
	res, _, err := idx.Search(ctx, "q")
	if err != nil {
		t.Fatal(err)
	}
	// 449 docs match (doc 42 is in two labels; label3's docs are not in the
	// query), but only the first MaxResults are returned.
	if len(res) != testOpts.MaxResults {
		t.Fatalf("got %d results, want %d", len(res), testOpts.MaxResults)
	}
	got := fmt.Sprintf("%s %v %s %v %s %v", res[0].ID, res[0].Score, res[1].ID, res[1].Score, res[2].ID, res[2].Score)
	if got != "42 5.25 43 4.5 a0 4.5" {
		t.Errorf("got %s", got)
	}
}

// The LLM is asked for MaxLabels labels, and if it returns more only the first
// MaxLabels are kept (labels dropped for being too long do not use up a slot);
// 0 keeps all.
func TestMaxLabels(t *testing.T) {
	llm := &promptLLM{}
	if _, err := New(llm, Options{MaxLabels: 3, MaxLabelWords: 1}).Add(context.Background(), "a", "x"); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(llm.systems[0], "Return from 1 to 3 labels") {
		t.Errorf("prompt lacks the label count:\n%s", llm.systems[0])
	}

	idx := New(fakeLLM{"x": "a, two words, b, c, d, e"}, Options{MaxLabels: 3, MaxLabelWords: 1})
	if labels, _ := idx.Add(context.Background(), "a", "x"); !reflect.DeepEqual(labels, []string{"a", "b", "c"}) {
		t.Errorf("labels = %v, want the first 3 one-word labels", labels)
	}
	idx = New(fakeLLM{"x": "a, b, c, d, e"}, Options{})
	if labels, _ := idx.Add(context.Background(), "a", "x"); len(labels) != 5 {
		t.Errorf("MaxLabels 0 should keep all, got %v", labels)
	}
}

// Documents and queries share one prompt, and it never lists existing labels.
func TestSinglePrompt(t *testing.T) {
	llm := &promptLLM{}
	idx := New(llm, testOpts)
	ctx := context.Background()
	idx.Add(ctx, "a", "first")
	idx.Add(ctx, "b", "second")
	idx.Search(ctx, "query")
	if llm.systems[0] != llm.systems[1] || llm.systems[1] != llm.systems[2] || strings.Contains(llm.systems[2], "already used") {
		t.Errorf("prompts differ or leak labels: %q", llm.systems)
	}
}

// Add, Search and Remove running at once must not race (run with -race).
func TestConcurrentAccess(t *testing.T) {
	idx := New(fakeLLM{"doc": "x, y", "q": "x, y"}, testOpts)
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		id := fmt.Sprint("d", i)
		wg.Add(3)
		go func() { defer wg.Done(); idx.Add(ctx, id, "doc") }()
		go func() { defer wg.Done(); idx.Search(ctx, "q") }()
		go func() { defer wg.Done(); idx.Remove(id) }()
	}
	wg.Wait()
}

// A label that only a doc or two has weighs like one with minLabelDocs docs, so
// it cannot dominate: x (1 doc) and y (5 docs) are worth the same.
func TestWeightFloor(t *testing.T) {
	llm := fakeLLM{"q": "x, y", "p": "x"}
	for i := 0; i < 5; i++ {
		llm[fmt.Sprint("y", i)] = "y"
	}
	idx := New(llm, testOpts)
	ctx := context.Background()
	for id := range llm {
		if id != "q" {
			idx.Add(ctx, id, id)
		}
	}
	res, _, err := idx.Search(ctx, "q")
	if err != nil {
		t.Fatal(err)
	}
	// total = 1 + 5 = 6; x counts as 5 docs, so both labels weigh 6/5 = 1.2.
	for _, r := range res {
		if r.Score != 1.2 {
			t.Errorf("%s score = %v, want 1.2", r.ID, r.Score)
		}
	}
}

// Search returns at most Options.MaxResults documents.
func TestSearchLimit(t *testing.T) {
	llm := fakeLLM{"q": "x"}
	for i := 0; i < testOpts.MaxResults+50; i++ {
		llm[fmt.Sprint("d", i)] = "x"
	}
	idx := New(llm, testOpts)
	ctx := context.Background()
	for id := range llm {
		if id != "q" {
			idx.Add(ctx, id, id)
		}
	}
	res, _, err := idx.Search(ctx, "q")
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != testOpts.MaxResults {
		t.Errorf("got %d results, want %d", len(res), testOpts.MaxResults)
	}
}

// Results under MinRelativeScore (20%) of the best score are dropped. All six
// labels have fewer than 5 docs, so each weighs the same (9/5): "top" has all
// six (100%), "keep" two (33%), "drop" one (16.7%).
func TestRelativeScoreCutoff(t *testing.T) {
	idx := New(fakeLLM{"top": "a, b, c, d, e, f", "keep": "a, b", "drop": "a", "q": "a, b, c, d, e, f"}, testOpts)
	ctx := context.Background()
	for _, id := range []string{"top", "keep", "drop"} {
		idx.Add(ctx, id, id)
	}
	res, _, err := idx.Search(ctx, "q")
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 || res[0].ID != "top" || res[1].ID != "keep" {
		t.Errorf("results = %+v, want [top keep]", res)
	}
}

// MaxLabelWords decides which labels are kept; <= 0 keeps all.
func TestParseLabelsMaxWords(t *testing.T) {
	raw := "kettle, Noise Cancelling, usb-c-hub"
	for maxWords, want := range map[int][]string{
		1: {"kettle"},
		2: {"kettle", "noise-cancelling"},
		0: {"kettle", "noise-cancelling", "usb-c-hub"},
	} {
		if got := parseLabels(raw, maxWords); !reflect.DeepEqual(got, want) {
			t.Errorf("parseLabels(%q, %d) = %q, want %q", raw, maxWords, got, want)
		}
	}
}

// Zero options switch the search limits off: no weight floor, no cutoff, no cap.
func TestSearchLimitsOff(t *testing.T) {
	llm := fakeLLM{"top": "a, b, c, d, e, f", "drop": "a", "q": "a, b, c, d, e, f"}
	idx := New(llm, Options{})
	ctx := context.Background()
	idx.Add(ctx, "top", "top")
	idx.Add(ctx, "drop", "drop")
	res, _, err := idx.Search(ctx, "q")
	if err != nil || len(res) != 2 {
		t.Fatalf("results = %+v, err %v; want both documents", res, err)
	}
}

// recordLLM answers by exact input and records every system prompt.
type recordLLM struct {
	answers map[string]string
	systems []string
}

func (r *recordLLM) Complete(_ context.Context, system, user string) (string, error) {
	r.systems = append(r.systems, system)
	return r.answers[user], nil
}

// With SendExistingLabels the prompt carries every label already in the index,
// most used first (ties by name); without it the prompt never does.
func TestSendExistingLabels(t *testing.T) {
	answers := map[string]string{"a": "rare, common", "b": "common, other", "c": "common", "q": "common"}
	ctx := context.Background()

	on := &recordLLM{answers: answers}
	idx := New(on, Options{MaxLabels: 50, MaxLabelWords: 1, SendExistingLabels: true})
	idx.Add(ctx, "a", "a")
	idx.Add(ctx, "b", "b")
	idx.Add(ctx, "c", "c")
	idx.Search(ctx, "q")
	// The first document sees an empty index, later ones see the labels so far.
	if !strings.Contains(on.systems[0], "(none yet)") {
		t.Errorf("first prompt should say the index is empty:\n%s", on.systems[0])
	}
	if !strings.Contains(on.systems[1], "most used first:\ncommon, rare\n") { // 1 doc each: by name
		t.Errorf("second prompt lacks the labels:\n%s", on.systems[1])
	}
	if !strings.Contains(on.systems[2], "most used first:\ncommon, other, rare\n") { // common has 2 docs
		t.Errorf("third prompt not ordered by doc count:\n%s", on.systems[2])
	}
	if !strings.Contains(on.systems[3], "most used first:\ncommon, other, rare\n") { // the search query too
		t.Errorf("search prompt lacks the labels:\n%s", on.systems[3])
	}

	off := &recordLLM{answers: answers}
	idx = New(off, Options{MaxLabels: 50, MaxLabelWords: 1})
	idx.Add(ctx, "a", "a")
	idx.Add(ctx, "b", "b")
	idx.Search(ctx, "q")
	for i, s := range off.systems {
		if strings.Contains(s, "already used") || s != ai.Prompt(50, 1) {
			t.Errorf("prompt %d without the flag must be the plain prompt:\n%s", i, s)
		}
	}
}
