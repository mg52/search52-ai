// Package e2e drives the whole service over HTTP: real handler, real index,
// real ai.LLMClient, with only the model replaced by mockllm.
package e2e

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"testing"

	"github.com/mg52/search52-ai/internal/ai"
	"github.com/mg52/search52-ai/internal/handler"
	"github.com/mg52/search52-ai/internal/index"
	"github.com/mg52/search52-ai/internal/mockllm"
)

type doc struct {
	ID      string   `json:"id"`
	Content string   `json:"content"`
	Labels  []string `json:"labels"`
}

// corpus is the 20 documents indexed by every test, with the labels the mock
// LLM is expected to produce for each.
var corpus = []doc{
	{"audio-01", "Sony WH-1000XM5 wireless over-ear headphones with industry-leading active noise cancellation.", []string{"audio", "sound", "headphones", "noise cancelling"}},
	{"audio-02", "Bose QuietComfort earbuds with adjustable noise cancelling and a secure in-ear fit for workouts.", []string{"audio", "sound", "earphones", "noise cancelling"}},
	{"audio-03", "Sennheiser HD 660S open-back audiophile headphones tuned for natural stereo sound.", []string{"audio", "sound", "headphones"}},
	{"audio-04", "Apple AirPods Pro with adaptive transparency mode and spatial audio.", []string{"audio", "sound", "earphones"}},
	{"audio-05", "JBL Flip portable bluetooth speaker with a waterproof design.", []string{"audio", "sound", "speakers"}},
	{"kitchen-01", "Ninja 1000W blender that crushes ice for frozen drinks.", []string{"kitchen", "appliances", "blender"}},
	{"kitchen-02", "Wusthof forged chef knife with a razor sharp edge.", []string{"kitchen", "cutlery", "knife"}},
	{"kitchen-03", "Lodge pre-seasoned cast iron skillet for searing and baking.", []string{"kitchen", "cookware", "skillet"}},
	{"kitchen-04", "Breville espresso machine with a built-in conical burr grinder.", []string{"kitchen", "appliances", "coffee"}},
	{"kitchen-05", "Fellow electric kettle with variable temperature control.", []string{"kitchen", "appliances", "kettle"}},
	{"medical-01", "Omron upper arm blood pressure monitor with irregular heartbeat detection.", []string{"medical", "health", "blood pressure monitor"}},
	{"medical-02", "Digital infrared thermometer for fast contactless fever checks.", []string{"medical", "health", "thermometer"}},
	{"medical-03", "Fingertip pulse oximeter showing blood oxygen saturation and pulse rate.", []string{"medical", "health", "pulse oximeter"}},
	{"medical-04", "Littmann classic stethoscope for clinical auscultation.", []string{"medical", "health", "stethoscope"}},
	{"garden-01", "Expandable 50ft garden hose with brass fittings.", []string{"gardening", "outdoor", "hose"}},
	{"garden-02", "Bypass pruner with an ergonomic grip for trimming roses.", []string{"gardening", "outdoor", "pruner"}},
	{"garden-03", "Cedar raised bed planter for growing vegetables.", []string{"gardening", "outdoor", "planter"}},
	{"dev-01", "Keychron mechanical keyboard with hot-swappable switches.", []string{"developer tools", "keyboard"}},
	{"dev-02", "Lightweight code editor with built-in git and debugging.", []string{"developer tools", "code editor"}},
	{"dev-03", "Anker USB-C hub with HDMI and ethernet ports for laptops.", []string{"developer tools", "hardware", "usb hub"}},
}

// newService starts the full service on a mock LLM and returns its base URL.
func newService(t *testing.T) string {
	t.Helper()
	llm := mockllm.NewServer(mockllm.New())
	t.Cleanup(llm.Close)
	svc := httptest.NewServer(handler.New(index.New(ai.NewLLMClient(llm.URL, "", "mock"))).Routes())
	t.Cleanup(svc.Close)
	return svc.URL
}

func send(t *testing.T, method, url string, body any, out any) int {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		json.NewEncoder(&buf).Encode(body)
	}
	req, _ := http.NewRequest(method, url, &buf)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, url, err)
	}
	defer resp.Body.Close()
	if out != nil {
		json.NewDecoder(resp.Body).Decode(out)
	}
	return resp.StatusCode
}

func indexCorpus(t *testing.T, base string) {
	t.Helper()
	for _, d := range corpus {
		var got doc
		if code := send(t, "POST", base+"/documents", map[string]string{"id": d.ID, "content": d.Content}, &got); code != http.StatusCreated {
			t.Fatalf("index %s: status %d", d.ID, code)
		}
		want := append([]string(nil), d.Labels...)
		gotLabels := append([]string(nil), got.Labels...)
		sort.Strings(want)
		sort.Strings(gotLabels)
		if !reflect.DeepEqual(gotLabels, want) {
			t.Errorf("%s labels = %v, want %v", d.ID, gotLabels, want)
		}
	}
}

type searchResp struct {
	Labels    []string `json:"labels"`
	Documents []doc    `json:"documents"`
}

func search(t *testing.T, base, q string) (searchResp, int) {
	t.Helper()
	var resp searchResp
	code := send(t, "POST", base+"/search", map[string]string{"q": q}, &resp)
	return resp, code
}

func ids(docs []doc) []string {
	out := make([]string, len(docs))
	for i, d := range docs {
		out[i] = d.ID
	}
	return out
}

// Results are ordered by number of shared labels, then by id.
func TestSearch(t *testing.T) {
	base := newService(t)
	indexCorpus(t, base)

	tests := []struct {
		query string
		want  []string
	}{
		// earphones+audio+sound+noise cancelling: audio-02 shares 4, audio-01/04 share 3, the rest 2.
		{"noise cancelling earbuds", []string{"audio-02", "audio-01", "audio-04", "audio-03", "audio-05"}},
		{"bluetooth speaker", []string{"audio-05", "audio-01", "audio-02", "audio-03", "audio-04"}},
		{"blood pressure", []string{"medical-01", "medical-02", "medical-03", "medical-04"}},
		{"sharp kitchen knife", []string{"kitchen-02", "kitchen-01", "kitchen-03", "kitchen-04", "kitchen-05"}},
		{"garden hose", []string{"garden-01", "garden-02", "garden-03"}},
		{"mechanical keyboard", []string{"dev-01", "dev-02", "dev-03"}},
	}
	for _, tc := range tests {
		t.Run(tc.query, func(t *testing.T) {
			resp, code := search(t, base, tc.query)
			if code != http.StatusOK {
				t.Fatalf("status %d", code)
			}
			if got := ids(resp.Documents); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("ids = %v, want %v", got, tc.want)
			}
			// Results carry the stored content, not just ids.
			for _, d := range resp.Documents {
				for _, c := range corpus {
					if c.ID == d.ID && c.Content != d.Content {
						t.Errorf("%s content = %q, want %q", d.ID, d.Content, c.Content)
					}
				}
			}
		})
	}
}

func TestSearchNoMatch(t *testing.T) {
	base := newService(t)
	indexCorpus(t, base)

	// The mock finds no labels in the query, so labelling fails like an empty model answer.
	if _, code := search(t, base, "quantum physics"); code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", code)
	}
}

func TestDeleteRemovesFromSearch(t *testing.T) {
	base := newService(t)
	indexCorpus(t, base)

	if code := send(t, "DELETE", base+"/documents/audio-02", nil, nil); code != http.StatusNoContent {
		t.Fatalf("delete status %d", code)
	}
	resp, _ := search(t, base, "noise cancelling earbuds")
	if want := []string{"audio-01", "audio-04", "audio-03", "audio-05"}; !reflect.DeepEqual(ids(resp.Documents), want) {
		t.Errorf("ids = %v, want %v", ids(resp.Documents), want)
	}
}

func TestReindexReplacesLabels(t *testing.T) {
	base := newService(t)
	indexCorpus(t, base)

	// Same id, new content: audio-05 stops being a speaker and becomes a garden hose.
	send(t, "POST", base+"/documents", map[string]string{"id": "audio-05", "content": "Heavy duty garden hose."}, nil)

	speakers, _ := search(t, base, "bluetooth speaker")
	if got := ids(speakers.Documents); contains(got, "audio-05") {
		t.Errorf("audio-05 still matches speakers: %v", got)
	}
	hoses, _ := search(t, base, "garden hose")
	if !contains(ids(hoses.Documents), "audio-05") {
		t.Errorf("reindexed audio-05 missing from garden hose results: %v", ids(hoses.Documents))
	}
}

func TestBatchIndexing(t *testing.T) {
	base := newService(t)

	var resp struct{ Indexed, Failed int }
	docs := []map[string]string{}
	for _, d := range corpus {
		docs = append(docs, map[string]string{"id": d.ID, "content": d.Content})
	}
	docs = append(docs, map[string]string{"content": "no id"})
	if code := send(t, "POST", base+"/documents/batch", map[string]any{"documents": docs}, &resp); code != http.StatusCreated {
		t.Fatalf("status %d", code)
	}
	if resp.Indexed != 20 || resp.Failed != 1 {
		t.Errorf("indexed/failed = %d/%d, want 20/1", resp.Indexed, resp.Failed)
	}
	if got, _ := search(t, base, "garden hose"); len(got.Documents) != 3 {
		t.Errorf("garden hose results = %v", ids(got.Documents))
	}
}

func contains(s []string, v string) bool {
	for _, x := range s {
		if x == v {
			return true
		}
	}
	return false
}
