// Package mockllm is a deterministic stand-in for the labelling LLM, for tests.
// It labels text by keyword rules instead of calling a model, and can be used
// directly (it satisfies index.Chatter) or behind a fake OpenAI-compatible
// /chat/completions server (NewServer) to exercise the real ai.LLMClient.
package mockllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
)

// Rule adds Labels to any text containing one of Keywords (case-insensitive).
type Rule struct {
	Keywords []string
	Labels   []string
}

// DefaultRules cover the audio, kitchen, medical, gardening and developer
// tools products used across the tests.
var DefaultRules = []Rule{
	{[]string{"headphone"}, []string{"audio", "sound", "headphones"}},
	{[]string{"earbud", "airpods"}, []string{"audio", "sound", "earphones"}},
	{[]string{"noise cancel"}, []string{"noise cancelling"}},
	{[]string{"speaker"}, []string{"audio", "sound", "speakers"}},

	{[]string{"blender", "smoothie"}, []string{"kitchen", "appliances", "blender"}},
	{[]string{"knife"}, []string{"kitchen", "cutlery", "knife"}},
	{[]string{"skillet"}, []string{"kitchen", "cookware", "skillet"}},
	{[]string{"espresso"}, []string{"kitchen", "appliances", "coffee"}},
	{[]string{"kettle"}, []string{"kitchen", "appliances", "kettle"}},

	{[]string{"blood pressure"}, []string{"medical", "health", "blood pressure monitor"}},
	{[]string{"thermometer"}, []string{"medical", "health", "thermometer"}},
	{[]string{"oximeter"}, []string{"medical", "health", "pulse oximeter"}},
	{[]string{"stethoscope"}, []string{"medical", "health", "stethoscope"}},

	{[]string{"garden hose"}, []string{"gardening", "outdoor", "hose"}},
	{[]string{"pruner"}, []string{"gardening", "outdoor", "pruner"}},
	{[]string{"planter"}, []string{"gardening", "outdoor", "planter"}},

	{[]string{"keyboard"}, []string{"developer tools", "keyboard"}},
	{[]string{"code editor"}, []string{"developer tools", "code editor"}},
	{[]string{"usb-c hub"}, []string{"developer tools", "hardware", "usb hub"}},
}

// LLM labels text with the labels of every rule it matches. Text matching no
// rule gets an empty label list, like a model that found nothing to say.
type LLM struct {
	Rules []Rule
}

func New() *LLM { return &LLM{Rules: DefaultRules} }

// Complete ignores the system prompt and answers "label1, label2" for user.
func (m *LLM) Complete(_ context.Context, _, user string) (string, error) {
	text := strings.ToLower(user)
	labels := []string{}
	for _, r := range m.Rules {
		for _, k := range r.Keywords {
			if strings.Contains(text, k) {
				labels = append(labels, r.Labels...)
				break
			}
		}
	}
	return strings.Join(labels, ", "), nil
}

// NewServer serves m behind an OpenAI-compatible /chat/completions endpoint.
// Point ai.NewLLMClient at its URL; the caller closes it.
func NewServer(m *LLM) *httptest.Server {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Messages []struct{ Role, Content string } `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) == 0 {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		system, user := req.Messages[0].Content, req.Messages[len(req.Messages)-1].Content
		content, _ := m.Complete(r.Context(), system, user)
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": content}}},
		})
	})
	return httptest.NewServer(mux)
}
