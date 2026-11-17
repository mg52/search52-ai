package handler

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mg52/search52-ai/internal/index"
)

// fakeLLM labels every text by its first word, e.g. "audio gear" -> ["audio"].
type fakeLLM struct{}

func (fakeLLM) Complete(_ context.Context, _, user string) (string, error) {
	return strings.Fields(user)[0], nil
}

func do(t *testing.T, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	h := New(index.New(fakeLLM{}, index.Options{MaxPromptLabels: 1000, MinLabels: 5, MaxLabels: 10}))
	return doWith(t, h, method, path, body)
}

func doWith(t *testing.T, h *Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	h.Routes().ServeHTTP(w, httptest.NewRequest(method, path, strings.NewReader(body)))
	return w
}

func TestCreateDocumentValidation(t *testing.T) {
	for _, body := range []string{`{"content": "audio gear"}`, `{"id": "a"}`, `not json`} {
		if w := do(t, "POST", "/documents", body); w.Code != http.StatusBadRequest {
			t.Errorf("body %q: status = %d, want 400", body, w.Code)
		}
	}
}

func TestIndexSearchDelete(t *testing.T) {
	h := New(index.New(fakeLLM{}, index.Options{MaxPromptLabels: 1000, MinLabels: 5, MaxLabels: 10}))
	if w := doWith(t, h, "POST", "/documents", `{"id": "a", "content": "audio gear"}`); w.Code != http.StatusCreated {
		t.Fatalf("create: %d %s", w.Code, w.Body)
	}
	w := doWith(t, h, "POST", "/documents/batch", `{"documents": [{"id": "b", "content": "audio cable"}, {"id": "c", "content": "coffee beans"}, {"content": "no id"}]}`)
	if w.Code != http.StatusCreated || !strings.Contains(w.Body.String(), `"indexed":2`) || !strings.Contains(w.Body.String(), `"failed":1`) {
		t.Fatalf("batch: %d %s", w.Code, w.Body)
	}

	w = doWith(t, h, "POST", "/search", `{"q": "audio please"}`)
	var resp struct {
		Documents []struct{ ID, Content string }
	}
	json.Unmarshal(w.Body.Bytes(), &resp)
	if len(resp.Documents) != 2 || resp.Documents[0].ID != "a" || resp.Documents[0].Content != "audio gear" || resp.Documents[1].ID != "b" {
		t.Fatalf("search: %d %s", w.Code, w.Body)
	}

	if w := doWith(t, h, "DELETE", "/documents/a", ""); w.Code != http.StatusNoContent {
		t.Errorf("delete: %d", w.Code)
	}
	if w := doWith(t, h, "DELETE", "/documents/a", ""); w.Code != http.StatusNotFound {
		t.Errorf("second delete: %d, want 404", w.Code)
	}
}

func TestSearchRequiresQuery(t *testing.T) {
	if w := do(t, "POST", "/search", `{"q": ""}`); w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}
