package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func chatServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", handler)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func okReply(w http.ResponseWriter, content string) {
	json.NewEncoder(w).Encode(map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"content": content}}},
	})
}

func TestCompleteSuccess(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("missing/incorrect auth header: %q", r.Header.Get("Authorization"))
		}
		var req chatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if len(req.Messages) != 2 || req.Messages[0].Role != "system" || req.Messages[1].Role != "user" {
			t.Errorf("unexpected messages: %+v", req.Messages)
		}
		okReply(w, "hello back")
	})
	c := NewLLMClient(srv.URL, "secret", "m")
	got, err := c.Complete(context.Background(), "sys", "usr")
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got != "hello back" {
		t.Errorf("Complete = %q, want %q", got, "hello back")
	}
}

func TestCompleteEmptyChoices(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"choices": []any{}})
	})
	c := NewLLMClient(srv.URL, "k", "m")
	if _, err := c.Complete(context.Background(), "sys", "usr"); err == nil {
		t.Fatal("expected error for empty choices")
	}
}

func TestCompleteRetriesThenSucceeds(t *testing.T) {
	var calls int32
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		okReply(w, "ok")
	})
	c := NewLLMClient(srv.URL, "k", "m")
	if _, err := c.Complete(context.Background(), "sys", "usr"); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if calls < 2 {
		t.Errorf("expected at least 2 calls, got %d", calls)
	}
}

func TestCompleteDoesNotRetryPermanent4xx(t *testing.T) {
	var calls int32
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		http.Error(w, "bad request", http.StatusBadRequest)
	})
	c := NewLLMClient(srv.URL, "k", "m")
	if _, err := c.Complete(context.Background(), "sys", "usr"); err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Errorf("calls = %d, want 1", calls)
	}
}

func TestCompleteNoAuthHeaderWhenKeyEmpty(t *testing.T) {
	hadAuth := true
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		hadAuth = r.Header.Get("Authorization") != ""
		okReply(w, "ok")
	})
	c := NewLLMClient(srv.URL, "", "m")
	if _, err := c.Complete(context.Background(), "sys", "usr"); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if hadAuth {
		t.Fatal("Authorization header should be omitted when apiKey is empty")
	}
}
