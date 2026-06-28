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

func embedServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/embeddings", handler)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func writeChat(w http.ResponseWriter, content string) {
	json.NewEncoder(w).Encode(map[string]any{
		"choices": []map[string]any{{"message": map[string]any{"content": content}}},
	})
}

func TestCompleteSuccess(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("missing/incorrect auth header: %q", r.Header.Get("Authorization"))
		}
		writeChat(w, "hello")
	})
	c := NewLLMClient(srv.URL, "secret", "m")
	got, err := c.Complete(context.Background(), "sys", "user")
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got != "hello" {
		t.Errorf("Complete = %q, want hello", got)
	}
}

func TestCompleteEmptyChoices(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"choices": []any{}})
	})
	c := NewLLMClient(srv.URL, "k", "m")
	if _, err := c.Complete(context.Background(), "s", "u"); err == nil {
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
		writeChat(w, "recovered")
	})
	c := NewLLMClient(srv.URL, "k", "m")
	got, err := c.Complete(context.Background(), "s", "u")
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got != "recovered" {
		t.Errorf("Complete = %q, want recovered", got)
	}
	if calls < 2 {
		t.Errorf("expected at least 2 calls, got %d", calls)
	}
}

func TestCompleteContextCancelled(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	c := NewLLMClient(srv.URL, "k", "m")
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled before the call; retry backoff must observe it
	if _, err := c.Complete(ctx, "s", "u"); err == nil {
		t.Fatal("expected error with cancelled context")
	}
}

func TestEmbedSuccess(t *testing.T) {
	srv := embedServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["input"] != "hello" {
			t.Errorf("input = %v, want hello", req["input"])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1, 2, 3}}},
		})
	})
	c := NewEmbeddingClient(srv.URL, "k", "m")
	got, err := c.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Errorf("Embed = %v, want [1 2 3]", got)
	}
}

func TestEmbedEmptyData(t *testing.T) {
	srv := embedServer(t, func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []any{}})
	})
	c := NewEmbeddingClient(srv.URL, "k", "m")
	if _, err := c.Embed(context.Background(), "x"); err == nil {
		t.Fatal("expected error for empty data")
	}
}
