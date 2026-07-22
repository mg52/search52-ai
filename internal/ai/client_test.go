package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func embedServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/embeddings", handler)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestEmbedSuccess(t *testing.T) {
	srv := embedServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Errorf("missing/incorrect auth header: %q", r.Header.Get("Authorization"))
		}
		var req map[string]any
		json.NewDecoder(r.Body).Decode(&req)
		if req["input"] != "hello" {
			t.Errorf("input = %v, want hello", req["input"])
		}
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1, 2, 3}}},
		})
	})
	c := NewEmbeddingClient(srv.URL, "secret", "m")
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

func TestEmbedRetriesThenSucceeds(t *testing.T) {
	var calls int32
	srv := embedServer(t, func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			http.Error(w, "boom", http.StatusInternalServerError)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1}}},
		})
	})
	c := NewEmbeddingClient(srv.URL, "k", "m")
	got, err := c.Embed(context.Background(), "x")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(got) != 1 || got[0] != 1 {
		t.Errorf("Embed = %v, want [1]", got)
	}
	if calls < 2 {
		t.Errorf("expected at least 2 calls, got %d", calls)
	}
}

func TestEmbedDoesNotRetryPermanent4xx(t *testing.T) {
	var calls int32
	srv := embedServer(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		http.Error(w, "bad request", http.StatusBadRequest)
	})
	c := NewEmbeddingClient(srv.URL, "k", "m")
	if _, err := c.Embed(context.Background(), "x"); err == nil {
		t.Fatal("expected error for 400 response")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1 (a 400 must not be retried)", calls)
	}
}

func TestEmbedContextCancelled(t *testing.T) {
	srv := embedServer(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	c := NewEmbeddingClient(srv.URL, "k", "m")
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled before the call; retry backoff must observe it
	if _, err := c.Embed(ctx, "x"); err == nil {
		t.Fatal("expected error with cancelled context")
	}
}

func TestEmbedNoAuthHeaderWhenKeyEmpty(t *testing.T) {
	hadAuth := true
	srv := embedServer(t, func(w http.ResponseWriter, r *http.Request) {
		hadAuth = r.Header.Get("Authorization") != ""
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{{"embedding": []float64{1}}},
		})
	})
	c := NewEmbeddingClient(srv.URL, "", "m")
	if _, err := c.Embed(context.Background(), "x"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if hadAuth {
		t.Fatal("Authorization header should be omitted when apiKey is empty")
	}
}

func chatServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", handler)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv
}

func TestCompleteSuccess(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if len(req.Messages) != 2 || req.Messages[0].Role != "system" || req.Messages[1].Role != "user" {
			t.Errorf("unexpected messages: %+v", req.Messages)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "hello back"}}},
		})
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
