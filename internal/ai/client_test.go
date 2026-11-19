package ai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
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

// The reasoning_effort field defaults to "none" and is left out when empty.
func TestCompleteReasoningEffort(t *testing.T) {
	var got map[string]any
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		got = map[string]any{}
		json.NewDecoder(r.Body).Decode(&got)
		okReply(w, "ok")
	})
	c := NewLLMClient(srv.URL, "k", "m")
	c.Complete(context.Background(), "sys", "usr")
	if got["reasoning_effort"] != "none" {
		t.Errorf("default reasoning_effort = %v, want none", got["reasoning_effort"])
	}
	c.ReasoningEffort = ""
	c.Complete(context.Background(), "sys", "usr")
	if _, ok := got["reasoning_effort"]; ok {
		t.Errorf("empty ReasoningEffort should omit the field, got %v", got)
	}
}

// MaxAttempts is the total number of tries, and Timeout limits each one.
func TestCompleteMaxAttemptsAndTimeout(t *testing.T) {
	var calls int32
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		http.Error(w, "boom", http.StatusInternalServerError)
	})
	c := NewLLMClient(srv.URL, "k", "m")
	c.MaxAttempts = 1
	if _, err := c.Complete(context.Background(), "sys", "usr"); err == nil || atomic.LoadInt32(&calls) != 1 {
		t.Errorf("MaxAttempts 1: err=%v calls=%d, want an error after 1 call", err, calls)
	}

	slow := chatServer(t, func(w http.ResponseWriter, r *http.Request) { time.Sleep(300 * time.Millisecond) })
	c = NewLLMClient(slow.URL, "k", "m")
	c.MaxAttempts, c.Timeout = 1, 50*time.Millisecond
	if _, err := c.Complete(context.Background(), "sys", "usr"); err == nil {
		t.Error("expected a timeout error")
	}
}

// temperature is always sent: 0 by default, the configured value otherwise.
func TestCompleteTemperature(t *testing.T) {
	var got map[string]any
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		got = map[string]any{}
		json.NewDecoder(r.Body).Decode(&got)
		okReply(w, "ok")
	})
	c := NewLLMClient(srv.URL, "k", "m")
	c.Complete(context.Background(), "sys", "usr")
	if v, ok := got["temperature"]; !ok || v != float64(0) {
		t.Errorf("default temperature = %v (present %v), want 0 sent", v, ok)
	}
	c.Temperature = 0.7
	c.Complete(context.Background(), "sys", "usr")
	if got["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", got["temperature"])
	}
}

// max_tokens is sent (300 by default) and left out when 0.
func TestCompleteMaxTokens(t *testing.T) {
	var got map[string]any
	srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
		got = map[string]any{}
		json.NewDecoder(r.Body).Decode(&got)
		okReply(w, "ok")
	})
	c := NewLLMClient(srv.URL, "k", "m")
	c.Complete(context.Background(), "sys", "usr")
	if got["max_tokens"] != float64(300) {
		t.Errorf("default max_tokens = %v, want 300", got["max_tokens"])
	}
	c.MaxTokens = 0
	c.Complete(context.Background(), "sys", "usr")
	if _, ok := got["max_tokens"]; ok {
		t.Errorf("MaxTokens 0 should omit the field, got %v", got)
	}
}

// A reply cut off by max_tokens loses its last label, which may be half written.
func TestCompleteTruncatedReply(t *testing.T) {
	for finish, want := range map[string]string{"length": "kitchen, appliance", "stop": "kitchen, appliance, ket"} {
		srv := chatServer(t, func(w http.ResponseWriter, r *http.Request) {
			json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{{"finish_reason": finish, "message": map[string]any{"content": "kitchen, appliance, ket"}}},
			})
		})
		got, err := NewLLMClient(srv.URL, "k", "m").Complete(context.Background(), "sys", "usr")
		if err != nil || got != want {
			t.Errorf("finish_reason %q: got %q, %v; want %q", finish, got, err, want)
		}
	}
}
