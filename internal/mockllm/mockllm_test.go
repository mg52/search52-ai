package mockllm

import (
	"context"
	"testing"

	"github.com/mg52/search52-ai/internal/ai"
)

func TestComplete(t *testing.T) {
	m := New()
	tests := []struct{ text, want string }{
		{"Bose earbuds with noise cancelling", "audio, sound, earphones, cancelling"},
		{"QUANTUM PHYSICS", ""},
	}
	for _, tc := range tests {
		got, err := m.Complete(context.Background(), "ignored", tc.text)
		if err != nil || got != tc.want {
			t.Errorf("Complete(%q) = %q, %v; want %q", tc.text, got, err, tc.want)
		}
	}
}

func TestServerWorksWithRealClient(t *testing.T) {
	srv := NewServer(New())
	defer srv.Close()

	got, err := ai.NewLLMClient(srv.URL, "", "mock").Complete(context.Background(), ai.Prompt(50, 1), "garden hose")
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if want := "gardening, outdoor, hose"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
