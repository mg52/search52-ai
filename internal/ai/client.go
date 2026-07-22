package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"
)

// EmbeddingClient calls an OpenAI-compatible /embeddings endpoint. The same
// endpoint is exposed by OpenAI and by Ollama's OpenAI-compatible API (base URL
// typically ".../v1").
type EmbeddingClient struct {
	baseURL string
	apiKey  string
	model   string
	http    *http.Client
}

func NewEmbeddingClient(baseURL, apiKey, model string) *EmbeddingClient {
	return &EmbeddingClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		model:   model,
		http:    &http.Client{Timeout: 60 * time.Second},
	}
}

type embeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

// Embed returns the embedding vector for text, retrying transient failures with
// exponential backoff. Vectors are float32 end-to-end: embedding similarities
// don't need float64 precision and the smaller footprint halves scan bandwidth.
func (c *EmbeddingClient) Embed(ctx context.Context, text string) ([]float32, error) {
	req := embeddingRequest{Input: text, Model: c.model}
	var resp embeddingResponse
	err := retry(ctx, 3, func() error {
		return postJSON(ctx, c.http, c.baseURL+"/embeddings", c.apiKey, req, &resp)
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Data) == 0 {
		return nil, fmt.Errorf("empty data in embedding response")
	}
	return resp.Data[0].Embedding, nil
}

// LLMClient calls an OpenAI-compatible /chat/completions endpoint.
type LLMClient struct {
	baseURL string
	apiKey  string
	model   string
	http    *http.Client
}

func NewLLMClient(baseURL, apiKey, model string) *LLMClient {
	return &LLMClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		model:   model,
		// Chat completions can run far longer than embeddings, especially
		// against local/self-hosted reasoning models whose generation time
		// is large and highly variable — 60s cuts those off before they
		// ever finish.
		http: &http.Client{Timeout: 10 * time.Minute},
	}
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// Complete sends a system/user message pair and returns the model's raw text
// response, retrying transient failures with exponential backoff.
func (c *LLMClient) Complete(ctx context.Context, system, user string) (string, error) {
	req := chatRequest{
		Model: c.model,
		Messages: []chatMessage{
			{Role: "system", Content: system},
			{Role: "user", Content: user},
		},
	}
	var resp chatResponse
	err := retry(ctx, 3, func() error {
		return postJSON(ctx, c.http, c.baseURL+"/chat/completions", c.apiKey, req, &resp)
	})
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("empty choices in LLM response")
	}
	return resp.Choices[0].Message.Content, nil
}

func postJSON(ctx context.Context, client *http.Client, url, apiKey string, body, out any) error {
	data, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return &apiError{status: resp.StatusCode, body: string(b)}
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

type apiError struct {
	status int
	body   string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("API error %d: %s", e.status, e.body)
}

// retryable reports whether an attempt is worth repeating: network failures
// and transient statuses (5xx, 408, 429) are; other 4xx responses (bad
// request, auth, missing model) will fail identically every time.
func retryable(err error) bool {
	var ae *apiError
	if errors.As(err, &ae) {
		return ae.status >= 500 || ae.status == http.StatusRequestTimeout || ae.status == http.StatusTooManyRequests
	}
	return true
}

func retry(ctx context.Context, attempts int, fn func() error) error {
	var err error
	for i := 0; i < attempts; i++ {
		if err = fn(); err == nil {
			return nil
		}
		if !retryable(err) {
			return err
		}
		if i < attempts-1 {
			wait := time.Duration(math.Pow(2, float64(i))) * time.Second
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(wait):
			}
		}
	}
	return err
}
