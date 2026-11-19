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
	"strings"
	"time"
)

// LLMClient calls an OpenAI-compatible /chat/completions endpoint.
type LLMClient struct {
	baseURL string
	apiKey  string
	model   string
	http    *http.Client

	// Timeout limits one attempt. Chat completions can run far longer than
	// embeddings, especially against local/self-hosted reasoning models whose
	// generation time is large and highly variable, so the default is generous.
	Timeout time.Duration
	// MaxAttempts is how many times a call is tried in total; only transient
	// failures (network, 5xx, 408, 429) are retried, with exponential backoff.
	MaxAttempts int
	// ReasoningEffort is sent as "reasoning_effort". "none" switches off
	// chain-of-thought output on reasoning models (OpenAI o-series, Qwen3/3.5
	// via Ollama): the labelling prompt is a fixed task, so reasoning tokens
	// only add latency. Empty omits the field, for endpoints that reject it.
	ReasoningEffort string
	// Temperature is the sampling temperature. 0 makes the labels as repeatable
	// as the model allows, which matters because documents and queries must end
	// up with the same label vocabulary. It is always sent, 0 included.
	Temperature float64
	// MaxTokens limits the length of one reply, sent as "max_tokens" (<= 0 omits
	// it). A label list is short, so this mostly stops a model that never ends the
	// list (a repetition loop) from running for minutes.
	MaxTokens int
}

// NewLLMClient returns a client with the defaults: 10 minute timeout, 3
// attempts, reasoning effort "none", temperature 0, 300 max tokens. Set the exported fields to
// change them.
func NewLLMClient(baseURL, apiKey, model string) *LLMClient {
	return &LLMClient{
		baseURL:         baseURL,
		apiKey:          apiKey,
		model:           model,
		http:            &http.Client{},
		Timeout:         10 * time.Minute,
		MaxAttempts:     3,
		ReasoningEffort: "none",
		MaxTokens:       300,
	}
}

type chatRequest struct {
	Model           string        `json:"model"`
	Messages        []chatMessage `json:"messages"`
	ReasoningEffort string        `json:"reasoning_effort,omitempty"`
	// No omitempty: temperature 0 must be sent, or the server uses its own default.
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
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
		FinishReason string `json:"finish_reason"`
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
		ReasoningEffort: c.ReasoningEffort,
		Temperature:     c.Temperature,
		MaxTokens:       c.MaxTokens,
	}
	var resp chatResponse
	err := retry(ctx, c.MaxAttempts, func() error {
		ctx, cancel := context.WithTimeout(ctx, c.Timeout)
		defer cancel()
		return postJSON(ctx, c.http, c.baseURL+"/chat/completions", c.apiKey, req, &resp)
	})
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("empty choices in LLM response")
	}
	content := resp.Choices[0].Message.Content
	if resp.Choices[0].FinishReason == "length" {
		// Cut off by MaxTokens: the last label may be half written, drop it.
		if i := strings.LastIndex(content, ","); i >= 0 {
			content = content[:i]
		}
	}
	return content, nil
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
