package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"
)

type LLMClient struct {
	baseURL string
	apiKey  string
	model   string
	http    *http.Client
}

type EmbeddingClient struct {
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
		http:    &http.Client{Timeout: 60 * time.Second},
	}
}

func NewEmbeddingClient(baseURL, apiKey, model string) *EmbeddingClient {
	return &EmbeddingClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		model:   model,
		http:    &http.Client{Timeout: 60 * time.Second},
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

type embeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
	} `json:"data"`
}

func (c *EmbeddingClient) Embed(ctx context.Context, text string) ([]float64, error) {
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
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error %d: %s", resp.StatusCode, string(b))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func retry(ctx context.Context, attempts int, fn func() error) error {
	var err error
	for i := 0; i < attempts; i++ {
		if err = fn(); err == nil {
			return nil
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
