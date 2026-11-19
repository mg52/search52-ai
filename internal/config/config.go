package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

// Config is read from environment variables by Load. Each field names its
// variable; README.md has the same table.
type Config struct {
	// PORT: HTTP port the server listens on.
	Port string

	// LLM_BASE_URL (required): base URL of the OpenAI-compatible chat endpoint,
	// e.g. http://localhost:11434/v1 for Ollama.
	LLMBaseURL string
	// LLM_MODEL (required): chat model name, e.g. qwen3.5:9b.
	LLMModel string
	// LLM_API_KEY (optional): sent as a bearer token; omit for unauthenticated
	// endpoints (Ollama accepts any value).
	LLMAPIKey string
	// LLM_TIMEOUT: time limit for one LLM call attempt, as a Go duration
	// ("90s", "10m"). Local reasoning models can be slow, so keep it generous.
	LLMTimeout time.Duration
	// LLM_MAX_ATTEMPTS: how many times an LLM call is tried in total. Only
	// transient failures (network errors, 5xx, 408, 429) are retried, with
	// exponential backoff; 1 disables retries.
	LLMMaxAttempts int
	// LLM_REASONING_EFFORT: value of the "reasoning_effort" request field.
	// "none" turns off chain-of-thought on reasoning models (Qwen3/3.5 via Ollama,
	// OpenAI o-series), which only adds latency for this task. Set it to an empty
	// string to leave the field out, for endpoints that reject it.
	LLMReasoningEffort string
	// LLM_TEMPERATURE: sampling temperature (0 to 2). 0 keeps the labels as
	// repeatable as the model allows, so the same text gets the same labels.
	LLMTemperature float64
	// LLM_MAX_TOKENS: most tokens one LLM reply may have, sent as "max_tokens".
	// A label list is short (about 3 tokens per label), so this is a safety net:
	// it stops a model that never ends the list from running for minutes. If a
	// reply is cut off, its last, possibly half written label is dropped. Keep it
	// well above MAX_LABELS x 3.
	LLMMaxTokens int

	// MAX_LABELS: the LLM is asked for 1 to MAX_LABELS labels per document or query
	// (it may use fewer), broad categories first. The first MAX_LABELS labels it returns (after labels that
	// break MAX_LABEL_WORDS are removed) go into the index, so extras are dropped.
	MaxLabels int
	// SEND_EXISTING_LABELS: true or false. When true, every labelling call (for
	// documents and for search queries) also gets the labels the index already
	// has, most used first, and the LLM is told to reuse them instead of inventing
	// synonyms. That makes the vocabulary more consistent but the prompt grows with
	// the index (it carries every label), so each call gets slower and costlier.
	SendExistingLabels bool
	// MAX_LABEL_WORDS: most words in one label. With 1 the LLM is asked for
	// single words and any longer label is ignored. Words are split on spaces,
	// underscores and hyphens.
	MaxLabelWords int

	// MIN_LABEL_DOCS: when weighing a search, a label counts as having at least
	// this many documents, so a label that only one or two documents happen to
	// have cannot dominate the ranking. Raise it for bigger indexes.
	MinLabelDocs int
	// MIN_RELATIVE_SCORE: search results scoring less than this fraction of the
	// best result are dropped (0 to 1; 0.2 = under 20% of the top score). The best
	// result always stays. 0 keeps every match.
	MinRelativeScore float64
	// MAX_RESULTS: most documents one search returns, best first.
	MaxResults int
}

func Load() (*Config, error) {
	c := &Config{
		Port:               "8080",
		LLMBaseURL:         os.Getenv("LLM_BASE_URL"),
		LLMModel:           os.Getenv("LLM_MODEL"),
		LLMAPIKey:          os.Getenv("LLM_API_KEY"),
		LLMTimeout:         10 * time.Minute,
		LLMMaxAttempts:     3,
		LLMReasoningEffort: "none",
		LLMTemperature:     0,
		LLMMaxTokens:       300,
		MaxLabels:          50,
		MaxLabelWords:      1,
		MinLabelDocs:       5,
		MinRelativeScore:   0.2,
		MaxResults:         100,
	}
	if v := os.Getenv("PORT"); v != "" {
		c.Port = v
	}
	if v, ok := os.LookupEnv("LLM_REASONING_EFFORT"); ok {
		c.LLMReasoningEffort = v // empty is allowed: it omits the field
	}
	for name, dst := range map[string]*int{
		"LLM_MAX_ATTEMPTS": &c.LLMMaxAttempts,
		"LLM_MAX_TOKENS":   &c.LLMMaxTokens,
		"MAX_LABELS":       &c.MaxLabels,
		"MAX_LABEL_WORDS":  &c.MaxLabelWords,
		"MIN_LABEL_DOCS":   &c.MinLabelDocs,
		"MAX_RESULTS":      &c.MaxResults,
	} {
		if v := os.Getenv(name); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n < 1 {
				return nil, fmt.Errorf("%s must be a positive integer, got %q", name, v)
			}
			*dst = n
		}
	}
	if v := os.Getenv("LLM_TIMEOUT"); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil || d <= 0 {
			return nil, fmt.Errorf("LLM_TIMEOUT must be a positive duration like 90s or 10m, got %q", v)
		}
		c.LLMTimeout = d
	}
	if v := os.Getenv("LLM_TEMPERATURE"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f < 0 || f > 2 {
			return nil, fmt.Errorf("LLM_TEMPERATURE must be a number from 0 to 2, got %q", v)
		}
		c.LLMTemperature = f
	}
	if v := os.Getenv("SEND_EXISTING_LABELS"); v != "" {
		b, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("SEND_EXISTING_LABELS must be true or false, got %q", v)
		}
		c.SendExistingLabels = b
	}
	if v := os.Getenv("MIN_RELATIVE_SCORE"); v != "" {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil || f < 0 || f > 1 {
			return nil, fmt.Errorf("MIN_RELATIVE_SCORE must be a number from 0 to 1, got %q", v)
		}
		c.MinRelativeScore = f
	}
	if c.LLMBaseURL == "" || c.LLMModel == "" {
		return nil, fmt.Errorf("LLM_BASE_URL and LLM_MODEL are required")
	}
	return c, nil
}
