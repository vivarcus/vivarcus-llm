// Package llm provides an OpenAI-compatible Chat Completions client (CAP-INT).
package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const defaultTimeout = 120 * time.Second

// Message is one chat message.
// Content is either a string or []ContentPart (multimodal).
type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	Name       string     `json:"name,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// Tool is an OpenAI-compatible tool definition (function calling).
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes one callable function tool.
type ToolFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

// ToolCall is one model-requested tool invocation.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the function name and JSON arguments string.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatRequest is an OpenAI-compatible chat completions request.
type ChatRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  any       `json:"tool_choice,omitempty"`
}

// Usage captures token counts when the provider returns them.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatResponse is the subset of the OpenAI response we need.
type ChatResponse struct {
	Content   string
	ToolCalls []ToolCall
	Model     string
	Usage     Usage
	Raw       []byte
}

// Connection holds resolved credentials for one LLM connection.
type Connection struct {
	Name    string
	BaseURL string
	APIKey  string
	Model   string
}

// Client calls OpenAI-compatible Chat Completions endpoints.
type Client struct {
	HTTP    *http.Client
	Timeout time.Duration
}

// NewClient returns a client with default timeouts.
func NewClient() *Client {
	return &Client{
		HTTP:    &http.Client{Timeout: defaultTimeout},
		Timeout: defaultTimeout,
	}
}

// ChatCompletions POSTs to {baseURL}/chat/completions.
func (c *Client) ChatCompletions(ctx context.Context, conn Connection, req ChatRequest) (ChatResponse, error) {
	if c == nil {
		return ChatResponse{}, fmt.Errorf("llm client not configured")
	}
	base := strings.TrimRight(strings.TrimSpace(conn.BaseURL), "/")
	if base == "" {
		return ChatResponse{}, fmt.Errorf("llm base_url required")
	}
	if strings.TrimSpace(conn.APIKey) == "" {
		return ChatResponse{}, fmt.Errorf("llm api_key required")
	}
	if strings.TrimSpace(req.Model) == "" {
		req.Model = strings.TrimSpace(conn.Model)
	}
	if strings.TrimSpace(req.Model) == "" {
		return ChatResponse{}, fmt.Errorf("llm model required")
	}
	if len(req.Messages) == 0 {
		return ChatResponse{}, fmt.Errorf("llm messages required")
	}
	req.Messages = SanitizeMessagesForModel(req.Model, req.Messages)

	body, err := json.Marshal(req)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("marshal chat request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, fmt.Errorf("build chat request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+conn.APIKey)

	httpClient := c.HTTP
	if httpClient == nil {
		httpClient = &http.Client{Timeout: defaultTimeout}
	}
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("llm chat completions: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return ChatResponse{}, fmt.Errorf("read llm response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ChatResponse{}, fmt.Errorf("llm chat completions status %d: %s", resp.StatusCode, truncate(string(raw), 512))
	}

	var parsed openAIChatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return ChatResponse{}, fmt.Errorf("decode llm response: %w", err)
	}
	content := ""
	var toolCalls []ToolCall
	if len(parsed.Choices) > 0 {
		content = assistantContent(parsed.Choices[0].Message.Content)
		toolCalls = parsed.Choices[0].Message.ToolCalls
	}
	if strings.TrimSpace(content) == "" && len(toolCalls) == 0 {
		return ChatResponse{}, fmt.Errorf("llm returned empty content")
	}
	return ChatResponse{
		Content:   content,
		ToolCalls: toolCalls,
		Model:     firstNonEmpty(parsed.Model, req.Model),
		Usage: Usage{
			PromptTokens:     parsed.Usage.PromptTokens,
			CompletionTokens: parsed.Usage.CompletionTokens,
			TotalTokens:      parsed.Usage.TotalTokens,
		},
		Raw: raw,
	}, nil
}

// StreamDelta is one streaming content chunk (and optional final usage).
type StreamDelta struct {
	Content string
	Done    bool
	Usage   Usage
	Model   string
}

// ChatCompletionsStream POSTs with stream=true and invokes onDelta for each content chunk.
func (c *Client) ChatCompletionsStream(ctx context.Context, conn Connection, req ChatRequest, onDelta func(StreamDelta) error) (ChatResponse, error) {
	if c == nil {
		return ChatResponse{}, fmt.Errorf("llm client not configured")
	}
	if onDelta == nil {
		return ChatResponse{}, fmt.Errorf("onDelta required")
	}
	base := strings.TrimRight(strings.TrimSpace(conn.BaseURL), "/")
	if base == "" {
		return ChatResponse{}, fmt.Errorf("llm base_url required")
	}
	if strings.TrimSpace(conn.APIKey) == "" {
		return ChatResponse{}, fmt.Errorf("llm api_key required")
	}
	if strings.TrimSpace(req.Model) == "" {
		req.Model = strings.TrimSpace(conn.Model)
	}
	if strings.TrimSpace(req.Model) == "" {
		return ChatResponse{}, fmt.Errorf("llm model required")
	}
	if len(req.Messages) == 0 {
		return ChatResponse{}, fmt.Errorf("llm messages required")
	}
	req.Stream = true
	req.Messages = SanitizeMessagesForModel(req.Model, req.Messages)

	body, err := json.Marshal(req)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("marshal chat request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ChatResponse{}, fmt.Errorf("build chat request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+conn.APIKey)
	httpReq.Header.Set("Accept", "text/event-stream")

	httpClient := c.HTTP
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 0} // stream; rely on ctx
	} else {
		// Clone without per-request timeout so long streams are not cut off.
		httpClient = &http.Client{Transport: httpClient.Transport, CheckRedirect: httpClient.CheckRedirect, Jar: httpClient.Jar}
	}
	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return ChatResponse{}, fmt.Errorf("llm chat completions stream: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
		return ChatResponse{}, fmt.Errorf("llm chat completions status %d: %s", resp.StatusCode, truncate(string(raw), 512))
	}

	var (
		full   strings.Builder
		model  = req.Model
		usage  Usage
		reader = bufio.NewReader(resp.Body)
	)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return ChatResponse{}, fmt.Errorf("read llm stream: %w", err)
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			break
		}
		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue
		}
		if chunk.Model != "" {
			model = chunk.Model
		}
		delta := ""
		if len(chunk.Choices) > 0 {
			delta = chunk.Choices[0].Delta.Content
		}
		if chunk.Usage != nil {
			usage = *chunk.Usage
		}
		if delta != "" {
			full.WriteString(delta)
			if err := onDelta(StreamDelta{Content: delta, Model: model}); err != nil {
				return ChatResponse{}, err
			}
		}
	}
	content := full.String()
	if strings.TrimSpace(content) == "" {
		return ChatResponse{}, fmt.Errorf("llm returned empty content")
	}
	_ = onDelta(StreamDelta{Done: true, Usage: usage, Model: model, Content: ""})
	return ChatResponse{
		Content: content,
		Model:   model,
		Usage:   usage,
	}, nil
}

type openAIStreamChunk struct {
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *Usage `json:"usage"`
}

type openAIChatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Role      string          `json:"role"`
			Content   json.RawMessage `json:"content"`
			ToolCalls []ToolCall      `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
