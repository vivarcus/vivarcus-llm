package llm_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/vivarcus/vivarcus-llm"
)

func TestChatCompletions_success(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("Authorization = %q", got)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if body["model"] != "gpt-test" {
			t.Errorf("model = %v", body["model"])
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "gpt-test",
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": "hello from model"}},
			},
			"usage": map[string]int{"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
		})
	}))
	t.Cleanup(srv.Close)

	client := llm.NewClient()
	client.HTTP = srv.Client()
	out, err := client.ChatCompletions(context.Background(), llm.Connection{
		BaseURL: srv.URL,
		APIKey:  "test-key",
		Model:   "gpt-test",
	}, llm.ChatRequest{
		Messages: []llm.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}
	if out.Content != "hello from model" {
		t.Errorf("content = %q", out.Content)
	}
	if out.Usage.TotalTokens != 15 {
		t.Errorf("total tokens = %d", out.Usage.TotalTokens)
	}
}

func TestChatCompletions_httpError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad request"}`))
	}))
	t.Cleanup(srv.Close)

	client := llm.NewClient()
	client.HTTP = srv.Client()
	_, err := client.ChatCompletions(context.Background(), llm.Connection{
		BaseURL: srv.URL,
		APIKey:  "k",
		Model:   "m",
	}, llm.ChatRequest{Messages: []llm.Message{{Role: "user", Content: "x"}}})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestChatCompletions_multimodalRequest(t *testing.T) {
	t.Parallel()
	var sawUser any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []llm.Message `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if len(body.Messages) > 0 {
			sawUser = body.Messages[0].Content
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":   "m",
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": "seen"}}},
			"usage":   map[string]int{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	}))
	t.Cleanup(srv.Close)

	client := llm.NewClient()
	client.HTTP = srv.Client()
	out, err := client.ChatCompletions(context.Background(), llm.Connection{
		BaseURL: srv.URL,
		APIKey:  "k",
		Model:   "m",
	}, llm.ChatRequest{
		Messages: []llm.Message{{
			Role: "user",
			Content: []llm.ContentPart{
				llm.TextPart("describe"),
				llm.ImageDataPart("image/png", []byte{1, 2, 3}),
			},
		}},
	})
	if err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}
	if out.Content != "seen" {
		t.Fatalf("content = %q", out.Content)
	}
	raw, _ := json.Marshal(sawUser)
	if !strings.Contains(string(raw), `"type":"image_url"`) || !strings.Contains(string(raw), "data:image/png;base64,") {
		t.Fatalf("payload = %s", raw)
	}
}

func TestChatCompletions_toolCalls(t *testing.T) {
	t.Parallel()
	var sawTools bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body llm.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if len(body.Tools) != 1 || body.Tools[0].Function.Name != "lookup_sites" {
			t.Fatalf("tools = %+v", body.Tools)
		}
		sawTools = true
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "m",
			"choices": []map[string]any{
				{
					"finish_reason": "tool_calls",
					"message": map[string]any{
						"role":    "assistant",
						"content": nil,
						"tool_calls": []map[string]any{
							{
								"id":   "call_1",
								"type": "function",
								"function": map[string]string{
									"name":      "lookup_sites",
									"arguments": `{"study_id":"S1"}`,
								},
							},
						},
					},
				},
			},
			"usage": map[string]int{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
		})
	}))
	t.Cleanup(srv.Close)

	client := llm.NewClient()
	client.HTTP = srv.Client()
	out, err := client.ChatCompletions(context.Background(), llm.Connection{
		BaseURL: srv.URL,
		APIKey:  "k",
		Model:   "m",
	}, llm.ChatRequest{
		Messages: []llm.Message{{Role: "user", Content: "how many sites?"}},
		Tools: []llm.Tool{{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "lookup_sites",
				Description: "Query study sites",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"study_id": map[string]any{"type": "string"},
					},
				},
			},
		}},
	})
	if err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}
	if !sawTools {
		t.Fatal("expected tools in request")
	}
	if len(out.ToolCalls) != 1 || out.ToolCalls[0].Function.Name != "lookup_sites" {
		t.Fatalf("tool_calls = %+v", out.ToolCalls)
	}
	if out.ToolCalls[0].Function.Arguments != `{"study_id":"S1"}` {
		t.Fatalf("arguments = %q", out.ToolCalls[0].Function.Arguments)
	}
	if out.Content != "" {
		t.Fatalf("content = %q", out.Content)
	}
}

func TestPlatformDefaultFromEnv(t *testing.T) {
	t.Setenv("OPENVEEVA_LLM_BASE_URL", "https://example.test/v1")
	t.Setenv("OPENVEEVA_LLM_API_KEY", "secret")
	t.Setenv("OPENVEEVA_LLM_MODEL", "model-a")

	r := llm.NewResolver(nil)
	conn, err := r.Resolve(context.Background(), uuid.Nil, llm.PlatformDefaultName)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if conn.BaseURL != "https://example.test/v1" || conn.Model != "model-a" {
		t.Errorf("conn = %+v", conn)
	}
}
