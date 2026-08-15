package llm_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vivarcus/vivarcus-llm"
)

func TestModelAcceptsImageURL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		model string
		want  bool
	}{
		{"", true},
		{"gpt-4o", true},
		{"qwen-vl-max", true},
		{"qwen3-vl-plus", true},
		{"qwen2.5-vl-72b-instruct", true},
		{"qwen3.7-plus", true},
		{"qwen3.7-max", false},
		{"qwen-max", false},
		{"qwen3-max", false},
		{"qwen3-coder-plus", false},
	}
	for _, tc := range cases {
		if got := llm.ModelAcceptsImageURL(tc.model); got != tc.want {
			t.Fatalf("ModelAcceptsImageURL(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

func TestSanitizeMessagesForModel_stripsImagesForTextOnlyQwen(t *testing.T) {
	t.Parallel()
	msgs := []llm.Message{{
		Role: "user",
		Content: []llm.ContentPart{
			llm.TextPart("describe"),
			llm.ImageDataPart("image/png", []byte{1, 2, 3}),
		},
	}}
	out := llm.SanitizeMessagesForModel("qwen3.7-max", msgs)
	if len(out) != 1 {
		t.Fatalf("len = %d", len(out))
	}
	raw, _ := json.Marshal(out[0].Content)
	if string(raw) != `"describe"` {
		t.Fatalf("content = %s", raw)
	}
}

func TestSanitizeMessagesForModel_keepsImagesForVisionModel(t *testing.T) {
	t.Parallel()
	msgs := []llm.Message{{
		Role: "user",
		Content: []llm.ContentPart{
			llm.TextPart("describe"),
			llm.ImageDataPart("image/png", []byte{1, 2, 3}),
		},
	}}
	out := llm.SanitizeMessagesForModel("qwen-vl-max", msgs)
	raw, _ := json.Marshal(out[0].Content)
	if !jsonContains(raw, `"type":"image_url"`) {
		t.Fatalf("content = %s", raw)
	}
}

func TestSanitizeMessagesForModel_nilContentBecomesEmptyString(t *testing.T) {
	t.Parallel()
	out := llm.SanitizeMessagesForModel("qwen3.7-max", []llm.Message{{
		Role:    "assistant",
		Content: nil,
	}})
	raw, _ := json.Marshal(out[0])
	if !jsonContains(raw, `"content":""`) {
		t.Fatalf("message = %s", raw)
	}
}

func jsonContains(raw []byte, needle string) bool {
	return strings.Contains(string(raw), needle)
}
