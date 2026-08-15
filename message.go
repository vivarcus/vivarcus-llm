package llm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
)

// ContentPart is one OpenAI-compatible multimodal content part.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL holds a remote or data-URL image reference.
type ImageURL struct {
	URL string `json:"url"`
}

// TextPart returns a text content part.
func TextPart(text string) ContentPart {
	return ContentPart{Type: "text", Text: text}
}

// ImageDataPart encodes raw image bytes as a data URL content part.
func ImageDataPart(mediaType string, data []byte) ContentPart {
	mediaType = strings.TrimSpace(mediaType)
	if mediaType == "" {
		mediaType = "image/png"
	}
	return ContentPart{
		Type: "image_url",
		ImageURL: &ImageURL{
			URL: "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data),
		},
	}
}

// TextContent returns string content when Content is a plain string.
func (m Message) TextContent() string {
	switch v := m.Content.(type) {
	case string:
		return v
	case nil:
		return ""
	default:
		return strings.TrimSpace(fmt.Sprint(v))
	}
}

func assistantContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []ContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "text" && p.Text != "" {
				if b.Len() > 0 {
					b.WriteByte('\n')
				}
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return ""
}
