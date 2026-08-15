package llm

import "strings"

// ModelAcceptsImageURL reports whether the model accepts OpenAI-style image_url
// content parts. DashScope compatible-mode text-only Qwen models (e.g. qwen3.7-max)
// reject image_url with "Unexpected item type in content".
func ModelAcceptsImageURL(model string) bool {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return true
	}
	// Explicit multimodal markers (Qwen-VL / Omni / audio).
	if strings.Contains(m, "-vl") || strings.Contains(m, "vl-") ||
		strings.Contains(m, "-omni") || strings.Contains(m, "omni-") ||
		strings.Contains(m, "-audio") || strings.Contains(m, "audio-") {
		return true
	}
	if strings.Contains(m, "qwen") {
		// Text-only families on DashScope consumer compatible-mode.
		if strings.Contains(m, "coder") {
			return false
		}
		// qwen*-max without -vl (qwen3.7-max, qwen-max, qwen3-max, …).
		if strings.Contains(m, "-max") {
			return false
		}
	}
	return true
}

// SanitizeMessagesForModel adapts messages for the target model:
//   - replaces nil content with "" (some providers reject content:null)
//   - strips image_url parts for text-only models and collapses remaining text to a string
func SanitizeMessagesForModel(model string, msgs []Message) []Message {
	if len(msgs) == 0 {
		return msgs
	}
	acceptImages := ModelAcceptsImageURL(model)
	out := make([]Message, len(msgs))
	for i, m := range msgs {
		out[i] = m
		if m.Content == nil {
			out[i].Content = ""
			continue
		}
		if acceptImages {
			continue
		}
		parts, ok := contentParts(m.Content)
		if !ok {
			continue
		}
		var text strings.Builder
		droppedImage := false
		for _, p := range parts {
			if p.Type == "image_url" || p.ImageURL != nil {
				droppedImage = true
				continue
			}
			if p.Text == "" {
				continue
			}
			if text.Len() > 0 {
				text.WriteByte('\n')
			}
			text.WriteString(p.Text)
		}
		if droppedImage {
			out[i].Content = text.String()
		}
	}
	return out
}

func contentParts(content any) ([]ContentPart, bool) {
	switch v := content.(type) {
	case []ContentPart:
		return v, true
	case []any:
		out := make([]ContentPart, 0, len(v))
		for _, raw := range v {
			switch p := raw.(type) {
			case ContentPart:
				out = append(out, p)
			case map[string]any:
				part := ContentPart{Type: strings.TrimSpace(stringFromAny(p["type"]))}
				part.Text = stringFromAny(p["text"])
				if img, ok := p["image_url"].(map[string]any); ok {
					part.ImageURL = &ImageURL{URL: stringFromAny(img["url"])}
					if part.Type == "" {
						part.Type = "image_url"
					}
				}
				out = append(out, part)
			default:
				return nil, false
			}
		}
		return out, true
	default:
		return nil, false
	}
}

func stringFromAny(v any) string {
	s, _ := v.(string)
	return s
}
