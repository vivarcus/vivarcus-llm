# Vivarcus LLM

LLM integration layer for Go, extracted from the [Vivarcus](https://vivarcus.com/) platform — a configurable digital platform for life sciences.

## What's inside

| File | Purpose |
|------|---------|
| `model.go` | Model capability helpers (multimodal detection, DashScope/Qwen compatibility quirks) |
| `message.go` | OpenAI-style message structures |
| `resolve.go` | Connection resolution — PostgreSQL-backed config with env fallback (`VIVARCUS_LLM_*`; legacy `OPENVEEVA_LLM_*` accepted) |
| `client.go` | Streaming chat client |

## Usage

```go
import "github.com/vivarcus/vivarcus-llm"
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
