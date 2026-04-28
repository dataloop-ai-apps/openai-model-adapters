# Hosted Chat Completion

Chat completion adapter for **Dataloop-hosted** OpenAI-compatible services (e.g. Ollama running on Dataloop infrastructure). Unlike the base `chat_completion` adapter which calls OpenAI directly with an API key, this adapter connects to a service deployed inside the Dataloop platform via app-service routing.

## How it works

1. **No API key needed** -- authentication is handled through Dataloop's JWT-APP cookie mechanism, not an OpenAI API key.
2. **Endpoint resolution** -- on `load()`, the adapter resolves the actual service URL by following the Dataloop gateway redirect chain (gateway -> login -> service). The resolved URL is an OpenAI-compatible `/v1` endpoint.
3. **JWT lifecycle** -- before each `predict()` call, the JWT-APP cookie expiration is checked. If it's about to expire, the session and OpenAI client are automatically rebuilt.

## Warmup

Hosted model servers (e.g. Ollama) may need to load model weights into memory on the first request, which can take significant time and cause gateway timeouts (504).

To avoid this, the adapter performs a **native warmup** before the first `/v1/chat/completions` call:

- Sends a `POST /api/generate` request directly to the Ollama native API with an empty prompt
- Uses the `keep_alive` configuration value (default `10m`) to keep the model loaded in memory
- Runs before every `predict()` call — if the model is already loaded this returns immediately; if it was evicted it reloads before inference
- Failures are logged as warnings but do not block inference

## Configuration

Set these in the model entity configuration:

| Key | Default | Description |
|-----|---------|-------------|
| `app_id` | *(required)* | Dataloop app ID of the hosted service |
| `model_name` | `phi4-mini:latest` | Model identifier on the hosted service |
| `timeout` | `900` | HTTP read timeout in seconds for inference calls |
| `keep_alive` | `10m` | Ollama `keep_alive` duration for warmup requests |
| `stream` | `true` | Stream chat completion responses |
| `max_tokens` | `4096` | Maximum tokens in the response |
| `temperature` | `1.0` | Sampling temperature |
| `top_p` | `1.0` | Nucleus sampling parameter |
| `system_prompt` | `"You are a helpful assistant..."` | System prompt prepended to every conversation |

## Architecture

```
HostedChatCompletion (hosted_chat_completion.py)
  └── extends ModelAdapter (chat_completion.py)
        └── extends dl.BaseModelAdapter

DataloopAppServiceClient (common/dataloop_app_service.py)
  - Resolves app-service endpoint
  - Manages JWT-APP cookie session
  - Creates OpenAI client with cookie auth
  - Provides native warmup methods
```
