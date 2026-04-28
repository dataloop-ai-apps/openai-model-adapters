# Hosted Text Embeddings

Text embeddings adapter for **Dataloop-hosted** OpenAI-compatible services (e.g. Ollama running on Dataloop infrastructure). Unlike the base `text_embeddings` adapter which calls OpenAI directly with an API key, this adapter connects to a service deployed inside the Dataloop platform via app-service routing.

## How it works

1. **No API key needed** -- authentication is handled through Dataloop's JWT-APP cookie mechanism, not an OpenAI API key.
2. **Endpoint resolution** -- on `load()`, the adapter resolves the actual service URL by following the Dataloop gateway redirect chain (gateway -> login -> service). The resolved URL is an OpenAI-compatible `/v1` endpoint.
3. **JWT lifecycle** -- before each `embed()` call, the JWT-APP cookie expiration is checked. If it's about to expire, the session and OpenAI client are automatically rebuilt.
4. **No dimensions parameter** -- unlike the base adapter, this adapter does not send the `dimensions` parameter to the embeddings API, since hosted services like Ollama may not support it. The embedding size is determined by the model itself.

## Warmup

Hosted model servers (e.g. Ollama) may need to load model weights into memory on the first request, which can take significant time and cause gateway timeouts (504).

To avoid this, the adapter performs a **native warmup** before the first `/v1/embeddings` call:

- Sends a `POST /api/embeddings` request directly to the Ollama native API with a minimal input (`"."`)
- Runs once per session, and re-runs after JWT refresh triggers a session rebuild
- Failures are logged as warnings but do not block inference

## Configuration

Set these in the model entity configuration:

| Key | Default | Description |
|-----|---------|-------------|
| `app_id` | *(required)* | Dataloop app ID of the hosted service |
| `model_name` | *(required)* | Model identifier on the hosted service (e.g. `nomic-embed-text:latest`) |
| `timeout` | `900` | HTTP read timeout in seconds for inference calls |

## Architecture

```
HostedTextEmbeddings (hosted_text_embeddings.py)
  └── extends TextEmbeddings (text_embeddings.py)
        └── extends dl.BaseModelAdapter

DataloopAppServiceClient (common/dataloop_app_service.py)
  - Resolves app-service endpoint
  - Manages JWT-APP cookie session
  - Creates OpenAI client with cookie auth
  - Provides native warmup methods
```
