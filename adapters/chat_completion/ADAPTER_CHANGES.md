# Chat Completion Adapter: LLMTrace vs PromptItem

This document explains the differences between the legacy `chat_completion.py` (PromptItem-based) and the new `chat_completion_llm_trace.py` (LLMTrace-based) adapter.

Related Jira: **DAT-130200** / **DAT-130205**

---

## Data model

| Aspect | `chat_completion.py` | `chat_completion_llm_trace.py` |
|---|---|---|
| Item wrapper | `dl.PromptItem` | `dl.LLMTrace` |
| Message format | Custom prompts converted via `to_messages()` | Native OpenAI-format dicts (`trace.messages`) |
| Response storage | Annotations on the item (`prompt_item.add(...)`) | Messages in the trace JSON body (`trace.add_message(...)` + `trace.update()`) |
| Persistence mechanism | Annotation-based; each `add()` call creates/updates an annotation | Binary JSON update; `trace.update()` pushes the full body to the platform |

## Message handling

**Legacy (`chat_completion.py`)**

```python
messages = prompt_item.to_messages(model_name=model_name)
```

`PromptItem.to_messages()` converts the internal prompt structure into OpenAI-compatible message dicts. A separate conversion step is required.

**New (`chat_completion_llm_trace.py`)**

```python
messages = list(trace.messages)
```

`LLMTrace.messages` already returns a flat list of OpenAI-style dicts (`role` + `content` + extras). No conversion needed.

## Streaming update strategy

This is the most significant behavioral change.

**Legacy -- update on every token**

```python
for chunk in stream_response:
    response += chunk
    prompt_item.add(message={...}, model_info={...})
```

Every token triggers a platform HTTP request via `prompt_item.add()`. Under heavy load or long responses this creates excessive network traffic.

**New -- throttled updates (every ~3 seconds)**

```python
STREAM_THROTTLE_SECONDS = 3.0

for chunk in stream_response:
    response += chunk
    if time.time() - last_update >= STREAM_THROTTLE_SECONDS:
        # update trace message in-place, then persist
        trace.update()
        last_update = now

# guaranteed final update after stream ends
trace.update()
```

Tokens are accumulated locally. The platform is updated at most once every 3 seconds during streaming, plus one final update when the stream completes. Mid-stream `trace.update()` failures are caught and logged as warnings so the generation is not interrupted.

## RAG / nearest-items context

**Legacy** reads `nearestItems` from prompt metadata and appends a RAG context message:

```python
nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
if len(nearest_items) > 0:
    context = prompt_item.build_context(nearest_items=nearest_items, ...)
    messages.append({"role": "assistant", "content": context})
```

**New** does not include RAG context. `LLMTrace` does not have an equivalent metadata path for `nearestItems` yet. This can be added later when the platform supports it for traces.

## Model info

**Legacy** passes `model_info` into each `prompt_item.add()` call, which stores it as annotation metadata.

**New** attaches `model_info` as an extra field on the `LLMMessage` kwargs:

```python
dl.LLMMessage(
    role="assistant",
    content=response,
    model_info={"name": model_name, "model_id": self.model_entity.id},
)
```

Since LLMTrace has no annotations, model info lives directly on the message dict.

## Error handling

**Legacy** has no error handling around `prompt_item.add()` calls -- a platform failure during streaming will propagate and stop the generation.

**New** wraps mid-stream `trace.update()` in `try/except` with a warning log so transient platform errors do not abort the response. The final `trace.update()` after the stream completes does propagate errors normally.

## Summary of what stayed the same

- `load()` -- identical OpenAI client setup.
- `call_model()` -- identical streaming/non-streaming OpenAI API call.
- Overall predict loop structure (iterate batch, prepend system prompt, stream, accumulate response).
