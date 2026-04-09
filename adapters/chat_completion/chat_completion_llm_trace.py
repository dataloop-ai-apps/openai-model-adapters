from datetime import datetime, timezone
from openai import NOT_GIVEN
import dtlpy as dl
import logging
import openai
import json
import math
import time
import os

logger = logging.getLogger('openai-adapter')

STREAM_THROTTLE_SECONDS = 3.0
MAX_TOOL_ROUNDS = 10

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current UTC date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression and returns the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g. '2 ** 10 + math.sqrt(144)'.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        self.adapter_defaults.upload_annotations = False
        if os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError("Missing API key")
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call_model(self, messages, tools=NOT_GIVEN):
        stream = self.configuration.get("stream", True)
        max_tokens = self.configuration.get("max_tokens", NOT_GIVEN)
        temperature = self.configuration.get("temperature", NOT_GIVEN)
        top_p = self.configuration.get("top_p", NOT_GIVEN)
        model_name = self.configuration.get("model_name", "gpt-4o")

        stream_options = {"include_usage": True} if stream else NOT_GIVEN
        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            model=model_name,
            stream_options=stream_options,
            tools=tools,
        )
        self._completion_info = {}
        self._tool_calls = []

        if stream is True:
            tool_calls_accum = {}
            for chunk in response:
                if chunk.usage is not None:
                    self._completion_info["usage"] = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if not self._completion_info.get("completion_id"):
                    self._completion_info["completion_id"] = chunk.id
                    self._completion_info["model"] = chunk.model
                if choice.finish_reason is not None:
                    self._completion_info["finish_reason"] = choice.finish_reason
                if choice.delta.tool_calls:
                    for tc_delta in choice.delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = tool_calls_accum[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                yield choice.delta.content or ""
            self._tool_calls = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
        else:
            choice = response.choices[0]
            self._completion_info = {
                "completion_id": response.id,
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
            if choice.message.tool_calls:
                self._tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ]
            yield choice.message.content or ""

    def prepare_item_func(self, item: dl.Item):
        return dl.LLMTrace.from_item(item)

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get("system_prompt", "")
        model_name = self.model_entity.name

        for trace in batch:
            messages = list(trace.messages)
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            add_metadata = self.configuration.get("add_metadata")
            context = trace.build_context(add_metadata=add_metadata)
            if context:
                messages.append({"role": "assistant", "content": context})

            for tool_round in range(MAX_TOOL_ROUNDS):
                stream_response = self.call_model(messages=messages, tools=TOOLS)
                response = self._stream_to_trace(trace, stream_response, model_name)

                if self._completion_info.get("finish_reason") == "tool_calls" and self._tool_calls:
                    assistant_msg = {
                        "role": "assistant",
                        "content": response or None,
                        "tool_calls": self._tool_calls,
                    }
                    trace.add_message(assistant_msg)
                    messages.append(assistant_msg)

                    for tc in self._tool_calls:
                        result = self._execute_tool(tc["function"]["name"], tc["function"]["arguments"])
                        tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": result}
                        trace.add_message(tool_msg)
                        messages.append(tool_msg)

                    try:
                        trace.update()
                    except Exception:
                        logger.warning("Failed to update trace after tool round", exc_info=True)
                    continue

                self._finalize_assistant_message(trace, response, model_name)
                trace.update()
                break
            else:
                logger.warning("Reached max tool rounds (%d), forcing final response", MAX_TOOL_ROUNDS)
                self._finalize_assistant_message(trace, response, model_name)
                trace.update()

        return []

    def _stream_to_trace(self, trace, stream_response, model_name):
        """Consume a stream with throttled trace updates. Returns the full accumulated text."""
        response = ""
        last_update = time.time()
        assistant_msg_added = False

        for chunk in stream_response:
            response += chunk
            now = time.time()
            if now - last_update >= STREAM_THROTTLE_SECONDS:
                assistant_msg_added = self._update_trace_assistant_message(
                    trace, response, model_name, assistant_msg_added
                )
                try:
                    trace.update()
                except Exception:
                    logger.warning("Failed to update trace mid-stream", exc_info=True)
                last_update = now

        if assistant_msg_added:
            trace.messages[-1].pop("_streaming", None)
            trace.messages[-1]["content"] = response

        return response

    def _update_trace_assistant_message(self, trace, content, model_name, already_added):
        """Add or update the in-progress assistant message during streaming.

        Returns True to indicate the message has been added.
        """
        if not already_added:
            trace.add_message(dl.LLMMessage(
                role="assistant",
                content=content,
                _streaming=True,
                model_info={"name": model_name, "model_id": self.model_entity.id},
            ))
        else:
            trace.messages[-1]["content"] = content
        return True

    def _finalize_assistant_message(self, trace, content, model_name):
        """Add the final assistant message to the trace with model and completion info."""
        trace.add_message(dl.LLMMessage(
            role="assistant",
            content=content,
            model_info={"name": model_name, "model_id": self.model_entity.id},
            completion_info=self._completion_info,
        ))

    def _execute_tool(self, name, arguments_json):
        """Dispatch a tool call and return the result as a JSON string."""
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid arguments JSON: {arguments_json}"})

        if name == "get_current_time":
            return json.dumps({"utc_time": datetime.now(timezone.utc).isoformat()})

        if name == "calculate":
            expression = args.get("expression", "")
            try:
                safe_ns = {"__builtins__": {}, "math": math}
                result = eval(expression, safe_ns)  # noqa: S307
                return json.dumps({"result": result})
            except Exception as exc:
                return json.dumps({"error": str(exc)})

        return json.dumps({"error": f"Unknown tool: {name}"})


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    dl.setenv("prod")
    # dl.logout()
    if dl.token_expired():
        dl.login()

    model = dl.models.get(model_id="69c7a6d26b4d6a4adc88955b")
    item = dl.items.get(item_id="69c7a8ca9986f9db2631392a")
    adapter = ModelAdapter(model)
    adapter.predict_items([item])
