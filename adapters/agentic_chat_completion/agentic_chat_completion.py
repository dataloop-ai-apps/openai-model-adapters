"""Agentic hosted chat completion adapter.

Model adapter with a tool-calling loop backed by Jarvis.
LLM calls go through Jarvis (/api/v1/ai/chat/completions).
Tools are fetched and executed via the MCP protocol endpoint
(/api/v1/ai/mcps/{tool_set_id}/mcp) using JSON-RPC 2.0.

Configuration (on the model entity):
    tool_set_id   — Jarvis MCP tool set slug (e.g. "cloudops")
    system_prompt — agent system prompt
    llm_type      — "Agent" marker
    max_turns     — break guard for the agentic loop (default 20)
    model_name    — LLM model to use via Jarvis, provider-namespaced
                    (default "openai/gpt-4o")
    max_tokens    — max tokens per LLM call
    temperature   — LLM temperature

The AI Playground sends LLM trace items (dltype: llm_trace) with a
messages array. The adapter reads messages from the item, runs the
agentic loop, and writes the response back.
"""

import io
import json
import logging
import os

import dtlpy as dl
import httpx
import openai
import requests
from openai import NOT_GIVEN

logger = logging.getLogger("agentic-chat-completion")

DEFAULT_MAX_TURNS = 20
_TOOL_TRIM_KEEP_CHARS = 300


def _jarvis_base_url() -> str:
    """Derive the Jarvis base URL from the Dataloop environment."""
    gate = dl.environment().rstrip("/")
    if gate:
        return f"{gate}/ai"
    return "https://gate.dataloop.ai/api/v1/ai"


def _jarvis_auth_headers() -> dict:
    """Build auth headers for Jarvis HTTP calls (MCP endpoints)."""
    return dict(dl.client_api.auth)


def _trim_old_tool_results(messages, keep_chars=_TOOL_TRIM_KEEP_CHARS):
    """Truncate tool result content for older turns to save context window."""

    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= 1:
        return messages

    cutoff = user_indices[-1]
    result = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool" and i < cutoff:
            content = msg.get("content", "")
            if len(content) > keep_chars:
                msg = {**msg, "content": content[:keep_chars] + "\n[trimmed]"}
        result.append(msg)
    return result


class AgenticChatCompletion(dl.BaseModelAdapter):
    """Agentic chat completion adapter with a tool-calling loop via Jarvis.

    No API key needed — LLM calls and tool calls go through Jarvis.
    Auth is the Dataloop platform JWT (dl.token()).
    """

    def load(self, local_path, **kwargs):
        self.adapter_defaults.upload_annotations = False

        # Jarvis LLM client — OpenAI SDK pointed at Jarvis gateway
        self.jarvis_url = _jarvis_base_url()
        token = dl.token()
        if not token:
            raise ValueError("No Dataloop token available — cannot authenticate with Jarvis")

        ssl_verify = os.environ.get("SSL_VERIFY", "true").lower() != "false"
        http_client = httpx.Client(verify=ssl_verify)
        self.client = openai.OpenAI(
            api_key=token,
            base_url=self.jarvis_url,
            http_client=http_client,
        )

        # Agent configuration from model entity
        self.system_prompt = self.configuration.get("system_prompt", "")
        self.tool_set_id = self.configuration.get("tool_set_id", "")
        self.max_turns = self.configuration.get("max_turns", DEFAULT_MAX_TURNS)
        # Jarvis namespaces models by provider (e.g. "openai/gpt-4o"); a bare
        # model name returns 404 model_not_found from /ai/chat/completions.
        self.model_name = self.configuration.get("model_name", "openai/gpt-4o")
        self.max_tokens = self.configuration.get("max_tokens", NOT_GIVEN)
        self.temperature = self.configuration.get("temperature", NOT_GIVEN)

        logger.info(
            "Loaded agentic adapter: model=%s, tool_set=%s, max_turns=%d, jarvis=%s",
            self.model_name, self.tool_set_id, self.max_turns, self.jarvis_url,
        )

    def _refresh_token(self):
        """Refresh the Jarvis client token if the platform JWT has expired."""
        if dl.token_expired():
            token = dl.token()
            self.client = openai.OpenAI(
                api_key=token,
                base_url=self.jarvis_url,
                http_client=self.client._client,
            )

    def _mcp_url(self) -> str:
        """MCP protocol endpoint for the configured tool set."""
        return f"{self.jarvis_url}/mcps/{self.tool_set_id}/mcp"

    def _mcp_call(self, method: str, params: dict = None, timeout: int = 30) -> dict:
        """Send a JSON-RPC 2.0 request to the Jarvis MCP protocol endpoint."""
        body = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params:
            body["params"] = params
        resp = requests.post(
            self._mcp_url(),
            json=body,
            headers=_jarvis_auth_headers(),
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(
                f"MCP {method} error: {data['error'].get('message', data['error'])}"
            )
        return data.get("result", {})

    def _fetch_tools(self) -> list:
        """Fetch tool schemas via the MCP protocol endpoint (tools/list).

        Returns OpenAI-compatible tool schema list, converted from the
        MCP-native {name, description, inputSchema} shape.
        """
        if not self.tool_set_id:
            logger.warning("No tool set ID configured for agentic adapter")
            return []
        try:
            result = self._mcp_call("tools/list")
            mcp_tools = result.get("tools", [])
            if not mcp_tools:
                logger.warning(
                    "Tool set '%s' returned no tools — the agent will run without tools",
                    self.tool_set_id,
                )
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("inputSchema", {}),
                    },
                }
                for t in mcp_tools
            ]
            logger.info(
                "Fetched %d tools from tool set '%s': %s",
                len(tools),
                self.tool_set_id,
                [t["function"]["name"] for t in tools],
            )
            return tools
        except Exception as exc:
            logger.error("Failed to fetch tools from %s: %s", self._mcp_url(), exc)
            return []

    def _call_tool(self, tool_name: str, arguments: str) -> str:
        """Execute a tool call via the MCP protocol endpoint (tools/call)."""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            logger.warning("Could not parse tool arguments for %s: %r", tool_name, arguments)
            args = {}

        try:
            result = self._mcp_call(
                "tools/call",
                params={"name": tool_name, "arguments": args},
                timeout=120,
            )
            content = result.get("content", [])
            if isinstance(content, list):
                return "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            return json.dumps(content) if isinstance(content, dict) else str(content)
        except Exception as exc:
            error_msg = f"[ERROR] Tool {tool_name} failed: {exc}"
            logger.error(error_msg)
            return error_msg

    def _call_model(self, messages, tools):
        """Single LLM call to Jarvis. Returns the ChatCompletion response."""
        return self.client.chat.completions.create(
            messages=messages,
            tools=tools if tools else NOT_GIVEN,
            tool_choice="auto" if tools else NOT_GIVEN,
            model=self.model_name,
            stream=False,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def prepare_item_func(self, item: dl.Item):
        """Return the raw item — no PromptItem conversion needed for LLM trace items."""
        return item

    def predict(self, batch, **kwargs):
        self._refresh_token()

        model_name = self.model_entity.name

        for item in batch:
            # Load the LLMTrace object from the item
            trace = dl.LLMTrace.from_item(item)
            messages = [{"role": getattr(msg, "role", ""), "content": getattr(msg, "content", "")}
                for msg in trace.messages]
            logger.info("Read %d messages from trace item %s", len(messages), item.id)

            # Prepend system prompt
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            # Fetch tools from Jarvis MCP
            tools = self._fetch_tools()

            # Agentic loop
            final_content = ""
            for turn in range(self.max_turns):
                logger.info("Turn %d/%d", turn + 1, self.max_turns)

                response = self._call_model(messages, tools)
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                    # Append assistant message with tool calls.
                    # Only include fields the chat completions API accepts —
                    # model_dump() includes SDK extras (refusal, annotations,
                    # audio, reasoning, …) that Jarvis rejects as excess.
                    assistant_msg = {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in choice.message.tool_calls
                        ],
                    }
                    messages.append(assistant_msg)

                    # Execute each tool call via Jarvis MCP
                    for tc in choice.message.tool_calls:
                        logger.info("Calling tool: %s", tc.function.name)
                        result = self._call_tool(tc.function.name, tc.function.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })

                    # Trim older tool results to save context window
                    messages = _trim_old_tool_results(messages)
                else:
                    # Final answer
                    final_content = choice.message.content or ""
                    logger.info("Agent finished after %d turn(s)", turn + 1)
                    break
            else:
                logger.warning("Agent reached max_turns (%d) without final answer", self.max_turns)
                final_content = "Agent reached maximum turns without producing a final answer."

            # Append final assistant message to the trace and update the item
            trace.add_message(dl.LLMMessage(role="assistant", content=final_content))
            trace.update()
            logger.info("Updated trace item %s with assistant response (%d chars)", item.id, len(final_content))

            # Keep annotation as backup for compatibility
            prompt_id = str(sum(1 for m in messages if m.get("role") == "user"))
            builder = item.annotations.builder()
            builder.add(
                annotation_definition=dl.FreeText(text=final_content),
                prompt_id=prompt_id,
                model_info={
                    "name": model_name,
                    "model_id": self.model_entity.id,
                    "confidence": 1.0,
                },
            )
            item.annotations.upload(builder)
            logger.info("Uploaded response annotation as backup (promptId=%s)", prompt_id)

        return []
