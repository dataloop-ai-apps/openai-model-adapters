"""Agentic hosted chat completion adapter.

Model adapter with a tool-calling loop backed by Jarvis.
LLM calls go through Jarvis (/api/v1/ai/chat/completions).
Tools are fetched and executed via Jarvis MCP gateway
(/api/v1/ai/mcps/{tool_set_id}).

Configuration (on the model entity):
    tool_set_id   — Jarvis MCP tool set slug (e.g. "cloudops")
    system_prompt — agent system prompt
    llm_type      — "Agent" marker
    max_turns     — break guard for the agentic loop (default 20)
    model_name    — LLM model to use via Jarvis (default "gpt-4o")
    max_tokens    — max tokens per LLM call
    temperature   — LLM temperature

Loop (per prompt_item):
    messages = [system_prompt] + prompt_item.to_messages()
    tools    = GET /api/v1/ai/mcps/{tool_set_id}
    for turn in range(max_turns):
        response = jarvis.chat.completions.create(messages, tools)
        if tool_calls: execute via Jarvis MCP, append results, continue
        if stop: record final answer via prompt_item.add(), break
"""

import json
import logging
import os

import dtlpy as dl
import httpx
import openai
import requests
from openai import NOT_GIVEN

logger = logging.getLogger("agentic-hosted-chat-completion")

DEFAULT_MAX_TURNS = 20
_TOOL_TRIM_KEEP_CHARS = 300


def _jarvis_base_url() -> str:
    """Derive the Jarvis base URL from the Dataloop environment.

    Same logic as Donna v3 llm.py — dl.environment() + "/ai".
    JARVIS_BASE_URL env var overrides for testing.
    """
    override = os.environ.get("JARVIS_BASE_URL", "").strip()
    if override:
        return override.rstrip("/")
    try:
        gate = dl.environment().rstrip("/")
        if gate:
            return f"{gate}/ai"
    except Exception as exc:
        logger.debug("dl.environment() unavailable: %s", exc)
    return "https://gate.dataloop.ai/api/v1/ai"


def _jarvis_auth_headers() -> dict:
    """Build auth headers for Jarvis HTTP calls (MCP endpoints)."""
    return dict(dl.client_api.auth)


def _trim_old_tool_results(messages, keep_chars=_TOOL_TRIM_KEEP_CHARS):
    """Truncate tool result content for older turns to save context window.

    Same pattern as Donna v3 _trim_old_tool_results — the LLM has already
    synthesised older tool outputs into assistant messages, so full bytes
    only bloat the context.
    """
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


class AgenticHostedChatCompletion(dl.BaseModelAdapter):
    """Agentic model adapter with a tool-calling loop via Jarvis.

    No API key needed — LLM calls and tool calls go through Jarvis.
    Auth is the Dataloop platform JWT (dl.token()).
    """

    def load(self, local_path, **kwargs):
        self.adapter_defaults.upload_annotations = False

        # Jarvis LLM client — OpenAI SDK pointed at Jarvis gateway
        jarvis_url = _jarvis_base_url()
        token = dl.token()
        if not token:
            raise ValueError("No Dataloop token available — cannot authenticate with Jarvis")

        ssl_verify = os.environ.get("SSL_VERIFY", "true").lower() != "false"
        http_client = httpx.Client(verify=ssl_verify)
        self.client = openai.OpenAI(
            api_key=token,
            base_url=jarvis_url,
            http_client=http_client,
        )
        self.jarvis_url = jarvis_url

        # Agent configuration from model entity
        self.system_prompt = self.configuration.get("system_prompt", "")
        self.tool_set_id = self.configuration.get("tool_set_id", "")
        self.max_turns = self.configuration.get("max_turns", DEFAULT_MAX_TURNS)
        self.model_name = self.configuration.get("model_name", "gpt-4o")
        self.max_tokens = self.configuration.get("max_tokens", NOT_GIVEN)
        self.temperature = self.configuration.get("temperature", NOT_GIVEN)

        logger.info(
            "Loaded agentic adapter: model=%s, tool_set=%s, max_turns=%d, jarvis=%s",
            self.model_name, self.tool_set_id, self.max_turns, jarvis_url,
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

    def _fetch_tools(self) -> list:
        """Fetch tool schemas from Jarvis MCP gateway.

        GET /api/v1/ai/mcps/{tool_set_id} → {"tools": [...]}
        Returns OpenAI-compatible tool schema list.
        """
        if not self.tool_set_id:
            return []
        url = f"{self.jarvis_url}/mcps/{self.tool_set_id}"
        headers = _jarvis_auth_headers()
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            tools = data.get("tools", [])
            logger.info("Fetched %d tools from tool set '%s'", len(tools), self.tool_set_id)
            return tools
        except Exception as exc:
            logger.error("Failed to fetch tools from %s: %s", url, exc)
            return []

    def _call_tool(self, tool_name: str, arguments: str) -> str:
        """Execute a tool call via Jarvis MCP gateway.

        POST /api/v1/ai/mcps/{tool_set_id}
        Body: {"tool": "<tool_name>", "arguments": {...}}
        Returns the tool result as a string.
        """
        url = f"{self.jarvis_url}/mcps/{self.tool_set_id}"
        headers = _jarvis_auth_headers()
        headers["Content-Type"] = "application/json"

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            logger.warning("Could not parse tool arguments for %s: %r", tool_name, arguments)
            args = {}

        body = {"tool": tool_name, "arguments": args}
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # MCP result format: {"result": {"content": [...]}}
            result = data.get("result", data)
            if isinstance(result, dict):
                content = result.get("content", result)
                if isinstance(content, list):
                    return "\n".join(
                        item.get("text", str(item)) if isinstance(item, dict) else str(item)
                        for item in content
                    )
                return json.dumps(content)
            return str(result)
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
        return dl.PromptItem.from_item(item)

    def predict(self, batch, **kwargs):
        self._refresh_token()

        model_name = self.model_entity.name

        for prompt_item in batch:
            messages = prompt_item.to_messages(model_name=model_name)

            # Prepend system prompt
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            # Fetch tools from Jarvis MCP
            tools = self._fetch_tools()

            # Agentic loop
            for turn in range(self.max_turns):
                logger.info("Turn %d/%d", turn + 1, self.max_turns)

                response = self._call_model(messages, tools)
                choice = response.choices[0]

                if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                    # Append assistant message with tool calls
                    messages.append(choice.message.model_dump())

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

                    prompt_item.add(
                        message={"role": "assistant",
                                 "content": [{"mimetype": dl.PromptType.TEXT,
                                              "value": final_content}]},
                        model_info={"name": model_name,
                                    "confidence": 1.0,
                                    "model_id": self.model_entity.id},
                    )
                    break
            else:
                # Exhausted max_turns without a final answer
                logger.warning("Agent reached max_turns (%d) without final answer", self.max_turns)
                prompt_item.add(
                    message={"role": "assistant",
                             "content": [{"mimetype": dl.PromptType.TEXT,
                                          "value": "Agent reached maximum turns without producing a final answer."}]},
                    model_info={"name": model_name,
                                "confidence": 1.0,
                                "model_id": self.model_entity.id},
                )

        return []
