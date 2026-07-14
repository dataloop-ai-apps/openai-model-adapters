import os
import logging

import httpx
import dtlpy as dl
import openai
from openai import NOT_GIVEN

logger = logging.getLogger("openai-adapter")


def _to_openai_messages(messages):
    payload = []
    for msg in messages:
        payload.append(msg.to_json())
    return payload


def _get_role(message):
    return getattr(message, "role", None)


class ModelAdapter(dl.BaseModelAdapter):
    def _call_model(self, messages):
        response = self.client.chat.completions.create(
            messages=_to_openai_messages(messages),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=self.stream,
            model=self.model_name
        )
        if self.stream is False:
            yield response.choices[0].message.content or ""
        else:
            for chunk in response:
                if not chunk.choices:
                    continue
                yield chunk.choices[0].delta.content or ""
        
    def _apply_context(self, trace, messages):
        """Apply built context either by injecting into user message or adding a system message."""
        # trace.build_context will build context for the last message by default
        last_msg_context = trace.build_context(metadata_fields=self.add_metadata)
        logger.info("Last message context: %s", last_msg_context)
        if not last_msg_context:
            return

        if self.inject_context_to_user:
            if messages and _get_role(messages[-1]) == 'user':
                last_message = messages[-1]
                last_msg_content = last_message.content
                if isinstance(last_msg_content, list):
                    for part in last_msg_content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            part['text'] += f"\n\nContext:\n{last_msg_context}"
                            break
                else:
                    last_message.content = (last_msg_content or "") + f"\n\nContext:\n{last_msg_context}"
                logger.info("Injected context to user message: %s", last_message.content)
                if trace.messages and len(trace.messages) > 0:
                    trace.messages[-1].context = None
                return

            logger.warning(
                "Could not inject context: last message role is '%s', expected 'user'. Falling back to system message.",
                _get_role(messages[-1]) if messages else None
            )

        logger.info(
            "Adding context as system message: Use the following context to answer the user:\n%s",
            last_msg_context
        )
        messages.insert(-1, dl.LLMMessage(
            role="system",
            content=f"Use the following context to answer the user:\n{last_msg_context}"
        ))

        # Clear context from the message to prevent accumulation on re-runs
        if trace.messages and len(trace.messages) > 0:
            trace.messages[-1].context = None

    def load(self, local_path, **kwargs):
        self.adapter_defaults.upload_annotations = False
        self._app_service = None
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("Missing API key: set OPENAI_API_KEY env var")

        ssl_verify = os.environ.get("SSL_VERIFY", "true").lower() != "false"
        http_client = httpx.Client(verify=ssl_verify)
        self.client = openai.OpenAI(api_key=api_key, http_client=http_client)
        self.system_prompt = self.configuration.get('system_prompt')
        self.add_metadata = self.configuration.get("add_metadata")
        self.include_assistant = self.configuration.get("include_assistant", True)
        self.inject_context_to_user = self.configuration.get("inject_context_to_user", False)

        # Model completions configurations
        self.stream = self.configuration.get("stream", True)
        self.max_tokens = self.configuration.get("max_tokens", NOT_GIVEN)
        self.temperature = self.configuration.get("temperature", NOT_GIVEN)
        self.top_p = self.configuration.get("top_p", NOT_GIVEN)
        self.model_name = self.configuration.get("model_name", 'gpt-4o')
        self.stream_throttle_seconds = self.configuration.get("stream_throttle_seconds", 3.0)

    def generate(self, batch, **kwargs):
        if self._app_service is not None:
            self._app_service.check_jwt_expiration()
            self.client = self._app_service.client

        model_name = self.model_entity.name

        for trace in batch:
            messages = trace.messages
            if self.system_prompt:
                # Check if system_prompt already exists in messages; if not, insert at index 0
                has_matching = False
                for msg in messages:
                    if _get_role(msg) == 'system' and msg.content == self.system_prompt:
                        has_matching = True
                        break
                if not has_matching:
                    logger.info(f"Adding system prompt: {self.system_prompt}")
                    messages.insert(0, dl.LLMMessage(role="system", content=self.system_prompt))

            self._apply_context(trace, messages)

            if not self.include_assistant and messages and _get_role(messages[-1]) == 'assistant':
                messages = messages[:-1]

            logger.info(f"Sending messages to model: {messages}")
            stream = self._call_model(messages=messages)

            model_info = {
                'name': model_name,
                'confidence': 1.0,
                'model_id': self.model_entity.id
            }

            trace.stream_response(
                stream=stream,
                role="assistant",
                model_info=model_info,
                update_interval=self.stream_throttle_seconds,
            )

        return batch

