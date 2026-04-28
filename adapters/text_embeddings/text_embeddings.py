import logging
import os

import dtlpy as dl
import openai

logger = logging.getLogger("openai-text-embeddings")


class TextEmbeddings(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        self._app_service = None
        self.timeout = self.configuration.get("timeout", 900)

        if os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError("Missing API key: set OPENAI_API_KEY env var")

        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call_model(self, text):
        model_name = self.configuration.get("model_name", "text-embedding-3-large")
        create_kwargs = {
            "input": text,
            "model": model_name,
        }
        if self._app_service is None:
            create_kwargs["dimensions"] = self.model_entity.configuration.get(
                "embeddings_size", 256
            )
        create_kwargs["timeout"] = self.timeout
        response = self.client.embeddings.create(**create_kwargs)
        embedding = response.data[0].embedding
        return embedding

    def _hosted_inference_prepare(self):
        """Hosted adapters override (native warmup before first /v1 call)."""

    def embed(self, batch, **kwargs):
        # Re-sync client after JWT refresh (hosted subclass may rebuild it)
        if self._app_service is not None:
            self._app_service.check_jwt_expiration()
            self.client = self._app_service.client
            self._hosted_inference_prepare()

        hyde_model_name = self.configuration.get("hyde_model_name")
        embeddings = []

        for item in batch:
            if isinstance(item, str):
                text = item
            else:
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get("prompt", dict()).get("is_hyde", False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(model_name=hyde_model_name)[-1]
                        if messages["role"] == "assistant":
                            text = messages["content"][-1]["text"]
                        else:
                            raise ValueError(
                                "Only assistant messages are supported for hyde model"
                            )
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages["content"][-1]["text"]

                except ValueError as e:
                    raise ValueError(
                        f"Only mimetype text or prompt items are supported {e}"
                    )

            embedding = self.call_model(text=text)
            logger.info("Extracted embeddings for text %s: %s", item, embedding)
            embeddings.append(embedding)

        return embeddings
