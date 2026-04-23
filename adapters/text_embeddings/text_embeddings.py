import os
import sys

import dtlpy as dl
import openai
import logging

_adapters = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _adapters not in sys.path:
    sys.path.insert(0, _adapters)
from common.dataloop_app_service import DataloopAppServiceClient  # noqa: E402

logger = logging.getLogger("openai-text-embeddings")


class TextEmbeddings(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """Load configuration for OpenAI adapter or a Dataloop downloadable app (app_id)."""
        self._app_service = None
        self.using_app_service = False

        if self.configuration.get("app_id"):
            self.using_app_service = True
            self._app_service = DataloopAppServiceClient(
                self.configuration["app_id"],
                self.model_entity,
                logger,
            )
            self.client = self._app_service.client
            return

        if os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError(
                "Missing API key: set OPENAI_API_KEY or use app_id for an app service"
            )

        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call_model(self, text):
        model_name = self.configuration.get("model_name", "text-embedding-3-large")
        create_kwargs = {
            "input": text,
            "model": model_name,
        }
        if not self.using_app_service:
            create_kwargs["dimensions"] = self.model_entity.configuration.get(
                "embeddings_size", 256
            )
        response = self.client.embeddings.create(**create_kwargs)
        embedding = response.data[0].embedding
        return embedding

    def embed(self, batch, **kwargs):
        if self.using_app_service and self._app_service is not None:
            self._app_service.check_jwt_expiration()
            self.client = self._app_service.client

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
