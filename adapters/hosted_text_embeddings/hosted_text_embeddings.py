import logging

from adapters.text_embeddings.text_embeddings import TextEmbeddings
from adapters.common.dataloop_app_service import DataloopAppServiceClient

logger = logging.getLogger("hosted-text-embeddings")


class HostedTextEmbeddings(TextEmbeddings):
    """Text embeddings adapter for Dataloop-hosted models (OpenAI-compatible API, no API key)."""

    def load(self, local_path, **kwargs):
        app_id = self.configuration.get("app_id")
        if not app_id:
            raise ValueError("app_id is required for hosted text embeddings")

        model_name = self.configuration.get("model_name", "nomic-embed-text:latest")
        logger.info("Loading hosted text embeddings, model: %s, app_id: %s", model_name, app_id)

        timeout = self.configuration.get("timeout", 600)
        self._app_service = DataloopAppServiceClient(
            app_id,
            self.model_entity,
            logger,
            timeout=timeout,
        )
        self.client = self._app_service.client
