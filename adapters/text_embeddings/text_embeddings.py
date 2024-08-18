import openai
import dtlpy as dl
import os
import logging

logger = logging.getLogger('openai-text-embeddings')


class TextEmbeddings(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """ Load configuration for OpenAI adapter
        """
        if os.environ.get("OPENAI_API_KEY", None) is None:
            raise ValueError(f"Missing API key: OPENAI_API_KEY")
        self.model_name = self.configuration.get('model_name', 'text-embedding-3-large')
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def embed(self, batch, **kwargs):
        embeddings = []
        for text in batch:
            logger.info(f'Extracted text: {text}')
            if text is not None:
                response = self.model.embeddings.create(
                    input=text,
                    model=self.model_name,
                    dimensions=self.model_entity.configuration.get('embeddings_size', 256)
                )
                embedding = response.data[0].embedding
                logger.info(f'Extracted embeddings for text {text}: {embedding}')
                embeddings.append(embedding)
            else:
                logger.error(f'No text found in item')
                raise ValueError(f'No text found in item')

        return embeddings
