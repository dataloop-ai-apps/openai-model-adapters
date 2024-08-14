import openai
import dtlpy as dl
import os
import logging
from markdown_plain_text.extention import convert_to_plain_text
import json

logger = logging.getLogger('openai-text-embeddings')


class TextEmbeddings(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """ Load configuration for OpenAI adapter
        """
        if os.environ.get("OPENAI_API_KEY", None) is None:
            raise ValueError(f"Missing API key: OPENAI_API_KEY")
        self.model_name = self.configuration.get('model_name', 'text-embedding-3-large')
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def prepare_item_func(self, item):
        filename = item.download(overwrite=True)
        logger.info(f'Downloaded item: {filename}')
        text = None
        if item.mimetype == 'text/plain':
            with open(filename, 'r') as f:
                text = f.read()
                text = text.replace('\n', ' ')
        elif item.mimetype == 'text/markdown':
            with open(filename, 'r') as f:
                text = f.read()
                text = convert_to_plain_text(text)
                text = text.replace('\n', ' ')
        elif item.mimetype == 'application/json' and 'prompt' in item.system.get('shebang', dict()).get('dltype'):
            buffer = json.loads(filename)
            _, prompt_content = list(buffer['prompts'].items())[0]
            _, question = list(prompt_content.items())[0]

            if question["mimetype"] == dl.PromptType.TEXT:
                text = question["value"]
        else:
            raise ValueError(f'Unsupported mimetype: {item.mimetype}')

        return text

    def embed(self, batch, **kwargs):
        embeddings = []
        for text in batch:
            logger.info(f'Extracted text: {text}')
            if text is not None:
                response = self.model.embeddings.create(
                    input=text,
                    model=self.model_name,
                    dimensions=256
                )
                embedding = response.data[0].embedding
                logger.info(f'Extracted embeddings for text {text}: {embedding}')
                embeddings.append(embedding)
            else:
                logger.error(f'No text found in item')
                raise ValueError(f'No text found in item')

        return embeddings
