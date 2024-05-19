from openai import OpenAI
import dtlpy as dl
import os
import logging
from markdown_plain_text.extention import convert_to_plain_text
import json

logger = logging.getLogger('openai-text-embeddings')


class TextEmbeddings(dl.BaseModelAdapter):
    def __init__(self, model_entity, openai_key_name='openai_key'):
        self.openai_key_name = openai_key_name
        super().__init__(model_entity=model_entity)
        self.embeddings_size = self.configuration.get('embeddings_size', 256)
        self.model_name = self.configuration.get('model_name', 'text-embedding-3-large')

    def load(self, local_path, **kwargs):
        key = os.environ.get(self.openai_key_name)
        if key is None:
            raise ValueError("Cannot find a key for OPENAI")
        self.model = OpenAI(api_key=key)

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
            logger.info(f'Unsupported mimetype: {item.mimetype}, text = None')
        return text

    def extract_features(self, batch, **kwargs):
        feature_vectors = []
        for text in batch:
            logger.info(f'Extracted text: {text}')
            if text is not None:
                response = self.model.embeddings.create(
                    input=text,
                    model=self.model_name,
                    dimensions=256
                )
                embeddings = response.data[0].embedding
                feature_vectors.append(embeddings)
                logger.info(f'Extracted embeddings for text {text}: {embeddings}')
            else:
                feature_vectors.append([])
        return feature_vectors
