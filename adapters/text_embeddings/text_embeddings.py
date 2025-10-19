import dtlpy as dl
import logging
import openai
import os

logger = logging.getLogger('openai-text-embeddings')


class TextEmbeddings(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """ Load configuration for OpenAI adapter
        """
        if os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError(f"Missing API key: OPENAI_API_KEY")

        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call_model(self, text):
        model_name = self.configuration.get('model_name', 'text-embedding-3-large')
        dimensions = self.model_entity.configuration.get('embeddings_size', 256)

        response = self.client.embeddings.create(
            input=text,
            model=model_name,
            dimensions=dimensions
        )
        embedding = response.data[0].embedding
        return embedding
    
    def get_text(self, item):
        hyde_model_name = self.configuration.get('hyde_model_name')
        text = None
        if isinstance(item, str):
            text = item
        else:
            try:
                prompt_item = dl.PromptItem.from_item(item)
                is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
                if is_hyde is True:
                    messages = prompt_item.to_messages(model_name=hyde_model_name)[-1]
                    if messages['role'] == 'assistant':
                        text = messages['content'][-1]['text']
                    else:
                        raise ValueError(f'Only assistant messages are supported for hyde model')
                else:
                    messages = prompt_item.to_messages(include_assistant=False)[-1]
                    text = messages['content'][-1]['text']

            except ValueError as e:
                raise ValueError(f'Only mimetype text or prompt items are supported {e}')
        return text

    def embed(self, batch, **kwargs):
        embeddings = []

        for item in batch:
            text = self.get_text(item)
            if self.configuration.get('external_vector_db', False) is False:
                if isinstance(item, str):
                    self.adapter_defaults.upload_features = True
                else:
                    self.adapter_defaults.upload_features = False
            else:
                self.adapter_defaults.upload_features = False

            embedding = self.call_model(text=text)
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings
