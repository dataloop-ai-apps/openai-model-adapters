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
        for item in batch:
            if isinstance(item, str):
                self.adapter_defaults.upload_features = True
                text = item
            else:
                self.adapter_defaults.upload_features = False
                try:
                    prompt_item = dl.PromptItem.from_item(item)
                    is_hyde = item.metadata.get('prompt', dict()).get('is_hyde', False)
                    if is_hyde is True:
                        messages = prompt_item.to_messages(model_name=self.configuration.get('hyde_model_name'))[-1]
                        if messages['role'] == 'assistant':
                            text = messages['content'][0]['text']
                        else:
                            raise ValueError(f'Only assistant messages are supported for hyde model')
                    else:
                        messages = prompt_item.to_messages(include_assistant=False)[-1]
                        text = messages['content'][0]['text']

                except ValueError as e:
                    raise ValueError(f'Only mimetype text or prompt items are supported {e}')

            response = self.model.embeddings.create(
                input=text,
                model=self.model_name,
                dimensions=self.model_entity.configuration.get('embeddings_size', 256)
            )
            embedding = response.data[0].embedding
            logger.info(f'Extracted embeddings for text {item}: {embedding}')
            embeddings.append(embedding)

        return embeddings
