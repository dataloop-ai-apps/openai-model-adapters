import openai
import dtlpy as dl
import logging
import json
import os

logger = logging.getLogger('openai-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for OpenAI models',
                              init_inputs={'model_entity': dl.Model,
                                           'openai_key_name': "String"})
class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """ Load configuration for OpenAI adapter
        """
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def stream_response(self, messages):

        response = self.client.chat.completions.create(
            messages=messages,
            stream=True,
            model='gpt-4o'
        )
        for chunk in response:
            yield chunk.choices[0].delta.content or ""

    def prepare_item_func(self, item: dl.Item):
        return item

    def predict(self, batch: [dl.Item], **kwargs):
        """
        API call for Openai on the items batch
        :param batch: list of dl.Items
        :param kwargs:
        :return:
        """
        annotations = []
        for item in batch:
            for chunk in self.stream_response(messages=item.messages):
                item.add_response(chunk)
            return annotations
