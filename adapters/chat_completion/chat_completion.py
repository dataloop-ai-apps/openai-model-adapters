from openai import OpenAI
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
    def __init__(self, model_entity: dl.Model, openai_key_name):
        self.openai_key_name = openai_key_name
        super().__init__(model_entity=model_entity)

    def load(self, local_path, **kwargs):
        """ Load configuration for OpenAI adapter
        """
        key = os.environ.get(self.openai_key_name)
        if key is None:
            raise ValueError("Cannot find a key for OPENAI")
        self.model = OpenAI(api_key=key)

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item=item)
        return prompt_item

    def predict(self, batch: [dl.Item], **kwargs):
        """
        API call for Openai on the items batch
        :param batch: list of dl.Items
        :param kwargs:
        :return:
        """
        for prompt_item in batch:
            item_annotations = dl.AnnotationCollection()
            messages = prompt_item.messages(model_name=self.model_entity.name)
            response = self.model.chat.completions.create(
                model="gpt-3.5-turbo",  # gpt-4
                messages=messages)
            response_content = response.choices[0].message.content
            print("Response: {}".format(response_content))
            prompt_key = messages[-1].get('name')
            item_annotations.add(annotation_definition=dl.FreeText(text=response_content),
                                 prompt_id=prompt_key,
                                 model_info={
                                     "model_id": self.model_entity.id,
                                     "name": self.model_entity.name,
                                     "confidence": 1.0
                                 })
            prompt_item.add_responses(annotations=item_annotations)
