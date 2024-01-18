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
    def __init__(self, model_entity: dl.Model, openai_key_name):
        self.openai_key_name = openai_key_name
        super().__init__(model_entity=model_entity)

    def load(self, local_path, **kwargs):
        """ Load configuration for OpenAI adapter
        """
        key = os.environ.get(self.openai_key_name)
        if key is None:
            raise ValueError("Cannot find a key for OPENAI")
        openai.api_key = key

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
            buffer = json.load(item.download(save_locally=False))
            prompts = buffer["prompts"]
            item_annotations = item.annotations.builder()
            for prompt_key, prompt_content in prompts.items():
                for question in prompt_content.values():
                    print(f"User: {question['value']}")
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",  # gpt-4
                        messages=[
                            {"role": "system", "content": 'You are a helpful assistant who understands data science.'},
                            {"role": "user", "content": question['value']}
                        ])
                    response_content = response["choices"][0]["message"]["content"]
                    print("Response: {}".format(response_content))
                    item_annotations.add(annotation_definition=dl.FreeText(text=response_content),
                                         prompt_id=prompt_key,
                                         model_info={
                                             "name": "gpt-3.5-turbo",
                                             "confidence": 1.0
                                         })
            annotations.append(item_annotations)
        return annotations


def examples(self):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # gpt-4
        messages=[{"role": "system", "content": 'You are a helpful assistant who understands data science.'},
                  {"role": "user", "content": 'Why is Britain good?'}
                  ])

    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, "
               "and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help "
               "you today?\nHuman: I'd like to cancel my subscription.\nAI:",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )
