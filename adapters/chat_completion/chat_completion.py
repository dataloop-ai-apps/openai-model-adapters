import dtlpy as dl
import logging
import openai
import os

logger = logging.getLogger('openai-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for OpenAI models',
                              init_inputs={'model_entity': dl.Model})
class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """ Load configuration for OpenAI adapter
        """
        self.adapter_defaults.upload_annotations = False
        self.stream = self.configuration.get("stream", True)
        if os.environ.get("OPENAI_API_KEY", None) is None:
            raise ValueError(f"Missing API key: OPENAI_API_KEY")

        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def stream_response(self, messages):

        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=self.get_config_value("max_tokens"),
            temperature=self.get_config_value("temperature"),
            top_p=self.get_config_value("top_p"),
            stream=self.stream,
            model=self.configuration.get('model_name', 'gpt-4o')
        )
        for chunk in response:
            yield chunk.choices[0].delta.content or ""

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', '')
        for prompt_item in batch:
            # Get all messages including model annotations
            messages = prompt_item.to_messages(model_name=self.model_entity.name)
            messages.insert(0, {"role": "system",
                                "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=self.configuration.get("add_metadata"))
                messages.append({"role": "assistant", "content": context})

            stream_response = self.stream_response(messages=messages)
            response = ""
            for chunk in stream_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                stream=self.stream,
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})

        return []

    def get_config_value(self, key):
        value = self.configuration.get(key)
        return openai.NOT_GIVEN if value is None else value


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    dl.setenv('prod')
    model = dl.models.get(model_id="")
    item = dl.items.get(item_id="")
    a = ModelAdapter(model)
    a.predict_items([item])
