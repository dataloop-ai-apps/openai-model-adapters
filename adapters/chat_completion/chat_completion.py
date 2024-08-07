import dtlpy as dl
import logging
import openai
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
            model=self.configuration.get('model_name', 'gpt-4o')
        )

        ans = ""
        for chunk in response:
            yield ans + (chunk.choices[0].delta.content or "")

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', "")
        for prompt_item in batch:
            messages = prompt_item.to_messages(
                model_name=self.model_entity.name)  # Get all messages including model annotations
            messages.insert(0, {"role": "system",
                                "content": system_prompt})
            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                # build context
                context = ""
                for item_id in nearest_items:
                    context_item = dl.items.get(item_id=item_id)
                    source = context_item.metadata['system'].get('document', dict()).get('source', "missing")
                    with open(context_item.download(), 'r', encoding='utf-8') as f:
                        text = f.read()
                    context += f"\n<source>\n{source}\n</source>\n<text>\n{text}\n</text>"
                messages.append({"role": "assistant", "content": context})

            stream = self.stream_response(messages=messages)
            response = ""
            for chunk in stream:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                stream=True,
                                model_info={'name': self.model_entity.name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})

        return []


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    dl.setenv('prod')
    model = dl.models.get(model_id="66a8f9e2c7aab26441aa5869")
    item = dl.items.get(item_id="66b369a82e90de89dde976e0")
    a = ModelAdapter(model)
    a.predict_items([item])
