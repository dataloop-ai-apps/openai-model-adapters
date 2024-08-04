import dtlpy as dl
import logging
import openai
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

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', "")

        annotations = []
        for item in batch:
            if ('json' not in item.mimetype or
                    item.metadata.get('system', dict()).get('shebang', dict()).get('dltype') != 'prompt'):
                raise ValueError('Only prompt items are supported')
            prompt_item = dl.PromptItem.from_item(item)
            prompt_item_raw = json.load(item.download(save_locally=False))

            prompt_item._get_assistant_prompts(model_name=self.model_entity.name)
            messages = [{"role": "system",
                         "content": system_prompt},
                        *prompt_item.messages]
            last_key = prompt_item.prompts[-1].key
            nearest_items = [p['nearestItems'] for p in prompt_item_raw.get('prompts', dict()).get(last_key)
                             if 'metadata' in p['mimetype'] and 'nearestItems' in p]
            filters = dl.Filters(resource=dl.FiltersResource.ANNOTATION)
            filters.add(field='metadata.user.model.name', values=self.model_entity.name)
            collection = item.annotations.list(filters=filters)
            if len(nearest_items) > 0:
                nearest_items = nearest_items[0]
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
            ann = dl.Annotation.new(item=item,
                                    annotation_definition=dl.FreeText(text=' '),
                                    metadata={'system': {'promptId': last_key},
                                              'user': {'model': {'confidence': 1.0,
                                                                 'name': self.model_entity.name,
                                                                 'model_id': self.model_entity.id,
                                                                 }}})
            ann = ann.upload()
            for chunk in stream:
                ann.annotation_definition.coordinates += chunk
                ann = ann.update(True)
            collection.annotations.append(ann)
            annotations.append(collection)
        return annotations


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv('.env')
    self = ModelAdapter(model_entity=dl.models.get(model_id='66af57987c7ee9faa192b81c'))
    item = dl.items.get(None, '66af5ff21521b92ef9f5ce02')
    self.predict(batch=[item])
