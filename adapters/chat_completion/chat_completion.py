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
            prompt_item = json.load(item.download(save_locally=False))
            collection = dl.AnnotationCollection()
            for prompt_name, prompt_content in prompt_item.get('prompts').items():
                # get latest question
                question = [p['value'] for p in prompt_content if 'text' in p['mimetype']][0]
                messages = [{"role": "system",
                             "content": system_prompt},
                            {"role": "user",
                             "content": question}]
                nearest_items = [p['nearestItems'] for p in prompt_content if 'metadata' in p['mimetype'] and
                                 'nearestItems' in p]
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
                                        metadata={'system': {'promptId': prompt_name},
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
    self = ModelAdapter(model_entity=dl.models.get(model_id='66ae7a3d7c7ee9111392b81a'))
    item = dl.items.get(None, '66ae7bc324e0026d9b1206ca')
    self.predict(batch=[item])
