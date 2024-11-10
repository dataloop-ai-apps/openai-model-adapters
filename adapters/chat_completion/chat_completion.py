from openai import NOT_GIVEN
import dtlpy as dl
import logging
import openai
import os

logger = logging.getLogger('openai-adapter')


class ModelAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        """ Load configuration for OpenAI adapter
        """
        self.adapter_defaults.upload_annotations = False
        if os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError(f"Missing API key")

        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def call_model(self, messages):
        stream = self.configuration.get("stream", True)
        max_tokens = self.configuration.get("max_tokens", NOT_GIVEN)
        temperature = self.configuration.get("temperature", NOT_GIVEN)
        top_p = self.configuration.get("top_p", NOT_GIVEN)
        model_name = self.configuration.get("model_name", 'gpt-4o')

        response = self.client.chat.completions.create(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
            model=model_name
        )
        if stream is True:
            for chunk in response:
                yield chunk.choices[0].delta.content or ""
        else:
            yield response.choices[0].message.content or ""

    def prepare_item_func(self, item: dl.Item):
        prompt_item = dl.PromptItem.from_item(item)
        return prompt_item

    def predict(self, batch, **kwargs):
        system_prompt = self.model_entity.configuration.get('system_prompt', '')
        add_metadata = self.configuration.get("add_metadata")
        model_name = self.model_entity.name

        for prompt_item in batch:
            # Get all messages including model annotations
            messages = prompt_item.to_messages(model_name=model_name)
            messages.insert(0, {"role": "system",
                                "content": system_prompt})

            nearest_items = prompt_item.prompts[-1].metadata.get('nearestItems', [])
            if len(nearest_items) > 0:
                context = prompt_item.build_context(nearest_items=nearest_items,
                                                    add_metadata=add_metadata)
                logger.info(f"Nearest items Context: {context}")
                messages.append({"role": "assistant", "content": context})

            stream_response = self.call_model(messages=messages)
            response = ""
            for chunk in stream_response:
                #  Build text that includes previous stream
                response += chunk
                prompt_item.add(message={"role": "assistant",
                                         "content": [{"mimetype": dl.PromptType.TEXT,
                                                      "value": response}]},
                                stream=True,
                                model_info={'name': model_name,
                                            'confidence': 1.0,
                                            'model_id': self.model_entity.id})

        return []


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    model = dl.models.get(model_id="")
    item = dl.items.get(item_id="")
    a = ModelAdapter(model)
    a.predict_items([item])
