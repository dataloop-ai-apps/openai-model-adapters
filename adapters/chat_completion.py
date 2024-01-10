import openai
import dtlpy as dl
import logging
import json
import os

logger = logging.getLogger('openai-adapter')


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for OpenAI models',
                              init_inputs={'model_entity': dl.Model,
                                           'openai_key_name': dl.PACKAGE_INPUT_TYPE_STRING})
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
        prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )


def deploy():
    dl.setenv('prod')
    project = dl.projects.get(project_name='CVPR 2023 demo')
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={}
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='adapters/chat_completion.py')
    package = project.packages.push(package_name='openai-adapter',
                                    ignore_sanity_check=True,
                                    src_path=os.getcwd(),
                                    # scope='public',
                                    package_type='ml',
                                    # codebase=dl.GitCodebase(git_url='https://github.com/dataloop-ai/openai-model-adapters',
                                    #                         git_tag='1.0.0'),
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_HIGHMEM_XS,
                                                                        runner_image='python:3.8',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=1,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)
    s = package.services.list().items[0]
    s.package_revision = package.version
    s.update(True)
    print("Package created!")

    model = package.models.create(model_name='openai-gpt-3.5-turbo',
                                  description='OpenAI API call for gpt-3.5-turbo ',
                                  tags=['llm', 'openai', "chatgpt"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='project',
                                  configuration={},
                                  project_id=package.project.id
                                  )

    model.deploy()
    s = dl.services.get(service_id='648f741e2bcb3e2f232be84f')
    s.runtime.runner_image = 'python:3.8'
    s.update()


def predict_model():
    model = dl.models.get(model_id='648f740f7a887fa801b092ea')
    model.predict(item_ids=['648ed83b30a06d1dd16e31d5'])
