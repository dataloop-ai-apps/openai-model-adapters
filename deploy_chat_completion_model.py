import os
import dtlpy as dl
from adapters.chat_completion.chat_completion import ModelAdapter


def deploy(project_name):
    dl.setenv('prod')
    project = dl.projects.get(project_name=project_name)
    metadata = dl.Package.get_ml_metadata(cls=ModelAdapter,
                                          default_configuration={}
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='adapters/chat_completion/chat_completion.py')
    package = project.packages.push(package_name='openai-adapter',
                                    ignore_sanity_check=True,
                                    src_path=os.getcwd(),
                                    package_type='ml',
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_HIGHMEM_XS,
                                                                        runner_image='python:3.8',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=1,
                                                                            max_replicas=1),
                                                                        concurrency=1).to_json()},
                                    metadata=metadata)

    model = package.models.create(model_name='openai-gpt-3.5-turbo',
                                  description='OpenAI API call for gpt-3.5-turbo ',
                                  tags=['llm', 'openai', "chatgpt"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='project',
                                  configuration={},
                                  project_id=package.project.id
                                  )

    print(f'Model ID: {model.id}, Model Name: {model.name}')


if __name__ == '__main__':
    project_name = ''
    deploy(project_name=project_name)
