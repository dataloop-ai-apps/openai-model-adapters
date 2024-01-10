import os
import dtlpy as dl
from adapters.chat_completion import ModelAdapter


def deploy():
    dl.setenv('prod')
    project = dl.projects.get(project_name='ShadiProject')
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
    # s = package.services.list().items[0]
    # s.package_revision = package.version
    # s.update(True)
    # print("Package created!")

    model = package.models.create(model_name='openai-gpt-3.5-turbo',
                                  description='OpenAI API call for gpt-3.5-turbo ',
                                  tags=['llm', 'openai', "chatgpt"],
                                  dataset_id=None,
                                  status='trained',
                                  scope='project',
                                  configuration={},
                                  project_id=package.project.id
                                  )

    # model.deploy()
    # s = dl.services.get(service_id='659e9d1818fe7e17d0879ef3')
    # s.runtime.runner_image = 'python:3.8'
    # s.update()


if __name__ == '__main__':
    deploy()
