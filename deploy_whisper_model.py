import dtlpy as dl
from adapters.whisper.whisper import Whisper


def package_creation(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=Whisper,
                                          default_configuration={'max_new_tokens': 128,
                                                                 'chunk_length_s': 30,
                                                                 'batch_size': 16,
                                                                 'model_id': 'openai/whisper-large-v3'},
                                          output_type=dl.AnnotationType.SUBTITLE)
    modules = dl.PackageModule.from_entry_point(entry_point='model_adapter.py')

    package = project.packages.push(package_name='openai-whisper',
                                    src_path=os.getcwd(),
                                    description='Dataloop Openai/whisper pretrained implementation',
                                    is_global=False,
                                    package_type='ml',
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_L,
                                                                        runner_image='dataloopai/whisper-gpu.cuda.11.5.py3.8.pytorch2:1.0.1',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        preemptible=True,
                                                                        concurrency=3).to_json(),
                                        'executionTimeout': 1000 * 3600,
                                        'initParams': {'model_entity': None}
                                    },
                                    metadata=metadata)
    return package


def model_creation(package: dl.Package):
    model = package.models.create(model_name='openai-whisper',
                                  description='whisper arch, pretrained whisper-large-v3',
                                  tags=['whisper', 'pretrained', 'openai', 'audio', 'apache-2.0'],
                                  dataset_id=None,
                                  status='trained',
                                  scope='project',
                                  configuration={
                                      'max_new_tokens': 128,
                                      'chunk_length_s': 30,
                                      'batch_size': 16,
                                      'model_id': 'openai/whisper-large-v3'},
                                  project_id=package.project.id,
                                  labels=list(['Transcript']),
                                  input_type='audio',
                                  output_type='subtitle')
    return model


def deploy():
    project_name = '<enter your project name>'
    project = dl.projects.get(project_name)

    package = package_creation(project=project)
    print(f'new mode pushed. codebase: {package.codebase}')

    model = model_creation(package=package)
    model_entity = package.models.list().print()

    print(f'model and package deployed. package id: {package.id}, model id: {model_entity.id}')
