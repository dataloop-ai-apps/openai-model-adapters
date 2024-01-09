import dtlpy as dl
import logging
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os

logger = logging.getLogger('WhisperAdapter')


@dl.Package.decorators.module(description='Model Adapter for Whisper speech recognition model',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class Whisper(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):

        max_new_tokens = self.configuration.get('max_new_tokens', 128)
        chunk_length_s = self.configuration.get('chunk_length_s', 30)
        batch_size = self.configuration.get('batch_size', 16)
        model_id = self.configuration.get('model_id', 'openai/whisper-large-v3')

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id,
                                                               torch_dtype=torch_dtype,
                                                               low_cpu_mem_usage=True,
                                                               use_safetensors=True
                                                               )
        self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline("automatic-speech-recognition",
                             model=self.model,
                             tokenizer=self.processor.tokenizer,
                             feature_extractor=self.processor.feature_extractor,
                             max_new_tokens=max_new_tokens,
                             chunk_length_s=chunk_length_s,
                             batch_size=batch_size,
                             return_timestamps=True,
                             torch_dtype=torch_dtype,
                             device=device,
                             )
        logger.info('WhisperAdapter loaded')

    def prepare_item_func(self, item):
        return item

    def predict(self, batch: [dl.Item], **kwargs):
        logger.info('WhisperAdapter prediction started')
        batch_annotations = list()
        for item in batch:
            filename = item.download(overwrite=True)
            logger.info(f'WhisperAdapter predicting {filename},  started')
            result = self.pipe(filename)
            logger.info(f'WhisperAdapter predicting {filename}, done')
            # build the dtlpy annotations
            chunks = result['chunks']
            builder = item.annotations.builder()
            for chunk in chunks:
                text = chunk['text']
                timestamp = chunk['timestamp']
                start = timestamp[0]
                end = timestamp[1]
                builder.add(annotation_definition=dl.Subtitle(label=f'Transcript', text=text),
                            model_info={'name': 'openai/whisper-large-v3', 'confidence': 0.0},
                            start_time=start,
                            end_time=end)
            batch_annotations.append(builder)
            os.remove(filename)
        logger.info('WhisperAdapter prediction done')
        return batch_annotations


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


if __name__ == "__main__":
    deploy()
    # logger.info('WhisperAdapter started')
    # adapter = Whisper()
    # adapter.load()
    # annotations = adapter.predict([dl.items.get(item_id='')])
    # for collection in annotations:
    #     collection.upload()
    # print(annotations)
    # logger.info('WhisperAdapter finished')
