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
                            model_info={'name': 'openai/whisper-large-v3', 'confidence': 1.0},
                            start_time=start,
                            end_time=end)
            batch_annotations.append(builder)
            os.remove(filename)
        logger.info('WhisperAdapter prediction done')
        return batch_annotations
