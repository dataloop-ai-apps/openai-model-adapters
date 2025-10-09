FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv

USER 1000
WORKDIR /tmp
ENV HOME=/tmp
RUN pip install \
    transformers \
    accelerate \
    openai


# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/openai-model-adapters:0.0.14 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/openai-model-adapters:0.0.14

# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/openai-model-adapters:0.0.14 bash
