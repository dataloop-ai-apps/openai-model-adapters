FROM docker.io/dataloopai/dtlpy-agent:cpu.py3.10.opencv
USER root
RUN apt update && apt install -y curl gpg software-properties-common

USER 1000
WORKDIR /tmp
ENV HOME=/tmp
RUN pip install \
    transformers \
    accelerate \
    openai \
    git+https://github.com/kostyachum/python-markdown-plain-text.git#egg=plain-text-markdown-extention



# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/openai-model-adapters:0.0.13 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/openai-model-adapters:0.0.13

# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/openai-model-adapters:0.0.13 bash
