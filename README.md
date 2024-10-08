# OpenAI Adapters

This repository contains code for integrating `OpenAI` models into the `Dataloop` platform using direct API access.

## Using OpenAI Models in Dataloop Platform

1. Create an API key for using OpenAI client.
2. Install the model from [Dataloop Marketplace](https://docs.dataloop.ai/docs/marketplace) ( you can filter the
   available models by their provider) :

<img src="assets/market_place.png" alt="Image of the pipeline">

3. Insert your API key as a `Secret` in Dataloop platform under the name `OPENAI_API_KEY`:


* Navigate to Data Governance in the left option dialog.

<img src="assets/navigate.png" alt="Image of the pipeline">

* Choose `Create Secret`:

![img.png](assets/img.png)

* Insert your API Key and save:

<img src="assets/secret.png" alt="Image of the pipeline">

4. When using your model, insert your key to the service:

* Click on `Set Up Secrets`:

![img.png](assets/pipeline.png)

* Search for `OPENAI_API_KEY` and choose your `secret` and save.

<img src="assets/update api key.png" alt="Image of the pipeline">

# Adapters

## Whisper

- [Chat Completion](adapters/chat_completion/README.md)

## Chat Completion

- [Whisper](adapters/whisper/README.md)