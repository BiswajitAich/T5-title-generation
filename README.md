# T5 Fine-tuning for Title Generation

## About 

This project fine-tunes a FLAN-T5 small model to generate concise, context-aware titles from input text. The workflow includes LoRA-based parameter-efficient fine-tuning and 8-bit quantization to optimize inference for low-resource environments like Render free tier.

## Features

- Generate titles from plain text or code queries.
- Efficient LoRA fine-tuning to reduce GPU memory usage.
- Quantized model for faster inference on CPU/GPU.
- Hosted on Hugging Face Hub for easy deployment.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (HuggingFace)
- Datasets
- BitsAndBytes (for quantization)
- FastAPI + Uvicorn (for deployment)

## Model Info (opensource)

- Hugging Face Model: [aich007/T5-small-title-generation](https://huggingface.co/aich007/T5-small-title-generation)
- Task: Title Generation from input text or code.
- Quantization: 8-bit (for lightweight inference)
- Fine-tuning: LoRA-based

## References

- [T5 Paper](https://arxiv.org/abs/1910.10683)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Hugging Face Model Hub](https://huggingface.co/aich007/T5-small-title-generation)
