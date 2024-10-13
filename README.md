[![LinkedIn][linkedin-shield]][linkedin-url]

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.4](https://img.shields.io/badge/torch-v2.4-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![trl 0.10.1](https://img.shields.io/badge/trl-v0.10.1-violet)](https://huggingface.co/docs/trl/index)
[![Transformers 4.44.2](https://img.shields.io/badge/transformers-v4.44.2-red)](https://huggingface.co/docs/transformers/index)
[![PEFT 0.12.0](https://img.shields.io/badge/peft-v0.12.0-lightblue)](https://huggingface.co/docs/peft/index)
[![datasets 3.0.0](https://img.shields.io/badge/datasets-v2.15.0-orange)](https://huggingface.co/docs/datasets/index)
[![bitsandbytes 0.43.3](https://img.shields.io/badge/bitsandbytes-v0.43.3-green)](https://huggingface.co/blog/hf-bitsandbytes-integration)

# Finetuning Phi-3 for Multimodel

## Introduction

* Make a multimodel LLM that takes Image, Audio and Text as input and gives text as output.
* Use instruct 150K data for finetuning
* For Image input:
    * Use CLIP to get image embeddings
    * Train a projection layer to feed image embeddings to Phi model.
    * Fine tune Phi-3 on instruct 150K dataset using QLora.
* For Audio input:
    * Use Whisper or any other ASR model to convert audio to text.
* For text model:
    * Use Phi-3 tokenize and embeddings.

## :open_file_folder: Files
- [**generate_embeddings.py**](generate_embeddings.py)
    - Script to generate and store image embeddings using CLIP model.
- [**pretraining**](image_funetuning/pretraining/main.py)
    - Contains architecture of projection layer.
    - Class to load and process data.
    - Script to train projection layer.
- [**finetuning**](image_funetuning/finetuning/finetune.py)
    - Script to finetune Phi model on instruct 150K dataset

## :chart_with_upwards_trend: Projection layer training

    Epoch 1/3
    Loss: 6.5470 Batch_id=9856: 100%|██████████| 9857/9857 [1:49:31<00:00,  1.50it/s]
    Epoch 2/3
    Loss: 6.3709 Batch_id=9856: 100%|██████████| 9857/9857 [1:49:33<00:00,  1.50it/s]
    Epoch 3/3
    Loss: 6.3691 Batch_id=9856: 100%|██████████| 9857/9857 [1:49:31<00:00,  1.50it/s]
    Training completed and model saved.

## :chart_with_upwards_trend: Finetuning using QLora

    Step	Training Loss
    100	    12.643800
    200	    12.441300
    300	    11.495200
    400	    7.042100
    500	    3.107100
    600	    2.746800
    700	    2.546100
    800	    2.320600
    900	    2.036000
    1000	1.992700
    .
    .
    .
    5100	1.321800
    5200	1.321600
    5300	1.333300
    5400	1.341200
    5500	1.306900
    5600	1.318700
    5700	1.328000
    5800	1.317500
    5900	1.311600
    6000	1.308100

## Installation

1. Clone the repo
```
git clone https://github.com/AkashDataScience/multi_model_phi_3
```
2. Go inside folder
```
 cd multi_model_phi_3
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Training

```
cd image_finetuning

# Step-1: Generate and store image embeddings:
python generate_embeddings.py

# Step-2: Train projection layer
python pretraining/main.py

# Step-3: Finetune on instruck 150K
python finetuning/finetune.py

```

## Future improvements
* Approach to train projection layer can be improved (Training loss decreases from 12 to 6, but hets stuck at 6)
* Adding inferece after every few steps might be helpful to see what model is learning.
* Small paches of image can be passed instead of passing full image.
* Instead of finetuning only on image-instruct data, multiple datasources can be used to finetune model.

## Acknowledgments
This repo is developed using references listed below:
* [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485)
* [Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/pdf/2404.14219)
* [Parameter-Efficient Fine-Tuning Methods for Pretrained Language Models: A Critical Review and Assessment](https://arxiv.org/pdf/2312.12148)
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
* [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314)

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/akash-m-shah/
[Python.py]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[PyTorch.tensor]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/
[HuggingFace.transformers]: https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange
[huggingface-url]: https://huggingface.co/