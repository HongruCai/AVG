# Autoregressive “Voken” Generation for Text-to-image Retrieval

The official implementation of the paper "Autoregressive “Voken” Generation for Text-to-image Retrieval".

## Introduction

This repository contains the official implementation of the paper "Autoregressive “Voken” Generation for Text-to-image Retrieval". The code is based on PyTorch and is designed to generate "Voken" for text-to-image retrieval tasks.

## Requirements

The code is tested on Python 3.9.18, PyTorch 1.13.1 and CUDA 11.7. 

The required packages can be installed using the following command:

```bash 
pip install -r requirements.txt
```
## Data Preparation

1. The dataset used in the paper is the [COCO 2014](http://cocodataset.org/#download) dataset and the [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) dataset. The raw images should be downloaded and placed in the `RQ-VAE/data` directory.

2. Run the following command to preprocess the data:

```bash
cd RQ-VAE
bash scripts/prepare_emb.sh
```
## Training

### Training the RQ-VAE model
To train the RQ-VAE model, run the following command:

```bash
cd RQ-VAE
bash scripts/train_flickr_rqvae.sh
```
Then the model will be saved in the `RQ-VAE/outpur` directory.

Use the following command to generate the "Voken":

```bash
bash scripts/generate_codes.sh
```
### Training the Language Model

Prepare the data for the Language Model:

```bash
cd ..
bash scripts/prepare_dataset.sh
```
Train the Language Model:

```bash
bash scripts/train_t5.sh
or
bash scripts/finetune_llama.sh
```
Specially, the LLaMa model will need to be tested separately. 
```bash
bash scripts/test_llama.sh
```

## BibTeX

## Acknowledgements

## Contact
