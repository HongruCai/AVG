# Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation

## 🔍 Overview

This project implements AVG, a new approach for text-to-image retrieval that reformulates the task as token-to-voken generation to improve both retrieval effectiveness and efficiency. Unlike traditional methods based on cross-attention (one-tower) or shared embedding spaces (two-tower), AVG incorporates fine-grained token-level interactions while maintaining fast retrieval speed. It uses a semantically aligned image tokenizer and a hybrid generative-discriminative training objective to reduce semantic misalignment and bridge the gap to the retrieval target. Experimental results show that AVG achieves a 7.53% relative improvement in effectiveness and a 4× speedup compared to the widely used two-tower baseline, CLIP.

> For more details, refer to our paper accepted to **SIGIR 2025**: [Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation](https://arxiv.org/abs/2407.17274).

## 📦 Requirements

The code is tested on Python 3.9.18, PyTorch 1.13.1 and CUDA 11.7. 

The required packages can be installed using the following command:

```bash 
conda env create -f environment.yml
conda activate avg
```
## 🧾 Data

1. The dataset used in the paper is the [COCO 2014](http://cocodataset.org/#download) dataset and the [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset) dataset. The raw images should be downloaded and placed in the `RQ-VAE/data` directory.

2. Run the following command to preprocess the data to generate the image features and text features:

```bash
cd RQ-VAE
bash scripts/prepare_emb.sh
```

3. You can also use the simple tools/generate_psudo_query.py script to generate pseudo queries to augment the dataset. The psudo queries we used can be found [here]().


## 📈 Training

![x](https://hongrucai.github.io/images/avg.png)

### Tokenizer
To train the Tokenizer (RQ-VAE) model, run the following command:

```bash
cd RQ-VAE
bash scripts/train_rqvae.sh
```

Then the model will be saved in the `RQ-VAE/output` directory.

Use the following command to generate the "Voken":

```bash
bash scripts/generate_codes.sh
```
### Retriever

Use the codes genertaed to prepare the data for the Retriever (LLM) model:

```bash
cd ..
bash scripts/prepare_retriever_dataset.sh
```
Train the Retriever:

Stage 1: Generative training
```bash
bash scripts/train_retriever_t5.sh # the recall will be automatically recorded.
# or
bash scripts/finetune_retriever_llama.sh
```
Specially, the LLaMa model will need to be tested separately.
```bash
bash scripts/test_retriever_llama.sh
```
Stage 2: Discriminative training, load the checkpoint from the stage 1 and run the following command:

```bash
bash scripts/train_retriever_t5_stage2.sh # he hyper-parameters are sensitive, you may need to tune them.
```

### Reranker
We use the SEED-LLaMA model as the reranker. By adding a mlp layer, we can train the model with our retrieved results.

First, we need to prepare the data for the reranker.  and convert our image vokens to SEED-LLaMA ids.
```bash
# We need to generate the predictions from the retriever for training and test sets. 
bash scripts/prepare_reranker_dataset.sh 
# Then we need to convert the vokens to SEED-LLaMA ids.
bash scripts/convert_voken_to_seed.sh
```

Then we can train the reranker with the following command:
```bash
bash scripts/finetune_reranker_llama.sh
```
After training, we can test the reranker with the following command:
```bash
bash scripts/test_reranker_llama.sh
```

### Note
- For different datasets, you may need to adjust the hyper-parameters related to data path in the scripts.
- The other hyper-parameters are set to the default values, you may need to tune them for your own dataset or explore different settings.


## 📚 Citation

If you find this code useful, please consider citing our paper:

```bibtex
@inproceedings{li2025avg,
  title={Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation},
  author={Yongqi Li, Hongru Cai, Wenjie Wang, Leigang Qu, Yinwei Wei, Wenjie Li, Tat-Seng Chua},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series={SIGIR '25},
  year={2025}
}
```

## 📄 License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.

Part of the code is based on the following repo: [SEED-LLaMA](https://github.com/AILab-CVC/SEED). We continue to honor and adhere to its licensing terms for the portions derived from it.


## 📬 Contact

For inquiries, feel free to reach out to Hongru Cai at [henry.hongrucai@gmail.com](mailto:henry.hongrucai@gmail.com).