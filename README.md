# Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation

## 🔍 Overview

This project implements AVG, a new approach for text-to-image retrieval that reformulates the task as token-to-voken generation to improve both retrieval effectiveness and efficiency. Unlike traditional methods based on cross-attention (one-tower) or shared embedding spaces (two-tower), AVG incorporates fine-grained token-level interactions while maintaining fast retrieval speed. It uses a semantically aligned image tokenizer and a hybrid generative-discriminative training objective to reduce semantic misalignment and bridge the gap to the retrieval target. Experimental results show that AVG achieves a 7.53% relative improvement in effectiveness and a 4× speedup compared to the widely used two-tower baseline, CLIP.

> For more details, refer to our paper accepted to **SIGIR 2025**: [Revolutionizing Text-to-Image Retrieval as Autoregressive Token-to-Voken Generation](https://arxiv.org/abs/2407.17274).

## 📦 Requirements

The code is tested on Python 3.9.18, PyTorch 1.13.1 and CUDA 11.7. 

You can create a conda environment with the required dependencies using the provided `environment.yml` file.

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

###  🔵 Tokenizer (RQ-VAE)

#### Step 1: Train the Tokenizer (RQ-VAE)

```bash
cd RQ-VAE
bash scripts/train_rqvae.sh
```

The trained model will be saved in the `RQ-VAE/output` directory.

#### Step 2: Generate Voken Codes (Discrete Image Representations)

```bash
bash scripts/generate_codes.sh
```

This script encodes images into discrete token sequences, which will be used in downstream Retriever and Reranker stages.


### 🟡 Retriever (LLM)

#### Step 3: Prepare Retriever Training Data

Use the previously generated voken codes to construct the training data for the retriever:

```bash
cd ..
bash scripts/prepare_retriever_dataset.sh
```

#### Step 4: Train the Retriever

🔹 Stage 1: Generative Training (T5 or LLaMA)

To train a T5-based retriever:

```bash
bash scripts/train_retriever_t5.sh
# Recall metrics will be automatically recorded in the log file and wandb.
```

To train a LLaMA-based retriever:

```bash
bash scripts/finetune_retriever_llama.sh
```

For testing the LLaMA-based retriever (requires separate evaluation):

```bash
bash scripts/test_retriever_llama.sh
```

🔹 Stage 2: Discriminative Fine-Tuning (Optional)

Load the checkpoint from Stage 1 and perform discriminative training:

```bash
bash scripts/train_retriever_t5_stage2.sh
# Note: Hyperparameters are sensitive and may require tuning.
```


### 🟢 Reranker (SEED-LLaMA)

We use the [SEED-LLaMA](https://github.com/AILab-CVC/SEED) model as the reranker. A lightweight MLP head is added for scoring candidate responses retrieved by the retriever.

#### Step 5: Prepare Reranker Data

First, generate retrieval results for training set and test set:

```bash
bash scripts/prepare_reranker_dataset.sh
```

Then, convert the voken codes to SEED-LLaMA token IDs:

```bash
bash scripts/convert_voken_to_seed.sh
```

#### Step 6: Train the Reranker

```bash
bash scripts/finetune_reranker_llama.sh
```

#### Step 7: Evaluate the Reranker

```bash
bash scripts/test_reranker_llama.sh
```

### 🟥 Note
- Make sure to update all dataset-related paths in the training and evaluation scripts according to your local directory structure.

- The default hyperparameters (e.g., learning rate, batch size, number of epochs) are configured for a reference dataset. You must tune them for your own dataset.

- Some scripts assume a fixed number of retrieval candidates (e.g., top-50 or top-100). Adjust top_k accordingly based on your retriever output.


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

Part of the code (`./models`) is based on the [SEED-LLaMA](https://github.com/AILab-CVC/SEED). We continue to honor and adhere to its licensing terms.


## 📬 Contact

For inquiries, feel free to reach out to Hongru Cai at [henry.hongrucai@gmail.com](mailto:henry.hongrucai@gmail.com).