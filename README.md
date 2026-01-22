# Turkish Medical Visual Question Answering (TR VQA-Med)

**Paper title:**  

*Turkish Medical Visual Question Answering: Benchmarking Vision–Language Encoders and Contrastive Alignment*

This work has been accepted for presentation at the 2026 International Conference on Machine Learning Technologies (ICMLT 2026).
---

This repository provides experimental pipelines for **Turkish Medical Visual Question Answering (TR VQA-Med)** based on the **ImageCLEF 2019 VQA-Med** task formulation.  
The project focuses on **multimodal fusion of medical images and Turkish clinical questions**, supporting baseline training, partial fine-tuning of language models, and contrastive pretraining for image–text alignment.

The primary objective is to investigate how different **vision encoders** and **Turkish-compatible language models** interact in a classification-based VQA setting.

---

## Project Scope

- Medical Visual Question Answering (VQA-Med)
- Turkish question–answer pairs
- Image–text multimodal fusion
- Classification-based VQA formulation
- Transformer-based language models
- CNN / ViT-based image encoders
- Contrastive pretraining (CLIP-style)

This repository is intended for **research and experimental purposes**.

---
## Dataset Design Rationale

The ImageCLEF VQA-Med 2019 dataset defines four question categories: Modality, Plane, Organ System, and Abnormality. In this study, we focus only on Modality, Plane, and Organ System questions and exclude Abnormality.

This choice is motivated by our formulation of the task as a **classification-based VQA problem**. The selected three categories correspond to a **closed and well-defined answer set**, whereas Abnormality questions involve open-ended, free-text answers that are more suited to generative VQA models.

After removing the Abnormality category, the dataset forms a **67-class classification task**, enabling controlled and fair evaluation of multimodal vision–language models.

## Dataset Format

The scripts expect text files where each line follows the format:
synpicID|question(TR)|answer(TR)

Example:
synpic54082|hangi modalite gösterilmektedir?|bilgisayarlı tomografi


- `synpicID` is used to automatically locate the corresponding medical image.
- Questions and answers are provided in **Turkish**.
- Answers are treated as **categorical labels** (classification-based VQA).

> **Note:** Due to licensing restrictions, datasets and images are **not included** in this repository.

---

## Recommended Directory Structure

```text
Turkish-Medical-Visual-Question-Answering/
│
├── main_TR_r.py
├── fine_tune_TR_r.py
├── fine_tune_TR_con_pre.py
├── README.md
│
├── VQA_Med_TR/
│   ├── C1C2C3_train.txt
│   ├── C1C2C3_val.txt
│   └── C1C2C3_test.txt
│
├── ImageClef-2019-VQA-Med-Training/
│   └── Train_images/
│       └── synpicXXXXX.jpg
│
├── ImageClef-2019-VQA-Med-Validation/
│   └── Val_images/
│
├── ImageClef-2019-VQA-Med-Test/
│   └── Test_images/
│
├── models/
└── results/
```


---

## Code Overview

### `main_TR_r.py`
Implements a **baseline Turkish VQA-Med pipeline** using ImageCLEF-style question–answer data.  
The script evaluates multiple combinations of **text encoders** (e.g., BERT, T5, XLM-R, LaBSE, BERTurk) and **image encoders** (e.g., ResNet, DenseNet, ViT).

**Key characteristics:**
- Loads Turkish VQA data from TXT files
- Extracts visual and textual features
- Performs multimodal fusion via feature concatenation
- Uses an MLP-based classifier for answer prediction

---

### `fine_tune_TR_r.py`
Extends the baseline pipeline by enabling **end-to-end fine-tuning** of the multimodal VQA model with controlled language adaptation.

**Key characteristics:**
- Partially fine-tunes the text encoder by unfreezing only the last *N* transformer blocks, while keeping earlier layers frozen
- Fully fine-tunes image encoders (ResNet, DenseNet, ViT) initialized with ImageNet-pretrained weights
- Jointly optimizes visual encoders, partially unfrozen language encoders, fusion layers, and the classifier
- Trains the model using a classification-based VQA formulation
- Computes classification metrics (accuracy, macro/weighted precision, recall, F1-score)
- Saves trained model checkpoints and evaluation reports to disk
---

### `fine_tune_TR_con_pre.py`
Introduces an additional **contrastive pretraining stage** prior to VQA classification.

**Key characteristics:**
- Performs CLIP-style contrastive pretraining between image and text embeddings using paired medical images and Turkish questions
- Fully fine-tunes image encoders (ResNet, DenseNet, ViT) initialized with ImageNet-pretrained weights during the contrastive alignment stage
- Partially fine-tunes text encoders by unfreezing only the last *N* transformer blocks
- Uses normalized image and text projections with a temperature-scaled contrastive loss
- Transfers the aligned image and text encoders to a downstream VQA classification model
- Jointly optimizes visual encoders, partially unfrozen language encoders, fusion layers, and the classifier
- Computes classification metrics (accuracy, macro/weighted precision, recall, F1-score)
- Saves trained model checkpoints and evaluation reports to disk

---

## Installation

Python 3.8.3 is recommended.

Main dependencies include:

- `torch`
- `torchvision`
- `transformers`
- `sentence-transformers`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

Example installation:

```bash
pip install torch torchvision
pip install transformers sentence-transformers
pip install numpy pandas scikit-learn matplotlib seaborn tqdm pillow

