# Prompting-Based Fake News Detection Framework for Bengali

This repository contains the dataset structures, prompt templates, and evaluation framework for detecting misinformation in Bengali digital media using traditional machine learning classifiers and transformer-based architectures with prompt engineering techniques[cite: 1].

---

## Overview

The rapid spread of misinformation across Bengali digital media poses significant social, political, and public safety challenges[cite: 2]. Due to the low-resource nature of the Bengali language, standard supervised models often fail to generalize, while acquiring large annotated corpora remains expensive[cite: 2]. 

This project implements a hybrid framework evaluating traditional machine learning models alongside pretrained Transformer architectures (BERT, RoBERTa, T5) combined with prompt engineering paradigms (Zero-Shot, Few-Shot, and Chain-of-Thought)[cite: 1, 6, 7]. Results demonstrate that leveraging **T5 with Chain-of-Thought (CoT) prompting** achieves optimal classification performance and interpretability without requiring extensive fine-tuning[cite: 1, 12].

---

## Dataset

The dataset consists of **1,000 balanced Bengali news articles** categorized into four features with no missing values[cite: 5]:

* **`category`**: Domain of the news item (e.g., *National*, *Crime*, *Sports*, *Miscellaneous*)[cite: 5].
* **`headline`**: Short news titles in Bengali[cite: 5].
* **`content`**: Full text content or body of the article[cite: 5].
* **`label`**: Binary label where `0.0` denotes Real/Authentic News and `1.0` denotes Fake News[cite: 5].

### Dataset Sample

| Category | Headline | Content | Label |
| :--- | :--- | :--- | :--- |
| National | ঘূর্ণিঝড় ফণীর আঘাতে কলকাতা শহর টিকবে তো? | মানুষের স্পর্ধা দেখে বেশ অবাকই লাগে। সুপার সাইক... | `0.0` (Real) |
| Miscellaneous | একজন স্বামী যেভাবে চিড়িয়াখানার গরিলাকে জব্দ করলেন | স্বামী স্ত্রী গেছে চিড়িয়াখানায়। এক গরিলার খাঁচা... | `0.0` (Real) |
| Crime | দুই সাবেক সেনা কর্মকর্তার ফাঁসি, তিন জনের কারাদণ্ড | এনএসআইয়ের সাবেক মহাপরিচালক ব্রিগেডিয়ার জেনারেল... | `1.0` (Fake) |

---

## Methodology & Model Architectures

The framework compares traditional supervised baselines with prompt-conditioned transformer architectures[cite: 6, 7]:

### 1. Traditional Machine Learning Models
Traditional baselines rely on handcrafted text representations (TF-IDF vectors and bag-of-words features)[cite: 6]:
* **Logistic Regression**[cite: 6]
* **Decision Tree**[cite: 6]
* **Random Forest**[cite: 6]
* **Gradient Boosting**[cite: 6]

### 2. Transformer Architectures & Prompting
* **BERT**: Bidirectional encoder representations tested with masked language prompts[cite: 7, 8].
* **RoBERTa**: Optimized BERT variant evaluated across zero-shot and few-shot templates[cite: 7, 9].
* **T5 (Text-to-Text Transfer Transformer)**: Casts the task into a pure text generation setup and evaluated using Zero-Shot, Few-Shot, and Chain-of-Thought (CoT) prompting strategies[cite: 7, 10].
  ---

---

## Experimental Setup

* **Hardware Environment**: NVIDIA Tesla V100 (32 GB) & A100 (40 GB) GPUs, Dual Intel Xeon Processors, 256 GB RAM[cite: 8].
* **Frameworks**: PyTorch, Hugging Face Transformers[cite: 8].
* **Optimizer & Learning Rate**: AdamW optimizer, learning rate $2 \times 10^{-5}$ to $5 \times 10^{-5}$, batch size 16–32[cite: 8].
* **Data Split**: 70% Training, 15% Validation, 15% Testing (evaluated using 5-fold cross-validation)[cite: 8].

---

## Key Results & Evaluation

### 1. Traditional ML vs. Baseline Transformers Performance

| Task Category | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fake News** | Logistic Regression | 75 | 69 | 34 | 45 |
| | Decision Tree | 65 | 43 | 46 | 44 |
| | Gradient Boosting | 74 | 79 | 18 | 29 |
| | Random Forest | 73 | 65 | 24 | 35 |
| | BERT | 59 | 64 | 43 | 53 |
| | RoBERTa | 51 | 51 | 47 | 49 |
| | **T5** | **64** | **100** | **37** | **54** |
| **True News** | Logistic Regression | 75 | 76 | 84 | 45 |
| | Decision Tree | 65 | 75 | 73 | 74 |
| | Gradient Boosting | 74 | 73 | 98 | 84 |
| | Random Forest | 73 | 74 | 94 | 83 |
| | BERT | 59 | 57 | 76 | 65 |
| | RoBERTa | 51 | 51 | 55 | 53 |
| | **T5** | **64** | **100** | **100** | **100** |

### 2. Prompting Technique Comparison

| Task Category | Prompting Strategy | Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Fake News** | Zero-Shot | BERT | 70.0 | 74.0 | 62.0 | 67.5 |
| | Few-Shot | RoBERTa | 54.5 | 55.0 | 47.0 | 50.7 |
| | **Chain-of-Thought** | **T5** | **80.0** | **100.0** | **75.0** | **85.7** |
| **True News** | Zero-Shot | BERT | 60.0 | 64.0 | 43.0 | 53.0 |
| | Few-Shot | RoBERTa | 51.0 | 51.0 | 47.0 | 49.0 |
| | **Chain-of-Thought** | **T5** | **92.5** | **100.0** | **85.0** | **91.9** |
---

## Primary Findings & Error Analysis

1. **Superiority of Chain-of-Thought Reasoning**: T5 paired with Chain-of-Thought (CoT) prompting outperformed all traditional baselines and baseline transformers, achieving an F1-score of **85.7%** on fake news detection and **91.9%** on true news detection[cite: 1, 12].
2. **Data-Efficient Prompting**: Prompt-based LLM frameworks offer a cost-effective solution for detecting misinformation in low-resource languages without needing large-scale task-specific fine-tuning[cite: 1, 4].
3. **Common Error Patterns**:
   * **Satire & Humor**: Models frequently misclassified satirical articles as true news due to difficulty detecting irony[cite: 13].
   * **Ambiguous Political Content**: Politically charged articles lacking explicit factual anchors caused prediction errors[cite: 13].
   * **Code-Mixed Text**: Mixed English-Bengali tokens introduced tokenization issues, impacting consistency[cite: 13].

---

## Citation

If you use this work or framework in your research, please cite the original paper:

```bibtex
@inproceedings{mishrra2025prompting,
  title     = {Prompting-based fake news detection framework for Bengali},
  author    = {Mishrra, Shillpi and Dhar, Arpita and Mondal, Somnath and Ghosh, Aditi and Bal, Sauvik},
  booktitle = {Proceedings of the 4th International Conference on Innovation in Data Analytics (ICIDA)},
  year      = {2025},
  publisher = {Springer}
}
