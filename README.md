# IMDB Sentiment Analysis - Group Project

## Overview

Our team selected **Task C: Sentiment Analysis** (SA) using the **IMDB movie review dataset**, where the model is tasked to predict the sentiment of a given movie review. 

In this report, our team worked closely with transformer models, which is the SOTA deep learning architecture for natural language processing (NLP).

We intend to understand the overarching capabilities of transformer models in text sentiment analysis. In particular, we study the following capabilities:
- **Performance of different types of fine-tuning.** Between full fine-tuning, LoRA, prefix-tuning, and IA3, which one results in the most performant BERT model?
- **Comparison of various transformer architectures.** In text sentiment analysis tasks, which one performs the best, encoder-only, decoder-only or encoder-decoder model?
- **Comparison of various BERT models.** Between BERT, RoBERTa, and ALBERT, which one is the most suitable in text sentiment analysis tasks?
- **Domain adaptation.** How can I adapt a model pre-trained on a general dataset to our specific IMDB dataset?
- **Data augmentation.** How can I deal with small dataset availability?
- **Hyperparameter tuning.** How do I find the best hyperparameter for fine-tuning models?


## Instructions for Running Codes

### 1. Hardware and Environment

- All experiments were executed using the **CCDS GPU Cluster** and/or **NSCC**.
- We provide `./requirements.txt` which contains the dependency for our files. Install it on your virtual environment.

### 2. Quick Guide to Experiments

Each of our experiment is linked to its corresponding folder:

| **Experiment**                                      | **File Location**                                             |
| --------------------------------------------------- | -------------------------------------------------------- |
| 3. Performance of Different Types of Fine-tuning| `./finetune/peft` |
| 4. Comparison of Different Transformer Architectures        | `./finetune/architectures`             |
| 5. Comparison of Different Types of BERT Models           | `./finetune/bert_models`, `./mlm_eval`                           |
| 6. Domain Adaptation                             | `./pretraining/from_existing/HF_BERT_pro`, `./mlm_eval`                             |
| 7. Data Augmentation | `./data_augmentation`         |
| 8. Hyperparameter Tuning |`./hyperparam` |

**Note**: The serial numbers of the experiments correspond to the sections in the **Report.pdf**. All of our training results are located at `./logs`. The hyperparameter tuning figures can be found at `./hyperparameter_plots`.

### 3. Additional Code 
We also provide some extra code that did not make it to the final report due to space constraints:
- `./llama`, where we ran **LLaMA:1B, LLaMA:3B** and **LLaMA:8B** on the IMDB dataset. Results can be found at `./llama_results`.
- `./pretraining/from_scratch` which contains our implementation of BERT from scratch. The results can be found in `./logs/wes-bert`
- `./pretraining/from_existing/ALBERT`, which is our domain adaptation attempt for ALBERT. Training ALBERT takes more than 2 days **(You have been warned)**.
- `./pretraining/from_existing/RoBERTa` contains our domain adaptation attempt for RoBERTa. The results can be found in `./logs/roberta-pro`.
- `./.sh` and `./.pbs` are job files for **NTU GPU Cluster** and **NSCC** respectively. You may refer to this, or make your own.


## Conclusion

- Discovered that **full fine-tuning** is the most suitable fine-tuning method due to the small IMDB dataset.
- GPT-2 and BART outperformed BERT in our dataset, not due to insuitability of encoder-only model, but due to **inefficiency of BERT.**
- RoBERTa, another encoder-only model, was able to demonstrate that it was superior to BERT, and outperformed GPT-2 and BART.
- Demonstrated the importance of domain adaptation, by showing the significant improvements in model performance after fine-tuning BERT on the domain of IMDB movie reviews using the **MLM objective**.
- We also experimented with **back-translation** as a data augmentation technique to deal with small datasets, but found that back-translation was **largely unsuitable** for our IMDB dataset.
- Hyperparameter search using TPE reveals that **higher batch size**, such as 32, **is important** in ensuring higher accuracy, despite conventional wisdom.

**Contributors:**
1. Nadya Yuki Wangsajaya @yukiwukii
2. Shiu Lok Chun, Wesley @HamsterW

