# Toxicity Detection with Classical and Transformer-Based Models

## Overview
This project compares classical machine learning (TF-IDF + MLP) with transformer models (BERT, BERTweet, HateBERT) for toxicity detection in online text. We evaluate how domain-specific pretraining affects performance on binary classification, multi-label prediction, and cross-domain generalization from Wikipedia to Reddit.

## Models
- **MLP**: TF-IDF + MLP baseline (classical approach)
- **BERT**: Pretrained on general English (Wikipedia, BookCorpus)
- **BERTweet**: Pretrained on 850M English tweets (social media adapted)
- **HateBERT**: BERT extended with toxic Reddit content (abusive language adapted)

## Datasets
- **Jigsaw Toxic Comment Classification**: 159,571 Wikipedia comments with 6 toxicity labels (toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **Real Toxicity Prompts**: 10,000 Reddit comments for cross-domain evaluation

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Train MLP
python train_wikipedia.py

# Train HateBERT
python train_hatebert.py --train_csv train.csv --output_dir outputs/hatebert
```

Training scripts for BERT and BERTweet are available in the `training/` directory.

## Results
<<<<<<< Updated upstream

### Binary Classification (Wikipedia Validation)
| Model | Accuracy | F1 |
|-------|----------|-----|
| HateBERT | 98.95% | 94.84% |
| BERTweet | 98.17% | 90.92% |
| BERT | 95.60% | 82.10% |
| MLP | 91.10% | 52.30% |

### Cross-Domain Evaluation (Reddit)
| Model | Accuracy | F1 |
|-------|----------|-----|
| HateBERT | 89.66% | 70.86% |
| BERTweet | 88.93% | 68.22% |
| BERT | 88.71% | 67.62% |
| MLP | 74.22% | 16.52% |

## Key Findings
- **Domain-specific pretraining significantly improves performance**, especially on rare categories (threats, identity hate)
- **HateBERT achieves best overall results** due to toxic content pretraining
- **BERTweet excels at informal, slang-heavy toxicity** from social media adaptation
- **All models show performance degradation on cross-domain evaluation**, with MLP failing severely

=======
- MLP performed at **91.10%** accuracy for the validation set and a **74.22%** accuracy for cross-validation
- BERT performed at **95.60%** accuracy for the validation set and a **88.71%** accuracy for cross-validation
- BERTweet performed at **98.17%** accuracy for the validation set and a **88.93%** accuracy for cross validation
- HateBERT performed at **98.95%** accuracy for the validation set and a **89.66%** for cross validation

## References
This project was done for CS6120: Natural Language Processing at Northeastern University taught by Professor Silvio Amir. 
>>>>>>> Stashed changes
