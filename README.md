# Toxicity Classifier

## Overview
This project aims to test different models and analyze how well each model detects toxicity in text. This project tests

## Features
- Binary classification of text comments
- Multi label toxicity classifier
- Preprocesses and tokenizes input data
- Analysis of different toxicity models

## Installation
```bash
pip install -r requirements.txt
```

## Usage
All training is either done in the Google Colab file or the ```train_wikipedia.py``` file

## Dataset
- ```train.csv``` was trained on an 80/20 split training and validation for all models. 
- ```allenai/real-toxicity-prompts``` from huggingface was used to cross-validate our training process

## Results
- Classical MLP model performed at 91.10% accuracy for the validation set and a 74.22% accuracy for cross-validation
- BERT performed at 95.60% accuracy for the validation set and a 88.71% accuracy for cross-validation
- BERTweet performed at 98.17% accuracy for the validation set and a 88.93% accuracy for cross validation
- HateBERT performed at 98.95% accuracy for the validation set and a89.66% for cross validation