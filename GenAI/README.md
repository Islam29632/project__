# Question Answering System with BERT

This project implements a Question Answering (QA) system using BERT (Bidirectional Encoder Representations from Transformers) fine-tuned on the SQuAD v2 dataset. The system can extract answers from given contexts for user-provided questions.

## Features

- Fine-tuned BERT-base-uncased model for question answering
- Preprocessing pipeline for SQuAD v2 dataset
- Training and evaluation with Hugging Face Transformers
- Ready-to-use QA pipeline for inference
- Performance metrics (F1 score) tracking

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Datasets library
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy

Install requirements:
```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn pandas numpy
