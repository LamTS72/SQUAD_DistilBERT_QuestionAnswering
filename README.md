# QUESTION_ANSWERING_Text2Text
<p align="center"> <img width = "100%" height = "100%" src="imgs/demo.png"/>  </p>

## Introduction

This is finetuning model of BERT to handle SQUAD task from NLP.

## Description

Dataset: `*squad*`

Language: `*pytorch*`

## Notes:

`data_train.py`: represent the dataset

`preprocessing.py`: tokenizer dataset, align inputs and labels after tokenized that occur changed length of sentences.

`train.py`: train model

`predictor.py`: inference model

`config.py`: hyperparameter config

## How to use

1. Training:

`python3 train.py`

2. Inference:

`python3 predictor.py`

3. Pipeline for Gradio Deployment

`python3 main.py`
