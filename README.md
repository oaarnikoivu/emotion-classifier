# Ainoa

![Application](images/landing_page.png)

This model was trained to detect multiple emotions from text. The model achieves comparable results to the state-of-the-art by using an Attention LSTM with contextualized embeddings produced by a frozen BERT transformer model.

Inspired by the [DeepMoji](https://deepmoji.mit.edu/) website from MIT, a similar application has been implemented where you can view how the Attention LSTM model captures the most important semantic information from your given input text. You can play around with the model at <https://ainoa.netlify.app>.

## Overview

- [/client](https://github.com/oaarnikoivu/ainoa/tree/master/client) contains all client-side code related to the UI of the application.
- [/server](https://github.com/oaarnikoivu/ainoa/tree/master/server) contains the server-side code related to the application.
- [/notebooks](https://github.com/oaarnikoivu/ainoa/tree/master/notebooks) contains all the iPython notebooks to carry out the methods & experimental setup.

## Technologies

The model architecture is constructed using PyTorch. The [HuggingFace Transformers]() library is made use of in order to apply and retrive the embeddings from a pre-trained BERT model. [Scikit Learn]() is also made use of to assess model performance using the Jaccard index, and micro-averaged and macro-averaged F1 scores, as well as for the implemenation of the baseline machine learning algorithms. The models were trained with Google Colaboratory notebooks using an NVIDIA Tesla K80 GPU.

## Dataset

The dataset used is the [SemEval Task 1: Affect in Tweets]() emotion classification dataset where given a tweet, the task is to classify the text as having no emotion or as one, or more, emotions for eight of the [Plutchik]() categories plus optimism, pessimism, and love. The dataset consists of 6838 training examples, 886 validation examples and 3259 testing examples.

[Report](https://github.com/oaarnikoivu/ainoa/blob/master/1502639%20AARNIKOIVU%20Oliver%20-%20Thesis.pdf)
