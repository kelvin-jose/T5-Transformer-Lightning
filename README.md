# T5 for Sentiment Span Extraction
## T5 Overview
T5 is a recently released encoder-decoder model that reaches SOTA results by solving NLP problems with a text-to-text approach. This is where text is used as both an input and an output for solving all types of tasks. This was introduced in the recent paper, _Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer_ ([paper](https://arxiv.org/pdf/1910.10683.pdf)). I've been deeply interested in this model the moment I read about it.

I believe that the combination of text-to-text as a universal interface for NLP tasks paired with multi-task learning (single model learning multiple tasks) will have a huge impact on how NLP deep learning is applied in practice.

In this presentation I aim to give a brief overview of T5, explain some of its implications for NLP in industry, and demonstrate how it can be used for sentiment span extraction on tweets. I hope this material helps you guys use T5 for your own purposes!

## Key Points from T5 Paper
![T5 Usecases](https://camo.githubusercontent.com/58cef9e6b3da3b14fdc3782b5bdbefff63d0c678/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f69643d313558512d483749645433445674624c3766677759496d30384f5957624238565a)
1.  **Treats each NLP problem as a “text-to-text” problem and reaches SOTA results** - input: text, output: text

2.  **Unified approach for NLP Deep Learning** - Since the task is reflected purely in the text input and output, you can use the same model, objective, training procedure, and decoding process to ANY task. Above framework can be used for any task - show Q&A, summarization, etc.

3.  **Multiple NLP tasks can live in the same model** - E.g. Q&A, semantic similarity, etc. However, there is a problem called _task interference_ where good results on one task can also mean worse results on another task. E.g., a good summarizer may be bad at Q&A and vice versa. All the tasks above can live in the same model, which is how it works with the released T5 models (t5-small, t5-base, etc.)

## T5 for Sentiment Span Extraction (PyTorch)

1.  This is a dataset from an existing Kaggle [competition](https://www.kaggle.com/c/tweet-sentiment-extraction/data) - Tweet Sentiment Extraction
2.  Most of the existing model implementations use some sort of token classification task

    -   The index of the beginning and ending tokens are predicted and use to _extract_ the span
3.  T5 is an approach that is purely _generative_, like a classic language modelling task

    -   This is similar to abstractize summarization, translation, and overall text generation
    -   For our data, the span is not _extracted_ by predicting indices, but by generating the span from scratch

## Technology Stack
* PyTorch 1.5.x
* PyTorch Lightning
##  Brief about PyTorch-Lightning
![enter image description here](data:image/jpeg;base64,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)
Lightning is a way to organize your PyTorch code to decouple the science code from the engineering. It's more of a PyTorch style-guide than a framework.

In Lightning, you organize your code into 3 distinct categories:

1.  Research code (goes in the LightningModule).
2.  Engineering code (you delete, and is handled by the Trainer).
3.  Non-essential research code (logging, etc... this goes in Callbacks).

Here's an example of how to refactor your research code into a [LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/lightning-module.html).

## Reference
1. https://github.com/PyTorchLightning/pytorch-lightning/
2. https://pytorch.org/docs/stable/index.html
3. https://huggingface.co/transformers/model_doc/t5.html
