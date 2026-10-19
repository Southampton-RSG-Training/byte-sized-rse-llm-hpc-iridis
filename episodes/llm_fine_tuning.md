---
title: "Fine-Tuning an LLM on Iridis X"
teaching: 10 # teaching time in minutes
exercises: 2 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I fine-tune a large-language model on Iridis?
- How do I use multiple Iridis X nodes with SLURM?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

-

::::::::::::::::::::::::::::::::::::::::::::::::

## Fine-Tuning an LLM

Our starting point for the activity is some code that takes a trained LLM and fine-tunes it to to sentiment analysis on Yelp data. We'll be using HuggingFace's Transformers and Accelerate libraries to handle training models and working with different hardware.

``` python
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=num_labels,
    torch_dtype=torch.float32,
)
```


