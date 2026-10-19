---
title: "Local LLMs and Python"
teaching: 10 # teaching time in minutes
exercises: 2 # exercise time in minutes
---

:::::::::::::::::::::::::::::::::::::: questions

- How do I work with large language models in research?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Learn how to use the Transformers library to load models and perform inference.

::::::::::::::::::::::::::::::::::::::::::::::::

## Large Language Models on Python

There are a number of different libraries and applications that allow you to run and train large language models.  Many of these are optimised for inference or production use, rather than training or fine-tuning, and others lack the flexibility required for research use-cases. Some are aimed entirely at local use-cases and don't work well with HPC systems.

Since LLMs are deep neural networks of one architecture or another, the most basic way to work with them is to use a standard deep learning library, such as Torch, Tensorflow, or JAX. However for common tasks, such as inference, fine-tuning or even training, these libraries are likely lower-level than you would like to work with given the complexity of a modern local LLM: there is a lot of boilerplate code to set up the model. In particular for models with pre-trained weights, this means exactly matching the model architecture that the weights were generated from, which might not be present in the saved data.

Nvidia's HuggingFace website is a repository for model weights in a similar way that GitHub is a repository for code, and hosts most public open-weight models. To facilitate working with these models, HuggingFace has written several libraries which integrate with the HuggingFace website and make working with these models easier. These libraries sit on top of PyTorch, and so can integrate easily with Torch.

In previous sessions we've used the Accelerate library to handle differences in hardware and to distribute work across multiple GPUs. In this session we'll also use Accelerate to distribute the work, and even the models across multiple nodes in an HPC cluster.

We'll also use the Transformers library to handle the loading and preparation of the large language models themselves.

### Transformers

An easy way to use the Transformers library is via it's `AutoClass` API, where you specify the sort of task that you want to perform, and point the appropriate factory method at the weights or data.

For a typical LLM, you will need to set up a tokenizer and the model itself. For example, to load the Llama 2 7 billion parameter model and the corresponding tokeniser for inference, you would use:

``` python
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
```

This will download the model from the HuggingFace repository, create the model and tokeniser objects, and load weights and other configuration into them.

In addition to `AutoModelForCausalLM`, there are other ways that an LLM can be loaded:

- `AutoModel.from_pretrained` creates a basic model and loads the weights.
- `AutoModelForPreTraining.from_pretrained` sets the model up for pre-training (single token prediction).
- `AutoModelForSequenceClassification.from_pretrained` wraps an LLM so that it can be fine-tuned for classification.
- `AutoModelForQuestionAnswering.from_pretrained` wraps an LLM so it can be fine-tuned for question answering.
- `AutoModelForMultipleChoice.from_pretrained` wraps an LLM so it can be fine-tuned for multiple-choice phrase completion.
- `AutoModelForNextSentencePrediction.from_pretrained` wraps an LLM so it can be fine-tuned for prediction of the next sentence.
- `AutoModelForTokenClassification.from_pretrained` wraps an LLM so it can be fine-tuned to classify parts of a sequence.
- `AutoModelForSeq2SeqLM.from_pretrained` wraps an LLM so it can be fine-tuned for sequence to sequence transformation.
- `AutoModelForMaskedLM.from_pretrained` wraps an LLM so that it can be used for missing word prediction.

Despite its name, the Transformers library is not limited to transformer architecture LLMs, and can load and work with other types of LLMs. Additionally there are `AutoClass` APIs to support audio, computer vision and time-series models, as well as multimodal models.

While Transformers integrates closely with the HuggingFace repositories, these methods can load from appropriately structured local sources, such as "safetensors" format files.

### Datasets

The Datasets library provides access to training data in various formats suitable for use with particular tasks, primarily for use with Transformers, but also potentially with other tools and libraries as well.

### Inference

For most inference tasks using a pre-trained LLM, you need to tokenize the inputs using the tokenizer:

``` python
model_inputs = tokenizer(["The Python programming language is useful for "], return_tensors="pt")
```

For simple inference, you call the model with the appropriate inputs:
``` python
prediction = model(**model_inputs)
```

In this case the prediction is an object with relevant data attached, in particular, for LLM
``` python
prediction.logits
```
is a tensor with the logit weights for each token which can be turned into a probability distribution for the next token in the sequence with `softmax`.

For general text generation there are methods that simplify the process of repeatedly generating tokens:
``` python
tokens = model.generate(**model_inputs, max_length=100)
words = tokenizer.batch_decode(tokens)
print(words[0])
```

In fact for the most common tasks with pre-trained models, you can use the `pipeline` factory to set everything up to give an object that you can simply call to perform inference.

``` python
from transformers import pipeline

pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M-Instruct")
```

For instruct models (as opposed to foundation models) you can pass a message dictionary with the role and content and it will be automatically formatted as the model expects:

``` python
messages = [{'role': 'user', 'content': "The University of Southampton is?"}]
print(pipe(messages, max_new_tokens=100,))
```

You can also do this with the tokenizer for instruct models:
``` python
model_inputs = tokenizer.apply_chat_template(messages)
```

::: spoiler

### Foundation vs. Instruct or "Chat" Models

Foundation models are those which have simply been trained to predict the next word. Instruction-tuned models are foundation models which have further been trained to expect input to have a particular structure, usually in the form of a conversation: a user instruction followed by a response from the system. More advanced models may have additional roles such as "system" and "thinking". The exact format used to distinguish between the roles varies greatly across different models.

It is important to select the right sort of model for your task. For example, when performing inference an instruct model may be able to perform simple text-generation tasks, but it will not do so as well as a similar foundation model. Similarly, a foundation model may be able to infer the required structure for a instruct interface, but it will not perform as well as a model fine-tuned for that format.

Similarly, when fine-tuning, most often a foundation model is the most appropriate choice unless, the task is specifically to fine-tune the responses of an instruction model.

:::

::: spoiler

### Alternative Frameworks

We're using Transformers in this course because they give an API that is flexible and permits training, and integrates well with PyTorch.  However, this flexibility comes at some cost in terms of speed and resources. If all you need is inference on open weights models, there are libraries such as [vLLM](https://vllm.ai/) and [TensorRT LLM](https://developer.nvidia.com/tensorrt-llm) that give better performance for pure inference tasks.  However these may or may not work as well across multiple nodes on Iridis.

:::

### Fine-Tuning

Training an LLM is, in principle, the same as training any other deep neural network. For each epoch you iterate over the instances and then each batch in the training data, and perform a prediction. A loss is computed, and back-propagation performed to update the trainable weights.

One difference is that datasets typically store training data as plain text, where the model needs to be given tokenised data.  While you can do this on-demand that means re-computing the tokenized values again and again for each epoch. So it makes sense to pre-compute the tokenised values.

So, for example, you can train a classifier on top of an LLM with code like the following, which is not dissimilar to general deep learning training code:

``` python
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    num_labels=num_labels,
)
optimizer = AdamW(model.parameters(), lr=5e-5)

dataset = load_dataset("yelp/yelp_review_full")
train_dataloader = DataLoader(dataset["train"], shuffle=True)
```

The training loop is then exactly as you would do for a typical deep learning model: compute the forward, get the loss, back-propagate and update the optimizer's state.

``` python
for epoch in range(1, epochs+1):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
```

Other LLM-related models can be trained similarly with the PyTorch API with appropriate modifications to the loss functions, scheduling, optimizers and so forth. And the techniques using Accelerate discussed in previous sessions can be used to distribute work across available hardware.

This works the same way as discussed in previous sessions. The appropriate objects are given as arguments to Accelerate's `prepare` function, and the corresponding return values are wrapped to run on the available hardware.

## Distributed Logging


