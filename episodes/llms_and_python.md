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

## Large Language Models in Python

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

So, for example, you can train a classifier on top of an LLM with code like the following, which is not dissimilar to general deep learning training code:

``` python
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    num_labels=num_labels,
)
optimizer = AdamW(model.parameters(), lr=5e-5)

dataset = load_dataset("yelp/yelp_review_full")
```

One difference is that datasets typically store training data as plain text, where the model needs to be given tokenised data.  While you can do this on-demand that means re-computing the tokenized values again and again for each epoch. So it makes sense to pre-compute the tokenised values.  Code to do this looks something like this:

``` python
# This particular tokenizer doesn't have a pad token, so pad with sequence tokens
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    """Truncate/pad to 256 tokens."""
    return tokenizer(
        examples[columns["text"]],
        pad="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch")

train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True)
```

A note that in addition to tokenising, this function normalizes the lengths of the inputs to the first 256 tokens, truncating or padding as needed.  The padding is needed because each batch must have the same number of tokens for each sample, while the truncation is to make the code run in reasonable time.  Truncation may not be appropriate if you need to use the full text to train on.

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

Other LLM-related models can be trained similarly with the PyTorch API with appropriate modifications to the loss functions, scheduling, optimizers and so forth.

The techniques using Accelerate discussed in previous sessions can be used to distribute work across available hardware. This works the same way as discussed in previous sessions. The appropriate objects are given as arguments to Accelerate's `prepare` function, and the corresponding return values are wrapped to run on the available hardware.

``` python
train_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, model, optimizer
)
```

### Transformers `Trainer` class

There is frequently a lot of boilerplate in performing training loops, and so there are a number of libraries which allow you to avoid writing this code. As long as your research doesn't require non-standard steps in the training loop, you can simplify your code a lot by using these libraries.

[Keras](https://keras.io/) is popular because it allows you to use different backends (Torch, Tensorflow, or JAX) and has a high-level API for designing deep learning models, and so is a good choice where your research is into the architectures of the deep learning model.

However Transformers has its own approach to this problem, which is the `Trainer` class. The `Trainer` class is designed to handle more common, practical training tasks, and so is useful when you want to do something standard and then use the resulting model in your research.

To use the trainer class you need to:

- (optionally) create an appropriate `DataCollator` for the training;
- set up the hyper-parameters for your training run in a `TrainingArguments` class;
- pass those along with your tokeniser, model and data to the `Trainer` class;
- call the `train()` method of the `Trainer`

The `DataCollator` classes handle dynamically padding and formatting the data in each batch to the appropriate consistent shape. Different subclasses are useful for different cases, such as `DataCollatorForLanguageModeling` for fine-tuning a language model directly. The default collator simply pads to the length of the longest input in a batch.

In our example, we have already normalized the entire dataset to 256 tokens, but we could have the same effect by using a data collator like:

``` python
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding='max_length',
    max_length=256,
)
```

The `TrainingArguments` class accepts information about the number of epochs, batch sizes, optimizer type, learning rates, check-pointing, logging, evaluation metrics, Accelerate parameters, distributed training strategies, and more.

A typical configuration might look like:

``` python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    optim='adamw_torch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)
```

And then the actual training can be performed as follows:

``` python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

The `Trainer` class has Accelerate code embedded in it, so other than possibly setting some accelerate-related parameters on the `TrainingArguments` class, no additional work is needed to run the code using accelerate.

## Running on Multiple Nodes

To run on multiple nodes you need to specify the number of nodes you wish to use. You also need to specify the number of tasks and the number of GPUs and CPUs. Unlike the CPU clusters, there tend to be fairly strict limits on the number of nodes you can use, and it usually just one on Iridis but some partitions and users allow more - it is generally better strategy to move your work to a partition with more or more powerful GPUs than to split it across multiple nodes, so it's only when you start exceeding the capabilities of the most powerful machines available that you should consider running multi-node. If you try to use more nodes than you are allowed, you will see an `QOSMaxNodePerUserLimit` message when you query the status of your job.

So for example, to run on 2 nodes with 2 GPUs, you would configure the SLURM batch file something like:

``` bash
#SBATCH --nodes=2                         # Number of Nodes
#SBATCH --gpus=2                          # GPUs per Node
#SBATCH --ntasks=2                        # One per Node
#SBATCH --gpus-per-task=2                 # Every task to use two GPUs
#SBATCH --cpus-per-task=2                 # Every task to use two CPUs
```

Note that normally you would use all GPUs on a single node before you would move to multiple nodes.

You also need to have the `python`, `cuda` and `openmpi` modules loaded.

``` bash
module load python
module load cuda
module load openmpi
```

Beyond SLURM configuration, Accelerate needs to know which node is the "main" (rank 0) node's IP address, so we have to work that out from the SLURM environment:

``` bash
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Head node IP: $head_node_ip"
```

This then allows us to use `accelerate` to run the training:

``` bash
accelerate launch --multi-gpu --num_processes 4 --num_machines 2 --main_process_ip $head_node_ip --main_process_port 29500 llm_fine_tuning.py
```

### Types of Distributed Training

By default, multi-GPU training uses a data parallel approach to training.  This is fine for small models, such as the one we are using here.  However for large models, you are likely to run out of GPU memory. In this case you want to switch to fully-sharded data parallel (FSDP) which splits the model layers over multiple GPUs.

To enable FSDP with Accelerate you have to specify `--use_fsdp` in the command-line. There are [a number of additional parameters](https://huggingface.co/docs/accelerate/en/package_reference/cli) that you can specify to control the way that layers and weights are distributed.

