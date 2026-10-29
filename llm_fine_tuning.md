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

The goal is to fine tune a small LLM on the [Yelp reviews data](https://huggingface.co/datasets/Yelp/yelp_review_full), trying to predict the rating of a restaurant from the text of a review. We'll use the [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) from HuggingFace. We're deliberately using a tiny model for the example code, but the same code should work with a much larger model. The scripts take a `--model` argument which allows you to experiment with other models if you choose.

Ideally you want to be able to run and test your code locally before you try to run in the HPC environment.

::: challenge
### Install the code and LLM locally

To run the code locally, clone the github repository for the course and set up a virtual environment:

``` bash
git clone https://github.com/Southampton-RSG-Training/byte-sized-rse-llm-fine-tuning-example.git
cd byte-sized-rse-llm-fine-tuning-example
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Now we need to download the data and LLM. There is a Python script that will do this for us:

``` bash
python llm_download.py
```

:::

The setup for the training is largely as discussed earlier, using `AutoModelForSequenceClassification` since the fine-tuning is a classification exercise.

``` python
num_labels = len(dataset.unique('labels')['train'])
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=num_labels,
    torch_dtype=torch.float32,
)
```

::: challenge

### Run the fine-tuning locally

To fine-tune the model, you just need to run the script:

``` bash
python llm_fine_tuning.py
```

:::

### Training on Iridis

To run this on the HPC cluster, we go through much the same process as locally.  We need to install the code from GitHub and download the dataset and LLM weights. This has to be done on the login node, since we need internet access.

::: challenge

### Install the code and LLM on Iridis

Login to the Iridis X login node `loginX001.iridis.soton.ac.uk`. The bash command looks something like:

``` bash
ssh loginX001.iridis.soton.ac.uk
```

Once logged in, the steps to install the software and model are largely the same, however we need to remember to load the Python and CUDA module so that we have access to Python and everything installs correctly:

``` bash
module load python
module load cuda
git clone https://github.com/Southampton-RSG-Training/byte-sized-rse-llm-fine-tuning-example.git
cd byte-sized-rse-llm-fine-tuning-example
python3.14 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python llm_download.py
```

:::

You can run the code using sbatch as normal. There are a number of different batch files configured for different scenarios, including a simple single-node multi-GPU configuration for the `l4` partition, a multi-node configuration for Iridis 7, and a fully-sharded data parallel configuration for `l4`.

::: challenge

After this session is over, at your leisure, you can try running the single-node configuration on `l4`:

``` bash
sbatch single-node-l4.sh
```

:::

