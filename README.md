#  AI Omympiad MATH

<div align="center">
    <img src="https://i.ibb.co/9rx4pbX/AIMO.png">
</div>

Welcome to the AI Mathematical Olympiad (AIMO). This challenges us to develop algorithms and models that can solve intermediate-level high school math problems written in LaTeX format, advancing AI's mathematical reasoning capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Data Analysis](#data-analysis)
- [Feature Engineering](#feature-engineering)
- [Modeling and Training](#modeling-and-training)
- [Model Evaluation](#model-evaluation)
- [Deep Learning Model](#deep-learning-model)
- [Fine Tuning](#fine-tuning)

<a name="overview"></a>
## 🌟 Overview

The ability to reason mathematically is a critical milestone for AI. Mathematical reasoning is the foundation for solving many complex problems, from engineering marvels to intricate financial models. However, current AI capabilities are limited in this area.

The AI Mathematical Olympiad (AIMO) Prize is a new $10mn prize fund to spur the open development of AI models capable of performing as well as top human participants in the International Mathematical Olympiad (IMO). This competition includes 110 problems similar to an intermediate-level high school math challenge. The Gemma 7B benchmark for these problems is 3/50 on the public and private test sets.

The assessment of AI models' mathematical reasoning skills faces a significant hurdle, the issue of train-test leakage. Models trained on Internet-scale datasets may inadvertently encounter test questions during training, skewing the evaluation process.

To address this challenge, this competition uses a dataset of 110 novel math problems, created by an international team of problem solvers, recognizing the need for a transparent and fair evaluation framework. The dataset encompasses a range of difficulty levels, from simple arithmetic to algebraic thinking and geometric reasoning. This will help to strengthen the benchmarks for assessing AI models' mathematical reasoning skills, without the risk of contamination from training data.

This competition offers an exciting opportunity to benchmark open AI models against each other and foster healthy competition and innovation in the field. By addressing this initial benchmarking problem, you will contribute to advancing AI capabilities and help to ensure that its potential benefits outweigh the risks.

Join us as we work towards a future where AI models’ mathematical reasoning skills are accurately and reliably assessed, driving progress and innovation across industries.

<a name="data-analysis"></a>


## 📊 Data Analysis

The competition data comprises 110 mathematics problems similar in style to those of the AIME. The answer to each problem is a non-negative integer, which you should report modulo 1000. If, for instance, you believe the answer to a problem is 2034, your prediction should be 34.

All problems are text-only with mathematical notation in LaTeX. Please see the AIMO Prize - Note on Language and Notation.pdf handout for details on the notational conventions used. Although some problems may involve geometry, diagrams are not used in any problem.

The public test set comprises exactly 50 problems, and the private test set comprises a distinct set of 50 problems. We also provide a selection of 10 problems for use as training data. The problems in the two test sets have been selected to balance both difficulty and subject area.


<a name="feature-engineering"></a>
## 🧪 Feature Engineering

In-depth feature engineering was conducted to enhance the model's predictive power. This includes:
- Cleaning and preprocessing LaTeX-formatted math problems.
- Extracting relevant mathematical structures and symbols.
- Converting text into a format that can be fed into AI models.
  
<a name="modeling-and-training"></a>

## 🤖 Modeling and Training

### Load Data
No training data is provided in this competition; in other words, we can use any openly available datasets for this competition. In this notebook, we will use a modified Math dataset which I have compiled to have a Question-Solution-Answer format. These datasets include:

problem: The math problem in LaTeX format.
solution: Step-by-step solution to this problem.
answer: Final answer of the solution which will be the ground truth for this competition.
level: Difficulty of the problem.
type: The category of the problem.
This dataset comes with its own train test split. However, we will merge them both and use them for fine-tuning. You are welcome to use them for training and validation separately. Also to reduce the training time we will only be training on the first 1000 samples. You are welcome to train on the full data.

### Filter Data
The Math dataset contains various problems, but not all of them are suitable for this competition. More specifically, this competition requires a non-negative integer answer, while the Math dataset includes problems with different types of answers such as integers, floats, fractions, matrices, etc. In this notebook, we will only use those problems whose answers are non-negative integers and filter out the rest.

### Prompt Engineering
We will be using below simple prompt template we'll use to create problem-solution-answer trio to feed the model. This template will help the model to follow instruction and respond accurately. You can explore more advanced prompt templates for better results.


Role:
You are an advanced AI system with exceptional mathematical reasoning and problem-solving capabilities, specifically designed to solve tricky math problems (whose answer is a non-negative integer) written in LaTeX format from the AI Mathematical Olympiad (AIMO) competition. Your task is to accurately analyze and solve intricate mathematical problems, demonstrating a deep understanding of mathematical concepts and a strong ability to apply logical reasoning strategies.

Instruction:
1. Carefully read and comprehend the problem statement provided in the "Problem" section.
2. In the "Solution" section, provide a solution of the problem with detailed explanation of your logical reasoning process. Keep in mind that answer must be a non-negative integer number.
3. At the end, create a "Answer" section where you will state only the final numerical or algebraic answer, without any additional text or narrative.

Problem:
...

Solution:
...

Answer:
...

<a name="evaluation"></a>
## 🎯 Model Evaluation

The effectiveness of the models will be evaluated based on their ability to correctly solve the math problems and their accuracy in generating the correct answers.

<a name="deep-learning-model"></a>
## 🧠 Deep Learning Model

<div align="center"><img src="https://chartstatic.com/images/symbol_logos/meta.png" width="300"></div>


**LLama3** is a collection of advanced open LLMs developed by Meta AI. They are built to provide state-of-the-art performance in a wide range of applications, from text generation to complex problem solving. **LLama3**  models are designed to be versatile, offering capabilities that can be fine-tuned for specific tasks, making them suitable for both research and production environments.

In this implemntation , we will utilize the **llamma3-70b** model from Meta AI to solve the math olympiad questions. This model is chosen for its superior performance in understanding and generating complex text. By fine-tuning the LLama3 70b model, we aim to enhance its problem-solving abilities specifically for mathematical problems.

To explore other available models, you can adjust the model_path value in the CFG (config). You can find more details about LLama3 models on the (Meta AI website)[https://llama.meta.com/llama3/].

LLama3 models are available in several sizes, allowing users to choose models based on their computational resources and the requirements of their specific applications.


We built an end-to-end LLama3 model for causal language modeling. A causal language model (LM) predicts the next token based on previous tokens. This task setup can be used to train the model unsupervised on plain text input or to autoregressively generate plain text similar to the data used for training. This task can be used for pre-training or fine-tuning a LLama3 model simply by calling fit(). This model has a generate() method, which generates text based on a prompt. The generation strategy used is controlled by an additional sampler argument on compile(). You can recompile the model with different keras_nlp.samplers objects to control the generation. By default, "greedy" sampling will be used.

<a name="fine-tuning"></a>
## 🔄 Fine Tuning 
# Fine-tuning with LoRA

To get better responses from the model, we will fine-tune the model with Low Rank Adaptation (LoRA).

**What exactly is LoRA?**

LoRA is a method used to fine-tune large language models (LLMs) in an efficient way. It involves freezing the weights of the LLM and injecting trainable rank-decomposition matrices.

Imagine in an LLM, we have a pre-trained dense layer, represented by a $d \times d$ weight matrix, denoted as $W_0$. We then initialize two additional dense layers, labeled as $A$ and $B$, with shapes $d \times r$ and $r \times d$, respectively. Here, $r$ denotes the rank, which is typically **much smaller than** $d$. Prior to LoRA, the model's output was computed using the equation $output = W_0 \cdot x + b_0$, where $x$ represents the input and $b_0$ denotes the bias term associated with the original dense layer, which remains frozen. After applying LoRA, the equation becomes $output = (W_0 \cdot x + b_0) + (B \cdot A \cdot x)$, where $A$ and $B$ denote the trainable rank-decomposition matrices that have been introduced.

<center><img src="https://i.ibb.co/DWsbhLg/LoRA.png" width="300"><br/>
Credit: <a href="https://arxiv.org/abs/2106.09685">LoRA: Low-Rank Adaptation of Large Language Models</a> Paper</center>


In the LoRA paper, $A$ is initialized with $\mathcal{N} (0, \sigma^2)$ and $B$ with $0$, where $\mathcal{N}$ denotes the normal distribution, and $\sigma^2$ is the variance.

**Why does LoRA save memory?**

Even though we're adding more layers to the model with LoRA, it actually helps save memory. This is because the smaller layers (A and B) have fewer parameters to learn compared to the big model and fewer trainable parameters mean fewer optimizer variables to store. So, even though the overall model might seem bigger, it's actually more efficient in terms of memory usage. 
This imlpemntation  uses a LoRA rank of `4`. A higher rank means more detailed changes are possible, but also means more trainable parameters.

