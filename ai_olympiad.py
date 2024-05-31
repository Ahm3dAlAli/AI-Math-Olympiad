{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.14","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"tpu1vmV38","dataSources":[{"sourceId":73231,"databundleVersionId":8365361,"sourceType":"competition"},{"sourceId":8530140,"sourceType":"datasetVersion","datasetId":5094400},{"sourceId":8530144,"sourceType":"datasetVersion","datasetId":5094403},{"sourceId":33547,"sourceType":"modelInstanceVersion","isSourceIdPinned":true,"modelInstanceId":28079}],"dockerImageVersionId":30698,"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [markdown]\n# # AI Mathematical Olympiad \n# \n# <div align=\"center\">\n#     <img src=\"https://i.ibb.co/9rx4pbX/AIMO.png\">\n# </div>\n\n# %% [markdown]\n# In this competition, we aim is to build AI models that can solve tough math problems, in other words, creating LLM models capable of solving Math Olympiad problems. This notebook will guide you through the process of fine-tuning the **LLAMMA** LLM model with LoRA to solve math problems using KerasNLP. With KerasNLP, fine-tuning with LoRA becomes straightforward with just a few lines of code.\n# \n# **Did you know:**: This notebook is backend-agnostic? Which means it supports TensorFlow, PyTorch, and JAX backends. However, the best performance can be achieved with `JAX`. KerasNLP and Keras enable the choice of preferred backend. Explore further details on [Keras](https://keras.io/keras_3/).\n# \n# **Note**: For a deeper understanding of KerasNLP, refer to the [KerasNLP guides](https://keras.io/keras_nlp/).\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":true}}\n!pip install peft\n!pip install transformers accelerate\n!pip install git+https://github.com/huggingface/peft.git\n!pip install -q /kaggle/input/kerasnlp/keras-3.3.3-py3-none-any.whl --no-deps\n\n# %% [markdown]\n# # Import Libraries and Set Configuration\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false},\"execution\":{\"iopub.status.busy\":\"2024-05-31T09:00:28.972381Z\",\"iopub.execute_input\":\"2024-05-31T09:00:28.973245Z\",\"iopub.status.idle\":\"2024-05-31T09:00:28.977123Z\",\"shell.execute_reply.started\":\"2024-05-31T09:00:28.973204Z\",\"shell.execute_reply\":\"2024-05-31T09:00:28.976449Z\"}}\n# Import necessary libraries\nimport os\nos.environ[\"KERAS_BACKEND\"] = \"jax\"  # you can also use tensorflow or torch\nos.environ[\"XLA_PYTHON_CLIENT_MEM_FRACTION\"] = \"0.9\"  # avoid memory fragmentation on JAX backend.\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false},\"execution\":{\"iopub.status.busy\":\"2024-05-31T09:00:51.578615Z\",\"iopub.execute_input\":\"2024-05-31T09:00:51.578967Z\"}}\nimport keras\nimport keras_nlp\nfrom transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline\nimport numpy as np\nimport pandas as pd\nfrom tqdm.notebook import tqdm\ntqdm.pandas()  # progress bar for pandas\nimport plotly.graph_objs as go\nimport plotly.express as px\nfrom IPython.display import display, Markdown\nimport torch\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false},\"execution\":{\"iopub.status.busy\":\"2024-05-31T09:00:45.530713Z\",\"iopub.execute_input\":\"2024-05-31T09:00:45.531521Z\",\"iopub.status.idle\":\"2024-05-31T09:00:45.763692Z\",\"shell.execute_reply.started\":\"2024-05-31T09:00:45.531482Z\",\"shell.execute_reply\":\"2024-05-31T09:00:45.762775Z\"}}\n# Configuration\nclass CFG:\n    seed = 42\n    dataset_path = \"/kaggle/input/ai-mathematical-olympiad-prize\"\n    model_path = \"/kaggle/input/llama-3/transformers/8b-hf/1\"  # path to pretrained LLama3 model\n    sequence_length = 512  # max size of input sequence for training\n    batch_size = 1  # size of the input batch in training\n    epochs = 1  # number of epochs to train\n\n# Set random seed for reproducibility\nkeras.utils.set_random_seed(CFG.seed)\n\n# %% [markdown]\n# # Load Data \n# \n# No training data is provided in this competition; in other words, we can use any openly available datasets for this competition. In this notebook, we will use a modified **Math** dataset which I have compiled to have a `Question-Solution-Answer` format.\n# \n# **Data Format:**\n# \n# These datasets include:\n# - `problem`: The math problem in LaTeX format.\n# - `solution`: Step-by-step solution to this problem.\n# - `answer`: Final answer of the solution which will be the ground truth for this competition.\n# - `level`: Difficulty of the problem.\n# - `type`: The category of the problem.\n# \n# > This dataset comes with its own train test split. However, we will merge them both and use them for fine-tuning. You are welcome to use them for trainining and validation separately. Also to reduce the training time we will only be training on the first`1000` samples. You are welcome to train on the full data.\n# \n# # Filter Data\n# \n# The Math dataset contains various problems, but not all of them are suitable for this competition. More specifically, this competition requires a `non-negative integer` answer, while the Math dataset includes problems with different types of answers such as integers, floats, fractions, matrices, etc. In this notebook, we will only use those problems whose answers are non-negative integers and filter out the rest.\n# \n# \n# # Prompt Engineering\n# \n# We will be using below simple prompt template we'll use to create problem-solution-answer trio to feed the model. This template will help the model to follow instruction and respond accurately. You can explore more advanced prompt templates for better results. \n# \n# ```\n# Role:\n# You are an advanced AI system with exceptional mathematical reasoning and problem-solving capabilities, specifically designed to solve tricky math problems (whose answer is a non-negative integer) written in LaTeX format from the AI Mathematical Olympiad (AIMO) competition. Your task is to accurately analyze and solve intricate mathematical problems, demonstrating a deep understanding of mathematical concepts and a strong ability to apply logical reasoning strategies.\n# \n# Instruction:\n# 1. Carefully read and comprehend the problem statement provided in the \"Problem\" section.\n# 2. In the \"Solution\" section, provide a solution of the problem with detailed explanation of your logical reasoning process. Keep in mind that answer must be a non-negative integer number.\n# 3. At the end, create a \"Answer\" section where you will state only the final numerical or algebraic answer, without any additional text or narrative.\n# \n# Problem:\n# ...\n# \n# Solution:\n# ...\n# \n# Answer:\n# ...\n# ```\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false},\"execution\":{\"iopub.status.busy\":\"2024-05-31T08:17:58.840842Z\",\"iopub.execute_input\":\"2024-05-31T08:17:58.841179Z\",\"iopub.status.idle\":\"2024-05-31T08:17:59.095881Z\",\"shell.execute_reply.started\":\"2024-05-31T08:17:58.841148Z\",\"shell.execute_reply\":\"2024-05-31T08:17:59.094918Z\"}}\n# Load the Math dataset\ndf1 = pd.read_csv(\"/kaggle/input/mathqsa/train.csv\")\ndf2 = pd.read_csv(\"/kaggle/input/mathqsa/test.csv\")\ndf = pd.concat([df1, df2], axis=0)\ndf = df[:1000]  # take first 1000 samples\ndf.head(2)\n\n# Filter Data\ndef is_integer(text):\n    try:\n        if int(text) >= 0:\n            return True\n        else:\n            return False\n    except ValueError:\n        return False\n    \ndf[\"is_integer\"] = df.answer.map(is_integer)\ndf = df[df.is_integer].reset_index(drop=True)\ndf.head(2)\n\n# Prompt Engineering\ntemplate = \"\"\"Role:\\nYou are an advanced AI system with exceptional mathematical reasoning and problem-solving capabilities, specifically designed to solve tricky math problems (whose answer is a non-negative integer) written in LaTeX format from the AI Mathematical Olympiad (AIMO) competition. Your task is to accurately analyze and solve intricate mathematical problems, demonstrating a deep understanding of mathematical concepts and a strong ability to apply logical reasoning strategies.\\n\\nInstruction:\n1. Carefully read and comprehend the problem statement provided in the \"Problem\" section.\n2. In the \"Solution\" section, provide a solution of the problem with detailed explanation of your logical reasoning process. Keep in mind that answer must be a non-negative integer number.\n3. At the end, create a \"Answer\" section where you will state only the final numerical or algebraic answer, without any additional text or narrative.\\n\\nProblem:\\n{problem}\\n\\nSolution:\\n{solution}\"\"\"\n\ndf[\"prompt\"] = df.progress_apply(lambda row: template.format(problem=row.problem,\n                                                             solution=f\"{row.solution}\\n\\nAnswer:\\n{row.answer}\"),\n                                                             axis=1)\ndata = df.prompt.tolist()\n\n# %% [markdown]\n# Let's examine a sample prompt. As the answers in our dataset are curated with **markdown** format, we will render the sample using `Markdown()` to properly visualize the formatting.\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false}}\ndef colorize_text(text):\n    for word, color in zip([\"Role\", \"Instruction\", \"Problem\", \"Solution\", \"Answer\"],\n                           [\"blue\", \"yellow\", \"red\", \"cyan\", \"green\"]):\n        text = text.replace(f\"{word}:\", f\"\\n\\n**<font color='{color}'>{word}:</font>**\")\n    return text\n\n# Take a random sample\nsample = data[12]\n\n# Give colors to Instruction, Response and Category\nsample = colorize_text(sample)\n\n# Show sample in markdown\ndisplay(Markdown(sample))\n\n# %% [markdown]\n# # Modelling\n\n# %% [markdown]\n# <div align=\"center\"><img src=\"https://chartstatic.com/images/symbol_logos/meta.png\" width=\"300\"></div>\n\n# %% [markdown]\n# **LLama3** is a collection of advanced open **LLMs** developed by **Meta AI**. They are built to provide state-of-the-art performance in a wide range of applications, from text generation to complex problem solving. LLama3 models are designed to be versatile, offering capabilities that can be fine-tuned for specific tasks, making them suitable for both research and production environments.\n\n# %% [markdown]\n# LLama3 models are available in several sizes, allowing users to choose models based on their computational resources and the requirements of their specific applications.\n\n# %% [markdown]\n# | Parameters size | Tuned versions    | Intended platforms                 | Preset                 |\n# |-----------------|-------------------|------------------------------------|------------------------|\n# | 7B              | Pretrained        | Desktop computers and small servers        | `llama3-7b`          |\n# | 13B              | Pretrained | Desktop computers and small servers         | `llama3-13b` |\n# | 30B              | Pretrained        | Servers and high-end workstations   | `llama3-30b` |\n# | 70B              | Pretrained        | Servers and high-end workstations| `llama3-70b`          |\n\n# %% [markdown]\n# In this notebook, we will utilize the `llamma3-70b` model from Meta AI to solve the math olympiad questions. This model is chosen for its superior performance in understanding and generating complex text. By fine-tuning the LLama3 70b model, we aim to enhance its problem-solving abilities specifically for mathematical problems.\n# \n# To explore other available models, you can adjust the model_path value in the CFG (config). You can find more details about LLama3 models on the [Meta AI website](https://llama.meta.com/llama3/).\n\n# %% [markdown]\n# # LLAMA3 Casual LM \n# \n# The code below will build an end-to-end LLama3 model for causal language modeling. A causal language model (LM) predicts the next token based on previous tokens. This task setup can be used to train the model unsupervised on plain text input or to autoregressively generate plain text similar to the data used for training. This task can be used for pre-training or fine-tuning a LLama3 model simply by calling fit().\n# \n# This model has a generate() method, which generates text based on a prompt. The generation strategy used is controlled by an additional sampler argument on compile(). You can recompile the model with different keras_nlp.samplers objects to control the generation. By default, \"greedy\" sampling will be used.\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false}}\n# Load the tokenizer and model\ntokenizer = AutoTokenizer.from_pretrained(CFG.model_path)\nmodel = AutoModelForCausalLM.from_pretrained(CFG.model_path, torch_dtype=torch.float16)\n\n# Add a padding token if it doesn't exist\nif tokenizer.pad_token is None:\n    tokenizer.add_special_tokens({'pad_token': '[PAD]'})\n    model.resize_token_embeddings(len(tokenizer))\n\n# Move model to GPU\ndevice = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\nmodel = model.to(device)\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false}}\n# Tokenize the dataset\ndef tokenize_function(examples):\n    return tokenizer(examples, padding=\"max_length\", truncation=True, max_length=CFG.sequence_length)\n\ntokenized_data = tokenizer(data, padding=True, truncation=True, max_length=CFG.sequence_length, return_tensors=\"pt\")\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false}}\n# Define a function to colorize text\ndef colorize_text(text):\n    for word, color in zip([\"Role\", \"Instruction\", \"Problem\", \"Solution\", \"Answer\"],\n                           [\"blue\", \"yellow\", \"red\", \"cyan\", \"green\"]):\n        text = text.replace(f\"{word}:\", f\"\\n\\n**<font color='{color}'>{word}:</font>**\")\n    return text\n\n# Inference example\ndef infer_sample(index):\n    row = df.iloc[index]\n    prompt = template.format(problem=row.problem, solution=\"\")\n    inputs = tokenizer(prompt, return_tensors=\"pt\").to(device)  # Move inputs to GPU\n    outputs = model.generate(**inputs, max_length=1024)\n    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)\n    output_text = colorize_text(output_text)\n    display(Markdown(output_text))\n\n# Sample 1\ninfer_sample(12)\n\n# Sample 2\ninfer_sample(32)\n\n# %% [markdown]\n# # Fine-tuning with LoRA\n# \n# To get better responses from the model, we will fine-tune the model with Low Rank Adaptation (LoRA).\n# \n# **What exactly is LoRA?**\n# \n# LoRA is a method used to fine-tune large language models (LLMs) in an efficient way. It involves freezing the weights of the LLM and injecting trainable rank-decomposition matrices.\n# \n# Imagine in an LLM, we have a pre-trained dense layer, represented by a $d \\times d$ weight matrix, denoted as $W_0$. We then initialize two additional dense layers, labeled as $A$ and $B$, with shapes $d \\times r$ and $r \\times d$, respectively. Here, $r$ denotes the rank, which is typically **much smaller than** $d$. Prior to LoRA, the model's output was computed using the equation $output = W_0 \\cdot x + b_0$, where $x$ represents the input and $b_0$ denotes the bias term associated with the original dense layer, which remains frozen. After applying LoRA, the equation becomes $output = (W_0 \\cdot x + b_0) + (B \\cdot A \\cdot x)$, where $A$ and $B$ denote the trainable rank-decomposition matrices that have been introduced.\n# \n# <center><img src=\"https://i.ibb.co/DWsbhLg/LoRA.png\" width=\"300\"><br/>\n# Credit: <a href=\"https://arxiv.org/abs/2106.09685\">LoRA: Low-Rank Adaptation of Large Language Models</a> Paper</center>\n# \n# \n# In the LoRA paper, $A$ is initialized with $\\mathcal{N} (0, \\sigma^2)$ and $B$ with $0$, where $\\mathcal{N}$ denotes the normal distribution, and $\\sigma^2$ is the variance.\n# \n# **Why does LoRA save memory?**\n# \n# Even though we're adding more layers to the model with LoRA, it actually helps save memory. This is because the smaller layers (A and B) have fewer parameters to learn compared to the big model and fewer trainable parameters mean fewer optimizer variables to store. So, even though the overall model might seem bigger, it's actually more efficient in terms of memory usage. \n# \n# > This notebook uses a LoRA rank of `4`. A higher rank means more detailed changes are possible, but also means more trainable parameters.\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false}}\nfrom peft import get_peft_model, LoraConfig, TaskType\n\npeft_config = LoraConfig(\n    task_type=TaskType.CAUSAL_LM,\n    inference_mode=False,\n    r=4,\n    lora_alpha=32,\n    lora_dropout=0.1,\n)\n\nmodel = get_peft_model(model, peft_config)\n\n# Define training arguments\ntraining_args = TrainingArguments(\n    output_dir=\"./results\",\n    overwrite_output_dir=True,\n    num_train_epochs=CFG.epochs,\n    per_device_train_batch_size=CFG.batch_size,\n    save_steps=10_000,\n    save_total_limit=2,\n    fp16=True)\n\n# Define a custom trainer\ntrainer = Trainer(\n    model=model,\n    args=training_args,\n    train_dataset=tokenized_data,  # Assuming you have a train_dataset\n    eval_dataset=None  # Assuming you have an eval_dataset\n)\n\n# Train the model\ntrainer.train()\n\n# %% [markdown]\n# # Inference after tuning\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false}}\n# Sample 1 after fine-tuning\ninfer_sample(12)\n\n# Sample 2 after fine-tuning\ninfer_sample(32)\n\n# %% [markdown]\n# # Inference on IOMA Data\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false}}\nimport re\n\n# Extract answer from model response\ndef get_answer(text):\n    try:\n        answer = re.search(r'Answer:\\s*([\\s\\S]+)', text).group(1).strip()\n        answer = answer.replace(\",\", \"\")\n        if is_integer(answer):\n            return int(answer) % 1000\n        else:\n            return 0\n    except:\n        return 0\n\ndef infer(df):\n    preds = []\n    for i in tqdm(range(len(df))):\n        row = df.iloc[i]\n        prompt = template.format(problem=row.problem, solution=\"\")\n        inputs = tokenizer(prompt, return_tensors=\"pt\").to(device)  # Move inputs to GPU\n        outputs = model.generate(**inputs)\n        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)\n        pred = get_answer(output_text)\n        preds.append([row.id, pred])\n        if \"answer\" in row:\n            preds[-1] += [row.answer]\n    return preds\n\naimo_df = pd.read_csv(f\"{CFG.dataset_path}/train.csv\")\ntrain_preds = infer(aimo_df)\ntrain_pred_df = pd.DataFrame(train_preds, columns=[\"id\", \"prediction\", \"answer\"])\ntrain_pred_df.head()\n\n# %% [markdown]\n# # Submission File\n\n# %% [code] {\"jupyter\":{\"outputs_hidden\":false}}\ntest_df = pd.read_csv(f\"{CFG.dataset_path}/test.csv\")\ntest_preds = infer(test_df)\n\nsub_df = pd.DataFrame(test_preds, columns=[\"id\", \"answer\"])\nsub_df.to_csv(\"submission.csv\", index=False, header=True)\nsub_df.head()","metadata":{"_uuid":"640ac17c-8b74-4e88-a6f1-48587b8c8a65","_cell_guid":"0b282f36-4c81-46d5-ba45-e09037afef95","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}