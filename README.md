# BPEKit 🐍🦀 

A command-line tool and Python library for training and using efficient Byte Pair Encoding (BPE) tokenizers. It is built using Rust bindings and supports parallelization across multiple devices using [OpenMPI](https://www.open-mpi.org/).

## 📦 Installation
### 🎡 Install from wheel 
No Rust installation necessary.
```bash
pip install https://github.com/jamie-stephenson/bpekit/releases/download/v0.1.0-test/bpekit-0.1.0-cp310-abi3-linux_x86_64.whl
```
### 🛠️ Build from source 
Requires Rust installation and necessary build tools (build-essential, libclang).
```bash
pip install git+https://github.com/jamie-stephenson/bpekit.git
```

## 👩‍💻 Tokenization via the command line
### 1. `train` a new tokenizer on a dataset.
```bash
bpekit train PATH VOCAB_SIZE [OPTIONS] 
```
- `PATH`: The path to your dataset.
- `VOCAB_SIZE`: The vocabulary size of the tokenizer. The tokenizer will begin its training from the UTF-8 byte level meaning that we start with a vocab size of 256. Therefore `VOCAB_SIZE` must be greater than 256.
### 2. Use an existing tokenizer to `encode` a dataset.
```bash
bpekit encode PATH MERGES_PATH [OPTIONS] 
```
- `MERGES_PATH`: Path to tokenizer merges (the .pkl file saved from `train`).
## 📚 Tokenization via the Python library
As a Python library, `bpekit` matches the CLI functionality while also providing the `Tokenizer` class for more flexible usage.
The simple example below demonstrates training a tokenizer that can be reused wherever you need it.
```python
from bpekit import train_tokenizer, encode_dataset, Tokenizer

path = "/path/to/dataset/"
merges_path = "/path/to/merges.pkl" # We save the merges to this path
tokens_path = "/path/to/tokens/" # We save the tokens to this path
vocab_size = 2048

# Directly train a tokenizer:
tokenizer: Tokenizer = train_tokenizer(path, vocab_size, merges_path)

# We can use our tokenizer to encode a whole dataset:
encode_dataset(path, merges_path, tokens_path) 

# Or we can directly use our tokenizer more flexibly:
tokens = tokenizer.encode("BPEKit 🐍🦀")

# We can even build a new copy of our tokenizer elsewhere:
tokenizer2 = Tokenizer.from_pickled_merges(merges_path)
tokenizer2.decode(tokens) # Output: "BPEKit 🐍🦀"
``` 

## 📂 Dataset Sources
### 📄 .txt File
If your dataset is a plain text file, each example should be on its own line.

Example usage:
```bash
bpekit train data/my_text_dataset.txt 2048
```

### 🤗 Hugging Face Dataset 
If you want to use a Hugging Face dataset, you can download it with:

```bash
bpekit download HF_PATH PATH [OPTIONS]
```
Or with the library:
```python
from bpekit import download_dataset

download_dataset(
   hf_path = "username/dataset",
   path = "/path/to/dataset/" 
)
```
- `HF_PATH`: Path to the dataset on the Hugging Face Hub (e.g. `"username/dataset"`). 
- `PATH`: Directory in which dataset will be saved. This path need not exist: any necessary non-existent directories will be created. 

You can then use your dataset like so:
```bash
bpekit train /path/to/dataset/ 2048
```
## 👯‍♂️ Parallelization
`train` and `encode` (and their corresponding Python functions) support parallelization using OpenMPI.

Example usage:
```bash
mpirun -n 8 bpekit train data/my_dataset/ 8132
```

This allows for efficient tokenization of large datasets using multiple devices.

