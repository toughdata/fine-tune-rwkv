# Train RWKV for causal language modeling
This script will fine-tune a RWKV model with the HuggingFace transformers library.

## Why?
RWKV is a powerful model, but the tools used to train it are overcomplicated and it is not well-documented.
This script is a simple and *legible* implementation of training RWKV using HuggingFace's transformers library.

## Usage
- Install dependencies: `pip install datasets transformers[torch] accelerate`
- download the script
- edit hyperparameters as necessary in the `if __name__ == "__main__":` block
- run the script
  
## RWKV Original Repo
https://github.com/BlinkDL/RWKV-LM

## HuggingFace reference
Causal Language modeling: https://huggingface.co/docs/transformers/tasks/language_mod
eling
RWKV HuggingFace documentation: https://huggingface.co/docs/transformers/v4.30.0/en/model
_doc/rwkv#transformers.RwkvModel
