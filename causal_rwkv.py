"""Train a model for causal LM on the ELI5 dataset with RWKV

Causal Language Modeling: https://huggingface.co/docs/transformers/tasks/language_modeling
RWKV HuggingFace reference: https://huggingface.co/docs/transformers/v4.30.0/en/model_doc/rwkv#transformers.RwkvModel
"""

from transformers import (
    RwkvForCausalLM,
    RwkvConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from datasets import load_dataset
import torch
import numpy as np

import collections
from typing import Any, Dict
import math
import re


def remove_url_from_text(text: str):
    """Remove square brackets around linked text and (_URL_X_) after"""
    return re.sub(r"\[|\]|\(_URL_\d+_\)", "", text)


def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Concatenate and tokenize the answers in flattened ELI5 data"""
    concatenated = [remove_url_from_text(" ".join(x)) for x in examples["answers.text"]]
    return tokenizer(concatenated)


def chunk(examples: Dict[str, Any], chunk_size: int = 256) -> Dict[str, Any]:
    """Concatenate and chunk batches of data"""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    return {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated.items()
    }


def set_labels(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Add a labels column to the dataset which is a copy of input_ids"""
    examples["labels"] = examples["input_ids"].copy()
    return examples


if __name__ == "__main__":
    MODEL_NAME = "./rwkv-430M-pile-eli5"
    TOKENIZER_NAME = "sgugger/rwkv-430M-pile"
    DATASET = "eli5"
    CHUNK_SIZE = 128
    TRAIN_SIZE = 50000
    TEST_SIZE = 10000
    BATCH_SIZE = 8
    MODEL_OUTPUT_DIR="./rwkv-430M-pile-eli5"

    model = RwkvForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(DATASET, split="train_asks[:]")
    dataset = dataset.train_test_split(train_size=TRAIN_SIZE, test_size=TEST_SIZE, shuffle=True, seed=42)
    dataset = dataset.flatten()
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Encode
    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True, 
        num_proc=4,
        remove_columns=dataset["train"].column_names
    )

    # Chunk
    chunked_dataset = encoded_dataset.map(
        chunk,
        fn_kwargs={"chunk_size": CHUNK_SIZE},
        batched=True, 
    )

    # Label
    lm_dataset = chunked_dataset.map(
        set_labels,
        batched=True
    )

    training_args = TrainingArguments(
        output_dir = MODEL_OUTPUT_DIR,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        logging_steps=len(lm_dataset["train"]) // BATCH_SIZE,
        fp16=True,
        gradient_accumulation_steps=4,
        optim="adafactor",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
    )
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    # Evaluate before train
    eval_0 = trainer.evaluate()
    perplexity_0 = math.exp(eval_0["eval_loss"])

    # Train
    trainer.train()

    # Evaluate after train
    eval_f = trainer.evaluate()
    perplexity_f = math.exp(eval_f["eval_loss"])

    trainer.save_model("rwkv-430M-pile-eli5-20000-answers")

    with open("perplexity_results.txt", "w") as f:
        f.write(f"Inital perplexity: {perplexity_0}\nFinal perplexity: {perplexity_f}")
