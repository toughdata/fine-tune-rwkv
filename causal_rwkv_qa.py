"""Train a model for causal LM on the ELI5 dataset with RWKV

This time we will be feeding it data in the format:
text = "Question: {question}\nDetails: {question body text}:\nAnswer: "

We will not be using chunking here - We will feed the examples one by one.

Causal Language Modeling: https://huggingface.co/docs/transformers/tasks/language_modeling
Causal language Modeling (another): https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
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


def tokenize_function(examples: Dict[str, Any], max_length: int = 256) -> Dict[str, Any]:
    """Create a prompt from questions and answers and tokenize it"""
    questions = [question for question in examples['title']]
    selftexts = [selftext for selftext in examples['selftext']]
    answers = [answer for answer in examples['answers.text']]

    prompts = []
    for question, selftext, answer in list(zip(questions, selftexts, answers)):
        prompts.append(
            remove_url_from_text(
                f"Question: {question}"
                f"Question context: {selftext}"
                f"Answer: {answer}"
            )
        )

    return tokenizer(prompts, padding="max_length", truncation=True, max_length=max_length)


def set_labels(examples: Dict[str, Any]) -> Dict[str, Any]:
    """Add a labels column to the dataset which is a copy of input_ids"""
    examples["labels"] = examples["input_ids"].copy()
    return examples


if __name__ == "__main__":
    MODEL_NAME = "RWKV/rwkv-4-169m-pile"
    TOKENIZER_NAME = "RWKV/rwkv-4-169m-pile"
    DATASET = "eli5"
    TRAIN_SIZE = 100
    TEST_SIZE = 10
    BATCH_SIZE = 2
    MODEL_OUTPUT_DIR="./rwkv-169M-pile-eli5-qa"
    MAX_INPUT_LENGTH = 256

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
        remove_columns=dataset["train"].column_names,
        fn_kwargs={"max_length": MAX_INPUT_LENGTH}
    )

    # Label
    lm_dataset = encoded_dataset.map(
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
