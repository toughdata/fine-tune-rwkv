# -*- coding: utf-8 -*-
"""Training RWKV for Q/A on ELI5 dataset using Seq2SeqTrainer documented here: https://huggingface.co/docs/evaluate/transformers_integrations#seq2seqtrainer"""

from transformers import (
    AutoTokenizer,
    RwkvForCausalLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from datasets import (
    load_dataset,
    Dataset
)

import evaluate
import torch
import numpy as np
import nltk
import re

nltk.download("punkt")


def remove_url_from_text(text: str):
    """Remove square brackets around linked text and (_URL_0_) after"""
    return re.sub(r"\[|\]|\(_URL_\d+_\)", "", text)


def _encode(examples: list, max_length: int):
    """Encode the examples. For use in tokenize_function()"""
    return tokenizer(
        examples,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )


def tokenize_function(
        examples,
        max_length: int = 256,
        question_prefix: str = "Question: ",
        answer_prefix: str = "Answer: ",
    ):
    """Tokenize a batch of question/answer pairs.
    Questions are the post title and contents.
    We are extracting the first answer for each question"""
    titles = examples["title"]
    selftexts = examples["selftext"]
    questions = [f"{question_prefix}{t}\n{s}" for t, s in list(zip(titles, selftexts))]
    questions = [remove_url_from_text(question) for question in questions]

    first_answers = [answer_prefix + remove_url_from_text(x[0]) for x in examples["answers.text"]]
    encoding = _encode(questions, max_length)
    labels = _encode(first_answers, max_length)
    encoding["labels"] = labels.input_ids
    return encoding

_metric = evaluate.load("rouge")
def compute_metrics(eval_preds):
    """Compute eval metrics"""
    preds, labels = eval_preds
    labels = np.where(labels != 100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    result = _metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result

if __name__ == "__main__":
    MODEL_NAME = "toughdata/RWKV-ELI5-2500-examples"
    TOKENIZER_NAME = "RWKV/rwkv-4-430m-pile"
    DATASET = "eli5"
    SPLIT_START = 2500
    SPLIT_END = 5000
    BATCH_SIZE = 10
    NUM_EPOCHS = 2
    MODEL_OUTPUT_DIR = "./rwkv-430M-pile-eli5-seq2seq"
    MAX_LENGTH = 256
    MODEL_SAVENAME = "RWKV-ELI5-5000-examples"
    
    model = RwkvForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(DATASET, split=f"train_eli5[{SPLIT_START}:{SPLIT_END}]")
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    dataset = dataset.flatten()
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Encode the inputs
    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        fn_kwargs={
            "max_length": MAX_LENGTH
        }
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = MODEL_OUTPUT_DIR,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE // 2,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        predict_with_generate=True,
        optim="adafactor",
        gradient_accumulation_steps=4,
        generation_max_length=MAX_LENGTH
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    trainer.train()

    trainer.save_model(MODEL_SAVENAME)
