"""Inference with text generation model

https://huggingface.co/docs/transformers/generation_strategies
"""

from transformers import (
    AutoTokenizer,
    RwkvForCausalLM,
)

MODEL_NAME = "RWKV/rwkv-4-169m-pile"
TOKENIZER_NAME = "RWKV/rwkv-4-169m-pile"
PROMPT = ""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = RwkvForCausalLM.from_pretrained(MODEL_NAME)

encoded_question = tokenizer(PROMPT, return_tensors="pt").to("cuda")
encoded_answer = model.generate(**encoded_question, max_length=256, temperature=0.9, do_sample=True).squeeze()
answer = tokenizer.decode(encoded_answer, skip_special_tokens=True)
print(answer)
