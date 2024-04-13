from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
while True:
    mes = input("message: ")
    sequences = pipeline(
        mes,
        max_length=10000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        truncation=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(seq["generated_text"])
