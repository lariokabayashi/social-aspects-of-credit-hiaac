from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    load_in_8bit=False,
    device_map="auto",
)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

message = "Write a poem"

sequences = pipeline(
    message,
    max_length=10000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    truncation=True,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(seq["generated_text"])
