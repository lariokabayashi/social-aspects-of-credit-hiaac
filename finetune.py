from datasets import load_dataset
from peft import PeftConfig, PeftModel
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import LoraConfig
from peft import get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import prepare_model_for_kbit_training

# dataset_name = "timdettmers/openassistant-guanaco"
# dataset = load_dataset(dataset_name, split="train")

# model_name = "ybelkada/falcon-7b-sharded-bf16"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant = True
)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=bnb_config,
#     trust_remote_code=True,
#     device_map = "cuda:0"
# )

# model.config.use_cache = False
# model.gradient_checkpointing_enable()

# model = prepare_model_for_kbit_training(model)

# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

# lora_alpha = 16
# lora_dropout = 0.1
# lora_r = 16

# peft_config = LoraConfig(
#     lora_alpha=lora_alpha,
#     lora_dropout=lora_dropout,
#     r=lora_r,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=[
#         "query_key_value"
#     ]
# )

# model = get_peft_model(model,peft_config)

# training_arguments = TrainingArguments(
#     output_dir="./training_results",
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     num_train_epochs=1,
#     learning_rate=2e-4,
#     fp16=True,
#     optim="paged_adamw_32bit",
#     lr_scheduler_type="cosine",
#     warmup_ratio = 0.05
# )

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_config,
#     dataset_text_field="text",
#     max_seq_length=1024,
#     tokenizer=tokenizer,
#     args=training_arguments,
# )

# trainer.train()

trained_model_dir = "./trained_model"
# model.save_pretrained(trained_model_dir)

config = PeftConfig.from_pretrained(trained_model_dir)

trained_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict = True,
    quantization_config = bnb_config,
    trust_remote_code = True,
    device_map = "cuda:0",
)

trained_model = PeftModel.from_pretrained(trained_model, trained_model_dir)

trained_model_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code = True)
trained_model_tokenizer.pad_token = trained_model_tokenizer.eos_token

generation_config = trained_model.generation_config
generation_config.temperature = 0.7
generation_config.num_return_sequences = 1
generation_config.max_new_tokens = 1024
generation_config.top_p = 0.7
generation_config.pad_token_id = trained_model_tokenizer.pad_token_id
generation_config.eos_token_id = trained_model_tokenizer.eos_token_id

message = input("Question:")
encodings = trained_model_tokenizer(message, return_tensors='pt').to("cuda:0")

with torch.inference_mode():
    outputs = trained_model.generate(
        input_ids = encodings.input_ids,
        attention_mask = encodings.attention_mask,
        generation_config=generation_config,
        max_new_tokens = 1024
    )
outputs = trained_model_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(outputs)
