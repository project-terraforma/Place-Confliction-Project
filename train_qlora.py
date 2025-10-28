import os, json
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"

# 4-bit 量化配置（QLoRA 关键）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# LoRA 配置（按显存调整 r/alpha）
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

# 加载 JSONL 数据
train_ds = load_dataset("json", data_files="sft_data/train.jsonl")["train"]
val_ds   = load_dataset("json", data_files="sft_data/val.jsonl")["train"]

def format_example(ex):
    # 把 instruction + input + output 拼成单轮对话
    # 输出只训练助手答案（默认 SFTTrainer 会识别）
    prompt = f"{ex['instruction']}\n{ex['input']}\nAnswer:"
    return {"text": prompt + " " + ex["output"]}

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(format_example, remove_columns=val_ds.column_names)

def formatting_func(example):
    # 兼容单条/批处理两种输入结构
    if isinstance(example["text"], list):
        return example["text"]                # 批：直接返回一批字符串
    return [example["text"]]                  # 单条：包装成列表


# 训练超参（按你的 3K 对小数据设置）
sft_args = SFTConfig(
    output_dir="qwen2-7b-qlora-overture",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=1e-4,
    logging_steps=20,
    eval_strategy="steps",       # ✅ 注意：这个版本叫 eval_strategy
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    fp16=True,                   # Colab T4 建议用 fp16
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    gradient_checkpointing=True,
    report_to="none",
    dataset_text_field="text",   # ✅ 放在 SFTConfig，而不是 SFTTrainer
    max_length=1024,             # ✅ 不是 max_seq_length
    packing=False,               # 不把多条样本拼接
)

trainer = SFTTrainer(
    model=model,
    args=sft_args,               # 传 SFTConfig
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,  # ✅ 等价于以前的 tokenizer=...
    peft_config=peft_config,
)


trainer.train()
trainer.save_model()     # 保存 LoRA 适配器
tokenizer.save_pretrained("qwen2-7b-qlora-overture")
print("Done.")
