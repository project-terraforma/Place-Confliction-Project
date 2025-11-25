import json, re
from collections import Counter
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

BASE = "Qwen/Qwen2-7B-Instruct"
ADAPTER = "qwen2-7b-qlora-overture"

# ---- 建议的生成/量化设置 ----
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True, trust_remote_code=True)
tok.pad_token = tok.eos_token
tok.padding_side = "left"  # 生成任务更稳

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, ADAPTER)

# —— 确认 LoRA 适配器确实加载了 —— 
print("Active adapter:", getattr(model, "active_adapter", None))

# 进入评估模式，关闭梯度，防止额外显存占用
model.eval()
torch.set_grad_enabled(False)

def predict_yesno(prompt, max_new_tokens=2):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0
        )
    text = tok.decode(out[0], skip_special_tokens=True)

    # 只取 "Answer:" 后第一个词
    post = text.split("Answer:")[-1].strip()
    first = post.split()[0].strip(",.?!:;").upper() if post else ""

    mapping_yes = {"YES", "1", "TRUE"}
    mapping_no  = {"NO", "0", "FALSE"}

    if first in mapping_yes:
        return "YES", text
    if first in mapping_no:
        return "NO", text

    # 兜底：全句再搜一遍
    m = re.search(r"\b(YES|NO|1|0|TRUE|FALSE)\b", text.upper())
    if m:
        tok_ = m.group(1)
        return ("YES" if tok_ in mapping_yes else "NO"), text

    return "NO", text

# ---- 读取测试集 ----
ds = load_dataset("json", data_files="sft_data/test.jsonl")["train"]
y_true, y_pred = [], []

PRINT_N = 10         # 想看多少条就改这个
printed = 0

for idx, ex in enumerate(ds):
    prompt = f"{ex['instruction']}\n{ex['input']}\nAnswer:"
    pred_label, raw = predict_yesno(prompt)
    y_pred.append(1 if pred_label == "YES" else 0)
    gold_label = 1 if ex["output"].strip().upper() == "YES" else 0
    y_true.append(gold_label)

    # 只看前 10 条（你也可以改成只打印错例）
    if printed < PRINT_N:
        print(f"\n===== SAMPLE #{idx} =====")
        print("PROMPT:\n", prompt)
        print("RAW GENERATION:\n", raw)
        print("PARSED:", pred_label, "   GOLD:", "YES" if gold_label==1 else "NO")
        printed += 1

# ---- 指标与诊断输出 ----
acc = accuracy_score(y_true, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

print("Label balance (gold):", Counter(y_true))
print("Pred distribution:", Counter(y_pred))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=4))
print(f"Acc={acc:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")
