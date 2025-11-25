# make_sft_jsonl.py
import pandas as pd
import json
import os
import random

# 你可以改成命令行参数，这里用常量最简单
CSV_PATH = "overture_cleaned_places.csv"
OUT_DIR  = "sft_data"
SEED     = 42

random.seed(SEED)

def safe_str(x):
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()

def join_non_empty(parts, sep=" | "):
    """只拼接非空字段，避免出现一堆 | | |"""
    parts = [p for p in parts if p]
    return sep.join(parts)

def row_to_side(name, category, address, phone, website):
    """把单侧信息组合成紧凑字符串: name | category | address | phone | website"""
    return join_non_empty([
        safe_str(name),
        safe_str(category),
        safe_str(address),
        safe_str(phone),
        safe_str(website)
    ])

def to_record(A_text, B_text, y_int):
    """构造指令样本：只要求回答 YES/NO"""
    instr = (
        "Decide whether the two place records refer to the same real-world place. "
        "Answer ONLY 'YES' or 'NO'."
    )
    return {
        "instruction": instr,
        "input": f"Record A: {A_text}\nRecord B: {B_text}",
        "output": "YES" if int(y_int) == 1 else "NO"
    }

def main():
    # 读取 CSV
    df = pd.read_csv(CSV_PATH)

    # 兼容标签列名：优先 label_gt，其次 label；没有就报错更明确
    if "label_gt" in df.columns:
        labels = df["label_gt"]
    elif "label" in df.columns:
        labels = df["label"]
    else:
        raise ValueError("CSV 中找不到标签列：需要 'label_gt' 或 'label'。")

    # 检查必须字段是否存在（候选+基准）
    required_cols = [
        "name","category","address","phone","website",
        "base_name","base_category","base_address","base_phone","base_website"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}")

    data = []
    for _, r in df.iterrows():
        # 组装 Candidate / Base 侧文本（自动跳过空字段）
        A = row_to_side(r.get("name",""), r.get("category",""), r.get("address",""),
                        r.get("phone",""), r.get("website",""))
        B = row_to_side(r.get("base_name",""), r.get("base_category",""), r.get("base_address",""),
                        r.get("base_phone",""), r.get("base_website",""))

        # 如果两边都太空，跳过
        if not A and not B:
            continue

        y = int(r.get("label_gt", r.get("label", 0)))
        data.append(to_record(A, B, y))
        # 顺序交换增强（A↔B）
        data.append(to_record(B, A, y))

    # 打乱并切分
    random.shuffle(data)
    n = len(data)
    n_train = int(0.7 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "train.jsonl"), "w", encoding="utf-8") as f:
        for rec in data[:n_train]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "val.jsonl"), "w", encoding="utf-8") as f:
        for rec in data[n_train:n_train+n_val]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "test.jsonl"), "w", encoding="utf-8") as f:
        for rec in data[n_train+n_val:]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ Done. Samples: {n}  → train/val/test = {n_train}/{n_val}/{n_test}")
    # 可选：打印 2 条示例
    print("Example:")
    for e in data[:2]:
        print(json.dumps(e, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

