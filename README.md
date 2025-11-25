# Place-Confliction-Project

## ⭐ Overview
This project evaluates and compares multiple open-source Large Language Models (LLMs) for the task of place-record conflation using the Overture Maps 3K labeled place-pair dataset.
The goal is to determine which model—considering accuracy, AUC performance, and runtime efficiency—offers the best trade-off for real-world place deduplication.
This work is part of the CROWN 102 Overture Maps Foundation partnership, emphasizing open, reproducible, and cost-efficient evaluation rather than commercial API calls.

## 🧹 Dataset Pipeline
### Step 1 — Load and inspect data
- ReadData.py loads the Overture Maps dataset from parquet format and handles:
- JSON field normalization
- Address parsing
- Category extraction
- Positive/negative label creation

### Step 2 — Clean to CSV
- CreatCSV.py extracts:
- name
- category
- freeform address
- phone
- website for both candidate and base records.

### Step 3 — Generate SFT JSONL
- make_sft_jsonl.py produces:
- train.jsonl
- val.jsonl
- test.jsonl
    with reversible A/B pair augmentation.

## 🤖 Models Evaluated (Current Progress)
### ✅ 1. Qwen-2 7B Instruct (Fine-Tuned)
- Fine-tuning completed early.
- Strong classification:
  - Accuracy ≈ 0.918
  - F1 ≈ 0.9316
- Pending: Add AUC-ROC, PR-AUC, and latency.

## ⚠️ 2. Qwen-3 1.7B Zero-Shot
- Performed poorly on class 0:
  - Acc ≈ 0.60
  - F1 ≈ 0.388
  - AUC ≈ 0.582
- Conclusion: Model too small; no instruct version → Fine Tuning removed from refined OKRs.

## ✅ 3. Qwen-3 4B Instruct (Zero-Shot)
- Most complete results so far:
  - Accuracy ≈ 0.728
  - F1 ≈ 0.783
  - AUC-ROC ≈ 0.7802
  - PR-AUC ≈ 0.8286
- Includes ROC and PR curves.

## ⏳ 4. Qwen-3 4B Instruct (Fine-Tuned)
- Next model to train with QLoRA.
- Expected to outperform 4B zero-shot while being more efficient than 7B FT.

## 📊 Evaluation Metrics
Each model is evaluated using:
```
| Metric                 | Description               |
|:-----------------------|:--------------------------|
| Accuracy               | Overall correctness       |
|Precision / Recall / F1 |Balanced binary evaluation |
|AUC-ROC                 | Ranking quality in imbalanced data |
|PR-AUC                  | Precision–recall performance |
|Latency                 | Inference speed per 1,000 predictions (to be added)  |

```
Generated visualizations:
- ROC Curve
- PR Curve
- Confusion Matrix

## 📈 Current Results Snapshot
```
| Model            | Fine-Tuned? | Accuracy | Recall   | F1-score | AUC-ROC | PR-AUC | Notes |
|------------------|-------------|----------|----------|----------|---------|--------|----------|
| Qwen-2 7B        | Yes         | 0.9175   | 0.9361   | 0.9316   | TBA     | TBA    | Strongest so far |
| Qwen-3 4B        | Zero-Shot   | 0.7200   | 0.8458   | 0.7838   | 0.7892  | 0.8268 | Most complete metrics |
| Qwen-3 1.7B      | Zero-Shot   | 0.6008   | 0.9917   | 0.7488   | 0.5820  | 0.6528 | Weak baseline |
```

## 🔧 Next Steps
### Short-Term
- Fine-tune Qwen3-4B-Instruct
- Add AUC-ROC, PR-AUC, and latency to:
  - Qwen2-7B FT
  - Qwen3-1.7B zero-shot

### Medium-Term
- Produce combined comparison tables
- Finalize ROC/PR curves for all models
- Perform ≥10-sample error analysis
- Prepare slide deck + final report
