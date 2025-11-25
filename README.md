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
Fine-tuned early on the 3K Overture training data.  
Now fully re-evaluated using the updated evaluation script with AUC and latency.

- **Classification performance**
  - Accuracy ≈ **0.9175**
  - Precision ≈ **0.9271**
  - Recall ≈ **0.9361**
  - F1-score ≈ **0.9316**
- **Ranking quality**
  - AUC-ROC ≈ **0.9763**
  - PR-AUC ≈ **0.9837**
- **Latency**
  - Total samples: **1200**
  - Total inference time: **350.15 s**
  - Avg latency: **0.2918 s / sample**
  - Time per 1000 samples: **291.79 s**
- Artifacts:
  - `qwen2_7b_ft_roc_curve.png`
  - `qwen2_7b_ft_pr_curve.png`

This model now has complete benchmark data and serves as the strongest baseline for comparison against Qwen3-4B FT.

---

### ⚠️ 2. Qwen-3 1.7B Zero-Shot
- Accuracy ≈ **0.6008**
- AUC-ROC ≈ **0.5820**
- PR-AUC ≈ **0.6528**
- Extremely poor on negative class → removed from FT OKRs.

---

### ✅ 3. Qwen-3 4B Instruct (Zero-Shot)
- Accuracy ≈ **0.7288**
- F1-score ≈ **0.7838**
- AUC-ROC ≈ **0.7892**
- PR-AUC ≈ **0.8286**

---

### ✅ 4. Qwen-3 4B Instruct (Fine-Tuned, QLoRA)
Strong overall model with best AUC/PR metrics.

- **Classification**
  - Accuracy ≈ **0.9158**
  - F1-score ≈ **0.9293**
- **Ranking quality**
  - AUC-ROC ≈ **0.9755**
  - PR-AUC ≈ **0.9833**
- **Latency**
  - Avg latency: **0.2146 s / sample**

---

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
```text
| Model            | Setting    | Accuracy | Recall   | F1-score | AUC-ROC | PR-AUC | Latency (s/sample) | Notes                                    |
|------------------|----------- |----------|----------|----------|---------|--------|---------------------|------------------------------------------|
| Qwen-2 7B        | Fine-tuned | 0.9175   | 0.9361   | 0.9316   | 0.9763  | 0.9837 | 0.2918              | Highest accuracy; very strong baseline   |
| Qwen-3 4B        | Fine-tuned | 0.9158   | 0.9222   | 0.9293   | 0.9755  | 0.9833 | 0.2146              | Best speed/performance trade-off         |
| Qwen-3 4B        | Zero-shot  | 0.7288   | 0.8458   | 0.7838   | 0.7892  | 0.8286 | N/A                 | Strongest zero-shot baseline             |
| Qwen-3 1.7B      | Zero-shot  | 0.6008   | 0.9917   | 0.7488   | 0.5820  | 0.6528 | N/A                 | Poor performance → not used for FT       |
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
