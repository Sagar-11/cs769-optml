# -------------------------------------------------------------
# DEDIER ⚖️ with decoder‑only LMs on the Civil Comments dataset
# Teacher : GPT‑2‑large      (774 M params)
# Student : DistilGPT‑2      (82 M  params)  + early‑readout
#
# Requirements (install once in a fresh env):
#   pip install "transformers>=4.39" "datasets>=2.19" evaluate torch accelerate sentencepiece
# -------------------------------------------------------------

# 0. Imports
import torch, evaluate, numpy as np
import pandas as pd
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModelForCausalLM,
)
import transformers    # needed for TrainerCallback

# ------------------------------------------------------------------
# Device setup: two‑GPU split (teacher on cuda:0, student on cuda:1)
# ------------------------------------------------------------------
if torch.cuda.device_count() >= 2:
    TEACHER_DEV = torch.device("cuda:0")
    STUDENT_DEV = torch.device("cuda:1")
elif torch.cuda.is_available():
    TEACHER_DEV = STUDENT_DEV = torch.device("cuda:0")
else:
    TEACHER_DEV = STUDENT_DEV = torch.device("cpu")

# ------------------------------------------------------------------
# DeDiER schedule hyper‑parameters
L = 1   # recompute sample‑weights every L epochs
R = 1   # fine‑tune the auxiliary head for R epochs
# ------------------------------------------------------------------

# 1. Tokeniser – GPT‑style quirks
print("Loading Civil Comments dataset and preparing tokenizer...")
tok = AutoTokenizer.from_pretrained("gpt2-large")
tok.pad_token = tok.eos_token          # GPT‑2 has no <pad>
tok.padding_side = "right"             # causal mask needs right‑padding

# 2. Civil Comments splits (≈2 M rows → train/val/test)
# Using the built‑in HF dataset that already provides separate splits.
splits = load_dataset("civil_comments")   # returns a DatasetDict with train/validation/test

def tokenize(ex):
    return tok(
        ex["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

splits_tok = splits.map(tokenize, batched=True).rename_column("toxicity", "labels")

# initialise sample weights (wt_i ← 1 for all i)
splits_tok["train"] = splits_tok["train"].add_column(
    "wt", [1.0] * len(splits_tok["train"])
)
print(f"Tokenization complete. Training samples: {len(splits_tok['train']):,}, "
      f"Validation samples: {len(splits_tok['validation']):,}, "
      f"Test samples: {len(splits_tok['test']):,}")

# 3. Teacher – GPT‑2 large, sequence‑classification head
teacher = GPT2ForSequenceClassification.from_pretrained(
    "gpt2-large",
    num_labels=2,
    output_hidden_states=True,
)
teacher.to(TEACHER_DEV)

args_teacher = TrainingArguments(
    output_dir="gpt2_teacher",
    per_device_train_batch_size=2,          # big model ⇒ small batch
    gradient_accumulation_steps=4,          # effective 8
    fp16=True,
    learning_rate=5e-6,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

trainer_teacher = Trainer(
    model=teacher,
    args=args_teacher,
    train_dataset=splits_tok["train"],
    eval_dataset=splits_tok["validation"],
)
print(f"Starting teacher fine‑tuning on device {TEACHER_DEV}...")
trainer_teacher.train()
teacher.eval()
print("Teacher fine‑tuning finished and model set to eval mode.")
teacher.save_pretrained("gpt2_teacher_ckpt")

# 4. DEDIER student class for decoder‑only backbones
class DEDIERCausalStudent(nn.Module):
    """
    Decoder‑only backbone (e.g., DistilGPT‑2) with:
      • main CLS head on final token
      • auxiliary head on early layer (aux_layer)
      • DEDIER KD weighting
    """
    def __init__(self, base_ckpt="distilgpt2", num_labels=2, aux_layer=1):
        super().__init__()
        cfg = AutoConfig.from_pretrained(base_ckpt, output_hidden_states=True)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_ckpt, config=cfg
        ).transformer
        self.cls_head = nn.Linear(cfg.n_embd, num_labels)
        self.aux_layer = aux_layer
        self.aux_head = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, num_labels),
        )
        self.T = 2.0   # KD temperature

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        teacher_logits=None,
        wt=None,                     # NEW: pre‑computed sample weights
    ):
        out = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_token = attention_mask.sum(-1) - 1                 # index of final real token
        h_last = out.hidden_states[-1][torch.arange(input_ids.size(0)), last_token]
        logits = self.cls_head(h_last)

        h_aux = out.hidden_states[self.aux_layer][torch.arange(input_ids.size(0)), last_token]
        logits_aux = self.aux_head(h_aux)

        # inference mode
        if labels is None:
            return logits, logits_aux

        # ---- losses ---------------------------------------------------------
        ce = nn.functional.cross_entropy(logits, labels)

        kd = nn.functional.kl_div(
            nn.functional.log_softmax(logits / self.T, dim=-1),
            nn.functional.softmax(teacher_logits / self.T, dim=-1),
            reduction="batchmean",
        ) * (self.T ** 2)

        if wt is None:                                    # fall‑back for inference
            wt = torch.ones_like(kd)

        loss_dedier = (wt * kd).mean()                    # Algorithm 1, Eq. 5
        return ce + 0.05 * loss_dedier, logits, logits_aux

# 5. DistilGPT‑2 student
student = DEDIERCausalStudent(base_ckpt="distilgpt2")
student.to(STUDENT_DEV)

# 6. Data collator that injects teacher logits
def collate(batch):
    ids_cpu   = torch.tensor([b["input_ids"]      for b in batch])
    mask_cpu  = torch.tensor([b["attention_mask"] for b in batch])
    labels_cpu= torch.tensor([b["labels"]         for b in batch])
    wt_cpu    = torch.tensor([b["wt"]             for b in batch], dtype=torch.float32)

    # send a copy of the inputs to the TEACHER device to get logits
    ids_t   = ids_cpu.to(TEACHER_DEV)
    mask_t  = mask_cpu.to(TEACHER_DEV)
    with torch.no_grad():
        t_logits = teacher(input_ids=ids_t, attention_mask=mask_t).logits.cpu()

    # return everything on CPU; Trainer will move tensors to STUDENT_DEV
    return {
        "input_ids":      ids_cpu,
        "attention_mask": mask_cpu,
        "labels":         labels_cpu,
        "teacher_logits": t_logits,
        "wt":             wt_cpu,
    }

class DEDIERTrainer(Trainer):
    """Custom trainer that passes stored sample weights to the student."""

    def compute_loss(self, model, inputs, return_outputs=False):
        labels         = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits")
        wt             = inputs.pop("wt")

        teacher_logits = teacher_logits.to(model.device)

        loss, logits, _ = model(
            input_ids       = inputs["input_ids"],
            attention_mask  = inputs["attention_mask"],
            labels          = labels,
            teacher_logits  = teacher_logits,
            wt              = wt,            # <‑‑ cached weights
        )
        return (loss, (logits,)) if return_outputs else loss

# 7. Evaluation – AUROC, per‑group accuracy, and confidence margin
print("Student training complete. Beginning evaluation...")
roc = evaluate.load("roc_auc")
id_cols = [c for c in splits["test"].column_names if c.endswith("_identity")]

def predict_probs(ds, model):
    """Return positive‑class probabilities for each row in ds."""
    probs = []
    for batch in ds.map(tokenize, batched=True).iter(batch_size=128):
        ids  = torch.tensor(batch["input_ids"])
        mask = torch.tensor(batch["attention_mask"])
        with torch.no_grad():
            logits, _ = model(ids, attention_mask=mask)
        probs.append(logits.softmax(-1)[:, 1].cpu())         # p(y=1)
    return torch.cat(probs)

# ---- run inference on the student ----
y_prob = predict_probs(splits["test"], student)              # positive prob
y_true = torch.tensor(splits["test"]["labels"]).bool()       # ground truth
y_pred = (y_prob >= 0.5)                                     # hard labels
margin = torch.abs(2 * y_prob - 1)                           # |p1 − p2|

# ---- print metrics ----
print("\n=== GLOBAL METRICS ===")
print("AUROC  :", roc.compute(prediction_scores=y_prob, references=y_true)["roc_auc"])
print("Acc    :", (y_pred == y_true).float().mean().item())
print("Margin :", margin.mean().item())

rows = []
for g in id_cols:
    mask = torch.tensor(splits["test"][g]).bool()
    if mask.sum() == 0:
        continue
    auc  = roc.compute(prediction_scores=y_prob[mask], references=y_true[mask])["roc_auc"]
    acc  = (y_pred[mask] == y_true[mask]).float().mean().item()
    mar  = margin[mask].mean().item()
    rows.append((g, auc, acc, mar))

# pretty‑print as a table
if rows:
    df = pd.DataFrame(rows, columns=["group", "auroc", "accuracy", "avg_margin"])
    print("\n=== PER‑IDENTITY METRICS ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

print("Evaluation finished.")