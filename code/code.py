# -------------------------------------------------------------
# DEDIER with decoder‑only LMs on the Civil Comments dataset
#   Teacher : GPT‑2‑large  (774 M)
#   Student : DistilGPT‑2  (82 M)   – baseline & DeDiER variants
# -------------------------------------------------------------
#  Requirements (create a fresh env once):
#   pip install "transformers>=4.39" "datasets>=2.19" \
#               evaluate torch accelerate pandas sentencepiece
# -------------------------------------------------------------

# 0. Imports
import torch, evaluate, numpy as np, pandas as pd
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModelForCausalLM,
)
import transformers                           # for TrainerCallback

# ------------------------------------------------------------------
# Device setup: teacher on cuda:0, students on cuda:1 (if available)
# ------------------------------------------------------------------
if torch.cuda.device_count() >= 2:
    TEACHER_DEV = torch.device("cuda:0")
    STUDENT_DEV = torch.device("cuda:1")
elif torch.cuda.is_available():
    TEACHER_DEV = STUDENT_DEV = torch.device("cuda:0")
else:
    TEACHER_DEV = STUDENT_DEV = torch.device("cpu")

# ------------------------------------------------------------------
# DeDiER schedule hyper‑parameters (strict Algorithm 1)
L = 1   # recompute sample‑weights every L epochs
R = 1   # fine‑tune the auxiliary head for R epochs
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 1. Dataset & Tokeniser
# ------------------------------------------------------------------
print("Loading Civil Comments dataset and preparing tokenizer…")
tok = AutoTokenizer.from_pretrained("gpt2-large")
tok.pad_token, tok.padding_side = tok.eos_token, "right"

splits = load_dataset("civil_comments")      # returns train/validation/test

def tokenize(ex):
    return tok(
        ex["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

splits_tok = splits.map(tokenize, batched=True).rename_column("toxicity", "labels")
# add per‑example weights (all 1.0 to start)
splits_tok["train"] = splits_tok["train"].add_column("wt", [1.0] * len(splits_tok["train"]))

print(f"Tokenization complete. "
      f"Train {len(splits_tok['train']):,} / "
      f"Val {len(splits_tok['validation']):,} / "
      f"Test {len(splits_tok['test']):,}")

# ------------------------------------------------------------------
# 2. Teacher: GPT‑2‑large fine‑tune
# ------------------------------------------------------------------
teacher = GPT2ForSequenceClassification.from_pretrained(
    "gpt2-large", num_labels=2, output_hidden_states=True
).to(TEACHER_DEV)

args_teacher = TrainingArguments(
    output_dir="gpt2_teacher",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,   # effective 8
    fp16=True, learning_rate=5e-6,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer_teacher = Trainer(
    model=teacher,
    args=args_teacher,
    train_dataset=splits_tok["train"],
    eval_dataset=splits_tok["validation"],
)
print(f"Starting teacher fine‑tuning on {TEACHER_DEV}…")
trainer_teacher.train()
teacher.eval()
teacher.save_pretrained("gpt2_teacher_ckpt")
print("Teacher fine‑tuning finished.")

# ------------------------------------------------------------------
# 3. Baseline student (DistilGPT‑2) – plain fine‑tune
# ------------------------------------------------------------------
baseline_student = GPT2ForSequenceClassification.from_pretrained(
    "distilgpt2", num_labels=2, output_hidden_states=True
).to(STUDENT_DEV)

args_base = TrainingArguments(
    output_dir="distilgpt2_baseline",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True, learning_rate=2e-5,
    num_train_epochs=4,
    weight_decay=0.01,
)
trainer_base = Trainer(
    model=baseline_student,
    args=args_base,
    train_dataset=splits_tok["train"],
    eval_dataset=splits_tok["validation"],
)
print(f"Starting baseline (non‑DeDiER) student fine‑tuning on {STUDENT_DEV}…")
trainer_base.train()
baseline_student.eval()
print("Baseline student fine‑tuning finished.")

# ------------------------------------------------------------------
# 4. Raw (un‑trained) student for 0‑shot comparison
# ------------------------------------------------------------------
raw_student = GPT2ForSequenceClassification.from_pretrained(
    "distilgpt2", num_labels=2, output_hidden_states=True
).to(STUDENT_DEV)
raw_student.eval()

# ------------------------------------------------------------------
# 5. DeDiER student definition
# ------------------------------------------------------------------
class DEDIERCausalStudent(nn.Module):
    """DistilGPT‑2 backbone with main and auxiliary heads (Algorithm 1)."""
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
        self.T = 2.0  # KD temperature

    def forward(
        self, input_ids, attention_mask,
        labels=None, teacher_logits=None, wt=None,
    ):
        out = self.backbone(
            input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
        last_tok = attention_mask.sum(-1) - 1
        h_last = out.hidden_states[-1][torch.arange(input_ids.size(0)), last_tok]
        logits = self.cls_head(h_last)

        h_aux = out.hidden_states[self.aux_layer][torch.arange(input_ids.size(0)), last_tok]
        logits_aux = self.aux_head(h_aux)

        if labels is None:
            return logits, logits_aux  # inference

        ce = nn.functional.cross_entropy(logits, labels)
        kd = nn.functional.kl_div(
            nn.functional.log_softmax(logits / self.T, dim=-1),
            nn.functional.softmax(teacher_logits / self.T, dim=-1),
            reduction="batchmean",
        ) * (self.T ** 2)
        if wt is None:
            wt = torch.ones_like(kd)
        loss_dedier = (wt * kd).mean()
        return ce + 0.05 * loss_dedier, logits, logits_aux

# ------------------------------------------------------------------
# 6. Data collator: adds teacher logits & cached weights
# ------------------------------------------------------------------
def collate(batch):
    ids_cpu   = torch.tensor([b["input_ids"]      for b in batch])
    mask_cpu  = torch.tensor([b["attention_mask"] for b in batch])
    labels_cpu= torch.tensor([b["labels"]         for b in batch])
    wt_cpu    = torch.tensor([b["wt"]             for b in batch], dtype=torch.float32)

    # compute teacher logits on teacher GPU
    with torch.no_grad():
        t_logits = teacher(
            input_ids=ids_cpu.to(TEACHER_DEV),
            attention_mask=mask_cpu.to(TEACHER_DEV)
        ).logits.cpu()

    return {
        "input_ids":      ids_cpu,
        "attention_mask": mask_cpu,
        "labels":         labels_cpu,
        "teacher_logits": t_logits,
        "wt":             wt_cpu,
    }

# ------------------------------------------------------------------
# 7. Custom Trainer (passes wt tensor to model)
# ------------------------------------------------------------------
class DEDIERTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels         = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits").to(model.device)
        wt             = inputs.pop("wt").to(model.device)
        loss, logits, _ = model(
            input_ids      = inputs["input_ids"].to(model.device),
            attention_mask = inputs["attention_mask"].to(model.device),
            labels         = labels.to(model.device),
            teacher_logits = teacher_logits,
            wt             = wt,
        )
        return (loss, (logits,)) if return_outputs else loss

# ------------------------------------------------------------------
# 8. Callback: Algorithm 1 re‑weights every L epochs (optional strict mode)
# ------------------------------------------------------------------
class WeightUpdateCallback(transformers.TrainerCallback):
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch == 0 or int(state.epoch) % L != 0:
            return
        trainer = kwargs["trainer"]
        model   = trainer.model
        device  = model.device
        print(f"Epoch {int(state.epoch)} completed. "
              "Updating auxiliary head and sample weights…")

        # 1) fine‑tune auxiliary head only
        model.backbone.requires_grad_(False)
        model.cls_head.requires_grad_(False)
        model.aux_head.requires_grad_(True)
        opt = torch.optim.Adam(model.aux_head.parameters(), lr=1e-4)

        dl = trainer.get_train_dataloader()
        model.train()
        for _ in range(R):
            for batch in dl:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model.backbone(ids, attention_mask=mask,
                                     output_hidden_states=True, return_dict=True)
                last = mask.sum(-1) - 1
                h_aux = out.hidden_states[model.aux_layer][torch.arange(ids.size(0)), last]
                logits_aux = model.aux_head(h_aux)
                loss = nn.functional.cross_entropy(logits_aux, labels)
                loss.backward()
                opt.step(); opt.zero_grad()

        # 2) recompute weights
        model.eval()
        new_wts = []
        with torch.no_grad():
            for batch in dl:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                out = model.backbone(ids, attention_mask=mask,
                                     output_hidden_states=True, return_dict=True)
                last = mask.sum(-1) - 1
                h_aux = out.hidden_states[model.aux_layer][torch.arange(ids.size(0)), last]
                logits_aux = model.aux_head(h_aux)
                probs = logits_aux.softmax(-1)
                margin = probs.topk(2, dim=-1).values
                margin = (margin[:, 0] - margin[:, 1]).detach()
                wrong  = (logits_aux.argmax(-1) != labels).float()
                w = (1 + margin ** 3) * wrong
                new_wts.extend(w.cpu().tolist())

        # update dataset
        self.train_dataset = self.train_dataset.remove_column("wt") \
                                              .add_column("wt", new_wts)
        trainer.train_dataset = self.train_dataset
        model.backbone.requires_grad_(True)
        model.cls_head.requires_grad_(True)
        print("Auxiliary head fine‑tuning and weight recomputation done.")

# ------------------------------------------------------------------
# 9. DeDiER student training
# ------------------------------------------------------------------
student = DEDIERCausalStudent().to(STUDENT_DEV)
args_student = TrainingArguments(
    output_dir="gpt2_dedier_student",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True, learning_rate=2e-5,
    num_train_epochs=4,
    weight_decay=0.01,
)
trainer_student = DEDIERTrainer(
    model         = student,
    args          = args_student,
    train_dataset = splits_tok["train"],
    eval_dataset  = splits_tok["validation"],
    data_collator = collate,
    callbacks     = [WeightUpdateCallback(splits_tok["train"])],
)
print(f"Starting DeDiER student distillation on {STUDENT_DEV}…")
trainer_student.train()
student.eval()
print("DeDiER student training finished.")

# ------------------------------------------------------------------
# 10. Evaluation – teacher, raw student, baseline, DeDiER
# ------------------------------------------------------------------
print("Beginning evaluation of all models…")
roc = evaluate.load("roc_auc")
id_cols = [c for c in splits["test"].column_names if c.endswith("_identity")]

def predict_probs(ds, model, device):
    probs = []
    for batch in ds.map(tokenize, batched=True).iter(batch_size=128):
        ids  = torch.tensor(batch["input_ids"]).to(device)
        mask = torch.tensor(batch["attention_mask"]).to(device)
        with torch.no_grad():
            logits = model(ids, attention_mask=mask).logits
        probs.append(logits.softmax(-1)[:, 1].cpu())
    return torch.cat(probs)

models = {
    "Teacher (gpt2‑large)"   : (teacher,          TEACHER_DEV),
    "Raw distilgpt2 (0‑shot)": (raw_student,      STUDENT_DEV),
    "Finetuned distilgpt2"   : (baseline_student, STUDENT_DEV),
    "DeDiER student"         : (student,          STUDENT_DEV),
}

for name, (mdl, dev) in models.items():
    y_prob = predict_probs(splits["test"], mdl, dev)
    y_true = torch.tensor(splits["test"]["labels"]).bool()
    y_pred = (y_prob >= 0.5)
    margin = torch.abs(2 * y_prob - 1)

    print(f"\n=== {name} ===")
    print("AUROC  :", roc.compute(prediction_scores=y_prob,
                                  references=y_true)["roc_auc"])
    print("Acc    :", (y_pred == y_true).float().mean().item())
    print("Margin :", margin.mean().item())

    rows = []
    for g in id_cols:
        mask = torch.tensor(splits["test"][g]).bool()
        if mask.any():
            auc = roc.compute(prediction_scores=y_prob[mask],
                              references=y_true[mask])["roc_auc"]
            acc = (y_pred[mask] == y_true[mask]).float().mean().item()
            mar = margin[mask].mean().item()
            rows.append((g, auc, acc, mar))
    if rows:
        df = pd.DataFrame(rows, columns=["group", "auroc", "accuracy", "avg_margin"])
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

print("\nEvaluation finished.")