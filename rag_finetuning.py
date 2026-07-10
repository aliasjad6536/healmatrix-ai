"""
HealMatrix AI — RAG / BGE Embedding Fine-Tuning (FIXED)
========================================================

Fixed Issues:
- empathetic_dialogues: deprecated → uses trust_remote_code + fallback
- daily_dialog: deprecated → uses better_daily_dialog as fallback
- Added ShenLab/MentalChat16K as primary extra dataset
- Robust error handling for each dataset

Project:  HealMatrix AI — AI Mental Health Therapist
Model:    BAAI/bge-small-en-v1.5
Output:   checkpoints/bge_finetuned/

Usage:
    python rag_finetuning.py
"""

import os
import time
import json
import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

#  Config 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 60)
print("HealMatrix AI - RAG/BGE Fine-Tuning FIXED")
print("=" * 60)
print(f"Device : {device}")
if device == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

os.makedirs("checkpoints", exist_ok=True)

PAIRS = []
loaded_datasets = []

#  Dataset 1: Mental Health Counseling
print("\n[1/4] Mental Health Counseling Conversations...")
try:
    ds1 = load_dataset(
        "Amod/mental_health_counseling_conversations",
        cache_dir="hf_datasets/rag"
    )
    count = 0
    for item in ds1["train"]:
        q = str(item.get("Context") or item.get("context") or "")
        a = str(item.get("Response") or item.get("response") or "")
        if len(q) > 20 and len(a) > 20:
            PAIRS.append(InputExample(texts=[q, a], label=0.95))
            count += 1
    print(f"  {count:,} pairs added")
    loaded_datasets.append("Amod/mental_health_counseling_conversations")
except Exception as e:
    print(f"  {e}")

#  Dataset 2: MentalChat16K 
print("\n[2/4] MentalChat16K...")
try:
    ds2 = load_dataset(
        "ShenLab/MentalChat16K",
        cache_dir="hf_datasets/rag"
    )
    count = 0
    for item in ds2["train"]:
        q = str(item.get("question") or item.get("input") or
                item.get("Context") or "")
        a = str(item.get("answer") or item.get("output") or
                item.get("Response") or "")
        if len(q) > 20 and len(a) > 20:
            PAIRS.append(InputExample(texts=[q, a], label=0.92))
            count += 1
    print(f"   {count:,} pairs added")
    loaded_datasets.append("ShenLab/MentalChat16K")
except Exception as e:
    print(f"  not working {e}")

#  Dataset 3: Empathetic Dialogues
print("\n[3/4] Empathetic Dialogues")
emp_added = False
for emp_name in ["facebook/empathetic_dialogues", "empathetic_dialogues"]:
    try:
        ds3 = load_dataset(
            emp_name,
            trust_remote_code=True,
            cache_dir="hf_datasets/rag"
        )
        count = 0
        for item in ds3["train"]:
            ctx = str(item.get("context") or item.get("prompt") or "")
            utt = str(item.get("utterance") or item.get("response") or "")
            if len(ctx) > 20 and len(utt) > 20:
                PAIRS.append(InputExample(texts=[ctx, utt], label=0.90))
                count += 1
        print(f"  {count:,} pairs added ({emp_name})")
        loaded_datasets.append(emp_name)
        emp_added = True
        break
    except Exception as e:
        print(f"    {emp_name}: {str(e)[:60]}")

if not emp_added:
    print("  ℹ Empathetic dialogues skipped (counseling data sufficient)")

# ─ Dataset 4: Daily Dialog 
print("\n[4/4] Daily Dialog (FIXED)...")
dial_added = False
for dial_name in ["benjaminbeilharz/better_daily_dialog", "daily_dialog"]:
    try:
        ds4 = load_dataset(
            dial_name,
            trust_remote_code=True,
            cache_dir="hf_datasets/rag"
        )
        split = list(ds4.keys())[0]
        count = 0
        for item in ds4[split]:
            dialog = item.get("dialog") or item.get("turns") or []
            if isinstance(dialog, list) and len(dialog) >= 2:
                for i in range(len(dialog) - 1):
                    q = str(dialog[i])
                    a = str(dialog[i+1])
                    if len(q) > 10 and len(a) > 10:
                        PAIRS.append(InputExample(texts=[q, a], label=0.80))
                        count += 1
        print(f"  {count:,} pairs added ({dial_name})")
        loaded_datasets.append(dial_name)
        dial_added = True
        break
    except Exception as e:
        print(f"   {dial_name}: {str(e)[:60]}")

if not dial_added:
    print("  ℹ Daily dialog skipped (MentalChat sufficient)")

#  Negative Pairs 
negative_pairs = [
    ("I feel depressed every day", "The stock market increased today"),
    ("I have severe anxiety", "How to repair a motorcycle engine"),
    ("I feel lonely and hopeless", "The football tournament starts tomorrow"),
    ("I need emotional support", "Best pizza recipe for dinner"),
    ("I am having panic attacks", "Electric cars are becoming popular"),
    ("I want to hurt myself", "Weather is nice today"),
    ("I feel suicidal", "New movie released this weekend"),
]
for q, a in negative_pairs:
    PAIRS.append(InputExample(texts=[q, a], label=0.05))

#  Stats 
print(f"\n{'='*60}")
print(f"Training Statistics")
print(f"{'─'*40}")
print(f"Total Pairs      : {len(PAIRS):,}")
print(f"Datasets loaded  : {len(loaded_datasets)}")
for d in loaded_datasets:
    print(f"  ✅ {d}")

if len(PAIRS) == 0:
    print("\n No training pairs! Check internet connection.")
    exit(1)

#  Load BGE Model 
print("\nLoading BGE Model (BAAI/bge-small-en-v1.5)...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

# Save initial weights (BEFORE training)
initial_weights = (
    model[0].auto_model.embeddings.word_embeddings.weight.data.clone().cpu()
)
torch.save(initial_weights, "checkpoints/bge_weights_before.pt")
print(f"   Initial weights saved")
print(f"  Before[0][:4] = {initial_weights[0][:4].tolist()}")

#  Training 
BATCH  = 32 if device == "cuda" else 16
EPOCHS = 3  if device == "cuda" else 2

train_loader  = DataLoader(PAIRS, shuffle=True, batch_size=BATCH)
loss_function = losses.CosineSimilarityLoss(model)

print(f"\n Training for {EPOCHS} epochs on {device}...")
print(f"   Batch size: {BATCH}")
print(f"   Total pairs: {len(PAIRS):,}")
print("-" * 60)

start_time = time.time()

model.fit(
    train_objectives=[(train_loader, loss_function)],
    epochs=EPOCHS,
    warmup_steps=min(100, len(PAIRS)//BATCH),
    output_path="checkpoints/bge_finetuned",
    show_progress_bar=True,
)

# Save final weights (AFTER training)
final_weights = (
    model[0].auto_model.embeddings.word_embeddings.weight.data.clone().cpu()
)
torch.save(final_weights, "checkpoints/bge_weights_after.pt")

weight_change  = torch.abs(final_weights - initial_weights).mean().item()
total_minutes  = (time.time() - start_time) / 60

#  Report 
report = {
    "project": "HealMatrix AI",
    "model": "BAAI/bge-small-en-v1.5",
    "datasets": loaded_datasets,
    "training_pairs": len(PAIRS),
    "epochs": EPOCHS,
    "batch_size": BATCH,
    "device": device,
    "training_minutes": round(total_minutes, 2),
    "average_weight_change": round(weight_change, 8),
    "weight_proof": {
        "before": initial_weights[0][:4].tolist(),
        "after":  final_weights[0][:4].tolist(),
    }
}

with open("checkpoints/rag_training_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("\n" + "=" * 60)
print(" BGE FINE-TUNING COMPLETE")
print("=" * 60)
print(f"Weight Change : {weight_change:.8f}")
print(f"Training Time : {total_minutes:.2f} minutes")
print(f"\nBefore[0][:4]: {initial_weights[0][:4].tolist()}")
print(f"After [0][:4]: {final_weights[0][:4].tolist()}")
print("\n Saved Files:")
print("  checkpoints/bge_finetuned/")
print("  checkpoints/bge_weights_before.pt")
print("  checkpoints/bge_weights_after.pt")
print("  checkpoints/rag_training_report.json")
print("\nShow checkpoints/ folder to advisor!")
