from pathlib import Path
from datasets import load_dataset

print("=" * 60)
print("HealMatrix AI - Dataset Downloader FIXED")
print("=" * 60)

BASE = Path("hf_datasets")
for sub in ["emotion", "rag"]:
    (BASE / sub).mkdir(parents=True, exist_ok=True)

results = {}

# FER  
print("\n[1/5] FER Emotion Dataset...")

FER_OPTIONS = [
    "dima806/facial_emotions_image_detection",
    "Jeneral/fer_2013",
    "Francesco/fer2013",
]

fer_loaded = False
for fer_name in FER_OPTIONS:
    try:
        ds = load_dataset(fer_name, cache_dir=str(BASE / "emotion"))
        split = list(ds.keys())[0]
        count = len(ds[split])
        print(f"  {fer_name}")
        print(f"     Samples: {count:,}")
        results["emotion"] = f" {count:,} samples ({fer_name})"
        fer_loaded = True
        break
    except Exception as e:
        print(f"   {fer_name} failed: {str(e)[:50]}")

if not fer_loaded:
    print("  ℹ FER dataset unavailable online")
    print("     emotion_finetuning.py will use KB images instead")
    results["emotion"] = " Not available - will use KB fallback"

#  MENTAL HEALTH COUNSELING 
print("\n[2/5] Mental Health Counseling Conversations...")
try:
    ds2 = load_dataset(
        "Amod/mental_health_counseling_conversations",
        cache_dir=str(BASE / "rag")
    )
    count = len(ds2["train"])
    print(f"   {count:,} Q-A pairs")
    results["counseling"] = f"{count:,} Q-A pairs"
except Exception as e:
    print(f"  not working {e}")
    results["counseling"] = f"not working {e}"

#   MENTALCHAT16K 
print("\n[3/5] MentalChat16K...")
try:
    ds3 = load_dataset(
        "ShenLab/MentalChat16K",
        cache_dir=str(BASE / "rag")
    )
    count = len(ds3["train"])
    print(f"   {count:,} conversations")
    results["mentalchat"] = f"{count:,} conversations"
except Exception as e:
    print(f"  not wokring {e}")
    results["mentalchat"] = f"not wokring {e}"

# ── 4. EMPATHETIC DIALOGUES 
print("\n[4/5] Empathetic Dialogues")
emp_loaded = False

# Try facebook version first
for emp_name in [
    "facebook/empathetic_dialogues",
    "empathetic_dialogues",
]:
    try:
        ds4 = load_dataset(
            emp_name,
            trust_remote_code=True,
            cache_dir=str(BASE / "rag")
        )
        count = len(ds4["train"])
        print(f"  {emp_name}: {count:,} samples")
        results["empathetic"] = f"{count:,} ({emp_name})"
        emp_loaded = True
        break
    except Exception as e:
        print(f"  {emp_name}: {str(e)[:60]}")

if not emp_loaded:
    results["empathetic"] = "Skipped (not critical)"
    print("  ℹ Empathetic dialogues skipped - counseling data is sufficient")

# DAILY DIALOG (FIXED) 
print("\n[5/5] Daily Dialog (FIXED)...")
dial_loaded = False

for dial_name in [
    "benjaminbeilharz/better_daily_dialog",
    "daily_dialog",
]:
    try:
        ds5 = load_dataset(
            dial_name,
            trust_remote_code=True,
            cache_dir=str(BASE / "rag")
        )
        split = list(ds5.keys())[0]
        count = len(ds5[split])
        print(f"   {dial_name}: {count:,} dialogs")
        results["dailydialog"] = f" {count:,} ({dial_name})"
        dial_loaded = True
        break
    except Exception as e:
        print(f" {dial_name}: {str(e)[:60]}")

if not dial_loaded:
    results["dailydialog"] = " Skipped (not critical)"
    print("  ℹ Daily dialog skipped - MentalChat is sufficient")

#  SUMMARY 
print("DOWNLOAD SUMMARY")

labels = {
    "emotion":    "Emotion  (FER)",
    "counseling": "RAG      (Mental Health Q-A)",
    "mentalchat": "RAG      (MentalChat16K)",
    "empathetic": "RAG      (Empathetic)",
    "dailydialog":"RAG      (Daily Dialog)",
}
for key, label in labels.items():
    status = results.get(key, "Not attempted")
    print(f"  {label:35s}: {status}")

print("""
Saved to: hf_datasets/
  emotion/  ← Emotion images
  rag/      ← Mental health text

Next Steps:
  python emotion_finetuning.py
  python pose_finetuning.py
  python rag_finetuning.py
  OR
  python run_all_training.py
""")
