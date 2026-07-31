"""Quick sanity check on the fer2013-enhanced dataset."""
import numpy as np
from datasets import load_dataset
from PIL import Image
from pathlib import Path
from collections import Counter

print("Loading dataset...")
ds = load_dataset("abhilash88/fer2013-enhanced", cache_dir="hf_datasets/emotion")
d = ds[list(ds.keys())[0]]

print("\n=== BASIC INFO ===")
print("Total rows:", len(d))
print("Columns:", d.column_names)

print("\n=== USAGE SPLIT COUNTS ===")
usage_counts = Counter(item["usage"] for item in d)
print(usage_counts)

print("\n=== EMOTION LABEL COUNTS ===")
emo_counts = Counter(item["emotion_name"] for item in d)
for k, v in sorted(emo_counts.items()):
    print(f"  {k:<10}: {v}")

print("\n=== IMAGE ARRAY SANITY CHECK ===")
sample = d[0]["image"]
arr = np.array(sample, dtype="uint8")
print("Shape:", arr.shape, "dtype:", arr.dtype)
print("Min pixel:", arr.min(), "Max pixel:", arr.max(), "Mean:", arr.mean())

print("\n=== SAVING 6 SAMPLE IMAGES FOR VISUAL CHECK ===")
out_dir = Path("dataset_check")
out_dir.mkdir(exist_ok=True)

seen = set()
saved = 0
for item in d:
    emo = item["emotion_name"]
    if emo in seen:
        continue
    seen.add(emo)
    arr = np.array(item["image"], dtype="uint8")
    img = Image.fromarray(arr, mode="L").convert("RGB")
    # Upscale so it's actually viewable (48x48 is tiny)
    img = img.resize((192, 192), Image.NEAREST)
    path = out_dir / f"{emo}.jpg"
    img.save(path)
    print(f"  Saved {path}  (raw shape {arr.shape}, min={arr.min()}, max={arr.max()})")
    saved += 1
    if saved >= 7:
        break

print(f"\nDone. Open the images in: {out_dir.resolve()}")
print("Check in VS Code file explorer — do they look like real, recognizable faces?")
