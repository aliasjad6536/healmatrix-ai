"""
HealMatrix AI — Emotion Detection Diagnostic
"""
import sys
import traceback
from pathlib import Path

DATA_DIR = Path("data/emotions")
images = sorted(DATA_DIR.glob("emotion_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
if not images:
    print("No emotion images found in data/emotions/. Upload one via the UI first, then rerun this.")
    sys.exit(1)

img_path = str(images[0])
print(f"Testing with most recent image: {img_path}\n")

print("=" * 60)
print("TEST 1: FER (fer package)")
print("=" * 60)
try:
    import cv2
    from fer import FER
    print("  FER imported OK")
    detector = FER(mtcnn=False)
    print("  FER detector created OK")
    img = cv2.imread(img_path)
    print(f"  cv2.imread result: {'loaded, shape=' + str(img.shape) if img is not None else 'FAILED (None)'}")
    results = detector.detect_emotions(img)
    print(f"  detect_emotions() result: {results}")
except Exception as e:
    print(f"  FAILED with exception:")
    traceback.print_exc()

print()
print("=" * 60)
print("TEST 2: DeepFace")
print("=" * 60)
try:
    from deepface import DeepFace
    print("  DeepFace imported OK")
    result = DeepFace.analyze(
        img_path=img_path,
        actions=["emotion"],
        enforce_detection=False,
        silent=False,
        detector_backend="opencv",
    )
    print(f"  DeepFace.analyze() result: {result}")
except Exception as e:
    print(f"  FAILED with exception:")
    traceback.print_exc()

print()
print("=" * 60)
print("Diagnosis complete. Paste this FULL output back for a fix.")
print("=" * 60)
