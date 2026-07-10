"""
HealMatrix AI — Emotion Detection Module (v2, CPU-optimised)

ARCHITECTURE
------------
Primary  : FER (FER-2013 CNN)  — fast, lightweight, loads once, CPU-friendly.
Fallback : DeepFace            — heavier but more robust; only runs when FER is unsure.

CONFIDENCE-BASED ROUTING
------------------------
  FER confidence >= 0.60  ->  return FER result immediately (the fast path).
  FER confidence <  0.60  ->  run DeepFace and return its result (the accurate path).

WHY THIS DESIGN
---------------
• Why FER is PRIMARY: it's a small FER-2013 CNN. On a CPU it loads in
  ~1-2s once at startup and infers in tens of milliseconds per image — far
  lighter than DeepFace's full stack. Most everyday images are classified
  confidently by FER alone, so the fast path handles the majority of calls.
• Why DeepFace is FALLBACK only: DeepFace is more robust on hard images
  (odd angles, poor lighting) but is heavier to load and slower per call.
  Running it only when FER is unsure gives us DeepFace-level robustness on
  the hard cases without paying its cost on every call.
• Why HuggingFace was REMOVED: it downloaded a separate ~hundreds-of-MB
  transformers model that overlapped heavily with FER/DeepFace (all three
  are FER-2013-style emotion classifiers). It added install weight, memory,
  and a network dependency for almost no accuracy benefit.
• Why OpenCV emotion fallback was REMOVED: it never actually classified
  emotion — it only detected whether a face existed and then returned
  "neutral" with a fake 0.5 confidence. That's misleading data in a mental
  health app. OpenCV is still used internally for fast face detection.

PERFORMANCE NOTES (typical CPU laptop, no GPU)
----------------------------------------------
• FER primary:     model loads once (~1-2s startup); ~20-60 ms per image.
• DeepFace path:   only triggers on low-confidence images; ~0.3-1.5s when it runs.
• Memory:          one FER model held in RAM (~tens of MB). DeepFace weights
                   load lazily only the first time the fallback is needed.
• CPU suitability: high — single small model on the hot path, no GPU required.

IMPROVEMENT OVER THE OLD PIPELINE
---------------------------------
• vs DeepFace-only : DeepFace ran on EVERY image (slow + heavy). Now it runs
                     only on the minority of uncertain images.
• vs HuggingFace-only: removes a large redundant model download and its
                     network/RAM cost.
• vs OpenCV-only   : OpenCV alone cannot classify emotion at all (it returned
                     a placeholder). Now every result is a real emotion score.
"""

from pathlib import Path
from typing import Tuple, Dict
import threading 
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


FER_CONFIDENCE_THRESHOLD = 0.60

# Therapeutic insight map (PRESERVED from your original module)
_INSIGHTS = {
    "happy": (
        "You appear to be in a positive emotional state! ",
        "Keep nurturing what brings you joy. Consider journaling about "
        "what made you feel this way so you can revisit it.",
    ),
    "sad": (
        "You appear to be experiencing sadness or low mood. ",
        "It's okay to feel sad. Try a gentle walk, talking to someone "
        "you trust, or listening to calming music. Be kind to yourself.",
    ),
    "angry": (
        "You appear to be experiencing frustration or anger. ",
        "Take slow deep breaths — try 4-7-8 breathing: inhale 4s, "
        "hold 7s, exhale 8s. Physical movement can also help release tension.",
    ),
    "fear": (
        "You appear to be experiencing anxiety or fear. ",
        "Try the 5-4-3-2-1 grounding technique: name 5 things you see, "
        "4 you can touch, 3 you hear, 2 you smell, 1 you taste.",
    ),
    "surprise": (
        "You appear surprised or startled. ",
        "Take a moment to breathe and process. Deep breathing helps "
        "your nervous system settle.",
    ),
    "disgust": (
        "You appear to be experiencing discomfort or aversion. ",
        "Acknowledge your feelings without judgment. Consider whether "
        "you can create distance from what's triggering this response.",
    ),
    "neutral": (
        "Your expression appears calm and neutral. ",
        "You seem balanced right now — a great state for mindfulness, "
        "reflection, or tackling something that requires focus.",
    ),
}


def _format_result(emotion: str, confidence: float,
                   all_scores: Dict[str, float], source: str = "") -> str:
    """Build the readable result string (PRESERVED format from your original)."""
    desc, tip = _INSIGHTS.get(emotion, _INSIGHTS["neutral"])
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    lines = [
        f" Detected Emotion : **{emotion.capitalize()}**",
        f" Confidence       : {confidence * 100:.1f}%",
        "",
        f" {desc}",
        "",
        "All Emotion Scores:",
    ]
    for emo, score in sorted_scores[:7]:
        pct = score if score <= 100 else score * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"  {emo.capitalize():12s} [{bar}] {pct:.1f}%")

    lines += ["", f" Tip: {tip}"]
    if source:
        lines += ["", f"🔬 Engine: {source}"]
    return "\n".join(lines)


_fer_detector = None
_fer_lock = threading.Lock()
_deepface_ready = False


def _get_fer():
    """Load the FER detector once and cache it."""
    global _fer_detector
    if _fer_detector is not None:
        return _fer_detector
    with _fer_lock:
        if _fer_detector is None:
            # The `fer` package: FER(mtcnn=False) uses the fast OpenCV Haar
            # detector instead of the heavier MTCNN — better for CPU.
            from fer import FER  
            _fer_detector = FER(mtcnn=False)
    return _fer_detector

# PRIMARY ENGINE — FER
def _analyze_with_fer(path: str):
    """
    Run FER on the image.
    Returns (emotion, confidence_0_to_1, all_scores_0_to_1) or None on failure
    / no face detected.
    """
    try:
        import cv2  
        detector = _get_fer()
        img = cv2.imread(path)
        if img is None:
            return None

        results = detector.detect_emotions(img)
        if not results:
            return None

        face = max(results, key=lambda r: r["box"][2] * r["box"][3])
        scores = {k.lower(): float(v) for k, v in face["emotions"].items()}

        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]
        return emotion, confidence, scores
    except Exception:
        return None

#apply deep face here
#
def _analyze_with_deepface(path: str):
    """
    Run DeepFace on the image (only called when FER is unsure).
    Returns (emotion, confidence_0_to_1, all_scores_0_to_1) or None on failure.
    """
    global _deepface_ready
    try:
        from deepface import DeepFace 

        result = DeepFace.analyze(
            img_path=path,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
            detector_backend="opencv",  
        )
        _deepface_ready = True
        if isinstance(result, list):
            result = result[0]

        emotion = result["dominant_emotion"].lower()
        raw = result["emotion"]                     
        total = sum(raw.values())
        scores = ({k.lower(): v / total for k, v in raw.items()}
                  if total > 1.01 else {k.lower(): float(v) for k, v in raw.items()})
        confidence = scores.get(emotion, 0.0)
        return emotion, confidence, scores
    except Exception:
        return None


def analyze_facial_emotion(image_path: str) -> Tuple[str, str, float]:
    """
    Analyse the dominant facial emotion in *image_path* using
    confidence-based routing (FER primary -> DeepFace fallback).

    Returns
    -------
    (result_text, emotion_label, confidence_0_to_1)
    """
    path = str(image_path)

    # STEP 1  PRIMARY: FER (the fast path) 
    fer_out = _analyze_with_fer(path)

    if fer_out is not None:
        emotion, confidence, scores = fer_out

        # High confidence -> trust FER and return immediately.
        if confidence >= FER_CONFIDENCE_THRESHOLD:
            return _format_result(emotion, confidence, scores, "FER"), emotion, confidence

        #  STEP 2  Low confidence -> FALLBACK to DeepFace 
        df_out = _analyze_with_deepface(path)
        if df_out is not None:
            d_emotion, d_conf, d_scores = df_out
            return (_format_result(d_emotion, d_conf, d_scores,
                                   "DeepFace (FER was uncertain)"),
                    d_emotion, d_conf)

        # DeepFace failed  return FER's best guess rather than nothing.
        return (_format_result(emotion, confidence, scores,
                               "FER (low confidence; DeepFace unavailable)"),
                emotion, confidence)

    #  FER found no face / failed entirely -> try DeepFace directly 
    df_out = _analyze_with_deepface(path)
    if df_out is not None:
        d_emotion, d_conf, d_scores = df_out
        return _format_result(d_emotion, d_conf, d_scores, "DeepFace"), d_emotion, d_conf

    #  Both engines failed (almost always: no detectable face) 
    return (
        " No face detected in the image.\n\n"
        "Tips:\n"
        "  • Ensure your face is clearly visible\n"
        "  • Use good, even lighting\n"
        "  • Face the camera directly",
        "neutral", 0.0,
    )


def analyze_facial_emotion_detailed(image_path: str) -> Dict:
    """
    Same analysis, but returns a structured dict — useful for the API / React
    frontend. Includes which engine produced the result.
    """
    text, emotion, confidence = analyze_facial_emotion(image_path)
    engine = "none"
    if "Engine: FER" in text:
        engine = "FER"
    elif "DeepFace" in text:
        engine = "DeepFace"
    return {
        "result_text": text,
        "emotion": emotion,
        "confidence": confidence,
        "engine": engine,
    }


try:
    _get_fer()
    print("   Emotion engine    warmed up (FER primary, DeepFace fallback)")
except Exception:
    pass