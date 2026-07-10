"""
Uses PyTorch KeypointRCNN (17 body keypoints) — BlazePose-style skeleton
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import threading
import math

# COCO 17 keypoints (PyTorch KeypointRCNN)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),            # face
    (5,6),                               # shoulders
    (5,7),(7,9),                         # left arm
    (6,8),(8,10),                        # right arm
    (5,11),(6,12),(11,12),              # torso
    (11,13),(13,15),                     # left leg
    (12,14),(14,16),                     # right leg
]

# Keypoint colors (BGR for OpenCV)
KP_COLOR   = (0, 200, 80)    # green
BONE_COLOR = (0, 180, 255)   # cyan
TEXT_COLOR = (0, 200, 80)    # green

_INSIGHTS = {
    "confident": (
        "Your posture suggests confidence and openness.",
        "Maintaining this posture reinforces a positive emotional state. Keep it up!",
    ),
    "tense": (
        "Your posture suggests tension or stress.",
        "Try rolling your shoulders back and down. Take 3 deep belly breaths.",
    ),
    "slouched": (
        "You appear to be slouching, which can reflect or worsen low mood.",
        "Try lifting your chest gently and pulling your shoulders back.",
    ),
    "neutral": (
        "Your posture appears relaxed and neutral.",
        "You look physically at ease. Notice how this connects to your emotional state.",
    ),
}

# Model singleton
_model      = None
_model_lock = threading.Lock()
_device     = None


def _get_model():
    """Load KeypointRCNN once and cache it."""
    global _model, _device
    if _model is not None:
        return _model, _device

    with _model_lock:
        if _model is None:
            import torch
            from torchvision.models.detection import (
                keypointrcnn_resnet50_fpn,
                KeypointRCNN_ResNet50_FPN_Weights,
            )

            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            _model  = keypointrcnn_resnet50_fpn(weights=weights)
            _model.to(_device)
            _model.eval()

    return _model, _device


def _angle_deg(ax, ay, bx, by) -> float:
    return abs(math.degrees(math.atan2(by - ay, bx - ax)))


def _classify_posture(kps: dict) -> Tuple[str, dict]:
    """
    Classify posture from 17 COCO keypoints.
    kps = {name: (x, y, score)}
    """
    def get(name):
        return kps.get(name, (0, 0, 0))

    ls  = get("left_shoulder")
    rs  = get("right_shoulder")
    lh  = get("left_hip")
    rh  = get("right_hip")
    nose = get("nose")
    le  = get("left_ear")
    re  = get("right_ear")

    sh_mid_x = (ls[0] + rs[0]) / 2
    sh_mid_y = (ls[1] + rs[1]) / 2
    hi_mid_x = (lh[0] + rh[0]) / 2
    hi_mid_y = (lh[1] + rh[1]) / 2

    shoulder_width = max(abs(ls[0] - rs[0]), 1.0)

    # Feature 1: shoulder tilt (0 = level)
    sh_tilt = _angle_deg(ls[0], ls[1], rs[0], rs[1])
    if sh_tilt > 90:
        sh_tilt = 180 - sh_tilt

    # Feature 2: ear tilt (head tilt)
    ear_tilt = _angle_deg(le[0], le[1], re[0], re[1])
    if ear_tilt > 90:
        ear_tilt = 180 - ear_tilt

    # Feature 3: forward head (nose vs shoulder midline)
    fwd_head = (nose[0] - sh_mid_x) / shoulder_width

    # Feature 4: torso lean
    torso_h = max(abs(sh_mid_y - hi_mid_y), 1.0)
    torso_lean = abs(sh_mid_x - hi_mid_x) / torso_h

    # Feature 5: head height (higher = better posture)
    head_drop = (sh_mid_y - nose[1]) / shoulder_width

    data = {
        "shoulder_tilt": round(sh_tilt, 1),
        "head_tilt":     round(ear_tilt, 1),
        "forward_head":  round(fwd_head, 3),
        "torso_lean":    round(torso_lean, 3),
        "head_height":   round(head_drop, 3),
    }

    if abs(fwd_head) > 0.35 or torso_lean > 0.25 or head_drop < 0.55:
        posture = "slouched"
    elif sh_tilt > 8 or ear_tilt > 10:
        posture = "tense"
    elif sh_tilt <= 5 and torso_lean <= 0.12 and head_drop >= 0.75:
        posture = "confident"
    else:
        posture = "neutral"

    return posture, data


def _draw_skeleton(img, kps_xy: list, scores: list, threshold=0.4):
    """Draw 17 keypoints + skeleton connections on image."""
    import cv2

    h, w = img.shape[:2]
    pts = []
    for i, (x, y) in enumerate(kps_xy):
        pts.append((int(x), int(y), scores[i] if i < len(scores) else 1.0))

    # Draw bones
    for (a, b) in COCO_SKELETON:
        if a < len(pts) and b < len(pts):
            if pts[a][2] > threshold and pts[b][2] > threshold:
                cv2.line(img, (pts[a][0], pts[a][1]),
                              (pts[b][0], pts[b][1]),
                         BONE_COLOR, 2, cv2.LINE_AA)

    # Draw keypoints
    for i, (x, y, sc) in enumerate(pts):
        if sc > threshold:
            cv2.circle(img, (x, y), 5, KP_COLOR, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 5, (255,255,255), 1, cv2.LINE_AA)

    return img


def analyze_body_language(image_path: str) -> Tuple[str, Optional[str], Dict]:
    """
    Detect 17 body keypoints using PyTorch KeypointRCNN.
    Returns (result_text, annotated_image_path, data_dict)
    """
    path = Path(image_path)

    try:
        import torch
        import cv2
        from PIL import Image
        from torchvision import transforms as T

        # Load image
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return "Could not load image.", None, {}

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(img_rgb)
        transform = T.Compose([T.ToTensor()])
        tensor = transform(pil_img)

        model, device = _get_model()
        with torch.no_grad():
            outputs = model([tensor.to(device)])

        # Get best detection (highest score)
        output = outputs[0]
        if len(output["scores"]) == 0 or output["scores"][0] < 0.5:
            return (
                "No person detected in the image.\n\n"
                "Tips:\n"
                "  - Ensure your full body is visible\n"
                "  - Use good, even lighting\n"
                "  - Face the camera with plain background",
                None, {}
            )

        # Best detection
        idx     = 0
        kps     = output["keypoints"][idx].cpu().numpy()   # [17, 3] (x, y, score)
        kps_xy  = [(k[0], k[1]) for k in kps]
        scores  = [k[2] for k in kps]

        # Build named keypoint dict
        kp_named = {
            COCO_KEYPOINTS[i]: (kps[i][0], kps[i][1], kps[i][2])
            for i in range(len(COCO_KEYPOINTS))
        }

        # Classify posture
        posture, data = _classify_posture(kp_named)
        desc, tip     = _INSIGHTS.get(posture, _INSIGHTS["neutral"])

        # Draw skeleton on image
        annotated = img_bgr.copy()
        annotated = _draw_skeleton(annotated, kps_xy, scores)

        # Add posture label
        cv2.putText(annotated, f"Posture: {posture.capitalize()}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Shoulder tilt: {data['shoulder_tilt']}  |  Head tilt: {data['head_tilt']}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Forward head: {data['forward_head']}  |  Torso lean: {data['torso_lean']}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

        out_path = path.with_name(path.stem + "_pose_annotated.jpg")
        cv2.imwrite(str(out_path), annotated)

        # Visible keypoints count
        visible = sum(1 for s in scores if s > 0.4)

        lines = [
            f"Detected Posture : **{posture.capitalize()}**",
            "",
            f"{desc}",
            "",
            "Body Keypoint Analysis (17 COCO landmarks):",
            f"  Keypoints detected : {visible}/17",
            f"  Shoulder tilt      : {data['shoulder_tilt']}° (0° = level)",
            f"  Head tilt          : {data['head_tilt']}°",
            f"  Forward head       : {data['forward_head']} (0 = aligned, high = slouch)",
            f"  Torso lean         : {data['torso_lean']} (0 = upright)",
            f"  Head height        : {data['head_height']} (higher = better)",
            "",
            f"Tip: {tip}",
            "",
            f"Engine: PyTorch KeypointRCNN (ResNet50) — {visible}/17 keypoints — {str(device).upper()}",
        ]
        return "\n".join(lines), str(out_path), {**data, "posture": posture}

    except ImportError as e:
        return _opencv_fallback(image_path)
    except Exception as e:
        return f"Error analysing body language: {e}", None, {}


def _opencv_fallback(image_path: str) -> Tuple[str, Optional[str], Dict]:
    """Simple OpenCV fallback."""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return "Could not load image.", None, {}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img[:h//2, :], cv2.COLOR_BGR2GRAY)
        fc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = fc.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return "No person detected. Ensure full upper body is visible with good lighting.", None, {}

        fx, fy, fw, fh = faces[0]
        offset  = abs((fx + fw//2) - w//2) / w
        posture = "confident" if offset < 0.1 else "neutral" if offset < 0.2 else "tense"
        _, tip  = _INSIGHTS.get(posture, _INSIGHTS["neutral"])

        out = img.copy()
        cv2.rectangle(out, (fx, fy), (fx+fw, fy+fh), KP_COLOR, 2)
        cv2.putText(out, f"Posture: {posture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, KP_COLOR, 2)
        out_path = image_path.replace(".jpg", "_pose_annotated.jpg")
        cv2.imwrite(out_path, out)

        return (f"Detected Posture: {posture.capitalize()}\n\nTip: {tip}\n\n"
                "Engine: OpenCV fallback"), out_path, {"posture": posture}
    except Exception as e:
        return f"Pose analysis failed: {e}", None, {}


#  Warm up model at import 
try:
    _get_model()
    import torch
    dev = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"   Pose engine       warmed up (PyTorch KeypointRCNN — {dev})")
except Exception as e:
    print(f"   Pose engine       will load on first use ({e})")