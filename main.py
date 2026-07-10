"""
HealMatrix AI — Emotional Wellness Intelligence System
Professional Light Theme UI | v1.4
"""

import json, os, tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# ── Directories ───────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
DATA_DIR          = BASE_DIR / "data"
CHAT_LOGS_DIR     = DATA_DIR / "chat_logs"
EMOTIONS_DIR      = DATA_DIR / "emotions"
CRISIS_ALERTS_DIR = DATA_DIR / "crisis_alerts"
SESSION_DIR       = DATA_DIR / "session"
for _d in [CHAT_LOGS_DIR, EMOTIONS_DIR, CRISIS_ALERTS_DIR, SESSION_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config import (
        GROQ_API_KEY, GROQ_MODEL,
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
        TWILIO_PHONE_NUMBER, TWILIO_WHATSAPP_NUMBER,
        EMERGENCY_CONTACT, EMERGENCY_WHATSAPP,
    )
except ImportError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   = "llama-3.3-70b-versatile"
    TWILIO_ACCOUNT_SID = TWILIO_AUTH_TOKEN = ""
    TWILIO_PHONE_NUMBER = TWILIO_WHATSAPP_NUMBER = ""
    EMERGENCY_CONTACT = EMERGENCY_WHATSAPP = ""

# ── Groq client ───────────────────────────────────────────────────────────────
try:
    from groq import Groq as _Groq
    _groq_client = _Groq(api_key=GROQ_API_KEY)
except Exception:
    _groq_client = None

# ── Twilio alert ──────────────────────────────────────────────────────────────
def send_twilio_alert(severity: str, message: str):
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=(f"HealMatrix Crisis Alert [{severity.upper()}]\n\n"
                  f"User: {message[:200]}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
            from_=TWILIO_WHATSAPP_NUMBER, to=EMERGENCY_WHATSAPP,
        )
        print(f"  WhatsApp alert sent [{severity.upper()}]")
        if severity == "high":
            client.calls.create(
                twiml='<Response><Say>Emergency Alert from HealMatrix AI. A user is in crisis.</Say></Response>',
                from_=TWILIO_PHONE_NUMBER, to=EMERGENCY_CONTACT,
            )
    except Exception as e:
        print(f"  Twilio failed: {e}")

# ── Load AI modules ───────────────────────────────────────────────────────────
print("=" * 60)
print("  HealMatrix AI v1.4 — loading modules ...")
print("=" * 60)
_F: Dict[str, bool] = {}

try:
    from crisis_detection import detect_crisis, get_crisis_severity, get_crisis_response_prefix
    _F["crisis"] = True;   print("  crisis_detection    OK")
except ImportError as e:
    _F["crisis"] = False;  print(f"  crisis_detection    FALLBACK ({e})")

try:
    from rag_system import query_with_rag, get_relevant_context
    _F["rag"] = True;      print("  rag_system           OK")
except ImportError as e:
    _F["rag"] = False;     print(f"  rag_system           FALLBACK ({e})")

try:
    from agi_engine import agi_query, decide_action, get_action_badge
    _F["agi"] = True;      print("  agi_engine           OK")
except ImportError as e:
    _F["agi"] = False;     print(f"  agi_engine           FALLBACK ({e})")

try:
    from emotion_detection import analyze_facial_emotion
    _F["emotion"] = True;  print("  emotion_detection    OK")
except ImportError as e:
    _F["emotion"] = False; print(f"  emotion_detection    FALLBACK ({e})")

try:
    from pose_detection import analyze_body_language
    _F["pose"] = True;     print("  pose_detection       OK")
except ImportError as e:
    _F["pose"] = False;    print(f"  pose_detection       FALLBACK ({e})")

try:
    from sentiment_analysis import analyze_sentiment
    _F["sentiment"] = True; print("  sentiment_analysis   OK")
except ImportError as e:
    _F["sentiment"] = False; print(f"  sentiment_analysis   FALLBACK ({e})")

try:
    from voice_input import transcribe_audio, save_gradio_audio
    _F["voice"] = True;    print("  voice_input          OK")
except ImportError as e:
    _F["voice"] = False;   print(f"  voice_input          FALLBACK ({e})")

try:
    from therapist_finder import search_therapists, get_therapist_cards_html
    _F["therapist"] = True; print("  therapist_finder     OK")
except ImportError as e:
    _F["therapist"] = False; print(f"  therapist_finder     FALLBACK ({e})")

try:
    from gtts import gTTS
    _F["tts"] = True;      print("  gTTS                 OK")
except ImportError as e:
    _F["tts"] = False;     print(f"  gTTS                 FALLBACK ({e})")

# ── Fallback functions ────────────────────────────────────────────────────────
if not _F.get("crisis"):
    def get_crisis_severity(t): return "none"
    def get_crisis_response_prefix(s): return ""
    def detect_crisis(t): return False

if not _F.get("sentiment"):
    def analyze_sentiment(t): return {"sentiment":"neutral","confidence":0.5,"result_text":"Sentiment unavailable.","color":"#63b3ed"}

if not _F.get("voice"):
    def transcribe_audio(p): return ""
    def save_gradio_audio(a): return None

if not _F.get("tts"):
    def text_to_speech(t): return None
else:
    def text_to_speech(text: str) -> Optional[str]:
        try:
            tts = gTTS(text=text[:500], lang="en", slow=False)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=str(DATA_DIR))
            tts.save(tmp.name)
            return tmp.name
        except Exception as e:
            print(f"  TTS failed: {e}"); return None

if not _F.get("rag"):
    def query_with_rag(q, k=3): return "RAG system unavailable."
    def get_relevant_context(q, k=3): return []

if not _F.get("agi"):
    def agi_query(message, severity, **kwargs):
        return query_with_rag(message), "GUIDE"
    def get_action_badge(a): return ""

if not _F.get("therapist"):
    def search_therapists(l, **kwargs): return []
    def get_therapist_cards_html(r, l): return "<p>Therapist finder unavailable.</p>"

if not _F.get("emotion"):
    def analyze_facial_emotion(p): return "Emotion detection unavailable.", "neutral", 0.0

if not _F.get("pose"):
    def analyze_body_language(p): return "Pose detection unavailable.", None, {}

# ── POSE DETECTION FIX ────────────────────────────────────────────────────────
# mediapipe.solutions fix — newer mediapipe uses different API
def safe_analyze_body_language(image_path: str):
    """Wrapper that fixes mediapipe solutions attribute error"""
    try:
        import mediapipe as mp
        # Fix: newer mediapipe may not have mp.solutions directly
        # Try to patch it
        if not hasattr(mp, 'solutions'):
            try:
                from mediapipe.python.solutions import pose as mp_pose_mod
                from mediapipe.python.solutions import drawing_utils
                from mediapipe.python.solutions import drawing_styles

                class _Solutions:
                    pose = mp_pose_mod
                    drawing_utils = drawing_utils
                    drawing_styles = drawing_styles
                    class PoseLandmark:
                        pass

                mp.solutions = _Solutions()
            except Exception:
                pass

        result, annotated, data = analyze_body_language(image_path)
        return result, annotated, data
    except AttributeError as e:
        if "solutions" in str(e):
            return _fallback_pose_analysis(image_path)
        return f"Pose error: {e}", None, {}
    except Exception as e:
        return f"Pose error: {e}", None, {}

def _fallback_pose_analysis(image_path: str):
    """Simple OpenCV fallback when mediapipe.solutions is unavailable"""
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return "Could not load image.", None, {}

        h, w = img.shape[:2]
        # Simple heuristic: analyze upper body region
        upper = img[:h//2, :]
        gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)

        # Face detection for head position
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return (
                "No person clearly detected.\n\n"
                "Tips:\n"
                "  - Ensure full upper body is visible\n"
                "  - Use good lighting\n"
                "  - Face the camera directly",
                None, {}
            )

        # Basic posture from face position
        fx, fy, fw, fh = faces[0]
        face_center_x = fx + fw // 2
        img_center_x = w // 2
        offset = abs(face_center_x - img_center_x) / w

        if offset < 0.1:
            posture = "confident"
            desc = "Your posture appears confident and centered."
            tip = "Great posture! Keep it up."
        elif offset < 0.2:
            posture = "neutral"
            desc = "Your posture appears relaxed and neutral."
            tip = "You look comfortable. Notice how this connects to your emotional state."
        else:
            posture = "tense"
            desc = "Your posture suggests some tension."
            tip = "Try rolling your shoulders back and taking 3 deep breaths."

        result = (
            f"Detected Posture : {posture.capitalize()}\n\n"
            f"{desc}\n\n"
            f"Tip: {tip}\n\n"
            f"Engine: OpenCV (MediaPipe fallback mode)"
        )

        # Draw on image
        out = img.copy()
        cv2.rectangle(out, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)
        cv2.putText(out, f"Posture: {posture}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        out_path = image_path.replace(".jpg", "_pose_annotated.jpg")
        cv2.imwrite(out_path, out)

        return result, out_path, {"posture": posture}
    except Exception as e:
        return f"Pose analysis failed: {e}", None, {}

# ── Session manager ───────────────────────────────────────────────────────────
class SessionManager:
    def __init__(self):
        self.sessions: Dict = {}
        self.current: Optional[str] = None
        self.history: List = []
        self.emotion_history: List = []
        self.sentiment_history: List = []
        self.crisis_history: List = []
        self.agi_actions: List = []
        self.last_emotion: Optional[str] = None
        self.last_posture: Optional[str] = None
        self._start: datetime = datetime.now()

    def create(self) -> str:
        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current = sid
        self.history = []
        self.emotion_history = []
        self.sentiment_history = []
        self.crisis_history = []
        self.agi_actions = []
        self.last_emotion = None
        self.last_posture = None
        self._start = datetime.now()
        self.sessions[sid] = {"total_messages": 0}
        return sid

    def add(self, role: str, content: str, meta: dict = {}):
        self.history.append({"role": role, "content": content, **meta})
        if self.current and self.current in self.sessions:
            self.sessions[self.current]["total_messages"] += 1

    def get_history(self): return self.history[-20:]
    def log_emotion(self, e, c):
        self.last_emotion = e
        self.emotion_history.append({"time": datetime.now().strftime("%H:%M"), "emotion": e, "confidence": round(c*100,1)})
    def log_sentiment(self, s, c):
        self.sentiment_history.append({"time": datetime.now().strftime("%H:%M"), "sentiment": s, "confidence": round(c*100,1)})
    def log_crisis(self, sev, msg):
        self.crisis_history.append({"time": datetime.now().strftime("%H:%M"), "severity": sev, "message": msg[:80]})
    def log_agi(self, action):
        self.agi_actions.append({"time": datetime.now().strftime("%H:%M"), "action": action})

    def stats(self):
        total = sum(1 for h in self.history if h.get("role") == "user")
        dur = datetime.now() - self._start
        mins = int(dur.total_seconds() // 60)
        return {
            "total_messages": total,
            "emotions_detected": len(self.emotion_history),
            "crisis_alerts": len(self.crisis_history),
            "duration": f"{mins}m",
        }

sm = SessionManager()

# ── Evaluation metrics (from JSON if available) ───────────────────────────────
def _load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except: return {}

def _img_path(name):
    p = DATA_DIR / name
    return str(p) if p.exists() else None

_crisis_metrics  = _load_json(DATA_DIR / "crisis_evaluation.json")
_emotion_metrics = _load_json(DATA_DIR / "emotion_evaluation.json")
_rag_metrics     = _load_json(DATA_DIR / "rag_evaluation.json")

# ── Core functions ────────────────────────────────────────────────────────────
def chat_with_ai(message, history, tts_enabled):
    if not message or not message.strip():
        return history, "", None, ""

    severity = get_crisis_severity(message)
    is_crisis = severity != "none"

    if is_crisis:
        sm.log_crisis(severity, message)
        send_twilio_alert(severity, message)

    # Sentiment analysis
    sent_result = None
    if _F.get("sentiment"):
        sent_result = analyze_sentiment(message)
        sm.log_sentiment(sent_result["sentiment"], sent_result["confidence"])

    # RAG context
    rag_context = ""
    if _F.get("rag"):
        try:
            chunks = get_relevant_context(message, k=3)
            rag_context = "\n".join(f"- {c}" for c in chunks)
        except Exception:
            pass

    # AGI reasoning
    try:
        ai_text, action = agi_query(
            message=message,
            severity=severity,
            conversation_history=sm.get_history(),
            emotion=sm.last_emotion,
            sentiment=sent_result["sentiment"] if sent_result else None,
            posture=sm.last_posture,
            crisis_history=sm.crisis_history,
            rag_context=rag_context,
        )
        sm.log_agi(action)
        action_html = get_action_badge(action)
    except Exception as e:
        ai_text = query_with_rag(message) if _F.get("rag") else f"I'm here to help. (Error: {e})"
        action = "GUIDE"
        action_html = ""

    response = get_crisis_response_prefix(severity) + ai_text
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ]
    sm.add("user", message, {"severity": severity})
    sm.add("assistant", response, {"agi_action": action})

    audio_path = text_to_speech(ai_text) if tts_enabled and _F.get("tts") else None
    return history, "", audio_path, action_html


def chat_with_voice(audio, history, tts_enabled):
    if audio is None:
        return history, "", None, "", "No audio recorded."
    audio_path = save_gradio_audio(audio)
    if not audio_path:
        return history, "", None, "", "Could not process audio."
    transcribed = transcribe_audio(audio_path)
    if not transcribed or transcribed.startswith("["):
        return history, "", None, "", f"Transcription failed: {transcribed}"
    new_history, _, audio_out, action_html = chat_with_ai(transcribed, history, tts_enabled)
    return new_history, "", audio_out, action_html, f'You said: "{transcribed}"'


def analyze_emotion_fn(image):
    if image is None:
        return "Please upload or capture an image.", None
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
        p = EMOTIONS_DIR / f"emotion_{ts}.jpg"
        img.save(p)
        result, emotion, conf = analyze_facial_emotion(str(p))
        sm.log_emotion(emotion, conf)
        return result, str(p)
    except Exception as e:
        return f"Error: {e}", None


def analyze_pose_fn(image):
    if image is None:
        return "Please upload or capture an image.", None
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
        p = EMOTIONS_DIR / f"pose_{ts}.jpg"
        img.save(p)
        # Use safe wrapper that handles mediapipe.solutions error
        result, annotated, data = safe_analyze_body_language(str(p))
        if data and data.get("posture"):
            sm.last_posture = data["posture"]
        return result, annotated
    except Exception as e:
        return f"Error: {e}", None


def run_sentiment(text):
    if not text.strip():
        return "Please enter text to analyse."
    return analyze_sentiment(text)["result_text"]


def find_therapists_fn(location, radius):
    if not location.strip():
        return "<p style='color:#dc2626'>Please enter a location.</p>"
    results = search_therapists(location, radius_km=int(radius))
    return get_therapist_cards_html(results, location)


def stats_str():
    s = sm.stats()
    return (f"Messages: {s['total_messages']}  |  "
            f"Emotions: {s['emotions_detected']}  |  "
            f"Alerts: {s['crisis_alerts']}  |  "
            f"Duration: {s['duration']}")


def get_stats_dashboard():
    s = sm.stats()
    sid = sm.current or "None"
    lines = [
        f"### Live Session Statistics\n",
        f"**Session ID:** `{sid}`\n", "---",
        "#### Chat Summary",
        f"- **Total Messages:** {s['total_messages']}",
        f"- **Session Duration:** {s['duration']}",
        f"- **Crisis Alerts:** {s['crisis_alerts']}",
        f"- **Emotions Detected:** {s['emotions_detected']}",
    ]
    if sm.agi_actions:
        lines += ["", "---", "#### AGI Decisions"]
        from collections import Counter
        counts = Counter(a["action"] for a in sm.agi_actions)
        for action, count in counts.most_common():
            lines.append(f"- **{action}**: {count} time{'s' if count > 1 else ''}")
    if sm.emotion_history:
        lines += ["", "---", "#### Emotion History"]
        for e in sm.emotion_history[-10:]:
            lines.append(f"- `{e['time']}` — **{e['emotion'].capitalize()}** ({e['confidence']}%)")
    if sm.crisis_history:
        lines += ["", "---", "#### Crisis Alert History"]
        for c in sm.crisis_history:
            lines.append(f"- `{c['time']}` [{c['severity'].upper()}] — {c['message']}")
    return "\n".join(lines)


def new_session():
    sid = sm.create()
    welcome = (
        "**Welcome to HealMatrix AI** — Session `{}`\n\n"
        "I am **Dr. Emily Hartman**, your AI mental health companion.\n\n"
        "You can **type** your message or use **Voice** to speak.\n\n"
        "**How are you feeling today?**"
    ).format(sid)
    sm.add("assistant", welcome)
    return [{"role": "assistant", "content": welcome}], stats_str()


# ── Professional Light Theme CSS ──────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;600&display=swap');

:root {
  --bg: #f8fafc;
  --bg2: #f1f5f9;
  --bg3: #e2e8f0;
  --card: #ffffff;
  --border: #e2e8f0;
  --border2: #cbd5e1;
  --navy: #1e3a5f;
  --blue: #2563eb;
  --teal: #0891b2;
  --green: #059669;
  --red: #dc2626;
  --amber: #d97706;
  --purple: #7c3aed;
  --text: #0f172a;
  --text2: #475569;
  --text3: #94a3b8;
  --radius: 12px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 4px 16px rgba(0,0,0,0.06);
  --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 10px 30px rgba(0,0,0,0.1);
}

* { box-sizing: border-box; }

.gradio-container {
  background: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
  max-width: 1400px !important;
  margin: 0 auto !important;
  color: var(--text) !important;
}

/* Hero */
.hm-hero {
  background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 50%, #0891b2 100%);
  border-radius: 16px;
  padding: 3rem 2rem 2.5rem;
  margin-bottom: 1.5rem;
  text-align: center;
  box-shadow: var(--shadow-md);
}
.hm-logo {
  font-family: 'Playfair Display', serif;
  font-size: 2.8rem;
  font-weight: 600;
  color: #ffffff;
  margin: 0 0 0.4rem;
  letter-spacing: -0.5px;
}
.hm-tagline {
  color: rgba(255,255,255,0.8);
  font-size: 1rem;
  font-weight: 300;
  margin: 0 0 1.5rem;
}
.hm-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  justify-content: center;
}
.hm-pill {
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25);
  color: #ffffff;
  padding: 0.3rem 0.9rem;
  border-radius: 100px;
  font-size: 0.8rem;
  font-weight: 500;
  backdrop-filter: blur(4px);
}

/* Stats bar */
.hm-stats {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.8rem 1.2rem;
  margin-bottom: 0.8rem;
  font-size: 0.85rem;
  color: var(--text2);
  box-shadow: var(--shadow);
}

/* Crisis banner */
.hm-crisis {
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-left: 4px solid var(--red);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  margin: 0.8rem 0;
  font-size: 0.88rem;
  color: #7f1d1d;
}
.hm-crisis strong { color: var(--red); }

/* Section header */
.hm-section {
  font-family: 'Playfair Display', serif;
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--navy);
  margin: 0 0 1rem;
  padding-bottom: 0.6rem;
  border-bottom: 2px solid var(--border);
}

/* Cards */
.hm-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.3rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow);
}
.hm-card h3 {
  font-family: 'Playfair Display', serif;
  color: var(--navy);
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 0.7rem;
}
.hm-card p, .hm-card div {
  color: var(--text2);
  font-size: 0.85rem;
  line-height: 1.7;
}

/* Tip box */
.hm-tip {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  padding: 0.7rem 1rem;
  font-size: 0.84rem;
  color: #1e40af;
  margin-top: 0.8rem;
}
.hm-tip strong { color: var(--blue); }

/* Metric cards */
.hm-metric-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 0.8rem;
  margin-bottom: 1rem;
}
.hm-metric-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.2rem 1rem;
  text-align: center;
  box-shadow: var(--shadow);
}
.hm-metric-val {
  font-family: 'Playfair Display', serif;
  font-size: 2rem;
  font-weight: 600;
  color: var(--navy);
  line-height: 1;
  margin-bottom: 0.3rem;
}
.hm-metric-label {
  font-size: 0.75rem;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-weight: 600;
}

/* Footer */
.hm-footer {
  text-align: center;
  padding: 1.5rem 0 1rem;
  color: var(--text3);
  font-size: 0.82rem;
  border-top: 1px solid var(--border);
  margin-top: 1rem;
}
.hm-footer strong { color: var(--text2); }

/* Gradio overrides — Light Theme */
.gradio-container .tabs { background: transparent !important; }

.gradio-container .tab-nav {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  margin-bottom: 1rem !important;
  gap: 2px !important;
  box-shadow: var(--shadow) !important;
}
.gradio-container .tab-nav button {
  background: transparent !important;
  color: var(--text2) !important;
  border-radius: 8px !important;
  border: none !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  padding: 0.45rem 1rem !important;
  transition: all 0.15s !important;
}
.gradio-container .tab-nav button.selected {
  background: var(--navy) !important;
  color: #ffffff !important;
  box-shadow: 0 2px 8px rgba(30,58,95,0.3) !important;
}
.gradio-container .tab-nav button:hover:not(.selected) {
  background: var(--bg2) !important;
  color: var(--text) !important;
}

.gradio-container input[type="text"],
.gradio-container textarea {
  background: var(--card) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.9rem !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

.gradio-container .block {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow) !important;
}

.gradio-container label span {
  color: var(--text2) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.5px !important;
}

.gradio-container button.primary {
  background: linear-gradient(135deg, #1e3a5f, #2563eb) !important;
  border: none !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  border-radius: 8px !important;
  transition: all 0.15s !important;
  box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
}
.gradio-container button.primary:hover {
  opacity: 0.92 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}
.gradio-container button.secondary {
  background: var(--bg2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text2) !important;
  font-family: 'Inter', sans-serif !important;
  border-radius: 8px !important;
}
.gradio-container button.secondary:hover {
  background: var(--bg3) !important;
  color: var(--text) !important;
}

.gradio-container .chatbot {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
.gradio-container .message.user {
  background: linear-gradient(135deg, #1e3a5f, #1e40af) !important;
  color: #ffffff !important;
  border-radius: 14px 14px 4px 14px !important;
  border: none !important;
}
.gradio-container .message.bot {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 14px 14px 14px 4px !important;
  box-shadow: var(--shadow) !important;
}

.gradio-container .markdown { color: var(--text) !important; }
.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown h3 {
  color: var(--navy) !important;
  font-family: 'Playfair Display', serif !important;
}
.gradio-container .markdown a { color: var(--blue) !important; }
.gradio-container .markdown code {
  background: var(--bg2) !important;
  color: var(--teal) !important;
  border-radius: 4px !important;
  padding: 0.1rem 0.4rem !important;
  font-size: 0.85rem !important;
}
.gradio-container .markdown table { width: 100%; border-collapse: collapse; }
.gradio-container .markdown th {
  background: var(--navy) !important;
  color: #ffffff !important;
  font-size: 0.8rem !important;
  padding: 0.6rem 0.8rem !important;
}
.gradio-container .markdown td {
  padding: 0.5rem 0.8rem !important;
  border: 1px solid var(--border) !important;
  color: var(--text2) !important;
  font-size: 0.87rem !important;
}
.gradio-container .markdown tr:nth-child(even) td {
  background: var(--bg2) !important;
}
"""

# ── Build UI ──────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════
# HEALMATRIX AI — PROFESSIONAL FRONTEND v2.0
# Design System: Clinical SaaS | Inter + JetBrains Mono
# ══════════════════════════════════════════════════════════════════

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --hm-bg:         #F5F6F8;
  --hm-surface:    #FFFFFF;
  --hm-surface2:   #F0F2F5;
  --hm-surface3:   #E8EBF0;
  --hm-navy:       #0B1D3A;
  --hm-navy2:      #152E54;
  --hm-blue:       #1A56DB;
  --hm-blue2:      #1E429F;
  --hm-blue-pale:  #EBF0FF;
  --hm-blue-mid:   #BFDBFE;
  --hm-teal:       #0284C7;
  --hm-teal-pale:  #E0F2FE;
  --hm-green:      #059669;
  --hm-green-pale: #ECFDF5;
  --hm-amber:      #B45309;
  --hm-amber-pale: #FFFBEB;
  --hm-red:        #DC2626;
  --hm-red-pale:   #FEF2F2;
  --hm-purple:     #7C3AED;
  --hm-purple-pale:#EDE9FE;
  --hm-text:       #0F172A;
  --hm-text2:      #334155;
  --hm-text3:      #64748B;
  --hm-text4:      #94A3B8;
  --hm-border:     rgba(15,23,42,0.08);
  --hm-border2:    rgba(15,23,42,0.05);
  --hm-r:  12px;
  --hm-r2:  8px;
  --hm-r3: 20px;
  --hm-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.04);
  --hm-shadow2: 0 4px 6px rgba(0,0,0,0.04), 0 10px 30px rgba(0,0,0,0.08);
}

* { box-sizing: border-box; }

/* ── Base ── */
.gradio-container {
  background: var(--hm-bg) !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  color: var(--hm-text) !important;
  -webkit-font-smoothing: antialiased !important;
}
footer { display: none !important; }
.gradio-container > .contain { padding: 0 !important; max-width: 100% !important; }

/* ── Topbar ── */
.hm-topbar {
  height: 56px;
  background: var(--hm-surface);
  border-bottom: 1px solid var(--hm-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 32px;
  position: sticky;
  top: 0;
  z-index: 300;
  margin-bottom: 0;
}
.hm-brand { display: flex; align-items: center; gap: 10px; }
.hm-brand-mark {
  width: 32px; height: 32px;
  background: var(--hm-navy);
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.hm-brand-mark svg { display: block; }
.hm-brand-name { font-size: 14px; font-weight: 600; color: var(--hm-text); letter-spacing: -0.3px; }
.hm-brand-ver {
  font-size: 9.5px; color: var(--hm-text4);
  background: var(--hm-surface2); border: 1px solid var(--hm-border);
  padding: 1px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace;
}
.hm-topbar-right { display: flex; align-items: center; gap: 18px; }
.hm-status { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--hm-text3); }
.hm-status-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--hm-green);
  animation: hm-pulse 2.5s infinite;
}
@keyframes hm-pulse {
  0%   { box-shadow: 0 0 0 0 rgba(5,150,105,0.4); }
  70%  { box-shadow: 0 0 0 6px rgba(5,150,105,0); }
  100% { box-shadow: 0 0 0 0 rgba(5,150,105,0); }
}
.hm-vdiv { width: 1px; height: 20px; background: var(--hm-border); }
.hm-topbar-crisis { display: flex; align-items: center; gap: 6px; font-size: 11.5px; }
.hm-crsis-lbl { font-size: 10px; font-weight: 600; color: var(--hm-text4); text-transform: uppercase; letter-spacing: 0.6px; }
.hm-crsis-num { font-weight: 700; color: var(--hm-red); font-size: 12px; }
.hm-crsis-sep { color: var(--hm-text4); }

/* ── Main content wrapper ── */
.hm-main { padding: 28px 32px; max-width: 1440px; margin: 0 auto; }

/* ── Page header ── */
.hm-page-hd { margin-bottom: 22px; }
.hm-crumb {
  display: flex; align-items: center; gap: 5px;
  font-size: 11.5px; color: var(--hm-text4); margin-bottom: 6px;
}
.hm-crumb-active { color: var(--hm-text3); }
.hm-page-title {
  font-size: 20px; font-weight: 600; color: var(--hm-text);
  letter-spacing: -0.5px; margin: 0 0 3px;
}
.hm-page-sub { font-size: 12.5px; color: var(--hm-text4); margin: 0; }

/* ── Stat cards ── */
.hm-stats-row {
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 12px; margin-bottom: 22px;
}
.hm-stat {
  background: var(--hm-surface);
  border: 1px solid var(--hm-border);
  border-radius: var(--hm-r);
  padding: 15px 17px;
  box-shadow: var(--hm-shadow);
}
.hm-stat-lbl {
  font-size: 10.5px; font-weight: 600; color: var(--hm-text4);
  text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 7px;
}
.hm-stat-val {
  font-size: 26px; font-weight: 600; color: var(--hm-text);
  letter-spacing: -1px; line-height: 1;
}
.hm-stat-sm { font-size: 15px; padding-top: 5px; }
.hm-stat-red { color: var(--hm-red); }
.hm-stat-sub { font-size: 11px; color: var(--hm-text4); margin-top: 5px; }
.hm-sub-green { color: var(--hm-green); }
.hm-sub-red   { color: var(--hm-red); }

/* ── Crisis banner ── */
.hm-crisis-banner {
  background: var(--hm-red-pale);
  border: 1px solid #FECACA;
  border-radius: var(--hm-r);
  padding: 11px 16px;
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 20px; font-size: 12.5px;
  box-shadow: var(--hm-shadow);
}
.hm-crisis-banner svg { color: var(--hm-red); flex-shrink: 0; }
.hm-crisis-banner strong { color: var(--hm-red); }
.hm-crisis-banner span { color: #7F1D1D; }
.hm-crisis-nums { display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }
.hm-cnum {
  font-size: 11px; font-weight: 600; color: var(--hm-red);
  background: #ffffff; border: 1px solid #FECACA;
  padding: 3px 10px; border-radius: 20px;
  white-space: nowrap;
}

/* ── AGI badge ── */
.hm-agi-badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 10.5px; font-weight: 600;
  padding: 4px 10px; border-radius: 20px;
  border: 1px solid; letter-spacing: 0.3px;
  text-transform: uppercase; margin-top: 6px;
}

/* ── Info cards (sidebar panels) ── */
.hm-info-card {
  background: var(--hm-surface);
  border: 1px solid var(--hm-border);
  border-radius: var(--hm-r);
  overflow: hidden;
  margin-bottom: 12px;
  box-shadow: var(--hm-shadow);
}
.hm-info-card-hd {
  padding: 10px 14px;
  border-bottom: 1px solid var(--hm-border);
  font-size: 10.5px; font-weight: 600;
  color: var(--hm-text3);
  text-transform: uppercase; letter-spacing: 0.6px;
  display: flex; align-items: center; gap: 7px;
  background: var(--hm-surface2);
}
.hm-info-card-bd {
  padding: 13px 14px;
  font-size: 12.5px; color: var(--hm-text2);
  line-height: 1.85;
}
.hm-info-card-bd strong { color: var(--hm-text); }

/* ── Metric progress bars ── */
.hm-metric-row { margin-bottom: 14px; }
.hm-metric-top {
  display: flex; justify-content: space-between;
  align-items: center; margin-bottom: 5px;
}
.hm-metric-name { font-size: 12.5px; color: var(--hm-text2); }
.hm-metric-pct { font-size: 15px; font-weight: 600; color: var(--hm-text); }
.hm-metric-track {
  height: 5px; background: var(--hm-surface3);
  border-radius: 3px; overflow: hidden;
}
.hm-metric-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }

/* ── Feature chips ── */
.hm-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.hm-chip {
  font-size: 11px; font-weight: 500;
  padding: 3px 10px; border-radius: 20px; border: 1px solid;
}
.hm-chip-blue   { background: var(--hm-blue-pale);   color: var(--hm-blue2);  border-color: var(--hm-blue-mid); }
.hm-chip-teal   { background: var(--hm-teal-pale);   color: #0369A1;           border-color: #BAE6FD; }
.hm-chip-green  { background: var(--hm-green-pale);  color: #065F46;           border-color: #A7F3D0; }
.hm-chip-amber  { background: var(--hm-amber-pale);  color: var(--hm-amber);  border-color: #FDE68A; }
.hm-chip-red    { background: var(--hm-red-pale);    color: var(--hm-red);    border-color: #FECACA; }
.hm-chip-purple { background: var(--hm-purple-pale); color: var(--hm-purple); border-color: #DDD6FE; }
.hm-chip-slate  { background: var(--hm-surface2);    color: var(--hm-text3);  border-color: var(--hm-border); }

/* ── Pill tags (features row) ── */
.hm-pills-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px; }
.hm-pill {
  font-size: 11px; font-weight: 500; color: var(--hm-text3);
  background: var(--hm-surface); border: 1px solid var(--hm-border);
  padding: 4px 12px; border-radius: 20px;
}

/* ── Section divider ── */
.hm-section-hd {
  font-size: 11px; font-weight: 600; color: var(--hm-text4);
  text-transform: uppercase; letter-spacing: 0.6px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--hm-border);
  margin-bottom: 14px;
}

/* ── Tip box ── */
.hm-tip {
  background: var(--hm-blue-pale);
  border: 1px solid var(--hm-blue-mid);
  border-radius: var(--hm-r2);
  padding: 8px 12px;
  font-size: 12px; color: var(--hm-blue2);
  margin-top: 10px;
}
.hm-tip strong { color: var(--hm-blue); }

/* ── Alert card ── */
.hm-alert {
  border-radius: var(--hm-r2);
  padding: 10px 13px;
  font-size: 12.5px;
  margin-bottom: 12px;
  border: 1px solid;
}
.hm-alert-red    { background: var(--hm-red-pale);    border-color: #FECACA; color: #7F1D1D; }
.hm-alert-green  { background: var(--hm-green-pale);  border-color: #A7F3D0; color: #065F46; }
.hm-alert-blue   { background: var(--hm-blue-pale);   border-color: var(--hm-blue-mid); color: var(--hm-blue2); }
.hm-alert-amber  { background: var(--hm-amber-pale);  border-color: #FDE68A; color: #713F12; }
.hm-alert strong { font-weight: 600; }

/* ── Therapist card ── */
.hm-th-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 14px; }
.hm-th-card {
  background: var(--hm-surface);
  border: 1px solid var(--hm-border);
  border-radius: var(--hm-r);
  padding: 16px;
  transition: border-color 0.15s, box-shadow 0.15s;
  box-shadow: var(--hm-shadow);
}
.hm-th-card:hover { border-color: var(--hm-blue-mid); box-shadow: 0 4px 16px rgba(26,86,219,0.08); }
.hm-th-top { display: flex; align-items: center; gap: 11px; margin-bottom: 10px; }
.hm-th-av {
  width: 40px; height: 40px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 600; flex-shrink: 0;
}
.hm-th-name { font-size: 13px; font-weight: 600; color: var(--hm-text); }
.hm-th-spec { font-size: 11.5px; color: var(--hm-text4); }
.hm-th-meta { font-size: 11.5px; color: var(--hm-text3); line-height: 2; margin-bottom: 12px; }

/* ── Session ID ── */
.hm-session-id {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10.5px; color: var(--hm-text4);
  background: var(--hm-surface2); border: 1px solid var(--hm-border);
  padding: 2px 8px; border-radius: 4px;
  display: inline-block; margin-bottom: 12px;
}

/* ── Feature status row ── */
.hm-feat-row {
  display: flex; align-items: center;
  justify-content: space-between;
  font-size: 12px; color: var(--hm-text2);
  padding: 5px 0; border-bottom: 1px solid var(--hm-border2);
}
.hm-feat-row:last-child { border-bottom: none; }
.hm-feat-ok  { font-size: 10px; font-weight: 600; color: var(--hm-green); background: var(--hm-green-pale); border: 1px solid #A7F3D0; padding: 1px 7px; border-radius: 10px; }
.hm-feat-no  { font-size: 10px; font-weight: 600; color: var(--hm-text4); background: var(--hm-surface2); border: 1px solid var(--hm-border); padding: 1px 7px; border-radius: 10px; }

/* ── Footer ── */
.hm-footer {
  background: var(--hm-surface);
  border-top: 1px solid var(--hm-border);
  padding: 16px 32px;
  display: flex; align-items: center; justify-content: space-between;
  font-size: 11.5px; color: var(--hm-text4);
  margin-top: 32px; flex-wrap: wrap; gap: 8px;
}
.hm-footer-brand { font-weight: 600; color: var(--hm-text3); font-size: 12px; }
.hm-footer-crisis strong { color: var(--hm-red); }
.hm-footer-stack { font-size: 10.5px; color: var(--hm-text4); }

/* ── Gradio overrides ── */
.gradio-container .tabs { background: transparent !important; }

.gradio-container .tab-nav {
  background: var(--hm-surface) !important;
  border-bottom: 1px solid var(--hm-border) !important;
  border-radius: 0 !important;
  padding: 0 32px !important;
  gap: 0 !important;
  position: sticky !important;
  top: 56px !important;
  z-index: 200 !important;
  box-shadow: none !important;
}
.gradio-container .tab-nav button {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color: var(--hm-text3) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 14px 18px !important;
  margin-bottom: -1px !important;
  transition: all 0.15s !important;
}
.gradio-container .tab-nav button:hover {
  color: var(--hm-text2) !important;
  background: var(--hm-surface2) !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--hm-blue) !important;
  border-bottom-color: var(--hm-blue) !important;
  background: transparent !important;
  font-weight: 600 !important;
}

.gradio-container input[type="text"],
.gradio-container textarea {
  background: var(--hm-surface2) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r2) !important;
  color: var(--hm-text) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13.5px !important;
  padding: 10px 13px !important;
  transition: all 0.15s !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
  border-color: var(--hm-blue) !important;
  background: #ffffff !important;
  box-shadow: 0 0 0 3px rgba(26,86,219,0.08) !important;
  outline: none !important;
}

.gradio-container label span {
  font-family: 'Inter', sans-serif !important;
  font-size: 10.5px !important;
  font-weight: 600 !important;
  color: var(--hm-text4) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.6px !important;
}

.gradio-container button.primary {
  background: var(--hm-navy) !important;
  border: none !important;
  border-radius: var(--hm-r2) !important;
  color: #ffffff !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 9px 20px !important;
  transition: opacity 0.15s, transform 0.1s !important;
  letter-spacing: 0.1px !important;
  box-shadow: 0 1px 4px rgba(11,29,58,0.25) !important;
}
.gradio-container button.primary:hover {
  opacity: 0.88 !important;
  transform: translateY(-1px) !important;
}

.gradio-container button.secondary {
  background: var(--hm-surface) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r2) !important;
  color: var(--hm-text2) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: background 0.15s !important;
}
.gradio-container button.secondary:hover {
  background: var(--hm-surface2) !important;
}

.gradio-container .block {
  background: var(--hm-surface) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r) !important;
  box-shadow: var(--hm-shadow) !important;
}

.gradio-container .chatbot {
  background: var(--hm-bg) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r) !important;
}
.gradio-container .message.user {
  background: var(--hm-navy) !important;
  color: #ffffff !important;
  border-radius: 14px 14px 3px 14px !important;
  border: none !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  max-width: 76% !important;
  padding: 10px 14px !important;
}
.gradio-container .message.bot {
  background: var(--hm-surface) !important;
  color: var(--hm-text2) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: 14px 14px 14px 3px !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  max-width: 76% !important;
  padding: 10px 14px !important;
  box-shadow: var(--hm-shadow) !important;
}

.gradio-container .markdown { color: var(--hm-text2) !important; font-size: 13.5px !important; line-height: 1.65 !important; }
.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown h3 { color: var(--hm-text) !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
.gradio-container .markdown strong { color: var(--hm-text) !important; }
.gradio-container .markdown a { color: var(--hm-blue) !important; }
.gradio-container .markdown code {
  background: var(--hm-surface2) !important; color: var(--hm-navy) !important;
  font-family: 'JetBrains Mono', monospace !important;
  border-radius: 4px !important; padding: 2px 6px !important; font-size: 12px !important;
}
.gradio-container .markdown table { width: 100%; border-collapse: collapse; font-size: 13px; }
.gradio-container .markdown th {
  background: var(--hm-surface2) !important; color: var(--hm-text3) !important;
  font-size: 10.5px !important; font-weight: 600 !important; text-transform: uppercase !important;
  letter-spacing: 0.5px !important; padding: 8px 12px !important; border: 1px solid var(--hm-border) !important;
}
.gradio-container .markdown td {
  padding: 8px 12px !important; border: 1px solid var(--hm-border) !important;
  color: var(--hm-text2) !important;
}
.gradio-container .markdown tr:nth-child(even) td { background: var(--hm-surface2) !important; }

input[type="range"] { accent-color: var(--hm-blue) !important; }
input[type="checkbox"] { accent-color: var(--hm-blue) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }

@media (max-width: 768px) {
  .hm-stats-row { grid-template-columns: repeat(2, 1fr); }
  .hm-topbar { padding: 0 16px; }
}
"""

# ── HTML Components ───────────────────────────────────────────────

_TOPBAR = """
<div class="hm-topbar">
  <div class="hm-brand">
    <div class="hm-brand-mark">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
           stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    </div>
    <span class="hm-brand-name">HealMatrix AI</span>
    <span class="hm-brand-ver">v1.4</span>
  </div>
  <div class="hm-topbar-right">
    <div class="hm-status">
      <div class="hm-status-dot"></div>
      All systems operational
    </div>
    <div class="hm-vdiv"></div>
    <div class="hm-topbar-crisis">
      <span class="hm-crsis-lbl">Crisis</span>
      <span class="hm-crsis-num">988</span>
      <span class="hm-crsis-sep">·</span>
      <span class="hm-crsis-num">0800-00-002</span>
    </div>
  </div>
</div>
"""

_CRISIS_BANNER = """
<div class="hm-crisis-banner">
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
       stroke="currentColor" stroke-width="2" style="flex-shrink:0;color:#DC2626">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
    <path d="M12 9v4"/><path d="M12 17h.01"/>
  </svg>
  <div>
    <strong>24/7 Crisis Support</strong>
    <span> — Immediate help is always available. You are not alone.</span>
  </div>
  <div class="hm-crisis-nums">
    <span class="hm-cnum">988 (USA)</span>
    <span class="hm-cnum">0800-00-002 (Pakistan)</span>
    <span class="hm-cnum">Text HOME to 741741</span>
  </div>
</div>
"""


# ══════════════════════════════════════════════════════════════════
# HEALMATRIX AI — PROFESSIONAL FRONTEND v3.0
# Design: Clinical SaaS (ChatGPT × Linear × Stripe × Headspace)
# Stack: Gradio 6.x + Custom CSS Design System
# ══════════════════════════════════════════════════════════════════

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --hm-bg:         #F5F6F8;
  --hm-surface:    #FFFFFF;
  --hm-surface2:   #F0F2F5;
  --hm-surface3:   #E8EBF0;
  --hm-navy:       #0B1D3A;
  --hm-navy2:      #152E54;
  --hm-blue:       #1A56DB;
  --hm-blue2:      #1E429F;
  --hm-blue-pale:  #EBF0FF;
  --hm-blue-mid:   #BFDBFE;
  --hm-teal:       #0284C7;
  --hm-teal-pale:  #E0F2FE;
  --hm-green:      #059669;
  --hm-green-pale: #ECFDF5;
  --hm-amber:      #B45309;
  --hm-amber-pale: #FFFBEB;
  --hm-red:        #DC2626;
  --hm-red-pale:   #FEF2F2;
  --hm-purple:     #7C3AED;
  --hm-purple-pale:#EDE9FE;
  --hm-text:       #0F172A;
  --hm-text2:      #334155;
  --hm-text3:      #64748B;
  --hm-text4:      #94A3B8;
  --hm-border:     rgba(15,23,42,0.08);
  --hm-r:  12px;
  --hm-r2:  8px;
  --hm-shadow: 0 1px 3px rgba(0,0,0,0.04),0 4px 12px rgba(0,0,0,0.04);
}

/* ── Reset ── */
* { box-sizing: border-box; }

.gradio-container {
  background: var(--hm-bg) !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  color: var(--hm-text) !important;
  -webkit-font-smoothing: antialiased !important;
}
footer { display: none !important; }
.gradio-container > .contain { padding: 0 !important; max-width: 100% !important; }

/* ── Topbar ── */
.hm-topbar {
  height: 56px;
  background: var(--hm-surface);
  border-bottom: 1px solid var(--hm-border);
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 32px;
  position: sticky; top: 0; z-index: 300;
}
.hm-brand { display: flex; align-items: center; gap: 10px; }
.hm-brand-mark {
  width: 32px; height: 32px; background: var(--hm-navy);
  border-radius: 8px; display: flex; align-items: center; justify-content: center;
}
.hm-brand-name { font-size: 14px; font-weight: 600; color: var(--hm-text); letter-spacing: -0.3px; }
.hm-brand-ver {
  font-size: 9.5px; color: var(--hm-text4);
  background: var(--hm-surface2); border: 1px solid var(--hm-border);
  padding: 1px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace;
}
.hm-topbar-right { display: flex; align-items: center; gap: 18px; }
.hm-status { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--hm-text3); }
.hm-status-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--hm-green);
  animation: hm-pulse 2.5s infinite;
}
@keyframes hm-pulse {
  0%  { box-shadow: 0 0 0 0 rgba(5,150,105,0.4); }
  70% { box-shadow: 0 0 0 6px rgba(5,150,105,0); }
  100%{ box-shadow: 0 0 0 0 rgba(5,150,105,0); }
}
.hm-vdiv { width: 1px; height: 20px; background: var(--hm-border); }
.hm-crisis-top { display: flex; align-items: center; gap: 5px; font-size: 11.5px; }
.hm-crisis-lbl { font-size: 10px; font-weight: 600; color: var(--hm-text4); text-transform: uppercase; letter-spacing: 0.6px; }
.hm-crisis-num { font-weight: 700; color: var(--hm-red); font-size: 12px; }

/* ── Tabs ── */
.gradio-container .tabs { background: transparent !important; }
.gradio-container .tab-nav {
  background: var(--hm-surface) !important;
  border-bottom: 1px solid var(--hm-border) !important;
  border-radius: 0 !important;
  padding: 0 32px !important;
  gap: 0 !important;
  position: sticky !important;
  top: 56px !important;
  z-index: 200 !important;
  box-shadow: none !important;
}
.gradio-container .tab-nav button {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color: var(--hm-text3) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 14px 18px !important;
  margin-bottom: -1px !important;
  transition: all 0.15s !important;
}
.gradio-container .tab-nav button:hover {
  color: var(--hm-text2) !important;
  background: var(--hm-surface2) !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--hm-blue) !important;
  border-bottom-color: var(--hm-blue) !important;
  background: transparent !important;
  font-weight: 600 !important;
}

/* ── Content wrapper ── */
.hm-content { padding: 28px 32px; max-width: 1440px; margin: 0 auto; }

/* ── Page header ── */
.hm-page-hd { margin-bottom: 22px; }
.hm-crumb { display: flex; align-items: center; gap: 5px; font-size: 11.5px; color: var(--hm-text4); margin-bottom: 5px; }
.hm-crumb-sep { font-size: 10px; }
.hm-crumb-active { color: var(--hm-text3); }
.hm-page-title { font-size: 20px; font-weight: 600; color: var(--hm-text); letter-spacing: -0.5px; margin: 0 0 3px; }
.hm-page-sub { font-size: 12.5px; color: var(--hm-text4); margin: 0; }

/* ── Stat cards ── */
.hm-stats-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.hm-stat { background: var(--hm-surface); border: 1px solid var(--hm-border); border-radius: var(--hm-r); padding: 15px 17px; box-shadow: var(--hm-shadow); }
.hm-stat-lbl { font-size: 10.5px; font-weight: 600; color: var(--hm-text4); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 7px; }
.hm-stat-val { font-size: 26px; font-weight: 600; color: var(--hm-text); letter-spacing: -1px; line-height: 1; }
.hm-stat-sm  { font-size: 15px; padding-top: 5px; }
.hm-stat-sub { font-size: 11px; color: var(--hm-text4); margin-top: 5px; }
.hm-sub-green { color: var(--hm-green) !important; }
.hm-sub-red   { color: var(--hm-red) !important; }

/* ── Crisis banner ── */
.hm-crisis-banner {
  background: var(--hm-red-pale);
  border: 1px solid #FECACA;
  border-radius: var(--hm-r);
  padding: 11px 16px;
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 20px; font-size: 12.5px;
}
.hm-crisis-banner strong { color: var(--hm-red); }
.hm-crisis-banner span { color: #7F1D1D; }
.hm-crisis-nums { display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }
.hm-cnum {
  font-size: 11px; font-weight: 600; color: var(--hm-red);
  background: #fff; border: 1px solid #FECACA;
  padding: 3px 10px; border-radius: 20px;
}

/* ── Info cards ── */
.hm-card {
  background: var(--hm-surface);
  border: 1px solid var(--hm-border);
  border-radius: var(--hm-r);
  overflow: hidden; margin-bottom: 12px;
  box-shadow: var(--hm-shadow);
}
.hm-card-hd {
  padding: 10px 14px;
  border-bottom: 1px solid var(--hm-border);
  font-size: 10.5px; font-weight: 600; color: var(--hm-text3);
  text-transform: uppercase; letter-spacing: 0.6px;
  background: var(--hm-surface2);
}
.hm-card-bd {
  padding: 13px 14px;
  font-size: 12.5px; color: var(--hm-text2); line-height: 1.85;
}

/* ── Feature status rows ── */
.hm-feat-row { display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--hm-text2); padding: 4px 0; border-bottom: 1px solid rgba(15,23,42,0.04); }
.hm-feat-row:last-child { border-bottom: none; }
.hm-feat-ok { font-size: 10px; font-weight: 600; color: var(--hm-green); background: var(--hm-green-pale); border: 1px solid #A7F3D0; padding: 1px 7px; border-radius: 10px; }
.hm-feat-off { font-size: 10px; font-weight: 600; color: var(--hm-text4); background: var(--hm-surface2); border: 1px solid var(--hm-border); padding: 1px 7px; border-radius: 10px; }

/* ── AGI badge ── */
.hm-agi-wrap { padding: 8px 14px; border-top: 1px solid var(--hm-border); background: var(--hm-surface2); min-height: 36px; }

/* ── Tip box ── */
.hm-tip {
  background: var(--hm-blue-pale); border: 1px solid var(--hm-blue-mid);
  border-radius: var(--hm-r2); padding: 9px 13px;
  font-size: 12px; color: var(--hm-blue2); margin-top: 10px;
}
.hm-tip strong { color: var(--hm-blue); }

/* ── Alert boxes ── */
.hm-alert { border-radius: var(--hm-r2); padding: 10px 13px; font-size: 12.5px; margin-bottom: 12px; border: 1px solid; }
.hm-alert-red   { background: var(--hm-red-pale);    border-color: #FECACA; color: #7F1D1D; }
.hm-alert-blue  { background: var(--hm-blue-pale);   border-color: var(--hm-blue-mid); color: var(--hm-blue2); }
.hm-alert-green { background: var(--hm-green-pale);  border-color: #A7F3D0; color: #065F46; }

/* ── Metric bars ── */
.hm-metric { margin-bottom: 13px; }
.hm-metric-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.hm-metric-name { font-size: 12.5px; color: var(--hm-text2); }
.hm-metric-pct  { font-size: 14px; font-weight: 600; color: var(--hm-text); }
.hm-metric-track { height: 5px; background: var(--hm-surface3); border-radius: 3px; overflow: hidden; }
.hm-metric-fill  { height: 100%; border-radius: 3px; }

/* ── Session ID ── */
.hm-sid { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--hm-text4); background: var(--hm-surface2); border: 1px solid var(--hm-border); padding: 2px 8px; border-radius: 4px; display: inline-block; margin-bottom: 12px; }

/* ── Footer ── */
.hm-footer {
  background: var(--hm-surface); border-top: 1px solid var(--hm-border);
  padding: 14px 32px; display: flex; align-items: center; justify-content: space-between;
  font-size: 11.5px; color: var(--hm-text4); flex-wrap: wrap; gap: 8px; margin-top: 32px;
}
.hm-footer-brand { font-weight: 600; color: var(--hm-text3); font-size: 12px; }
.hm-footer-crisis strong { color: var(--hm-red); }

/* ── Gradio component overrides ── */
.gradio-container input[type="text"],
.gradio-container textarea {
  background: var(--hm-surface2) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r2) !important;
  color: var(--hm-text) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13.5px !important;
  transition: all 0.15s !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
  border-color: var(--hm-blue) !important;
  background: #ffffff !important;
  box-shadow: 0 0 0 3px rgba(26,86,219,0.08) !important;
  outline: none !important;
}
.gradio-container label span {
  font-family: 'Inter', sans-serif !important;
  font-size: 10.5px !important;
  font-weight: 600 !important;
  color: var(--hm-text4) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.6px !important;
}
.gradio-container button.primary {
  background: var(--hm-navy) !important;
  border: none !important;
  border-radius: var(--hm-r2) !important;
  color: #ffffff !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 9px 20px !important;
  transition: opacity 0.15s, transform 0.1s !important;
  box-shadow: 0 1px 4px rgba(11,29,58,0.25) !important;
}
.gradio-container button.primary:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.gradio-container button.secondary {
  background: var(--hm-surface) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r2) !important;
  color: var(--hm-text2) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: background 0.15s !important;
}
.gradio-container button.secondary:hover { background: var(--hm-surface2) !important; }
.gradio-container .block {
  background: var(--hm-surface) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r) !important;
  box-shadow: var(--hm-shadow) !important;
}
.gradio-container .chatbot {
  background: var(--hm-bg) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r) !important;
}
.gradio-container .message.user {
  background: var(--hm-navy) !important;
  color: #ffffff !important;
  border-radius: 14px 14px 3px 14px !important;
  border: none !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  max-width: 76% !important;
  padding: 10px 14px !important;
}
.gradio-container .message.bot {
  background: var(--hm-surface) !important;
  color: var(--hm-text2) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: 14px 14px 14px 3px !important;
  font-size: 13.5px !important;
  line-height: 1.65 !important;
  max-width: 76% !important;
  padding: 10px 14px !important;
  box-shadow: var(--hm-shadow) !important;
}
.gradio-container .markdown { color: var(--hm-text2) !important; font-size: 13.5px !important; }
.gradio-container .markdown h1,.gradio-container .markdown h2,.gradio-container .markdown h3 { color: var(--hm-text) !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
.gradio-container .markdown strong { color: var(--hm-text) !important; }
.gradio-container .markdown a { color: var(--hm-blue) !important; }
.gradio-container .markdown code { background: var(--hm-surface2) !important; color: var(--hm-navy) !important; font-family: 'JetBrains Mono', monospace !important; border-radius: 4px !important; padding: 2px 6px !important; font-size: 12px !important; }
.gradio-container .markdown table { width: 100%; border-collapse: collapse; font-size: 13px; }
.gradio-container .markdown th { background: var(--hm-surface2) !important; color: var(--hm-text3) !important; font-size: 10.5px !important; font-weight: 600 !important; text-transform: uppercase !important; padding: 8px 12px !important; border: 1px solid var(--hm-border) !important; }
.gradio-container .markdown td { padding: 8px 12px !important; border: 1px solid var(--hm-border) !important; color: var(--hm-text2) !important; }
.gradio-container .markdown tr:nth-child(even) td { background: var(--hm-surface2) !important; }
input[type="range"] { accent-color: var(--hm-blue) !important; }
input[type="checkbox"] { accent-color: var(--hm-blue) !important; }
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
"""

# ── HTML Components ────────────────────────────────────────────

_TOPBAR = """
<div class="hm-topbar">
  <div class="hm-brand">
    <div class="hm-brand-mark">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    </div>
    <span class="hm-brand-name">HealMatrix AI</span>
    <span class="hm-brand-ver">v1.4</span>
  </div>
  <div class="hm-topbar-right">
    <div class="hm-status">
      <span class="hm-status-dot"></span>
      All systems operational
    </div>
    <div class="hm-vdiv"></div>
    <div class="hm-crisis-top">
      <span class="hm-crisis-lbl">Crisis</span>
      <span class="hm-crisis-num">988</span>
      <span style="color:#94A3B8">·</span>
      <span class="hm-crisis-num">0800-00-002</span>
    </div>
  </div>
</div>
"""

_CRISIS_BANNER = """
<div class="hm-crisis-banner">
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#DC2626" stroke-width="2" style="flex-shrink:0">
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
    <path d="M12 9v4"/><path d="M12 17h.01"/>
  </svg>
  <div><strong>24/7 Crisis Support</strong><span> — Immediate help is always available</span></div>
  <div class="hm-crisis-nums">
    <span class="hm-cnum">988 (USA)</span>
    <span class="hm-cnum">0800-00-002 (PK)</span>
    <span class="hm-cnum">HOME to 741741</span>
  </div>
</div>
"""

_FOOTER = """
<div class="hm-footer">
  <span class="hm-footer-brand">HealMatrix AI &nbsp;·&nbsp; v1.4 &nbsp;·&nbsp; 2026</span>
  <span class="hm-footer-crisis">Crisis: <strong>988</strong> (USA) &nbsp;·&nbsp; <strong>0800-00-002</strong> (Pakistan) &nbsp;·&nbsp; Text HOME to 741741</span>
  <span style="font-size:10.5px">Gradio &nbsp;·&nbsp; Groq LLaMA 3.3 &nbsp;·&nbsp; Whisper &nbsp;·&nbsp; DeepFace &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; RoBERTa &nbsp;·&nbsp; gTTS &nbsp;·&nbsp; Twilio</span>
</div>
"""

def _page_hd(title, subtitle, crumb="Clinical"):
    return f"""
    <div class="hm-page-hd">
      <div class="hm-crumb">
        <span>HealMatrix</span>
        <span class="hm-crumb-sep">›</span>
        <span>{crumb}</span>
        <span class="hm-crumb-sep">›</span>
        <span class="hm-crumb-active">{title}</span>
      </div>
      <h1 class="hm-page-title">{title}</h1>
      <p class="hm-page-sub">{subtitle}</p>
    </div>"""

def _stats_html(msgs, dur, alerts, emotions):
    return f"""
    <div class="hm-stats-row">
      <div class="hm-stat">
        <div class="hm-stat-lbl">Messages</div>
        <div class="hm-stat-val">{msgs}</div>
        <div class="hm-stat-sub hm-sub-green">This session</div>
      </div>
      <div class="hm-stat">
        <div class="hm-stat-lbl">Duration</div>
        <div class="hm-stat-val">{dur}</div>
        <div class="hm-stat-sub">Active</div>
      </div>
      <div class="hm-stat">
        <div class="hm-stat-lbl">Emotions</div>
        <div class="hm-stat-val">{emotions}</div>
        <div class="hm-stat-sub">Detected</div>
      </div>
      <div class="hm-stat">
        <div class="hm-stat-lbl">Crisis Alerts</div>
        <div class="hm-stat-val" style="color:{'#DC2626' if alerts>0 else '#0F172A'}">{alerts}</div>
        <div class="hm-stat-sub {'hm-sub-red' if alerts>0 else ''}">{'Alert sent' if alerts>0 else 'None'}</div>
      </div>
    </div>"""

def _agi_badge_new(action):
    cfg = {
        "ESCALATE": ("#FEF2F2","#DC2626","#FECACA"),
        "REASSURE": ("#EBF0FF","#1E429F","#BFDBFE"),
        "GUIDE":    ("#E0F2FE","#0369A1","#BAE6FD"),
        "REFER_THERAPIST": ("#FFFBEB","#B45309","#FDE68A"),
        "MOTIVATE": ("#ECFDF5","#059669","#A7F3D0"),
        "ASSESS":   ("#EDE9FE","#7C3AED","#DDD6FE"),
    }
    bg,col,bd = cfg.get(action, ("#F0F2F5","#64748B","#E2E8F0"))
    return f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:10.5px;font-weight:600;padding:4px 10px;border-radius:20px;border:1px solid {bd};background:{bg};color:{col};text-transform:uppercase;letter-spacing:0.3px">AGI: {action}</span>'

def _features_card_new(F):
    items = [
        ("AGI Reasoning Engine", F.get("agi")),
        ("RAG Knowledge Base", F.get("rag")),
        ("Voice Input (Whisper)", F.get("voice")),
        ("Sentiment Analysis", F.get("sentiment")),
        ("Text-to-Speech", F.get("tts")),
        ("Twilio Alerts", F.get("twilio")),
        ("Therapist Finder", F.get("therapist")),
    ]
    rows = "".join(
        f'<div class="hm-feat-row"><span>{n}</span><span class="{"hm-feat-ok" if ok else "hm-feat-off"}">{"Active" if ok else "Off"}</span></div>'
        for n, ok in items
    )
    return f'<div class="hm-card"><div class="hm-card-hd">Active Modules</div><div class="hm-card-bd" style="padding:10px 14px">{rows}</div></div>'

def _agi_matrix_card():
    rows_html = ""
    for action, col, desc in [
        ("ESCALATE","#DC2626","Crisis detected"),
        ("REASSURE","#1E429F","Emotional validation"),
        ("GUIDE","#0369A1","CBT/DBT guidance"),
        ("REFER_THERAPIST","#B45309","Professional help"),
        ("MOTIVATE","#059669","Encouragement"),
        ("ASSESS","#7C3AED","Gather information"),
    ]:
        rows_html += f'<div class="hm-feat-row"><span style="font-weight:600;font-size:11.5px;color:{col}">{action}</span><span style="color:var(--hm-text3);font-size:11px">{desc}</span></div>'
    return f'<div class="hm-card"><div class="hm-card-hd">AGI Decision Matrix</div><div class="hm-card-bd" style="padding:10px 14px">{rows_html}</div></div>'

def _info_card(title, content):
    return f'<div class="hm-card"><div class="hm-card-hd">{title}</div><div class="hm-card-bd">{content}</div></div>'

def _metric_bar(name, pct, color="#1A56DB"):
    return f"""<div class="hm-metric">
      <div class="hm-metric-top"><span class="hm-metric-name">{name}</span><span class="hm-metric-pct">{pct}%</span></div>
      <div class="hm-metric-track"><div class="hm-metric-fill" style="width:{pct}%;background:{color}"></div></div>
    </div>"""


# ── Build UI ───────────────────────────────────────────────────
def build_app():
    with gr.Blocks(title="HealMatrix AI") as app:

        # Topbar
        gr.HTML(_TOPBAR)

        # Session init
        sm.create()

        with gr.Tabs():

            # ══ TAB 1: THERAPY CHAT ═══════════════════════════════
            with gr.Tab("Therapy Chat"):
                gr.HTML(_page_hd("Therapy Chat", "Talk with Dr. Emily Hartman — AGI-powered clinical reasoning", "Clinical"))
                gr.HTML(_CRISIS_BANNER)

                stats_out = gr.HTML(_stats_html(0,"0m",0,0))

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            value=[{"role":"assistant","content":(
                                "**Hello, I am Dr. Emily Hartman.**\n\n"
                                "I am your AI mental health companion powered by **AGI reasoning** — "
                                "trained in **CBT**, **DBT**, **ACT**, and mindfulness-based therapy.\n\n"
                                "Everything you share here is private and confidential.\n\n"
                                "**How are you feeling today?**"
                            )}],
                            height=430, show_label=False,
                        )
                        agi_display = gr.HTML('<div class="hm-agi-wrap"></div>')

                        with gr.Tabs():
                            with gr.Tab("Text"):
                                with gr.Row():
                                    msg_box = gr.Textbox(
                                        placeholder="Share what is on your mind...",
                                        show_label=False, scale=5, lines=2,
                                        container=False,
                                    )
                                    send_btn = gr.Button("Send", variant="primary", scale=1)
                            with gr.Tab("Voice"):
                                voice_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your message")
                                voice_btn   = gr.Button("Transcribe and Send", variant="primary")
                                voice_status= gr.Textbox(label="Transcription", interactive=False, lines=1)

                        tts_toggle = gr.Checkbox(label="Read response aloud (Text-to-Speech)", value=False)
                        tts_audio  = gr.Audio(label="AI Voice Response", type="filepath", autoplay=True)

                    with gr.Column(scale=1):
                        gr.Button("New Session", variant="secondary").click(
                            lambda: (new_session()[0], _stats_html(0,"0m",0,0), ""),
                            outputs=[chatbot, stats_out, agi_display]
                        )
                        gr.HTML(_agi_matrix_card())
                        gr.HTML(_features_card_new(_F))

                def _chat_update(message, history, tts):
                    h, msg, audio, agi_html = chat_with_ai(message, history, tts)
                    s = sm.stats()
                    stats = _stats_html(s["total_messages"], s["duration"], s["crisis_alerts"], s["emotions_detected"])
                    # Extract action for new badge
                    import re
                    m = re.search(r'AGI: (\w+)', agi_html or "")
                    badge = f'<div class="hm-agi-wrap">{_agi_badge_new(m.group(1))}</div>' if m else '<div class="hm-agi-wrap"></div>'
                    return h, msg, audio, badge, stats

                def _voice_update(audio, history, tts):
                    h, msg, audio_out, agi_html, status = chat_with_voice(audio, history, tts)
                    s = sm.stats()
                    stats = _stats_html(s["total_messages"], s["duration"], s["crisis_alerts"], s["emotions_detected"])
                    import re
                    m = re.search(r'AGI: (\w+)', agi_html or "")
                    badge = f'<div class="hm-agi-wrap">{_agi_badge_new(m.group(1))}</div>' if m else '<div class="hm-agi-wrap"></div>'
                    return h, msg, audio_out, badge, status, stats

                msg_box.submit(_chat_update, [msg_box, chatbot, tts_toggle], [chatbot, msg_box, tts_audio, agi_display, stats_out])
                send_btn.click(_chat_update,  [msg_box, chatbot, tts_toggle], [chatbot, msg_box, tts_audio, agi_display, stats_out])
                voice_btn.click(_voice_update, [voice_input, chatbot, tts_toggle], [chatbot, msg_box, tts_audio, agi_display, voice_status, stats_out])

            # ══ TAB 2: SENTIMENT ══════════════════════════════════
            with gr.Tab("Sentiment Analysis"):
                gr.HTML(_page_hd("Sentiment Analysis", "Emotional tone detection using RoBERTa transformer model", "Clinical"))
                with gr.Row():
                    with gr.Column(scale=1):
                        sentiment_input = gr.Textbox(
                            label="Text to Analyse",
                            placeholder="Type or paste any message here — journal entry, recent conversation, or how you feel right now...",
                            lines=7
                        )
                        sentiment_btn = gr.Button("Analyse Sentiment", variant="primary")
                        gr.HTML(_info_card("About this module",
                            "Uses <strong>cardiffnlp/twitter-roberta-base-sentiment-latest</strong> — "
                            "a RoBERTa model trained on 124M real-world texts. Detects positive, negative, "
                            "or neutral tone with confidence scores. Each therapy message is also automatically analysed."
                        ))
                        gr.HTML('<div class="hm-tip"><strong>Try:</strong> Paste a journal entry or describe how you feel today.</div>')
                    with gr.Column(scale=1):
                        sentiment_result = gr.Textbox(label="Analysis Result", lines=18, interactive=False)
                sentiment_btn.click(run_sentiment, [sentiment_input], [sentiment_result])
                sentiment_input.submit(run_sentiment, [sentiment_input], [sentiment_result])

            # ══ TAB 3: EMOTION DETECTION ══════════════════════════
            with gr.Tab("Emotion Detection"):
                gr.HTML(_page_hd("Emotion Detection", "7-class facial emotion recognition using FER + DeepFace CV models", "Clinical"))
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.Tab("Webcam"):
                                cam_img     = gr.Image(sources=["webcam"], type="numpy", label="Capture Photo")
                                cam_emo_btn = gr.Button("Analyse Emotion", variant="primary")
                            with gr.Tab("Upload"):
                                up_img      = gr.Image(sources=["upload"], type="numpy", label="Upload Photo")
                                up_emo_btn  = gr.Button("Analyse Emotion", variant="primary")
                        gr.HTML('<div class="hm-tip" style="margin-bottom:12px"><strong>Tips:</strong> Face camera directly, good lighting, clear view. Results feed into AGI reasoning.</div>')
                        gr.HTML(_info_card("Detectable Emotions",
                            '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px">' +
                            "".join(f'<span style="font-size:11px;font-weight:500;padding:3px 10px;border-radius:20px;border:1px solid;background:{bg};color:{col};border-color:{bd}">{e}</span>'
                            for e,col,bg,bd in [
                                ("Happy","#065F46","#ECFDF5","#A7F3D0"),
                                ("Sad","#1E429F","#EBF0FF","#BFDBFE"),
                                ("Angry","#DC2626","#FEF2F2","#FECACA"),
                                ("Fear","#7C3AED","#EDE9FE","#DDD6FE"),
                                ("Surprise","#B45309","#FFFBEB","#FDE68A"),
                                ("Neutral","#64748B","#F0F2F5","#E2E8F0"),
                                ("Disgust","#0369A1","#E0F2FE","#BAE6FD"),
                            ]) + "</div>"
                        ))
                        gr.HTML(_info_card("Model Pipeline",
                            "<strong>Primary:</strong> FER (FER-2013 CNN) — fast, lightweight<br>"
                            "<strong>Fallback:</strong> DeepFace — robust for difficult images<br>"
                            "<strong>Routing:</strong> Confidence-based (threshold: 60%)"
                        ))
                    with gr.Column(scale=1):
                        emo_result = gr.Textbox(label="Analysis Result", lines=18, interactive=False)
                        emo_img    = gr.Image(label="Processed Image", type="filepath")
                cam_emo_btn.click(analyze_emotion_fn, [cam_img], [emo_result, emo_img])
                up_emo_btn.click(analyze_emotion_fn,  [up_img],  [emo_result, emo_img])

            # ══ TAB 4: BODY LANGUAGE ══════════════════════════════
            with gr.Tab("Body Language"):
                gr.HTML(_page_hd("Body Language", "Posture analysis using PyTorch KeypointRCNN (17 COCO body keypoints)", "Clinical"))
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.Tab("Webcam"):
                                pose_cam     = gr.Image(sources=["webcam"], type="numpy", label="Capture Full Body")
                                pose_cam_btn = gr.Button("Analyse Posture", variant="primary")
                            with gr.Tab("Upload"):
                                pose_up      = gr.Image(sources=["upload"], type="numpy", label="Upload Photo")
                                pose_up_btn  = gr.Button("Analyse Posture", variant="primary")
                        gr.HTML('<div class="hm-tip" style="margin-bottom:12px"><strong>Tips:</strong> Full body visible, good lighting, plain background.</div>')
                        gr.HTML(_info_card("Posture Classification",
                            '<div style="line-height:2.2">'
                            '<span style="font-weight:600;color:#059669">Confident</span> — Open stance, upright, head held high<br>'
                            '<span style="font-weight:600;color:#B45309">Tense</span> — Raised or uneven shoulders<br>'
                            '<span style="font-weight:600;color:#1A56DB">Neutral</span> — Relaxed, balanced posture<br>'
                            '<span style="font-weight:600;color:#DC2626">Slouched</span> — Forward head, rounded back'
                            '</div>'
                        ))
                        gr.HTML(_info_card("Detection Engine",
                            "<strong>Primary:</strong> PyTorch KeypointRCNN (17 COCO keypoints, GPU)<br>"
                            "<strong>Fallback:</strong> OpenCV face position analysis<br>"
                            "Result feeds into AGI multimodal reasoning."
                        ))
                    with gr.Column(scale=1):
                        pose_result = gr.Textbox(label="Posture Analysis", lines=18, interactive=False)
                        pose_img    = gr.Image(label="Annotated Image", type="filepath")
                pose_cam_btn.click(analyze_pose_fn, [pose_cam], [pose_result, pose_img])
                pose_up_btn.click(analyze_pose_fn,  [pose_up],  [pose_result, pose_img])

            # ══ TAB 5: THERAPIST FINDER ════════════════════════════
            with gr.Tab("Therapist Finder"):
                gr.HTML(_page_hd("Therapist Finder", "Locate licensed mental health professionals near you via Google Maps", "Support"))
                with gr.Row():
                    with gr.Column(scale=1):
                        location_input = gr.Textbox(label="Your Location", placeholder="e.g. Lahore, Pakistan", value="Lahore, Pakistan")
                        radius_input   = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Search Radius (km)")
                        find_btn       = gr.Button("Find Therapists", variant="primary")
                        gr.HTML(_info_card("Configuration Required",
                            'Add to <code>config.py</code>:<br><br>'
                            '<code style="background:#F0F2F5;padding:2px 6px;border-radius:4px">GOOGLE_MAPS_API_KEY = "your_key"</code><br><br>'
                            'Get free key at <a href="https://console.cloud.google.com" target="_blank" style="color:#1A56DB">console.cloud.google.com</a><br>'
                            'Enable: <strong>Places API</strong> + <strong>Geocoding API</strong>'
                        ))
                        gr.HTML('<div class="hm-alert hm-alert-blue"><strong>Remember:</strong> Seeking help is a sign of strength, not weakness.</div>')
                    with gr.Column(scale=2):
                        therapist_results = gr.HTML('<div style="color:var(--hm-text4);padding:20px;font-size:13px">Enter your location and click Find Therapists.</div>')
                find_btn.click(find_therapists_fn, [location_input, radius_input], [therapist_results])

            # ══ TAB 6: STATISTICS ══════════════════════════════════
            with gr.Tab("Statistics"):
                gr.HTML(_page_hd("Statistics", "Real-time session analytics and AGI decision tracking", "Analytics"))
                with gr.Row():
                    with gr.Column(scale=2):
                        stats_dashboard = gr.Markdown(get_stats_dashboard())
                        gr.Button("Refresh", variant="secondary").click(get_stats_dashboard, outputs=stats_dashboard)
                    with gr.Column(scale=1):
                        s = sm.stats()
                        gr.HTML(f"""
                        <div class="hm-stats-row" style="grid-template-columns:repeat(2,1fr)">
                          <div class="hm-stat"><div class="hm-stat-lbl">Messages</div><div class="hm-stat-val">{s['total_messages']}</div></div>
                          <div class="hm-stat"><div class="hm-stat-lbl">Emotions</div><div class="hm-stat-val">{s['emotions_detected']}</div></div>
                          <div class="hm-stat"><div class="hm-stat-lbl">Alerts</div><div class="hm-stat-val" style="color:#DC2626">{s['crisis_alerts']}</div></div>
                          <div class="hm-stat"><div class="hm-stat-lbl">Duration</div><div class="hm-stat-val hm-stat-sm">{s['duration']}</div></div>
                        </div>""")
                        gr.HTML(_info_card("Crisis Alert Routing",
                            '<div class="hm-feat-row"><span>Low severity</span><span style="color:#1E429F;font-size:11px;font-weight:500">WhatsApp SMS</span></div>'
                            '<div class="hm-feat-row"><span>Medium severity</span><span style="color:#B45309;font-size:11px;font-weight:500">WhatsApp SMS</span></div>'
                            '<div class="hm-feat-row"><span>High severity</span><span style="color:#DC2626;font-size:11px;font-weight:500">WhatsApp + Call</span></div>'
                        ))
                        gr.HTML(_info_card("Docker Deployment",
                            '<code style="display:block;background:#F0F2F5;padding:8px;border-radius:6px;font-size:11.5px;margin-bottom:6px">docker build -t healmatrix .</code>'
                            '<code style="display:block;background:#F0F2F5;padding:8px;border-radius:6px;font-size:11.5px;margin-bottom:6px">docker run -p 7860:7860 healmatrix</code>'
                            '<code style="display:block;background:#F0F2F5;padding:8px;border-radius:6px;font-size:11.5px">docker-compose up --build</code>'
                        ))

            # ══ TAB 7: EVALUATION ══════════════════════════════════
            with gr.Tab("Evaluation"):
                gr.HTML(_page_hd("Model Evaluation", "Training reports, performance metrics and dataset information", "Analytics"))

                # Load training reports
                import json as _json
                def _load_rpt(name):
                    p = BASE_DIR / "checkpoints" / name
                    if p.exists():
                        try: return _json.loads(p.read_text())
                        except: pass
                    return {}
                e_rpt = _load_rpt("emotion_training_report.json")
                p_rpt = _load_rpt("pose_training_report.json")
                r_rpt = _load_rpt("rag_training_report.json")

                with gr.Tabs():
                    with gr.Tab("Emotion Model"):
                        with gr.Row():
                            with gr.Column():
                                if e_rpt:
                                    wc = e_rpt.get("weight_change_proof",{}).get("avg_change",0)
                                    bef= e_rpt.get("weight_change_proof",{}).get("before",[])
                                    aft= e_rpt.get("weight_change_proof",{}).get("after",[])
                                    va = e_rpt.get("best_val_accuracy_pct",0)
                                    gr.HTML(f"""
                                    <div class="hm-alert hm-alert-green" style="margin-bottom:12px">
                                      <strong>Proof of Training</strong><br>
                                      Weight change: <code>{wc:.6f}</code> — model actually updated!
                                    </div>
                                    {_metric_bar("Validation Accuracy", round(va,1), "#059669")}
                                    {_info_card("Weight Change Proof",
                                      f"<strong>Before:</strong> <code>{bef[:4]}</code><br>"
                                      f"<strong>After:</strong> <code>{aft[:4]}</code><br>"
                                      f"<strong>Avg Change:</strong> <code>{wc:.6f}</code>"
                                    )}
                                    {_info_card("Training Config",
                                      f"Model: <strong>{e_rpt.get('model','MobileNetV2')}</strong><br>"
                                      f"Dataset: <strong>{e_rpt.get('dataset','FER-2013')}</strong><br>"
                                      f"Epochs: <strong>{e_rpt.get('epochs',5)}</strong><br>"
                                      f"Device: <strong>{e_rpt.get('device','GPU')}</strong><br>"
                                      f"Time: <strong>{e_rpt.get('training_minutes',0)} min</strong>"
                                    )}""")
                                else:
                                    gr.HTML('<div class="hm-alert hm-alert-blue">No training data. Run <code>python3 emotion_finetuning.py</code></div>')
                            with gr.Column():
                                ei = _img_path("emotion_model_evaluation.png")
                                if ei: gr.Image(value=ei, label="Evaluation Chart", type="filepath", interactive=False)
                                else: gr.HTML('<div class="hm-alert hm-alert-blue">Run evaluation notebook to generate charts.</div>')

                    with gr.Tab("Pose Model"):
                        with gr.Row():
                            with gr.Column():
                                if p_rpt:
                                    wc = p_rpt.get("weight_change_proof",{}).get("avg_change",0)
                                    va = p_rpt.get("best_val_accuracy_pct",0)
                                    gr.HTML(f"""
                                    <div class="hm-alert hm-alert-green" style="margin-bottom:12px">
                                      <strong>Proof of Training</strong><br>Weight change: <code>{wc:.6f}</code>
                                    </div>
                                    {_metric_bar("Validation Accuracy", round(va,1), "#7C3AED")}
                                    {_info_card("Training Config",
                                      f"Model: <strong>MobileNetV2 (MPII Pose)</strong><br>"
                                      f"Epochs: <strong>{p_rpt.get('epochs',4)}</strong><br>"
                                      f"Time: <strong>{p_rpt.get('training_minutes',0)} min</strong><br>"
                                      f"Device: <strong>{p_rpt.get('device','GPU')}</strong>"
                                    )}""")
                                else:
                                    gr.HTML('<div class="hm-alert hm-alert-blue">No training data. Run <code>python3 pose_finetuning.py</code></div>')
                            with gr.Column():
                                gr.HTML('<div class="hm-alert hm-alert-blue">Run evaluation notebook to generate charts.</div>')

                    with gr.Tab("RAG Model"):
                        with gr.Row():
                            with gr.Column():
                                if r_rpt:
                                    wc  = r_rpt.get("average_weight_change", r_rpt.get("weight_change",0))
                                    gr.HTML(f"""
                                    <div class="hm-alert hm-alert-green" style="margin-bottom:12px">
                                      <strong>Proof of Training</strong><br>Weight change: <code>{wc:.8f}</code>
                                    </div>
                                    {_info_card("BGE Fine-tuning Report",
                                      f"Model: <strong>{r_rpt.get('model','BAAI/bge-small-en-v1.5')}</strong><br>"
                                      f"Pairs: <strong>{r_rpt.get('training_pairs',0):,}</strong><br>"
                                      f"Epochs: <strong>{r_rpt.get('epochs',3)}</strong><br>"
                                      f"Time: <strong>{r_rpt.get('training_minutes',0)} min</strong>"
                                    )}
                                    {_info_card("Datasets Used",
                                      "<br>".join(f"• {d}" for d in r_rpt.get("datasets",[]))
                                    )}""")
                                else:
                                    gr.HTML('<div class="hm-alert hm-alert-blue">No training data. Run <code>python3 rag_finetuning.py</code></div>')
                            with gr.Column():
                                gr.HTML(_info_card("Total Dataset Summary",
                                    f"""<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px">
                                    {"".join(f'<div style="background:var(--hm-surface2);border-radius:8px;padding:10px;text-align:center"><div style="font-size:18px;font-weight:700;color:{col}">{val}</div><div style="font-size:11px;color:var(--hm-text3)">{lbl}</div></div>'
                                    for lbl,val,col in [
                                        ("Total Samples","207,338+","#1A56DB"),
                                        ("Dataset Size","7.86 GB","#059669"),
                                        ("Emotion Images","25K+","#1E429F"),
                                        ("RAG Conversations","182K+","#059669"),
                                        ("Pose Images","25K","#7C3AED"),
                                        ("Datasets Used","10","#B45309"),
                                    ])}</div>"""
                                ))

        gr.HTML(_FOOTER)

    return app


if __name__ == "__main__":
    import gradio as gr
    print(f"\n  Project : {BASE_DIR}")
    print(f"  Gradio  : {gr.__version__}")
    print("\n  Modules:")
    for k, v in _F.items():
        print(f"    {k:20s}: {'OK' if v else 'fallback'}")
    print("\n  Starting at http://localhost:7860 ...\n")
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=_CSS,
    )