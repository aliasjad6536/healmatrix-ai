"""
HealMatrix AI — Emotional Wellness Intelligence System
Professional Light Theme UI | v1.5
Fixes: text input visibility, adds language selector + screen recording
FIXED: safe_analyze_body_language() no longer fails when mediapipe is
       absent (e.g. inside the Docker image) — it now falls through to
       the real PyTorch KeypointRCNN engine in pose_detection.py, which
       never needed mediapipe in the first place.
"""

import json, os, tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image

# ── Directories ─────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
DATA_DIR          = BASE_DIR / "data"
CHAT_LOGS_DIR     = DATA_DIR / "chat_logs"
EMOTIONS_DIR      = DATA_DIR / "emotions"
CRISIS_ALERTS_DIR = DATA_DIR / "crisis_alerts"
SESSION_DIR       = DATA_DIR / "session"
for _d in [CHAT_LOGS_DIR, EMOTIONS_DIR, CRISIS_ALERTS_DIR, SESSION_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Config ──────────────────────────────────────────────────────
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

try:
    from groq import Groq as _Groq
    _groq_client = _Groq(api_key=GROQ_API_KEY)
except Exception:
    _groq_client = None

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

# ── Load AI modules ─────────────────────────────────────────────
print("=" * 60)
print("  HealMatrix AI v1.5 — loading modules ...")
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

_F["twilio"] = bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)

# ── Fallback functions ──────────────────────────────────────────
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

# ── Pose detection safety wrapper (mediapipe optional, FIXED) ───
# mediapipe is LEGACY / OPTIONAL here. The real pose engine
# (PyTorch KeypointRCNN, in pose_detection.py) does not need mediapipe
# at all. Previously, a missing mediapipe install (e.g. inside the
# Docker image, which does not include it) caused the whole function
# to fail before ever reaching analyze_body_language(), because the
# `import mediapipe` line and the real call were both inside one
# try/except block. This version isolates the optional mediapipe
# patch from the real call, so a missing mediapipe package no longer
# breaks pose detection.
def safe_analyze_body_language(image_path: str):
    try:
        import mediapipe as mp
        if not hasattr(mp, 'solutions'):
            try:
                from mediapipe.python.solutions import pose as mp_pose_mod
                from mediapipe.python.solutions import drawing_utils
                from mediapipe.python.solutions import drawing_styles
                class _Solutions:
                    pose = mp_pose_mod
                    drawing_utils = drawing_utils
                    drawing_styles = drawing_styles
                    class PoseLandmark: pass
                mp.solutions = _Solutions()
            except Exception:
                pass
    except ImportError:
        # mediapipe not installed (e.g. in the Docker image) — this is
        # fine, the real KeypointRCNN engine below does not need it.
        pass

    try:
        return analyze_body_language(image_path)
    except AttributeError as e:
        if "solutions" in str(e):
            return _fallback_pose_analysis(image_path)
        return f"Pose error: {e}", None, {}
    except Exception as e:
        return f"Pose error: {e}", None, {}

def _fallback_pose_analysis(image_path: str):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Could not load image.", None, {}
        h, w = img.shape[:2]
        upper = img[:h//2, :]
        gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return ("No person clearly detected.\n\nTips:\n"
                    "  - Ensure full upper body is visible\n"
                    "  - Use good lighting\n"
                    "  - Face the camera directly", None, {})
        fx, fy, fw, fh = faces[0]
        offset = abs((fx + fw // 2) - w // 2) / w
        if offset < 0.1:
            posture, desc, tip = "confident", "Your posture appears confident and centered.", "Great posture. Keep it up."
        elif offset < 0.2:
            posture, desc, tip = "neutral", "Your posture appears relaxed and neutral.", "You look comfortable."
        else:
            posture, desc, tip = "tense", "Your posture suggests some tension.", "Try rolling your shoulders back and taking three deep breaths."
        result = f"Detected Posture: {posture.capitalize()}\n\n{desc}\n\nTip: {tip}\n\nEngine: OpenCV (MediaPipe fallback mode)"
        out = img.copy()
        cv2.rectangle(out, (fx, fy), (fx+fw, fy+fh), (0,255,0), 2)
        cv2.putText(out, f"Posture: {posture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out_path = image_path.replace(".jpg", "_pose_annotated.jpg")
        cv2.imwrite(out_path, out)
        return result, out_path, {"posture": posture}
    except Exception as e:
        return f"Pose analysis failed: {e}", None, {}

# ── Session manager ──────────────────────────────────────────────
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

def _load_json(path):
    try:
        with open(path) as f: return json.load(f)
    except Exception: return {}

def _img_path(name):
    p = DATA_DIR / name
    return str(p) if p.exists() else None

# ── Language support ────────────────────────────────────────────
LANGUAGES = ["English", "Urdu", "Roman Urdu", "Arabic", "Spanish", "French"]

def _apply_language(message: str, language: str) -> str:
    if not language or language == "English":
        return message
    return f"{message}\n\n[System instruction: respond in {language} language.]"

# ── Core functions ───────────────────────────────────────────────
def chat_with_ai(message, history, tts_enabled, language="English"):
    if not message or not message.strip():
        return history, "", None, ""

    severity = get_crisis_severity(message)
    is_crisis = severity != "none"

    if is_crisis:
        sm.log_crisis(severity, message)
        send_twilio_alert(severity, message)

    sent_result = None
    if _F.get("sentiment"):
        sent_result = analyze_sentiment(message)
        sm.log_sentiment(sent_result["sentiment"], sent_result["confidence"])

    rag_context = ""
    if _F.get("rag"):
        try:
            chunks = get_relevant_context(message, k=3)
            rag_context = "\n".join(f"- {c}" for c in chunks)
        except Exception:
            pass

    message_for_ai = _apply_language(message, language)

    try:
        ai_text, action = agi_query(
            message=message_for_ai,
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
        ai_text = query_with_rag(message_for_ai) if _F.get("rag") else f"I'm here to help. (Error: {e})"
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


def chat_with_voice(audio, history, tts_enabled, language="English"):
    if audio is None:
        return history, "", None, "", "No audio recorded."
    audio_path = save_gradio_audio(audio)
    if not audio_path:
        return history, "", None, "", "Could not process audio."
    transcribed = transcribe_audio(audio_path)
    if not transcribed or transcribed.startswith("["):
        return history, "", None, "", f"Transcription failed: {transcribed}"
    new_history, _, audio_out, action_html = chat_with_ai(transcribed, history, tts_enabled, language)
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
        "### Live Session Statistics\n",
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
        "I am Dr. Emily Hartman, your AI mental health companion.\n\n"
        "You can type your message or use Voice to speak.\n\n"
        "How are you feeling today?"
    ).format(sid)
    sm.add("assistant", welcome)
    return [{"role": "assistant", "content": welcome}], stats_str()


# ══════════════════════════════════════════════════════════════
# CSS DESIGN SYSTEM — Clinical SaaS light theme (unchanged palette)
# ══════════════════════════════════════════════════════════════

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

* { box-sizing: border-box; }

html, body { color-scheme: light !important; }

.gradio-container {
  background: var(--hm-bg) !important;
  font-family: 'Inter', -apple-system, sans-serif !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  color: var(--hm-text) !important;
  -webkit-font-smoothing: antialiased !important;
  color-scheme: light !important;
}
footer { display: none !important; }
.gradio-container > .contain { padding: 0 !important; max-width: 100% !important; }

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
.hm-status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--hm-green); animation: hm-pulse 2.5s infinite; }
@keyframes hm-pulse {
  0%  { box-shadow: 0 0 0 0 rgba(5,150,105,0.4); }
  70% { box-shadow: 0 0 0 6px rgba(5,150,105,0); }
  100%{ box-shadow: 0 0 0 0 rgba(5,150,105,0); }
}
.hm-vdiv { width: 1px; height: 20px; background: var(--hm-border); }
.hm-crisis-top { display: flex; align-items: center; gap: 5px; font-size: 11.5px; }
.hm-crisis-lbl { font-size: 10px; font-weight: 600; color: var(--hm-text4); text-transform: uppercase; letter-spacing: 0.6px; }
.hm-crisis-num { font-weight: 700; color: var(--hm-red); font-size: 12px; }

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
.gradio-container .tab-nav button:hover { color: var(--hm-text2) !important; background: var(--hm-surface2) !important; }
.gradio-container .tab-nav button.selected {
  color: var(--hm-blue) !important;
  border-bottom-color: var(--hm-blue) !important;
  background: transparent !important;
  font-weight: 600 !important;
}

.hm-page-hd { margin-bottom: 22px; }
.hm-crumb { display: flex; align-items: center; gap: 5px; font-size: 11.5px; color: var(--hm-text4); margin-bottom: 5px; }
.hm-crumb-active { color: var(--hm-text3); }
.hm-page-title { font-size: 20px; font-weight: 600; color: var(--hm-text); letter-spacing: -0.5px; margin: 0 0 3px; }
.hm-page-sub { font-size: 12.5px; color: var(--hm-text4); margin: 0; }

.hm-stats-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
.hm-stat { background: var(--hm-surface); border: 1px solid var(--hm-border); border-radius: var(--hm-r); padding: 15px 17px; box-shadow: var(--hm-shadow); }
.hm-stat-lbl { font-size: 10.5px; font-weight: 600; color: var(--hm-text4); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 7px; }
.hm-stat-val { font-size: 26px; font-weight: 600; color: var(--hm-text); letter-spacing: -1px; line-height: 1; }
.hm-stat-sm  { font-size: 15px; padding-top: 5px; }
.hm-stat-sub { font-size: 11px; color: var(--hm-text4); margin-top: 5px; }
.hm-sub-green { color: var(--hm-green) !important; }
.hm-sub-red   { color: var(--hm-red) !important; }

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
.hm-cnum { font-size: 11px; font-weight: 600; color: var(--hm-red); background: #fff; border: 1px solid #FECACA; padding: 3px 10px; border-radius: 20px; }

.hm-card { background: var(--hm-surface); border: 1px solid var(--hm-border); border-radius: var(--hm-r); overflow: hidden; margin-bottom: 12px; box-shadow: var(--hm-shadow); }
.hm-card-hd { padding: 10px 14px; border-bottom: 1px solid var(--hm-border); font-size: 10.5px; font-weight: 600; color: var(--hm-text3); text-transform: uppercase; letter-spacing: 0.6px; background: var(--hm-surface2); }
.hm-card-bd { padding: 13px 14px; font-size: 12.5px; color: var(--hm-text2); line-height: 1.85; }

.hm-feat-row { display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: var(--hm-text2); padding: 4px 0; border-bottom: 1px solid rgba(15,23,42,0.04); }
.hm-feat-row:last-child { border-bottom: none; }
.hm-feat-ok { font-size: 10px; font-weight: 600; color: var(--hm-green); background: var(--hm-green-pale); border: 1px solid #A7F3D0; padding: 1px 7px; border-radius: 10px; }
.hm-feat-off { font-size: 10px; font-weight: 600; color: var(--hm-text4); background: var(--hm-surface2); border: 1px solid var(--hm-border); padding: 1px 7px; border-radius: 10px; }

.hm-agi-wrap { padding: 8px 14px; border-top: 1px solid var(--hm-border); background: var(--hm-surface2); min-height: 36px; }

.hm-tip { background: var(--hm-blue-pale); border: 1px solid var(--hm-blue-mid); border-radius: var(--hm-r2); padding: 9px 13px; font-size: 12px; color: var(--hm-blue2); margin-top: 10px; }
.hm-tip strong { color: var(--hm-blue); }

.hm-alert { border-radius: var(--hm-r2); padding: 10px 13px; font-size: 12.5px; margin-bottom: 12px; border: 1px solid; }
.hm-alert-red   { background: var(--hm-red-pale);    border-color: #FECACA; color: #7F1D1D; }
.hm-alert-blue  { background: var(--hm-blue-pale);   border-color: var(--hm-blue-mid); color: var(--hm-blue2); }
.hm-alert-green { background: var(--hm-green-pale);  border-color: #A7F3D0; color: #065F46; }

.hm-metric { margin-bottom: 13px; }
.hm-metric-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.hm-metric-name { font-size: 12.5px; color: var(--hm-text2); }
.hm-metric-pct  { font-size: 14px; font-weight: 600; color: var(--hm-text); }
.hm-metric-track { height: 5px; background: var(--hm-surface3); border-radius: 3px; overflow: hidden; }
.hm-metric-fill  { height: 100%; border-radius: 3px; }

.hm-footer { background: var(--hm-surface); border-top: 1px solid var(--hm-border); padding: 14px 32px; display: flex; align-items: center; justify-content: space-between; font-size: 11.5px; color: var(--hm-text4); flex-wrap: wrap; gap: 8px; margin-top: 32px; }
.hm-footer-brand { font-weight: 600; color: var(--hm-text3); font-size: 12px; }
.hm-footer-crisis strong { color: var(--hm-red); }

/* ── TEXT INPUT VISIBILITY FIX ────────────────────────────────
   Forces readable dark text on light background in every state,
   overriding Gradio's automatic dark-mode class. ────────────── */
.gradio-container input[type="text"],
.gradio-container textarea,
.gradio-container input {
  background: var(--hm-surface2) !important;
  border: 1px solid var(--hm-border) !important;
  border-radius: var(--hm-r2) !important;
  color: var(--hm-text) !important;
  caret-color: var(--hm-text) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13.5px !important;
  transition: all 0.15s !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
  border-color: var(--hm-blue) !important;
  background: #ffffff !important;
  color: var(--hm-text) !important;
  box-shadow: 0 0 0 3px rgba(26,86,219,0.08) !important;
  outline: none !important;
}
.dark .gradio-container input[type="text"],
.dark .gradio-container textarea,
body.dark .gradio-container input,
body.dark .gradio-container textarea {
  background: var(--hm-surface2) !important;
  color: var(--hm-text) !important;
  caret-color: var(--hm-text) !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder { color: var(--hm-text4) !important; opacity: 1 !important; }

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
.gradio-container .chatbot { background: var(--hm-bg) !important; border: 1px solid var(--hm-border) !important; border-radius: var(--hm-r) !important; }
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

# ── head script: forces light mode + screen recorder logic ─────
_HEAD = """
<script>
document.addEventListener('DOMContentLoaded', function() {
  document.body.classList.remove('dark');
  document.documentElement.classList.remove('dark');
});
let hmRecorder, hmChunks = [], hmStream;
async function hmStartRecording() {
  const statusEl = document.getElementById('hm-rec-status');
  try {
    hmStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
    hmChunks = [];
    hmRecorder = new MediaRecorder(hmStream);
    hmRecorder.ondataavailable = function(e) { if (e.data.size > 0) hmChunks.push(e.data); };
    hmRecorder.onstop = function() {
      const blob = new Blob(hmChunks, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'healmatrix_demo_' + Date.now() + '.webm';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      hmStream.getTracks().forEach(function(t) { t.stop(); });
      if (statusEl) statusEl.innerText = 'Recording saved to your Downloads folder.';
    };
    hmRecorder.start();
    if (statusEl) statusEl.innerText = 'Recording in progress. Click Stop and save when done.';
  } catch (err) {
    if (statusEl) statusEl.innerText = 'Could not start recording: ' + err.message;
  }
}
function hmStopRecording() {
  if (hmRecorder && hmRecorder.state !== 'inactive') { hmRecorder.stop(); }
}
</script>
"""

# ── HTML components ──────────────────────────────────────────
_TOPBAR = """
<div class="hm-topbar">
  <div class="hm-brand">
    <div class="hm-brand-mark">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    </div>
    <span class="hm-brand-name">HealMatrix AI</span>
    <span class="hm-brand-ver">v1.5</span>
  </div>
  <div class="hm-topbar-right">
    <div class="hm-status"><span class="hm-status-dot"></span>All systems operational</div>
    <div class="hm-vdiv"></div>
    <div class="hm-crisis-top">
      <span class="hm-crisis-lbl">Crisis</span>
      <span class="hm-crisis-num">988</span>
      <span style="color:#94A3B8">.</span>
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
  <span class="hm-footer-brand">HealMatrix AI &middot; v1.5 &middot; 2026</span>
  <span class="hm-footer-crisis">Crisis: <strong>988</strong> (USA) &middot; <strong>0800-00-002</strong> (Pakistan) &middot; Text HOME to 741741</span>
  <span style="font-size:10.5px">Gradio &middot; Groq LLaMA 3.3 &middot; Whisper &middot; DeepFace &middot; FAISS &middot; RoBERTa &middot; gTTS &middot; Twilio</span>
</div>
"""

_SCREEN_REC_HTML = """
<div class="hm-card">
  <div class="hm-card-hd">Screen recording</div>
  <div class="hm-card-bd">
    Record your screen directly in the browser for demo videos, thesis defense recordings, or bug reports.
    No installation needed — uses your browser's built-in screen capture.
    <div style="margin-top:14px; display:flex; gap:10px;">
      <button onclick="hmStartRecording()" style="background:#0B1D3A;color:#fff;border:none;padding:9px 18px;border-radius:8px;font-family:Inter,sans-serif;font-size:13px;font-weight:500;cursor:pointer;">Start recording</button>
      <button onclick="hmStopRecording()" style="background:#FFFFFF;color:#334155;border:1px solid #E2E8F0;padding:9px 18px;border-radius:8px;font-family:Inter,sans-serif;font-size:13px;font-weight:500;cursor:pointer;">Stop and save</button>
    </div>
    <p id="hm-rec-status" style="margin-top:12px;font-size:12.5px;color:#64748B;">Click Start recording, then choose the screen, tab, or window to capture in the browser prompt. Click Stop and save when finished — the video downloads automatically as a .webm file.</p>
    <div class="hm-tip"><strong>Tip:</strong> Choose "This tab" when the prompt appears if you only want the HealMatrix interface recorded.</div>
  </div>
</div>
"""

def _page_hd(title, subtitle, crumb="Clinical"):
    return f"""
    <div class="hm-page-hd">
      <div class="hm-crumb">
        <span>HealMatrix</span><span>&rsaquo;</span><span>{crumb}</span><span>&rsaquo;</span>
        <span class="hm-crumb-active">{title}</span>
      </div>
      <h1 class="hm-page-title">{title}</h1>
      <p class="hm-page-sub">{subtitle}</p>
    </div>"""

def _stats_html(msgs, dur, alerts, emotions):
    return f"""
    <div class="hm-stats-row">
      <div class="hm-stat"><div class="hm-stat-lbl">Messages</div><div class="hm-stat-val">{msgs}</div><div class="hm-stat-sub hm-sub-green">This session</div></div>
      <div class="hm-stat"><div class="hm-stat-lbl">Duration</div><div class="hm-stat-val">{dur}</div><div class="hm-stat-sub">Active</div></div>
      <div class="hm-stat"><div class="hm-stat-lbl">Emotions</div><div class="hm-stat-val">{emotions}</div><div class="hm-stat-sub">Detected</div></div>
      <div class="hm-stat"><div class="hm-stat-lbl">Crisis Alerts</div><div class="hm-stat-val" style="color:{'#DC2626' if alerts>0 else '#0F172A'}">{alerts}</div><div class="hm-stat-sub {'hm-sub-red' if alerts>0 else ''}">{'Alert sent' if alerts>0 else 'None'}</div></div>
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
    return f'<div class="hm-card"><div class="hm-card-hd">Active modules</div><div class="hm-card-bd" style="padding:10px 14px">{rows}</div></div>'

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
    return f'<div class="hm-card"><div class="hm-card-hd">AGI decision matrix</div><div class="hm-card-bd" style="padding:10px 14px">{rows_html}</div></div>'

def _info_card(title, content):
    return f'<div class="hm-card"><div class="hm-card-hd">{title}</div><div class="hm-card-bd">{content}</div></div>'

def _metric_bar(name, pct, color="#1A56DB"):
    return f"""<div class="hm-metric">
      <div class="hm-metric-top"><span class="hm-metric-name">{name}</span><span class="hm-metric-pct">{pct}%</span></div>
      <div class="hm-metric-track"><div class="hm-metric-fill" style="width:{pct}%;background:{color}"></div></div>
    </div>"""


# ── Build UI ─────────────────────────────────────────────────
def build_app():
    with gr.Blocks(title="HealMatrix AI", head=_HEAD) as app:

        gr.HTML(_TOPBAR)
        sm.create()

        with gr.Tabs():

            # ══ TAB 1: THERAPY CHAT ═══════════════════════════
            with gr.Tab("Therapy Chat"):
                gr.HTML(_page_hd("Therapy Chat", "Talk with Dr. Emily Hartman — AGI-powered clinical reasoning", "Clinical"))
                gr.HTML(_CRISIS_BANNER)

                stats_out = gr.HTML(_stats_html(0,"0m",0,0))

                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            value=[{"role":"assistant","content":(
                                "Hello, I am Dr. Emily Hartman.\n\n"
                                "I am your AI mental health companion powered by AGI reasoning — "
                                "trained in CBT, DBT, ACT, and mindfulness-based therapy.\n\n"
                                "Everything you share here is private and confidential.\n\n"
                                "How are you feeling today?"
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
                                voice_btn   = gr.Button("Transcribe and send", variant="primary")
                                voice_status= gr.Textbox(label="Transcription", interactive=False, lines=1)

                        with gr.Row():
                            language_dd = gr.Dropdown(choices=LANGUAGES, value="English", label="Response language", scale=1)
                            tts_toggle = gr.Checkbox(label="Read response aloud (Text-to-Speech)", value=False, scale=1)
                        tts_audio  = gr.Audio(label="AI Voice Response", type="filepath", autoplay=True)

                    with gr.Column(scale=1):
                        gr.Button("New Session", variant="secondary").click(
                            lambda: (new_session()[0], _stats_html(0,"0m",0,0), ""),
                            outputs=[chatbot, stats_out, agi_display]
                        )
                        gr.HTML(_agi_matrix_card())
                        gr.HTML(_features_card_new(_F))

                def _chat_update(message, history, tts, language):
                    h, msg, audio, agi_html = chat_with_ai(message, history, tts, language)
                    s = sm.stats()
                    stats = _stats_html(s["total_messages"], s["duration"], s["crisis_alerts"], s["emotions_detected"])
                    import re
                    m = re.search(r'AGI: (\w+)', agi_html or "")
                    badge = f'<div class="hm-agi-wrap">{_agi_badge_new(m.group(1))}</div>' if m else '<div class="hm-agi-wrap"></div>'
                    return h, msg, audio, badge, stats

                def _voice_update(audio, history, tts, language):
                    h, msg, audio_out, agi_html, status = chat_with_voice(audio, history, tts, language)
                    s = sm.stats()
                    stats = _stats_html(s["total_messages"], s["duration"], s["crisis_alerts"], s["emotions_detected"])
                    import re
                    m = re.search(r'AGI: (\w+)', agi_html or "")
                    badge = f'<div class="hm-agi-wrap">{_agi_badge_new(m.group(1))}</div>' if m else '<div class="hm-agi-wrap"></div>'
                    return h, msg, audio_out, badge, status, stats

                msg_box.submit(_chat_update, [msg_box, chatbot, tts_toggle, language_dd], [chatbot, msg_box, tts_audio, agi_display, stats_out])
                send_btn.click(_chat_update,  [msg_box, chatbot, tts_toggle, language_dd], [chatbot, msg_box, tts_audio, agi_display, stats_out])
                voice_btn.click(_voice_update, [voice_input, chatbot, tts_toggle, language_dd], [chatbot, msg_box, tts_audio, agi_display, voice_status, stats_out])

            # ══ TAB 2: SENTIMENT ═══════════════════════════════
            with gr.Tab("Sentiment Analysis"):
                gr.HTML(_page_hd("Sentiment Analysis", "Emotional tone detection using RoBERTa transformer model", "Clinical"))
                with gr.Row():
                    with gr.Column(scale=1):
                        sentiment_input = gr.Textbox(
                            label="Text to analyse",
                            placeholder="Type or paste any message here...",
                            lines=7
                        )
                        sentiment_btn = gr.Button("Analyse Sentiment", variant="primary")
                        gr.HTML(_info_card("About this module",
                            "Uses cardiffnlp/twitter-roberta-base-sentiment-latest — "
                            "a RoBERTa model trained on 124M real-world texts. Detects positive, negative, "
                            "or neutral tone with confidence scores."
                        ))
                    with gr.Column(scale=1):
                        sentiment_result = gr.Textbox(label="Analysis result", lines=18, interactive=False)
                sentiment_btn.click(run_sentiment, [sentiment_input], [sentiment_result])
                sentiment_input.submit(run_sentiment, [sentiment_input], [sentiment_result])

            # ══ TAB 3: EMOTION DETECTION ═══════════════════════
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
                        gr.HTML(_info_card("Model pipeline",
                            "Primary: FER-2013 CNN — fast, lightweight<br>"
                            "Fallback: DeepFace — robust for difficult images"
                        ))
                    with gr.Column(scale=1):
                        emo_result = gr.Textbox(label="Analysis Result", lines=18, interactive=False)
                        emo_img    = gr.Image(label="Processed Image", type="filepath")
                cam_emo_btn.click(analyze_emotion_fn, [cam_img], [emo_result, emo_img])
                up_emo_btn.click(analyze_emotion_fn,  [up_img],  [emo_result, emo_img])

            # ══ TAB 4: BODY LANGUAGE ═══════════════════════════
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
                    with gr.Column(scale=1):
                        pose_result = gr.Textbox(label="Posture Analysis", lines=18, interactive=False)
                        pose_img    = gr.Image(label="Annotated Image", type="filepath")
                pose_cam_btn.click(analyze_pose_fn, [pose_cam], [pose_result, pose_img])
                pose_up_btn.click(analyze_pose_fn,  [pose_up],  [pose_result, pose_img])

            # ══ TAB 5: THERAPIST FINDER ════════════════════════
            with gr.Tab("Therapist Finder"):
                gr.HTML(_page_hd("Therapist Finder", "Locate licensed mental health professionals near you via Google Maps", "Support"))
                with gr.Row():
                    with gr.Column(scale=1):
                        location_input = gr.Textbox(label="Your Location", placeholder="e.g. Lahore, Pakistan", value="Lahore, Pakistan")
                        radius_input   = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Search Radius (km)")
                        find_btn       = gr.Button("Find Therapists", variant="primary")
                    with gr.Column(scale=2):
                        therapist_results = gr.HTML('<div style="color:var(--hm-text4);padding:20px;font-size:13px">Enter your location and click Find Therapists.</div>')
                find_btn.click(find_therapists_fn, [location_input, radius_input], [therapist_results])

            # ══ TAB 6: SCREEN RECORDING (NEW) ═══════════════════
            with gr.Tab("Screen Recording"):
                gr.HTML(_page_hd("Screen Recording", "Record a live demo of the interface for reports or presentations", "Utility"))
                gr.HTML(_SCREEN_REC_HTML)

            # ══ TAB 7: STATISTICS ══════════════════════════════
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

        gr.HTML(_FOOTER)

    return app


if __name__ == "__main__":
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