"""
HealMatrix AI — Backend Server (Flask + ngrok)
Run:
    python backend.py            # terminal 2
    ngrok http 5000              # terminal 3
"""
#ssh -R 80:localhost:5000 nokey@localhost.run

import os
import json
import threading
from collections import defaultdict, deque
from datetime import datetime
from xml.sax.saxutils import escape

from flask import Flask, request, Response

# ── Config: env first, config.py as fallback ───────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import config as _cfg
except ImportError:
    _cfg = None


def _get(name, default=""):
    val = os.getenv(name)
    if val:
        return val
    if _cfg is not None:
        return getattr(_cfg, name, default)
    return default


TWILIO_ACCOUNT_SID     = _get("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN      = _get("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER    = _get("TWILIO_PHONE_NUMBER")
TWILIO_WHATSAPP_NUMBER = _get("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
EMERGENCY_CONTACT      = _get("EMERGENCY_CONTACT")
EMERGENCY_WHATSAPP     = _get("EMERGENCY_WHATSAPP")
FLASK_PORT             = int(_get("FLASK_PORT", "5000"))

# EMERGENCY_WHATSAPP must carry the whatsapp: prefix for messages.create()
if EMERGENCY_WHATSAPP and not EMERGENCY_WHATSAPP.startswith("whatsapp:"):
    EMERGENCY_WHATSAPP = "whatsapp:" + EMERGENCY_WHATSAPP

app = Flask(__name__)

os.makedirs("data/crisis_alerts", exist_ok=True)
os.makedirs("data/webhook_logs", exist_ok=True)

# ── Load the real AI modules (same ones main.py uses) ──────────────────────
_MOD = {}

try:
    from crisis_detection import get_crisis_severity, get_crisis_response_prefix
    _MOD["crisis"] = True
except ImportError as e:
    _MOD["crisis"] = False
    print(f"  crisis_detection unavailable: {e}")

    def get_crisis_severity(t):
        return "none"

    def get_crisis_response_prefix(s):
        return ""

try:
    from agi_engine import agi_query
    _MOD["agi"] = True
except ImportError as e:
    _MOD["agi"] = False
    print(f"  agi_engine unavailable: {e}")

try:
    from rag_system import get_relevant_context, query_with_rag
    _MOD["rag"] = True
except ImportError as e:
    _MOD["rag"] = False
    print(f"  rag_system unavailable: {e}")

try:
    from sentiment_analysis import analyze_sentiment
    _MOD["sentiment"] = True
except ImportError:
    _MOD["sentiment"] = False

# ── Per-sender state ──────────────────────────────────────────────────────
_history = defaultdict(lambda: deque(maxlen=16))   # sender -> recent turns
_crisis_log = defaultdict(list)                    # sender -> crisis events
_lock = threading.Lock()

WHATSAPP_LIMIT = 1500      # Twilio hard limit is 1600


def _log_webhook_event(name: str, payload: dict):
    path = f"data/webhook_logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"  [log] {e}")


def _send_crisis_alert(severity: str, message: str, sender: str):
    """Notify the emergency contact. Same behaviour as main.py."""
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and EMERGENCY_WHATSAPP):
        print("  [alert] Twilio not configured — skipped")
        return
    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=(f"HealMatrix Crisis Alert [{severity.upper()}]\n\n"
                  f"From: {sender}\nUser said: {message[:200]}\n"
                  f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
            from_=TWILIO_WHATSAPP_NUMBER,
            to=EMERGENCY_WHATSAPP,
        )
        print(f"  [alert] WhatsApp alert sent [{severity.upper()}]")

        if severity == "high" and TWILIO_PHONE_NUMBER and EMERGENCY_CONTACT:
            client.calls.create(
                twiml=('<Response><Say voice="Polly.Joanna">Emergency alert from Heal Matrix AI. '
                       'A person you are listed as an emergency contact for may be in crisis. '
                       'Please check on them as soon as possible.</Say></Response>'),
                from_=TWILIO_PHONE_NUMBER,
                to=EMERGENCY_CONTACT,
            )
            print("  [alert] Emergency voice call placed")
    except Exception as e:
        print(f"  [alert] Twilio failed: {e}")


def generate_reply(message: str, sender: str) -> str:
    """
    Full pipeline — identical to what the Gradio app does:
      crisis -> sentiment -> RAG -> AGI (Groq LLaMA) -> hotline prefix
    """
    severity = get_crisis_severity(message)

    if severity != "none":
        with _lock:
            _crisis_log[sender].append({"severity": severity, "message": message})
        _log_webhook_event("crisis", {"sender": sender, "severity": severity,
                                      "message": message,
                                      "timestamp": datetime.now().isoformat()})
        _send_crisis_alert(severity, message, sender)

    sentiment = None
    if _MOD["sentiment"]:
        try:
            s = analyze_sentiment(message)
            sentiment = s.get("sentiment") if s else None
        except Exception:
            pass

    rag_context = ""
    if _MOD["rag"]:
        try:
            chunks = get_relevant_context(message, k=3)
            if chunks:
                rag_context = "\n".join(f"• {c}" for c in chunks)
        except Exception as e:
            print(f"  [rag] {e}")

    ai_text = ""
    action = "GUIDE"
    if _MOD["agi"]:
        try:
            # CORRECT call: severity is required, history kwarg is
            # `conversation_history`, and the return value is a TUPLE.
            ai_text, action = agi_query(
                message=message,
                severity=severity,
                conversation_history=list(_history[sender]),
                emotion=None,          # no webcam over WhatsApp
                sentiment=sentiment,
                posture=None,
                crisis_history=_crisis_log[sender],
                rag_context=rag_context,
            )
        except Exception as e:
            print(f"  [agi] {e}")
            ai_text = ""

    if not ai_text and _MOD["rag"]:
        try:
            ai_text = query_with_rag(message)
        except Exception as e:
            print(f"  [rag fallback] {e}")

    if not ai_text:
        ai_text = ("I'm here for you. I'm having a brief technical issue — "
                   "please try again in a moment.\n\n"
                   "If you're in crisis: 988 (US) | 0800-00-002 (Pakistan)")

    with _lock:
        _history[sender].append({"role": "user", "content": message})
        _history[sender].append({"role": "assistant", "content": ai_text})

    print(f"  [reply] severity={severity} action={action} sentiment={sentiment}")
    return get_crisis_response_prefix(severity) + ai_text


# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    return {
        "status": "HealMatrix AI backend is running",
        "time": datetime.now().isoformat(),
        "modules": _MOD,
        "endpoints": {
            "/whatsapp/reply": "Webhook for incoming WhatsApp messages",
            "/voice/crisis": "TwiML for automated crisis voice call",
            "/status/call": "Call status callback",
        },
    }, 200


@app.route("/whatsapp/reply", methods=["POST"])
def whatsapp_reply():
    incoming_msg = (request.values.get("Body") or "").strip()
    from_number = request.values.get("From", "unknown")

    _log_webhook_event("whatsapp_reply", {
        "from": from_number, "message": incoming_msg,
        "timestamp": datetime.now().isoformat(),
    })
    print(f"[WHATSAPP IN] {from_number}: {incoming_msg[:80]}")

    if not incoming_msg:
        return Response('<?xml version="1.0" encoding="UTF-8"?><Response/>',
                        mimetype="text/xml")

    try:
        reply = generate_reply(incoming_msg, from_number)
    except Exception as e:
        print(f"[WHATSAPP ERROR] {e}")
        reply = ("I hit a technical problem. Please try again.\n\n"
                 "If you're in crisis: 988 (US) | 0800-00-002 (Pakistan)")

    if len(reply) > WHATSAPP_LIMIT:
        reply = reply[:WHATSAPP_LIMIT].rsplit(" ", 1)[0] + "..."

    print(f"[WHATSAPP OUT] {reply[:90]}...")

    twiml = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
             f'<Response>\n    <Message>{escape(reply)}</Message>\n</Response>')
    return Response(twiml, mimetype="text/xml")


@app.route("/voice/crisis", methods=["GET", "POST"])
def voice_crisis():
    severity = request.values.get("severity", "HIGH")
    _log_webhook_event("voice_crisis", {
        "severity": severity,
        "from": request.values.get("From"),
        "to": request.values.get("To"),
        "call_sid": request.values.get("CallSid"),
        "timestamp": datetime.now().isoformat(),
    })

    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">
        This is an automated alert from Heal Matrix AI.
        A person you are listed as an emergency contact for
        may be experiencing a mental health crisis.
        Please check on them as soon as possible.
    </Say>
    <Pause length="1"/>
    <Say voice="Polly.Joanna">
        This message will now repeat.
    </Say>
    <Pause length="1"/>
    <Say voice="Polly.Joanna">
        This is an automated alert from Heal Matrix AI.
        Please check on your emergency contact as soon as possible.
    </Say>
</Response>"""
    return Response(twiml, mimetype="text/xml")


@app.route("/status/call", methods=["POST"])
def call_status():
    status_data = {
        "call_sid": request.values.get("CallSid"),
        "call_status": request.values.get("CallStatus"),
        "to": request.values.get("To"),
        "from": request.values.get("From"),
        "duration": request.values.get("CallDuration"),
        "timestamp": datetime.now().isoformat(),
    }
    _log_webhook_event("call_status", status_data)
    print(f"[CALL STATUS] {status_data['call_status']} — SID: {status_data['call_sid']}")
    return ("", 204)


@app.route("/test", methods=["GET"])
def test_endpoint():
    return {"ok": True, "message": "Backend reachable via ngrok", "modules": _MOD}, 200


if __name__ == "__main__":
    print("=" * 62)
    print("  HealMatrix AI — Backend Server")
    print("=" * 62)
    for k, v in _MOD.items():
        print(f"  {k:<12}: {'OK' if v else 'UNAVAILABLE'}")
    print("-" * 62)
    print(f"  Local:            http://localhost:{FLASK_PORT}")
    print("  WhatsApp webhook: /whatsapp/reply")
    print("  Voice webhook:    /voice/crisis")
    print("  Status callback:  /status/call")
    print("=" * 62)
    print("  NEXT: in another terminal run:  ngrok http 5000")
    print("  Then paste  <ngrok-url>/whatsapp/reply  into")
    print("  Twilio Console -> Messaging -> Try it out ->")
    print("  Send a WhatsApp message -> Sandbox settings")
    print("  ('WHEN A MESSAGE COMES IN', method POST)")
    print("=" * 62)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)
