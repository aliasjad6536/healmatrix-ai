"""
HealMatrix AI — Backend Server (Flask + ngrok)
Serves TwiML for Twilio crisis voice calls.
"""

from flask import Flask, request, Response
from datetime import datetime
import json
import os

app = Flask(__name__)

os.makedirs("data/crisis_alerts", exist_ok=True)
os.makedirs("data/webhook_logs", exist_ok=True)


def _log_webhook_event(name: str, payload: dict):
    path = f"data/webhook_logs/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


@app.route("/", methods=["GET"])
def health_check():
    return {
        "status": "HealMatrix AI backend is running",
        "time": datetime.now().isoformat(),
        "endpoints": {
            "/voice/crisis": "TwiML for automated crisis voice call",
            "/whatsapp/reply": "Webhook for incoming WhatsApp replies",
            "/status/call": "Call status callback",
        },
    }, 200


@app.route("/voice/crisis", methods=["GET", "POST"])
def voice_crisis():
    severity = request.values.get("severity", "HIGH")

    _log_webhook_event(
        "voice_crisis",
        {
            "severity": severity,
            "from": request.values.get("From"),
            "to": request.values.get("To"),
            "call_sid": request.values.get("CallSid"),
            "timestamp": datetime.now().isoformat(),
        },
    )

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


@app.route("/whatsapp/reply", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")

    _log_webhook_event(
        "whatsapp_reply",
        {
            "from": from_number,
            "message": incoming_msg,
            "timestamp": datetime.now().isoformat(),
        },
    )

    print(f"[WHATSAPP REPLY] {from_number}: {incoming_msg}")

    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Thank you for confirming. HealMatrix AI has logged your response.</Message>
</Response>"""
    return Response(twiml, mimetype="text/xml")


@app.route("/test", methods=["GET"])
def test_endpoint():
    return {"ok": True, "message": "Backend reachable via ngrok"}, 200


if __name__ == "__main__":
    print("=" * 60)
    print("  HealMatrix AI — Backend Server")
    print("=" * 60)
    print("  Local:  http://localhost:5000")
    print("  Voice webhook:    /voice/crisis")
    print("  Status callback:  /status/call")
    print("  WhatsApp webhook: /whatsapp/reply")
    print("=" * 60)
    print("  NEXT: open a second terminal and run:")
    print("      ngrok http 5000")
    print("  Then copy the https:// forwarding URL into config.py")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
