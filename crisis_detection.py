import re
from typing import Tuple

_HIGH = [
    "kill myself", "killing myself", "end my life", "take my life",
    "want to die tonight", "going to kill", "plan to suicide",
    "already have a plan", "already decided", "goodbye forever",
    "last message", "won't be here anymore", "end it tonight",
    "overdose tonight", "jump off a", "hang myself",
    "shoot myself", "slit my wrists",
]

_MEDIUM = [
    "suicide", "suicidal", "want to die", "wish i was dead",
    "wish i were dead", "better off dead", "no reason to live",
    "can't go on", "can't keep going", "self-harm", "self harm",
    "hurt myself", "cut myself", "cutting myself", "overdose",
    "end it all", "end everything", "disappear forever",
    "don't want to be here", "done with life", "tired of living",
    "not worth living", "life is not worth", "pointless to live",
    "nobody would miss me", "everyone would be better without me",
]

_LOW = [
    "hopeless", "helpless", "no hope left", "giving up",
    "can't take it anymore", "can't cope", "falling apart",
    "breaking down", "at rock bottom", "nothing matters",
    "don't care anymore", "exhausted with life",
    "feel empty inside", "feel completely numb", "feel nothing",
    "so alone", "nobody cares", "trapped with no way out",
]


def detect_crisis(text: str) -> bool:
    """Return True if any crisis signal is found in *text*."""
    if not text:
        return False
    t = text.lower()
    return any(kw in t for kw in _HIGH + _MEDIUM + _LOW)


def get_crisis_severity(text: str) -> str:
    """
    Return 'none' | 'low' | 'medium' | 'high'.
    """
    if not text:
        return "none"
    t = text.lower()
    if any(kw in t for kw in _HIGH):
        return "high"
    if any(kw in t for kw in _MEDIUM):
        return "medium"
    if any(kw in t for kw in _LOW):
        return "low"
    return "none"


def get_crisis_response_prefix(severity: str) -> str:
    """Return a ready-to-prepend crisis message block for the given severity."""
    if severity == "high":
        return (
            "\n**IMMEDIATE CRISIS SUPPORT**\n\n"
            "I'm very concerned about your safety right now. "
            "Please reach out for help immediately:\n\n"
            "-  **988** — Suicide & Crisis Lifeline (call or text, 24/7)\n"
            "-  **0800-00-002** — Pakistan Mental Health Helpline\n"
            "-  Text **HOME** to **741741** — Crisis Text Line\n"
            "- **911** (US) / **1122** (Pakistan) — Emergency Services\n\n"
            "You are not alone. Please contact one of these right now.\n\n---\n\n"
        )
    if severity == "medium":
        return (
            "\n**CRISIS SUPPORT**\n\n"
            "I'm concerned about what you've shared. Your safety matters deeply.\n\n"
            "- 🇺 **988** | 🇵**0800-00-002** | Text **HOME** to **741741**\n\n"
            "You don't have to face this alone. I'm right here with you.\n\n---\n\n"
        )
    if severity == "low":
        return (
            "\n **I hear you, and I'm here.**\n\n"
            "It sounds like you're going through something really difficult. "
            "If things ever feel unbearable, free support is just a call away: "
            "**988** (US) | **0800-00-002** (Pakistan)\n\n---\n\n"
        )
    return ""

def send_whatsapp_alert(severity: str, message: str):
    try:
        from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER, EMERGENCY_WHATSAPP
        from twilio.rest import Client

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f" HealMatrix Crisis Alert [{severity.upper()}]\n\nUser said: {message[:200]}",
            from_=TWILIO_WHATSAPP_NUMBER,
            to=EMERGENCY_WHATSAPP
        )
        print(f"  WhatsApp alert sent [{severity}]")
    except Exception as e:
        print(f"   WhatsApp alert failed: {e}")