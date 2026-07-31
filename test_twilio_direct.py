"""
Direct Twilio test — bypasses main.py/backend.py entirely.
Isolates whether the problem is Twilio config or app code.
"""
from config import (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_NUMBER,
                    EMERGENCY_WHATSAPP, TWILIO_PHONE_NUMBER, EMERGENCY_CONTACT)

print("=" * 60)
print("  Config values being used:")
print("=" * 60)
print("  TWILIO_ACCOUNT_SID :", TWILIO_ACCOUNT_SID[:10] + "..." if TWILIO_ACCOUNT_SID else "MISSING")
print("  TWILIO_AUTH_TOKEN  :", "SET" if TWILIO_AUTH_TOKEN else "MISSING")
print("  TWILIO_WHATSAPP_NUMBER:", TWILIO_WHATSAPP_NUMBER)
print("  EMERGENCY_WHATSAPP    :", EMERGENCY_WHATSAPP)
print("  TWILIO_PHONE_NUMBER   :", TWILIO_PHONE_NUMBER)
print("  EMERGENCY_CONTACT     :", EMERGENCY_CONTACT)
print()

from twilio.rest import Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

print("=" * 60)
print("  TEST 1: Sending WhatsApp message directly")
print("=" * 60)
try:
    msg = client.messages.create(
        body="HealMatrix TEST — direct diagnostic message",
        from_=TWILIO_WHATSAPP_NUMBER,
        to=EMERGENCY_WHATSAPP,
    )
    print(f"  SUCCESS — Message SID: {msg.sid}")
    print(f"  Status: {msg.status}")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=" * 60)
print("  TEST 2: Checking recent message history")
print("=" * 60)
try:
    messages = client.messages.list(to=EMERGENCY_WHATSAPP, limit=5)
    print(f"  Recent messages to {EMERGENCY_WHATSAPP}: {len(messages)}")
    for m in messages:
        print(f"    - {m.date_created} | status={m.status} | error={m.error_code} {m.error_message or ''}")
except Exception as e:
    print(f"  Could not fetch message history: {e}")
