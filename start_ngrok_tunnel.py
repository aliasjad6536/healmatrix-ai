"""
HealMatrix AI — ngrok tunnel for backend.py (WhatsApp webhook)
"""
import time
from pyngrok import ngrok, conf
from config import NGROK_AUTHTOKEN

if not NGROK_AUTHTOKEN:
    print("NGROK_AUTHTOKEN missing in .env")
    raise SystemExit(1)

conf.get_default().auth_token = NGROK_AUTHTOKEN

try:
    ngrok.kill()
except Exception:
    pass

tunnel = ngrok.connect(5000, "http")

print("=" * 60)
print("  PUBLIC URL:", tunnel.public_url)
print("  Webhook for Twilio:", tunnel.public_url + "/whatsapp/reply")
print("=" * 60)
print("  Tunnel is running. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("\n  Stopping tunnel...")
    ngrok.kill()
    print("  Stopped.")
