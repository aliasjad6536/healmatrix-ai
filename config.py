"""
HealMatrix AI — Configuration
All API keys and settings in one place.
"""

GROQ_API_KEY       = ""
GROQ_MODEL         = ""
GROQ_WHISPER_MODEL = ""

TWILIO_ACCOUNT_SID     = ""
TWILIO_AUTH_TOKEN      = ""
TWILIO_FROM_NUMBER     = ""
TWILIO_PHONE_NUMBER    = ""
TWILIO_WHATSAPP_NUMBER = ""
EMERGENCY_CONTACT      = ""
EMERGENCY_WHATSAPP     = ""

GOOGLE_MAPS_API_KEY = ""  

NGROK_AUTHTOKEN = ""

NGROK_PUBLIC_URL     = ""
VOICE_WEBHOOK_URL    = f"{NGROK_PUBLIC_URL}/voice/crisis"
STATUS_CALLBACK_URL  = f"{NGROK_PUBLIC_URL}/status/call"
