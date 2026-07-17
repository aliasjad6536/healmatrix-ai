import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_WHISPER_MODEL  = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

TWILIO_ACCOUNT_SID     = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN      = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER     = os.getenv("TWILIO_PHONE_NUMBER", "")
TWILIO_PHONE_NUMBER    = os.getenv("TWILIO_PHONE_NUMBER", "")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
EMERGENCY_CONTACT      = os.getenv("EMERGENCY_CONTACT", "")
EMERGENCY_WHATSAPP     = os.getenv("EMERGENCY_WHATSAPP", "")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
NGROK_AUTHTOKEN     = os.getenv("NGROK_AUTHTOKEN", "")
