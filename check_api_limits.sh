#!/usr/bin/env bash
# HealMatrix AI — API limits / balance checker
set -uo pipefail
cd "$(dirname "$0")" 2>/dev/null || true

if [ -f .env ]; then
  set -a; source .env; set +a
else
  echo "❌ .env not found in current directory — run this from the project folder"
  exit 1
fi

echo "============================================================"
echo "  GROQ — rate limit status"
echo "============================================================"
if [ -z "${GROQ_API_KEY:-}" ]; then
  echo "  GROQ_API_KEY not set in .env"
else
  HEADERS=$(curl -s -D - -o /dev/null \
    https://api.groq.com/openai/v1/models \
    -H "Authorization: Bearer $GROQ_API_KEY")

  STATUS=$(echo "$HEADERS" | head -1)
  echo "  Response: $STATUS"
  echo "$HEADERS" | grep -i "x-ratelimit\|retry-after" | sed 's/^/  /'

  if echo "$STATUS" | grep -q "401"; then
    echo "  WARNING: 401 = key invalid/revoked"
  elif echo "$STATUS" | grep -q "429"; then
    echo "  WARNING: 429 = rate limit ALREADY hit right now"
  elif echo "$STATUS" | grep -q "200"; then
    echo "  OK: Key working. See remaining-requests / remaining-tokens above."
  fi
fi

echo
echo "============================================================"
echo "  TWILIO — account balance + trial status"
echo "============================================================"
if [ -z "${TWILIO_ACCOUNT_SID:-}" ] || [ -z "${TWILIO_AUTH_TOKEN:-}" ]; then
  echo "  TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not set in .env"
else
  BALANCE=$(curl -s -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN" \
    "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID/Balance.json")
  echo "  Balance response:"
  echo "$BALANCE" | python3 -m json.tool 2>/dev/null | sed 's/^/  /' || echo "  $BALANCE"

  ACCOUNT=$(curl -s -u "$TWILIO_ACCOUNT_SID:$TWILIO_AUTH_TOKEN" \
    "https://api.twilio.com/2010-04-01/Accounts/$TWILIO_ACCOUNT_SID.json")
  TYPE=$(echo "$ACCOUNT" | python3 -c "import json,sys; print(json.load(sys.stdin).get('type','?'))" 2>/dev/null)
  echo "  Account type: $TYPE   (Trial = disclaimer plays on calls; Full = no disclaimer)"
fi

echo
echo "============================================================"
echo "  GOOGLE MAPS — key validity + quota status"
echo "============================================================"
if [ -z "${GOOGLE_MAPS_API_KEY:-}" ]; then
  echo "  GOOGLE_MAPS_API_KEY not set in .env"
else
  RESP=$(curl -s "https://maps.googleapis.com/maps/api/geocode/json?address=Lahore&key=$GOOGLE_MAPS_API_KEY")
  GSTATUS=$(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null)
  echo "  API status: $GSTATUS"
  case "$GSTATUS" in
    OK) echo "  OK: Key working, quota available." ;;
    OVER_QUERY_LIMIT) echo "  WARNING: Daily/billing quota exceeded." ;;
    REQUEST_DENIED) echo "  WARNING: Key invalid, restricted, or billing not enabled." ;;
    *) echo "  Raw response: $RESP" ;;
  esac
  echo "  Note: Google doesn't expose exact dollar-credit remaining via API."
  echo "  Check console.cloud.google.com/billing for the real number."
fi

echo
echo "============================================================"
echo "  Done. Run this again any time before exhibition day."
echo "============================================================"
