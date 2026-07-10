
import os
from typing import List, Dict

try:
    from config import GOOGLE_MAPS_API_KEY
except ImportError:
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")


def search_therapists(location: str, radius_km: int = 10, max_results: int = 8) -> List[Dict]:
    if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == "your_google_maps_api_key_here":
        return []
    try:
        import googlemaps
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        geo = gmaps.geocode(location)
        if not geo: return []
        lat = geo[0]["geometry"]["location"]["lat"]
        lng = geo[0]["geometry"]["location"]["lng"]
        results = []; seen = set()
        for term in ["psychologist","mental health clinic","therapist","psychiatrist"]:
            if len(results) >= max_results: break
            places = gmaps.places_nearby(location=(lat,lng),radius=radius_km*1000,keyword=term,type="health")
            for place in places.get("results",[]):
                pid = place.get("place_id","")
                if pid in seen: continue
                seen.add(pid)
                try:
                    d = gmaps.place(pid,fields=["name","formatted_address","rating",
                        "formatted_phone_number","opening_hours","url"])["result"]
                except Exception: d = place
                results.append({
                    "name":    d.get("name","Unknown"),
                    "address": d.get("formatted_address",place.get("vicinity","N/A")),
                    "rating":  d.get("rating","N/A"),
                    "phone":   d.get("formatted_phone_number","N/A"),
                    "maps_url":d.get("url",f"https://maps.google.com/?q={lat},{lng}"),
                    "open_now":d.get("opening_hours",{}).get("open_now",None) if "opening_hours" in d else None,
                })
                if len(results) >= max_results: break
        return results
    except Exception as e:
        print(f"    Therapist search failed: {e}"); return []


def get_therapist_cards_html(results: List[Dict], location: str) -> str:
    if not results:
        if not GOOGLE_MAPS_API_KEY or GOOGLE_MAPS_API_KEY == "your_google_maps_api_key_here":
            return """<div style="background:#131b2e;border:1px solid rgba(251,113,133,0.3);border-radius:14px;padding:1.2rem;color:#fca5a5">
              <strong> Google Maps API key not configured.</strong><br><br>
              Add <code>GOOGLE_MAPS_API_KEY = "your_key"</code> to <code>config.py</code><br>
              Get a free key at: <a href="https://console.cloud.google.com/" style="color:#63b3ed" target="_blank">console.cloud.google.com</a><br>
              Enable: <strong>Places API</strong> + <strong>Geocoding API</strong>
            </div>"""
        return f'<div style="background:#131b2e;border:1px solid rgba(99,179,237,0.15);border-radius:14px;padding:1.2rem;color:#94a3b8">No therapists found near <strong style="color:#e2e8f0">{location}</strong>. Try a broader area.</div>'
    cards=f'<div style="color:#2dd4bf;font-size:1.1rem;font-weight:600;margin-bottom:1rem">🗺️ {len(results)} Therapists Near {location}</div>'
    cards+='<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:1rem">'
    for t in results:
        open_badge=""
        if t["open_now"] is True: open_badge='<span style="background:rgba(52,211,153,0.2);color:#34d399;padding:0.1rem 0.5rem;border-radius:20px;font-size:0.75rem;margin-left:0.5rem">🟢 Open</span>'
        elif t["open_now"] is False: open_badge='<span style="background:rgba(251,113,133,0.2);color:#fb7185;padding:0.1rem 0.5rem;border-radius:20px;font-size:0.75rem;margin-left:0.5rem">🔴 Closed</span>'
        rating_html=""
        if t["rating"]!="N/A":
            try: stars=min(int(float(t["rating"])),5); rating_html="⭐"*stars+f' <span style="color:#fbbf24">{t["rating"]}</span>'
            except: rating_html=str(t["rating"])
        cards+=f"""<div style="background:#131b2e;border:1px solid rgba(99,179,237,0.1);border-radius:14px;padding:1.2rem">
          <div style="font-weight:600;color:#e2e8f0;margin-bottom:0.5rem">{t['name']}{open_badge}</div>
          <div style="color:#94a3b8;font-size:0.83rem;line-height:1.8">
            📍 {t['address']}<br>{f'⭐ {rating_html}<br>' if rating_html else ''}
            📞 {t['phone']}<br>
            <a href="{t['maps_url']}" target="_blank" style="color:#63b3ed;font-size:0.8rem">🔗 View on Google Maps</a>
          </div></div>"""
    cards+="</div>"; return cards