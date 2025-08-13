# main.py
import os
import json
import requests
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

app = Flask(__name__)

# ===== CORS (restrict to your Bubble domains) =====
ALLOWED_ORIGINS = [
    "https://your-app.bubbleapps.io",   # <-- replace with your Bubble dev domain
    "https://www.yourdomain.com"        # <-- replace (or remove if not using custom domain)
]
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}}, supports_credentials=True)

# ===== Secrets =====
OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
APOLLO_API_KEY: Optional[str] = os.environ.get("APOLLO_API_KEY")
BACKEND_TOKEN: Optional[str] = os.environ.get("BACKEND_TOKEN")  # shared header token

# Create client only if the key exists (avoids None attribute errors)
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ===== Config =====
MODEL_INTENT = "gpt-4o-mini"
MODEL_FORMAT = "gpt-4o-mini"
APOLLO_BASE = "https://api.apollo.io/api/v1/mixed_people/search"

# ---------- Utility ----------
def http_json_error(msg, code=400, extra=None):
    payload = {"error": msg}
    if extra:
        payload.update(extra)
    return jsonify(payload), code

def ensure_int(n, default=5, low=1, high=20):
    try:
        n = int(n)
        if n < low:
            n = low
        if n > high:
            n = high
        return n
    except Exception:
        return default

def simplify_apollo_contacts(raw):
    """Map Apollo response into a clean list for Bubble + formatter."""
    contacts = []
    items = raw.get("contacts") or raw.get("people") or []
    for c in items:
        name = c.get("name") or "Unknown"
        title = c.get("title") or c.get("headline") or ""
        loc = c.get("city") or c.get("location") or ""
        linkedin = c.get("linkedin_url") or c.get("linkedin") or ""
        email_status = c.get("email_status") or c.get("contact_email_status") or ""
        contacts.append({
            "name": name,
            "title": title,
            "location": loc,
            "linkedin_url": linkedin,
            "email_status": email_status
        })
    return contacts

# ----- simple header auth for Bubble → backend -----
def require_backend_token():
    if not BACKEND_TOKEN:  # if you haven't set it, skip check (useful during local dev)
        return None
    token = request.headers.get("X-Backend-Token")
    if token != BACKEND_TOKEN:
        return jsonify({"error": "Unauthorized"}), 401
    return None

# ---------- OpenAI helper ----------
@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(Exception),
)
def chat_json(messages, model=MODEL_INTENT, max_tokens=150, temperature=0):
    if client is None:
        # Fail fast with clear message instead of AttributeError: 'NoneType'...
        raise RuntimeError("OPENAI_API_KEY is not set on the server.")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content  # guaranteed JSON string

# ---------- Friendly GET routes so browser tests don't 404 ----------
@app.get("/")
def index():
    return (
        "<h3>ShipLink Backend is running ✅</h3>"
        "<p>POST endpoints:</p>"
        "<ul>"
        "<li><code>/extract-intent</code> — body: {\"text\":\"...\"}</li>"
        "<li><code>/find-person</code> — body: {\"titles\":[],\"domains\":[],\"count\":5}</li>"
        "<li><code>/format-response</code> — body with contacts list</li>"
        "</ul>"
        "<p>Health: <a href='/health'>/health</a></p>",
        200
    )

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "openai": bool(OPENAI_API_KEY),
        "apollo": bool(APOLLO_API_KEY),
        "auth_required": bool(BACKEND_TOKEN)
    }), 200

# ---------- POST endpoints used by Bubble ----------
@app.post("/extract-intent")
def extract_intent():
    unauthorized = require_backend_token()
    if unauthorized:
        return unauthorized

    data = request.get_json(silent=True) or {}
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return http_json_error("Empty text", 400, {
            "intent": "unsupported", "titles": None, "domains": None, "count": None
        })

    system_msg = (
        'Classify the user request as one of: "Point of Contact", "Data Traffic", or "unsupported". '
        'Only if intent is "Point of Contact", also extract: '
        '`titles` (array), `domains` (array), and `count` (number, default 5). '
        'Return ONLY JSON: {"intent": string, "titles": array|null, "domains": array|null, "count": number|null}.'
    )

    try:
        content = chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_text}
            ],
            model=MODEL_INTENT,
            max_tokens=120,
            temperature=0
        )
        obj = json.loads(content)
    except Exception:
        obj = {"intent": "unsupported", "titles": None, "domains": None, "count": None}

    intent = obj.get("intent") or "unsupported"
    titles = obj.get("titles") if intent == "Point of Contact" else None
    domains = obj.get("domains") if intent == "Point of Contact" else None
    count = obj.get("count") if intent == "Point of Contact" else None
    count = ensure_int(count, default=5, low=1, high=20) if count is not None else None

    return jsonify({
        "intent": intent,
        "titles": titles,
        "domains": domains,
        "count": count
    }), 200

@app.post("/find-person")
def find_person():
    unauthorized = require_backend_token()
    if unauthorized:
        return unauthorized

    if not APOLLO_API_KEY:
        return http_json_error("APOLLO_API_KEY missing", 500)

    data = request.get_json(silent=True) or {}
    titles = data.get("titles") or []
    domains = data.get("domains") or []
    count = ensure_int(data.get("count"), default=5, low=1, high=20)

    payload = {
        "person_titles": titles,
        "q_organization_domains_list": domains,
        "contact_email_status": ["verified"],  # optional filter
        "per_page": count,
        "page": 1
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": APOLLO_API_KEY
    }

    try:
        r = requests.post(APOLLO_BASE, headers=headers, json=payload, timeout=30)
        if r.status_code >= 400:
            return http_json_error("Apollo error", r.status_code, {"apollo_response": r.text})
        apollo_json = r.json()
    except requests.RequestException as e:
        return http_json_error(f"Apollo request failed: {e}", 502)

    contacts = simplify_apollo_contacts(apollo_json)
    return jsonify({"contacts": contacts, "raw": apollo_json}), 200

@app.post("/format-response")
def format_response():
    unauthorized = require_backend_token()
    if unauthorized:
        return unauthorized

    data = request.get_json(silent=True) or {}
    intent = (data.get("intent") or "").strip()
    original_query = (data.get("original_query") or "").strip()
    contacts = (data.get("contacts") or [])[:10]
    notes = (data.get("notes") or "").strip()

    if client is None:
        return http_json_error("OPENAI_API_KEY is not set on the server.", 500)

    try:
        if intent == "Point of Contact":
            system_msg = (
                "You are a concise, friendly assistant. You will be given structured results. "
                "Write a short, conversational reply that summarizes the key info for the user. "
                "If contacts exist, list 3–6 with name, title, location, and a LinkedIn link if available. "
                "If no data is available, ask a brief clarifying question."
            )
            user_content = json.dumps({
                "intent": intent,
                "original_query": original_query,
                "contacts": contacts,
                "notes": notes
            }, ensure_ascii=False)

        else:
            # Fallback for "unsupported" or other custom intents → treat as ChatGPT-style Q&A
            system_msg = "You are ChatGPT, a helpful assistant. Respond to the user naturally and helpfully."
            user_content = original_query

        resp = client.chat.completions.create(
            model=MODEL_FORMAT,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}
            ],
            max_tokens=300,
            temperature=0.3
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return http_json_error(f"OpenAI formatting failed: {e}", 502)

    return jsonify({"message": text}), 200

if __name__ == "__main__":
    # Render/Replit proxy: bind to env PORT (falls back to 8000 locally)
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on 0.0.0.0:{port} (proxy-ready)")
    app.run(host="0.0.0.0", port=port)
