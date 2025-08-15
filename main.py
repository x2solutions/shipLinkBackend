# main.py
import os
import json
import requests
from typing import Optional, Any, List
from flask import Flask, request, jsonify
from flask_cors import CORS
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

app = Flask(__name__)

# ===== CORS (restrict to your Bubble domains) =====
ALLOWED_ORIGINS = [
    "https://slfinal.bubbleapps.io",    # <-- replace with your Bubble dev domain
    "https://www.yourdomain.com",        # <-- replace (or remove if not using custom domain)
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

def as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return default


# ---------- Apollo → compact contacts ----------
def simplify_apollo_contacts(raw):
    """
    Return a compact, uniform list of contacts with ONLY these keys:
    name, title, location, email_status, linkedin_url, email, phone
    (email/phone will be "" if not available)
    """
    def pick_location(c):
        city = (c.get("city") or c.get("location") or "").strip()
        state = (c.get("state") or "").strip()
        country = (c.get("country") or "").strip()
        parts = [p for p in [city, state or None, country or None] if p]
        return ", ".join(parts) if parts else ""

    def pick_email(c):
        # Free Apollo often returns "email_not_unlocked@domain.com"
        e = (c.get("email") or "").strip()
        if e.lower().startswith("email_not_unlocked"):
            return ""
        return e

    def pick_phone(c):
        phones = c.get("phone_numbers") or []
        if isinstance(phones, list):
            for p in phones:
                num = (p.get("sanitized_number") or p.get("number") or "").strip()
                if num:
                    return num
        # avoid leaking org phone numbers as person phones
        return ""

    items = raw.get("contacts") or raw.get("people") or []
    out = []
    for c in items:
        out.append({
            "name": (c.get("name") or "").strip(),
            "title": (c.get("title") or c.get("headline") or "").strip(),
            "location": pick_location(c),
            "email_status": (c.get("email_status") or c.get("contact_email_status") or "").strip(),
            "linkedin_url": (c.get("linkedin_url") or c.get("linkedin") or "").strip(),
            "email": pick_email(c),
            "phone": pick_phone(c),
        })
    return out


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
        # Fail fast with clear message instead of AttributeError
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
        "<li><code>/find-person</code> — body: {\"titles\":[],\"domains\":[],\"count\":5, \"include_raw\": false}</li>"
        "<li><code>/format-response</code> — body: {intent, original_query, contacts[], notes, count}</li>"
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

    # Tight, conservative classifier
    system_msg = (
        "Return ONLY JSON with keys exactly: "
        '{"intent": string, "titles": array|null, "domains": array|null, "count": number|null}. '

        'Intent rules: '
        '"Point of Contact" ONLY when the user clearly asks to find people at a company '
        '(e.g., “find 3 marketing managers at Mattel”, “get ecommerce managers at walmart.com”). '
        'This REQUIRES BOTH (1) at least one role/title AND (2) a company name or domain. '
        '"Data Traffic" is for questions about web/app traffic, visitors, MAUs, etc. '
        'All other general knowledge or chit‑chat is "unsupported". '

        'If intent is "Point of Contact", extract: '
        '`titles` (array of normalized titles from the text), '
        '`domains` (array of company domains inferred from company names; fix misspellings: Mattell→mattel.com), '
        '`count` (integer; default 5 if none). '

        'If either titles OR domains is missing/uncertain, set intent to "unsupported". '
        'Never guess extra titles or domains. Never return empty arrays; use null. '

        'Examples:\n'
        'User: "Find 3 marketing supervisors at Mattell" → '
        '{"intent":"Point of Contact","titles":["marketing supervisor"],"domains":["mattel.com"],"count":3}\n'
        'User: "What are the primary colors?" → '
        '{"intent":"unsupported","titles":null,"domains":null,"count":null}\n'
        'User: "Show web traffic for nike.com last year" → '
        '{"intent":"Data Traffic","titles":null,"domains":["nike.com"],"count":null}'
    )

    try:
        content = chat_json(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_text}
            ],
            model=MODEL_INTENT,
            max_tokens=180,
            temperature=0
        )
        obj = json.loads(content)
    except Exception:
        obj = {"intent": "unsupported", "titles": None, "domains": None, "count": None}

    # normalize + HARD GUARDS
    intent = (obj.get("intent") or "unsupported").strip()

    def nonempty_list(x: Any) -> bool:
        return isinstance(x, list) and any(str(i).strip() for i in x)

    if intent == "Point of Contact":
        titles = obj.get("titles")
        domains = obj.get("domains")
        count = obj.get("count")

        if not nonempty_list(titles) or not nonempty_list(domains):
            intent, titles, domains, count = "unsupported", None, None, None
        else:
            count = ensure_int(count, default=3, low=1, high=5)

        return jsonify({
            "intent": intent,
            "titles": titles,
            "domains": domains,
            "count": count
        }), 200

    elif intent == "Data Traffic":
        # keep domains if the model extracted them; Bubble can decide how to use
        domains = obj.get("domains") if nonempty_list(obj.get("domains")) else None
        return jsonify({
            "intent": "Data Traffic",
            "titles": None,
            "domains": domains,
            "count": None
        }), 200

    # Fallback: unsupported
    return jsonify({
        "intent": "unsupported",
        "titles": None,
        "domains": None,
        "count": None
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
    include_raw = as_bool(data.get("include_raw"), False)  # robust boolean parsing

    payload = {
        "person_titles": titles,
        "q_organization_domains_list": domains,
        "contact_email_status": ["verified"],  # you can relax this if needed
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

    resp = {
        "contacts": contacts,
        "count_requested": count,
        "count_returned": len(contacts),
    }
    # Only include a tiny debug raw when explicitly requested
    if include_raw:
        resp["raw"] = {
            "pagination": apollo_json.get("pagination"),
            "breadcrumbs": apollo_json.get("breadcrumbs"),
        }
    return jsonify(resp), 200


@app.post("/format-response")
def format_response():
    unauthorized = require_backend_token()
    if unauthorized:
        return unauthorized

    data = request.get_json(silent=True) or {}
    intent = (data.get("intent") or "").strip()
    original_query = (data.get("original_query") or "").strip()
    contacts: List[dict] = data.get("contacts") or []
    notes = (data.get("notes") or "").strip()

    # Resolve the desired count robustly
    count = ensure_int(
        data.get("count") or
        data.get("count_requested") or
        data.get("count_returned") or
        len(contacts),
        default=5, low=1, high=20
    )

    if client is None:
        return http_json_error("OPENAI_API_KEY is not set on the server.", 500)

    try:
        if intent == "Point of Contact":
            system_msg = (
                f"You are a concise, friendly assistant. You will be given structured results. "
                f"Write a short, conversational reply that summarizes the key info for the user. "
                f"List up to {count} contacts. For each contact, show: Name, Title, Location, Email status. "
                f"Include LinkedIn, Email, and Phone only if provided (omit blank). "
                f"If no contacts, ask a brief clarifying question."
            )
            user_content = json.dumps({
                "intent": intent,
                "original_query": original_query,
                "contacts": (contacts or [])[:count],  # enforce limit server-side
                "notes": notes,
                "count_requested": count
            }, ensure_ascii=False)

        else:
            # Generic path – act like normal ChatGPT
            system_msg = "You are ChatGPT, a helpful assistant. Respond to the user naturally and helpfully."
            user_content = original_query or "Hello"

        resp = client.chat.completions.create(
            model=MODEL_FORMAT,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content}
            ],
            max_tokens=320,
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
