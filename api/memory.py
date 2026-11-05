import os
import json
import tempfile
import httpx

# Local in-memory fallback
conversation_memory = {}
active_documents = {}
CACHE_FILE = os.path.join(tempfile.gettempdir(), "mitra_whatsapp_memory.json")
DOC_CACHE_FILE = os.path.join(tempfile.gettempdir(), "mitra_whatsapp_active_doc.json")

def _load_cache():
    global conversation_memory
    if not conversation_memory and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                conversation_memory = json.load(f)
        except Exception:
            conversation_memory = {}

def _save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(conversation_memory, f, ensure_ascii=False)
    except Exception:
        pass

def _load_doc_cache():
    global active_documents
    if not active_documents and os.path.exists(DOC_CACHE_FILE):
        try:
            with open(DOC_CACHE_FILE, "r", encoding="utf-8") as f:
                active_documents = json.load(f)
        except Exception:
            active_documents = {}

def _save_doc_cache():
    try:
        with open(DOC_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(active_documents, f, ensure_ascii=False)
    except Exception:
        pass

MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "50"))


def _redis_config():
    """Retrieve Upstash Redis REST credentials from environment if available."""
    url = os.getenv("UPSTASH_REDIS_REST_URL")
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        return url.rstrip("/"), token
    return None, None

def get_memory(user_id: str):
    """Retrieve message history: tries Upstash Redis cloud first, falls back to local cache."""
    url, token = _redis_config()
    if url and token:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=["GET", f"mitra_mem_{user_id}"]
                )
                if res.status_code == 200:
                    data = res.json().get("result")
                    if data:
                        return json.loads(data)
        except Exception as e:
            print(f"Redis get error (using local fallback): {e}")

    _load_cache()
    return conversation_memory.get(user_id, [])

def save_memory(user_id: str, role: str, content: str):
    """Save message to memory (keeping last 50 turns). Saves to Upstash Redis and local cache."""
    history = get_memory(user_id)

    # Cap individual memory turn length to 1200 chars to keep rich study & chat context
    stored_text = content[:1200] if len(content) > 1200 else content
    history.append({"role": role, "content": stored_text})

    # Keep last 50 messages (25 user + 25 assistant turns) for deep long-term context
    history = history[-MAX_MEMORY_MESSAGES:]

    # Save to local cache
    conversation_memory[user_id] = history
    _save_cache()

    # Save to Upstash Redis Cloud (stores for 30 days)
    url, token = _redis_config()
    if url and token:
        try:
            with httpx.Client(timeout=3.0) as client:
                client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=["SET", f"mitra_mem_{user_id}", json.dumps(history), "EX", 2592000]
                )
        except Exception as e:
            print(f"Redis save error: {e}")

def clear_memory(user_id: str):
    """Clears conversation memory and active document store in both Redis cloud and local cache."""
    _load_cache()
    conversation_memory[user_id] = []
    _save_cache()
    clear_active_document(user_id)

    url, token = _redis_config()
    if url and token:
        try:
            with httpx.Client(timeout=3.0) as client:
                client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=["DEL", f"mitra_mem_{user_id}"]
                )
        except Exception as e:
            print(f"Redis delete error: {e}")


# Active Document Store (stores full extracted text of the last sent PDF/PPT/Doc)
def get_active_document(user_id: str) -> dict | None:
    """Retrieves the active document metadata and extracted text for the user."""
    url, token = _redis_config()
    if url and token:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=["GET", f"mitra_active_doc_{user_id}"]
                )
                if res.status_code == 200:
                    data = res.json().get("result")
                    if data:
                        return json.loads(data)
        except Exception as e:
            print(f"Redis get active doc error: {e}")

    _load_doc_cache()
    return active_documents.get(user_id)

def save_active_document(user_id: str, doc_name: str, total_pages: int, extracted_text: str):
    """Saves the active document content (up to 12,000 chars) for subsequent Q&A follow-ups."""
    doc_data = {
        "doc_name": doc_name,
        "total_pages": total_pages,
        "extracted_text": extracted_text[:12000]
    }
    _load_doc_cache()
    active_documents[user_id] = doc_data
    _save_doc_cache()

    url, token = _redis_config()
    if url and token:
        try:
            with httpx.Client(timeout=3.0) as client:
                client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=["SET", f"mitra_active_doc_{user_id}", json.dumps(doc_data), "EX", 2592000]
                )
        except Exception as e:
            print(f"Redis save active doc error: {e}")

def clear_active_document(user_id: str):
    """Clears the active document store for the user."""
    _load_doc_cache()
    active_documents.pop(user_id, None)
    _save_doc_cache()

    url, token = _redis_config()
    if url and token:
        try:
            with httpx.Client(timeout=3.0) as client:
                client.post(
                    url,
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json=["DEL", f"mitra_active_doc_{user_id}"]
                )
        except Exception as e:
            print(f"Redis delete active doc error: {e}")
