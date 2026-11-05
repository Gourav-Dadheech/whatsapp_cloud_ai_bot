import os
import json
import tempfile
import httpx

# Local in-memory fallback
conversation_memory = {}
CACHE_FILE = os.path.join(tempfile.gettempdir(), "mitra_whatsapp_memory.json")

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

    # Cap individual memory turn length to 600 chars to keep AI prompt fast & focused
    stored_text = content[:600] if len(content) > 600 else content
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
    """Clears conversation memory in both Redis cloud and local cache."""
    _load_cache()
    conversation_memory[user_id] = []
    _save_cache()

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
