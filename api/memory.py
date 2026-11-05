import os
import json
import tempfile

# In-memory storage (per user)
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

def get_memory(user_id: str):
    """Retrieve the recent message history for a given user."""
    _load_cache()
    return conversation_memory.get(user_id, [])

def save_memory(user_id: str, role: str, content: str):
    """Save a user or assistant message to memory (keeping last 10 messages)."""
    _load_cache()
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    conversation_memory[user_id].append({
        "role": role,
        "content": content
    })

    # Keep only the last 10 messages (5 user + 5 assistant turns) to maintain speed and focus
    conversation_memory[user_id] = conversation_memory[user_id][-10:]
    _save_cache()

def clear_memory(user_id: str):
    """Clear memory for a given user."""
    _load_cache()
    if user_id in conversation_memory:
        conversation_memory[user_id] = []
        _save_cache()
