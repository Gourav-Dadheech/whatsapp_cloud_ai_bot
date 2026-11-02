# Simple in-memory storage (per user)
conversation_memory = {}

def get_memory(user_id: str):
    return conversation_memory.get(user_id, [])

def save_memory(user_id: str, role: str, content: str):
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    conversation_memory[user_id].append({
        "role": role,
        "content": content
    })

    # Keep only last 10 messages
    conversation_memory[user_id] = conversation_memory[user_id][-10:]

