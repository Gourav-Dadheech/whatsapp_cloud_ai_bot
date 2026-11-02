from groq import Groq
import os

# Groq client will automatically read API key from environment variable
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_ai(user_msg: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are Mitra, a friendly, helpful AI assistant. Answer clearly and completely."},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.7,
        max_tokens=300
    )

    return response.choices[0].message.content


