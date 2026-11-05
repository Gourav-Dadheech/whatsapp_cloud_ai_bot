import os
from groq import Groq

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)

def ask_ai(user_msg: str) -> str:
    """Synchronous AI response helper for testing or CLI scripts."""
    try:
        client = get_groq_client()
    except Exception as e:
        return f"⚠️ Configuration error: {e}"

    preferred_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    fallback_model = "openai/gpt-oss-20b"
    try:
        try:
            response = client.chat.completions.create(
                model=preferred_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Mitra, a friendly, warm, and supportive AI companion. "
                            "Match the user's language: reply in English if they write in English, "
                            "and Hinglish if they write in Hinglish. Format neatly for WhatsApp with WhatsApp markdown."
                        )
                    },
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.7,
                max_tokens=700
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if preferred_model != fallback_model:
                response = client.chat.completions.create(
                    model=fallback_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Mitra, a friendly, warm, and supportive AI companion. "
                                "Answer clearly, naturally, and politely. Format neatly for WhatsApp with WhatsApp markdown."
                            )
                        },
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.7,
                    max_tokens=600
                )
                return response.choices[0].message.content.strip()
            raise
    except Exception as e:
        return f"⚠️ Error: {e}"
