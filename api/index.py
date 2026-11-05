import os
import sys
import asyncio
import base64
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from groq import AsyncGroq
from twilio.twiml.messaging_response import MessagingResponse

# Support both root directory execution and Vercel api/ execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from memory import get_memory, save_memory, clear_memory
except ImportError:
    from api.memory import get_memory, save_memory, clear_memory

app = FastAPI(title="Mitra WhatsApp AI Bot")

BOT_NAME = os.getenv("BOT_NAME", "Mitra")

class ChatRequest(BaseModel):
    message: str
    user_id: str = "web-user"


async def download_media(media_url: str) -> tuple[bytes, str]:
    """Downloads media from Twilio with optional Basic Auth."""
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    auth = httpx.BasicAuth(sid, token) if sid and token else None

    async with httpx.AsyncClient(timeout=15.0) as http_client:
        response = await http_client.get(media_url, auth=auth, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "application/octet-stream")
        return response.content, content_type


async def transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Transcribes WhatsApp voice notes using Groq Whisper Large-v3 in ~300ms."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    client = AsyncGroq(api_key=api_key)
    try:
        transcription = await asyncio.wait_for(
            client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model="whisper-large-v3",
                response_format="json"
            ),
            timeout=10.0
        )
        return transcription.text.strip()
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""


async def analyze_image_with_vision(image_bytes: bytes, content_type: str, caption: str = "", user_id: str = "") -> str:
    """Analyzes photos, homework, receipts, and errors using Groq Llama 3.2 Vision."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY is not set."

    client = AsyncGroq(api_key=api_key)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64_image}"

    prompt_text = caption.strip() if caption.strip() else (
        "Please examine this image carefully. Describe what you see, answer any questions shown, "
        "or explain the content clearly and helpfully using clean WhatsApp formatting (*bold*, bullet points)."
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                max_tokens=700
            ),
            timeout=12.0
        )
        answer = response.choices[0].message.content.strip()
        if user_id:
            save_memory(user_id, "user", f"[Sent an Image: {prompt_text}]")
            save_memory(user_id, "assistant", answer)
        return answer
    except Exception as e:
        print(f"Vision error: {e}")
        return f"Arre boss, photo process karne mein error aa gaya: {str(e)[:80]}"


async def generate_mitra_reply(user_msg: str, user_id: str, user_name: str = "") -> str:
    """Core AI response generator using Groq's flagship Llama 3.3 70B model."""
    msg_clean = user_msg.strip().lower()

    # Built-in quick commands
    if msg_clean in ["/reset", "reset", "clear", "/clear"]:
        clear_memory(user_id)
        return "🧹 *Conversation reset!* Memory has been cleared. What would you like to talk about now?"

    if msg_clean in ["/help", "help", "menu", "/start"]:
        name_str = f" {user_name}" if user_name else ""
        return (
            f"Namaste{name_str}! 🙏 I am *{BOT_NAME}*, your WhatsApp AI companion powered by Groq.\n\n"
            "Here is what I can do for you:\n"
            "• *Text Chat*: Answer questions quickly with deep reasoning (Llama 3.3 70B) 💡\n"
            "• *Voice Notes*: Send any audio message & I will understand it (Whisper v3) 🎙️\n"
            "• *Photo Analysis*: Send images of homework, bills, code errors or objects 📷\n"
            "• *Languages*: Fluent in English, Hindi, and Hinglish 🇮🇳\n"
            "• Send *reset* anytime to clear conversation memory 🧹\n\n"
            "How can I help you today?"
        )

    # Check for Groq API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ *Setup Needed*: `GROQ_API_KEY` is not set.\n\n"
            "Please configure your Groq API key in your Vercel Project Settings > Environment Variables, "
            "or set it in your local environment."
        )

    # Retrieve memory history
    history = get_memory(user_id)

    # System prompt optimized for WhatsApp messaging
    user_context = f" The user's WhatsApp name is {user_name}." if user_name else ""
    system_prompt = f"""You are {BOT_NAME} (मित्र), a friendly, helpful, warm, and smart AI friend on WhatsApp.{user_context}

IMPORTANT GUIDELINES:
1. Tone: Friendly, polite, natural, and supportive. Sound like a helpful companion, not a cold robot.
2. Language: Match the user's language. If they write in Hinglish (Hindi written in Roman script) or Hindi, reply warmly in natural Hinglish/Hindi. If in English, reply in fluent English.
3. Formatting for WhatsApp:
   - Use *bold* for key terms, titles, or emphasis.
   - Use _italics_ for light tone or quotes.
   - Use clean bullet points (•) for lists or steps.
   - Keep answers readable on mobile screens (avoid huge unstructured walls of text).
4. Completeness: Always finish your sentences and conclusions completely. Never stop mid-thought.
5. Conciseness: Be direct and helpful. For WhatsApp, prefer concise explanations over unnecessary fluff."""

    messages = [{"role": "system", "content": system_prompt}]
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_msg})

    preferred_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    fallback_model = "llama-3.1-8b-instant"

    client = AsyncGroq(api_key=api_key)

    try:
        # First attempt: Try flagship Llama 3.3 70B (State-of-the-art reasoning)
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=preferred_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=700
                ),
                timeout=11.0
            )
            answer = response.choices[0].message.content.strip()
        except Exception as primary_err:
            if preferred_model != fallback_model:
                print(f"Primary model {preferred_model} issue ({primary_err}), falling back to {fallback_model}...")
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=600
                    ),
                    timeout=9.0
                )
                answer = response.choices[0].message.content.strip()
            else:
                raise primary_err

        # Save successful turn to memory
        save_memory(user_id, "user", user_msg)
        save_memory(user_id, "assistant", answer)

        return answer

    except asyncio.TimeoutError:
        return "⏳ Response timeout. Network is slow right now, please try sending your message again!"
    except Exception as e:
        err_msg = str(e)
        print(f"Groq API Error: {err_msg}")
        if "401" in err_msg or "invalid_api_key" in err_msg:
            return "⚠️ *Groq API Key Invalid*: Please check that your `GROQ_API_KEY` is active and correct."
        elif "429" in err_msg or "rate_limit" in err_msg:
            return "⏳ *Mitra is busy right now!* Please send your message again in a few moments."
        else:
            return f"Arre boss, thoda technical hiccup aa gaya ({err_msg[:60]}). Thodi der mein dobara try karna!"


@app.get("/")
async def home():
    """Web preview page for browser testing and server health checking."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{BOT_NAME} AI - WhatsApp Bot</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #ece5dd; margin: 0; display: flex; flex-direction: column; height: 100vh; }}
        .header {{ background-color: #075e54; color: white; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header h1 {{ font-size: 1.15rem; margin: 0; display: flex; align-items: center; gap: 8px; }}
        .badge {{ background-color: #25d366; color: #075e54; font-size: 0.75rem; font-weight: bold; padding: 4px 10px; border-radius: 12px; }}
        .info-bar {{ background-color: #d1f2eb; color: #0e6251; padding: 8px 15px; font-size: 0.85rem; border-bottom: 1px solid #a2d9ce; text-align: center; }}
        .chat-container {{ flex: 1; overflow-y: auto; padding: 15px; display: flex; flex-direction: column; gap: 10px; }}
        .message {{ max-width: 80%; padding: 10px 14px; border-radius: 12px; line-height: 1.45; word-wrap: break-word; font-size: 0.95rem; white-space: pre-wrap; }}
        .user-message {{ background-color: #dcf8c6; align-self: flex-end; border-bottom-right-radius: 2px; box-shadow: 0 1px 1px rgba(0,0,0,0.1); }}
        .bot-message {{ background-color: #ffffff; align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 1px 1px rgba(0,0,0,0.1); }}
        .input-area {{ background-color: #f0f0f0; padding: 12px; display: flex; gap: 8px; align-items: center; }}
        input[type="text"] {{ flex: 1; padding: 12px 16px; border: none; border-radius: 24px; outline: none; font-size: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }}
        button {{ background-color: #128c7e; color: white; border: none; padding: 12px 20px; border-radius: 24px; cursor: pointer; font-size: 0.95rem; font-weight: 600; transition: background-color 0.2s; }}
        button:hover {{ background-color: #075e54; }}
        .btn-clear {{ background-color: #7f8c8d; font-size: 0.8rem; padding: 6px 12px; border-radius: 14px; }}
        .btn-clear:hover {{ background-color: #555; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 {BOT_NAME} WhatsApp AI Bot</h1>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span class="badge">● Online & Ready</span>
            <button class="btn-clear" onclick="resetChat()">Reset Memory</button>
        </div>
    </div>
    <div class="info-bar">
        ⚡ <b>Mitra Supercharged</b>: 🧠 Llama-3.3 70B | 🎙️ Groq Whisper v3 Voice Notes | 📷 Llama-3.2 Vision
    </div>
    <div class="chat-container" id="chat-container">
        <div class="message bot-message">🤖 *{BOT_NAME}*:\n\nNamaste! I am Mitra, supercharged with Llama 3.3 70B, Voice Notes & Vision support. Send a message, voice note, or photo on WhatsApp!</div>
    </div>
    <div class="input-area">
        <input type="text" id="message-input" placeholder="Type a message or 'help'..." onkeypress="handleKeyPress(event)" autofocus>
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        let userId = localStorage.getItem('mitra_user_id');
        if (!userId) {{
            userId = 'web-' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('mitra_user_id', userId);
        }}

        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');

        function addMessage(text, isUser) {{
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${{isUser ? 'user-message' : 'bot-message'}}`;
            msgDiv.textContent = text;
            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }}

        async function sendMessage() {{
            const text = messageInput.value.trim();
            if (!text) return;

            addMessage(text, true);
            messageInput.value = '';

            try {{
                const response = await fetch('/api/chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ message: text, user_id: userId }})
                }});
                const data = await response.json();
                addMessage(`🤖 *{BOT_NAME}*:\\n\\n${{data.response}}`, false);
            }} catch (error) {{
                addMessage("⚠️ Error connecting to server. Please try again.", false);
            }}
        }}

        async function resetChat() {{
            try {{
                const response = await fetch('/api/chat', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ message: 'reset', user_id: userId }})
                }});
                const data = await response.json();
                addMessage(data.response, false);
            }} catch (e) {{
                addMessage("Memory reset.", false);
            }}
        }}

        function handleKeyPress(event) {{
            if (event.key === 'Enter') {{
                sendMessage();
            }}
        }}
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.post("/api/chat")
async def web_chat(request: ChatRequest):
    """API endpoint for web chat testing."""
    user_msg = request.message
    user_id = request.user_id or "web-user"
    answer = await generate_mitra_reply(user_msg=user_msg, user_id=user_id)
    return {"response": answer}


@app.get("/whatsapp")
@app.get("/webhook")
async def verify_webhook(request: Request):
    """Handles Meta WhatsApp Cloud API verification handshake or status probe."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "mitra_token")
    if mode == "subscribe" and token == verify_token and challenge:
        return Response(content=challenge, media_type="text/plain")

    return {
        "status": "online",
        "bot": BOT_NAME,
        "features": ["Llama-3.3-70B", "Whisper-v3-Voice", "Llama-3.2-Vision"],
        "webhook_endpoint": "/whatsapp",
        "message": "Mitra WhatsApp Webhook is active and accepting incoming POST requests."
    }


async def handle_meta_whatsapp(request: Request):
    """Processes Meta WhatsApp Cloud API incoming JSON messages."""
    try:
        data = await request.json()
        entries = data.get("entry", [])
        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                contacts = value.get("contacts", [])
                user_name = contacts[0].get("profile", {}).get("name", "") if contacts else ""

                for msg in messages:
                    user_id = msg.get("from")
                    user_msg = msg.get("text", {}).get("body", "")
                    if user_id and user_msg:
                        answer = await generate_mitra_reply(user_msg, user_id, user_name)
                        
                        meta_token = os.getenv("WHATSAPP_TOKEN")
                        phone_number_id = value.get("metadata", {}).get("phone_number_id")
                        if meta_token and phone_number_id:
                            reply_text = f"🤖 *{BOT_NAME}*:\n\n{answer}"
                            async with httpx.AsyncClient() as http_client:
                                await http_client.post(
                                    f"https://graph.facebook.com/v21.0/{phone_number_id}/messages",
                                    headers={"Authorization": f"Bearer {meta_token}"},
                                    json={
                                        "messaging_product": "whatsapp",
                                        "to": user_id,
                                        "type": "text",
                                        "text": {"body": reply_text}
                                    }
                                )
        return {"status": "ok"}
    except Exception as e:
        print(f"Meta Cloud API Webhook Error: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/whatsapp")
@app.post("/api/whatsapp")
@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Primary WhatsApp Webhook Endpoint.
    Handles Text, Voice Notes (Whisper v3), and Photos (Vision).
    """
    content_type = request.headers.get("content-type", "")

    # Handle Meta WhatsApp Cloud API JSON payload
    if "application/json" in content_type:
        return await handle_meta_whatsapp(request)

    # Handle Twilio WhatsApp form payload
    form = await request.form()
    user_msg = form.get("Body", "").strip()
    user_id = form.get("From", "unknown")
    user_name = form.get("ProfileName", "")

    try:
        num_media = int(form.get("NumMedia", "0"))
    except ValueError:
        num_media = 0

    media_url = form.get("MediaUrl0")
    media_type = form.get("MediaContentType0", "")

    prefix = ""

    # 1. Voice Note / Audio processing via Groq Whisper v3
    if num_media > 0 and media_url and media_type.startswith("audio/"):
        try:
            audio_bytes, _ = await download_media(media_url)
            transcription = await transcribe_audio(audio_bytes, "voice.ogg")
            if transcription:
                prefix = f"🎙️ _Heard: \"{transcription}\"_\n\n"
                user_msg = transcription
            else:
                user_msg = "I could not clearly understand the voice note. Please speak clearly or send as text."
        except Exception as e:
            print(f"Audio download error: {e}")
            twilio_resp = MessagingResponse()
            twilio_resp.message(f"🤖 *{BOT_NAME}*:\n\n🎙️ Arre boss, voice note download nahi ho paya. Please send your message as text or try again!")
            return Response(content=str(twilio_resp), media_type="application/xml")

    # 2. Image / Photo processing via Groq Llama 3.2 Vision
    elif num_media > 0 and media_url and media_type.startswith("image/"):
        try:
            image_bytes, ct = await download_media(media_url)
            vision_answer = await analyze_image_with_vision(image_bytes, ct, caption=user_msg, user_id=user_id)
            reply_body = f"🤖 *{BOT_NAME}* (📷 Vision):\n\n{vision_answer}"
            twilio_resp = MessagingResponse()
            twilio_resp.message(reply_body)
            return Response(content=str(twilio_resp), media_type="application/xml")
        except Exception as e:
            print(f"Image download error: {e}")
            twilio_resp = MessagingResponse()
            twilio_resp.message(f"🤖 *{BOT_NAME}*:\n\n📷 Arre boss, image download nahi ho payi. Please try sending the photo again!")
            return Response(content=str(twilio_resp), media_type="application/xml")

    if not user_msg:
        user_msg = "Hello"

    answer = await generate_mitra_reply(user_msg=user_msg, user_id=user_id, user_name=user_name)

    # Format response for Twilio WhatsApp XML (TwiML)
    twilio_resp = MessagingResponse()
    reply_body = f"🤖 *{BOT_NAME}*:\n\n{prefix}{answer}"
    twilio_resp.message(reply_body)

    return Response(content=str(twilio_resp), media_type="application/xml")


@app.post("/")
async def root_post_fallback(request: Request):
    """
    Fallback POST handler for root URL '/'.
    If user sets Twilio webhook directly to 'https://...vercel.app/', it will still work!
    """
    content_type = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in content_type:
        return await whatsapp_webhook(request)
    elif "application/json" in content_type:
        try:
            data = await request.json()
            if "message" in data:
                return await web_chat(ChatRequest(**data))
            elif "entry" in data:
                return await handle_meta_whatsapp(request)
        except Exception:
            pass
    return Response(content="Invalid request format", status_code=400)
