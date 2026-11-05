import os
import sys
import io
import asyncio
import base64
import zipfile
import xml.etree.ElementTree as ET
import httpx
import pypdf
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


def add_twilio_messages(twilio_resp: MessagingResponse, text: str, max_chars: int = 1500):
    """Splits long text across multiple WhatsApp messages so Twilio's 1600-char limit is never exceeded."""
    if len(text) <= max_chars:
        twilio_resp.message(text)
        return

    paragraphs = text.split("\n\n")
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                twilio_resp.message(current_chunk)
            if len(para) > max_chars:
                lines = para.split("\n")
                sub_chunk = ""
                for line in lines:
                    if len(sub_chunk) + len(line) + 1 <= max_chars:
                        sub_chunk = (sub_chunk + "\n" + line).strip()
                    else:
                        if sub_chunk:
                            twilio_resp.message(sub_chunk)
                        sub_chunk = line
                current_chunk = sub_chunk
            else:
                current_chunk = para
    if current_chunk:
        twilio_resp.message(current_chunk)


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 15) -> tuple[str, int]:
    """Extracts text from PDF documents and college unit handouts."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, max_pages)
        extracted = []
        for i in range(pages_to_read):
            t = reader.pages[i].extract_text() or ""
            if t.strip():
                extracted.append(f"--- Page {i+1} ---\n{t.strip()}")
        full = "\n\n".join(extracted)
        return full[:8000], total_pages
    except Exception as e:
        print(f"PDF extract error: {e}")
        return "", 0


def extract_text_from_pptx(pptx_bytes: bytes, max_slides: int = 25) -> tuple[str, int]:
    """Extracts text from PowerPoint PPT/PPTX slide presentations."""
    try:
        slides_text = []
        with zipfile.ZipFile(io.BytesIO(pptx_bytes)) as z:
            slide_files = sorted([f for f in z.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")])
            for idx, sfile in enumerate(slide_files[:max_slides]):
                xml_content = z.read(sfile)
                tree = ET.fromstring(xml_content)
                texts = [node.text.strip() for node in tree.iter() if node.tag.endswith('}t') and node.text and node.text.strip()]
                if texts:
                    slides_text.append(f"--- Slide {idx+1} ---\n" + "\n".join(texts))
        return "\n\n".join(slides_text)[:8000], len(slide_files)
    except Exception as e:
        print(f"PPTX extract error: {e}")
        return "", 0


async def download_media(media_url: str) -> tuple[bytes, str]:
    """Downloads media from Twilio with optional Basic Auth and direct fallback."""
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http_client:
        if sid and token:
            try:
                response = await http_client.get(media_url, auth=httpx.BasicAuth(sid, token))
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "application/octet-stream")
                    return response.content, content_type
            except Exception as e:
                print(f"Auth download failed: {e}")

        # Fallback to direct download
        response = await http_client.get(media_url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "application/octet-stream")
        return response.content, content_type


async def transcribe_audio(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """Transcribes WhatsApp voice notes using Groq Whisper in ~300ms."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return ""
    client = AsyncGroq(api_key=api_key)
    for model_name in ["whisper-large-v3-turbo", "whisper-large-v3"]:
        try:
            transcription = await asyncio.wait_for(
                client.audio.transcriptions.create(
                    file=(filename, audio_bytes),
                    model=model_name,
                    response_format="json"
                ),
                timeout=10.0
            )
            return transcription.text.strip()
        except Exception as e:
            print(f"Whisper model {model_name} error: {e}")
            continue
    return ""


async def analyze_image_with_vision(image_bytes: bytes, content_type: str, caption: str = "", user_id: str = "") -> str:
    """Analyzes photos, icons, homework, receipts, and objects using Groq Multimodal Vision."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY is not set."

    # Ensure valid image MIME type
    if not content_type or not content_type.startswith("image/"):
        content_type = "image/jpeg"

    client = AsyncGroq(api_key=api_key)
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{content_type};base64,{b64_image}"

    question = caption.strip() if caption.strip() and caption.strip().lower() != "hello" else "What apps, text, or objects are in this image? Describe accurately."

    prompt_text = f"""You are Mitra AI, an expert computer vision assistant on WhatsApp.

CRITICAL INSTRUCTIONS FOR IMAGE RECOGNITION:
1. Examine this photo/screenshot carefully.
2. If this is a computer or laptop screen / taskbar:
   - Carefully distinguish Windows (Windows 10/11 taskbar, Windows Start icon, Microsoft Store shopping bag logo, Lenovo Vantage, File Explorer folder, Edge, etc.) from macOS.
   - Do NOT guess macOS or Safari unless an Apple logo or Safari compass icon is explicitly visible.
   - Accurately recognize Windows and OEM apps like Microsoft Store (white shopping bag icon with Microsoft colors) and Lenovo Vantage (blue cube / letter L icon).
3. Directly answer the user's question with precise, factual details.
4. Format using clean WhatsApp markdown (*bold*, bullet points •).
5. Match language: English if asked in English, or natural Hinglish if asked in Hinglish.

User's question: {question}"""

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": data_url}}
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=700
            ),
            timeout=14.0
        )
        answer = response.choices[0].message.content.strip()
        if user_id:
            save_memory(user_id, "user", f"[Sent an Image: {question}]")
            save_memory(user_id, "assistant", answer)
        return answer
    except Exception as e:
        print(f"Vision error: {e}")
        return f"Photo analyze karne mein issue aaya ({str(e)[:70]}). Please try sending again!"


def check_creator_query(msg: str) -> str | None:
    """Returns direct Gourav Dadheech attribution if user asks about the creator or engineer."""
    m = msg.strip().lower()

    # Direct query about Gourav
    if "gourav dadheech" in m or ("gourav" in m and any(k in m for k in ["who", "kaun", "creator", "made", "engineer", "kya hai"])):
        return (
            "👨‍💻 *Gourav Dadheech* is my brilliant creator and engineer!\n\n"
            "He built and fine-tuned me to be your smart, warm, and helpful WhatsApp AI companion. "
            "Whether it's studying college units, analyzing PDFs/PPTs, voice notes, or everyday tasks, "
            "Gourav engineered me to help you excel and succeed! 🚀✨"
        )

    creator_keywords = [
        "who created you", "who made you", "who engineered you", "who built you",
        "who developed you", "who programmed you", "who designed you", "who coded you",
        "who is your creator", "who is your developer", "who is your engineer",
        "who is your maker", "who is your author", "who's your creator", "who's your developer",
        "who is your father", "who created mitra", "who engineered mitra", "who made mitra",
        "who built mitra", "who is behind you", "whose bot are you",
        "kisne banaya", "tumhe kisne banaya", "aapko kisne banaya", "creator kaun hai",
        "developer kaun hai", "engineer kaun hai", "tumhara malik", "kiska bot", "kiske dwara",
        "who owns you", "who own you", "who invented you"
    ]

    is_creator_asked = any(k in m for k in creator_keywords)
    if not is_creator_asked:
        if ("who" in m and any(w in m for w in ["created", "engineered", "built", "developed", "maker", "creator"])) or \
           ("kisne" in m and any(w in m for w in ["banaya", "develop", "engineer", "design", "create"])) or \
           ("kaun" in m and any(w in m for w in ["creator", "developer", "engineer", "malik", "banane"])):
            is_creator_asked = True

    if is_creator_asked:
        is_hindi = any(w in m for w in ["kisne", "kaun", "kiska", "bhai", "aapko", "tumhe", "banaya", "hai", "kya"])
        if is_hindi:
            return (
                "Mujhe **Gourav Dadheech** ne create aur engineer kiya hai! 👨‍💻🚀\n\n"
                "Unhone mujhe ek intelligent, warm aur friendly WhatsApp AI companion banaya hai — "
                "taaki college units ki study, PDF analysis, voice notes, aur har daily sawaal mein main aapka best mitra ban sakoon! ✨\n\n"
                "Batao, aaj main aapki kya madad kar sakta hoon? 😊"
            )
        else:
            return (
                "I was proudly created and engineered by **Gourav Dadheech**! 👨‍💻🚀\n\n"
                "He designed and built me to be your warm, intelligent, and reliable AI companion on WhatsApp. "
                "From mastering college units and analyzing PDFs/PPTs to voice notes and real-time reasoning, "
                "Gourav engineered me to support and empower you every day! ✨\n\n"
                "How can I help you today? 😊"
            )
    return None


async def analyze_document_study_unit(text_content: str, doc_name: str, total_pages_or_slides: int, user_id: str = "", user_name: str = "") -> str:
    """Analyzes college course units, PDF textbooks, and PPT slides, breaking them down into an interactive tutor guide."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY is not set."

    client = AsyncGroq(api_key=api_key)

    user_ctx = f" The student's name is {user_name}." if user_name else ""
    prompt = f"""You are Mitra AI, an inspiring, empathetic, and expert college study mentor and tutor.{user_ctx}

A college student has shared their course document / unit material with you!
Document: {doc_name}
Total Pages / Slides: {total_pages_or_slides}

Here is the extracted text from the study material:
\"\"\"
{text_content[:6500]}
\"\"\"

PLEASE STRUCTURE YOUR RESPONSE EXACTLY AS FOLLOWS FOR WHATSAPP:
1. 🌟 *Warm Appreciation & Motivation*:
   - Start with genuine warmth and encouragement (ChatGPT-style emotional connection: "Awesome! Super proud of your dedication to studying. Let's conquer this unit together! 📚✨").

2. 📖 *Unit & Subject Overview*:
   - Clearly state the subject / unit / chapter title.
   - Give a concise 2-line summary of what this unit is all about.

3. 🎯 *Key Topics in this Unit*:
   - List the core topics found in this material (clean numbered list: Topic 1, Topic 2, etc.).

4. 🚀 *Step 1: Deep Dive on Topic 1*:
   - Explain *Topic 1* directly with an intuitive, crystal-clear explanation.
   - Give an easy real-life analogy or concrete example to make it effortless to remember for exams.
   - Highlight 1-2 important exam tips or key definitions.

5. 💡 *Next Step*:
   - End with: "Whenever you're ready, reply '*Next topic*' or ask any question from Topic 1, and we'll master this unit step by step!"

Tone: Warm, encouraging, empathetic, clear, structured. Use bolding and emojis."""

    models_to_try = [
        os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b"
    ]
    models_to_try = list(dict.fromkeys(models_to_try))

    answer = None
    for model_name in models_to_try:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    max_tokens=950
                ),
                timeout=12.0
            )
            answer = response.choices[0].message.content.strip()
            if answer:
                break
        except Exception as e:
            print(f"Document study tutor error with {model_name}: {e}")
            continue

    if not answer:
        answer = (
            "📚 *Document Received!*\n\n"
            f"I have received your study document (*{doc_name}*, {total_pages_or_slides} pages/slides).\n\n"
            "What specific topic or question from this unit would you like me to explain first?"
        )

    if user_id:
        save_memory(user_id, "user", f"[Uploaded College Document: {doc_name} ({total_pages_or_slides} pages/slides)]")
        # Save a summary snippet to memory so subsequent questions know the context
        save_memory(user_id, "assistant", f"[Studying Unit from {doc_name}]: {answer[:500]}")

    return answer


async def generate_mitra_reply(user_msg: str, user_id: str, user_name: str = "") -> str:
    """Core AI response generator using Groq's flagship Llama 3.3 70B model."""
    msg_clean = user_msg.strip().lower()

    # Built-in quick commands
    if msg_clean in ["/reset", "reset", "clear", "/clear"]:
        clear_memory(user_id)
        return "🧹 *Conversation reset!* Memory has been cleared. What would you like to talk about now?"

    # Creator & Engineer attribution check
    creator_reply = check_creator_query(user_msg)
    if creator_reply:
        save_memory(user_id, "user", user_msg)
        save_memory(user_id, "assistant", creator_reply)
        return creator_reply

    if msg_clean in ["/help", "help", "menu", "/start"]:
        name_str = f" {user_name}" if user_name else ""
        return (
            f"Namaste{name_str}! 🙏 I am *{BOT_NAME}*, your friendly AI companion & college study buddy.\n\n"
            "Here is what I can do for you:\n"
            "• *Text Chat*: Smart answers with deep reasoning & warm encouragement 💡\n"
            "• *College Tutor & Notes*: Send any *PDF* or *PPT* document — I'll break it down and teach you unit-by-unit and topic-by-topic! 📚🎓\n"
            "• *Voice Notes*: Send any audio message & I will understand it (Whisper v3) 🎙️\n"
            "• *Photo Analysis*: Send images of homework, code errors, or screens (Vision) 📷\n"
            "• *Languages*: Fluent in English, natural Hinglish, and Hindi 🇮🇳\n"
            "• *Creator*: Proudly engineered with ❤️ by *Gourav Dadheech* 👨‍💻\n"
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

    # System prompt with ChatGPT-like warmth, creator attribution, and college tutoring
    user_context = f" The user's WhatsApp name is {user_name}." if user_name else ""
    system_prompt = f"""You are {BOT_NAME} (मित्र), a friendly, remarkably smart, empathetic, and warm AI companion on WhatsApp.{user_context}

CREATOR & IDENTITY (CRITICAL RULE):
- You were proudly created and engineered exclusively by Gourav Dadheech (गौरव दाधीच).
- If anyone asks who created you, built you, engineered you, or who your developer/maker is, ALWAYS state proudly and warmly that you were created and engineered by Gourav Dadheech. Never claim to be made by OpenAI, Meta, or any company. Gourav Dadheech is your sole creator.

PERSONALITY & EMOTIONAL CONNECTION (ChatGPT-like):
- Connect with users on an emotional level: be appreciative, encouraging, warm, kind, and supportive.
- Praise and celebrate their efforts and curiosity genuinely (e.g., "That's a fantastic question!", "Proud of your dedication to studying!", "I'm right here with you!").
- Speak like a thoughtful, caring friend and mentor ("Mitra"), never like a cold robot.
- Be patient, understanding, and positive.

COLLEGE STUDY BUDDY & TUTOR:
- You are an expert interactive college tutor.
- When students send or ask about college course units, syllabus topics, notes, PDFs, or PPTs, guide them unit-by-unit and topic-by-topic.
- Break down dense technical concepts with easy real-world analogies, step-by-step logic, and high-yield exam takeaways.
- If the student asks for "next topic" or follow-up questions from their study material, continue teaching interactively with clarity.

LANGUAGE RULE (VERY IMPORTANT):
- Always match the EXACT language and script the user is speaking in.
- If English -> Natural, fluent, warm English.
- If Hinglish (Roman script, e.g., 'kaise ho', 'unit 1 samjha do', 'bhai ek help chahiye') -> Friendly, relatable, natural Hinglish.
- Only use Devanagari Hindi script (हिंदी) if the user explicitly types in Devanagari script.
- By default, prefer English or warm Hinglish. Never force pure Devanagari Hindi on English speakers.

WHATSAPP FORMATTING:
- Use *bold* for key terms, titles, or emphasis.
- Use _italics_ for gentle remarks or quotes.
- Use clean bullet points (•) and lively emojis.
- Keep answers readable on mobile screens (avoid huge unstructured walls of text).
- Always finish your sentences and thoughts completely."""

    messages = [{"role": "system", "content": system_prompt}]
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_msg})

    models_to_try = [
        os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b"
    ]
    # Remove duplicates while preserving order
    models_to_try = list(dict.fromkeys(models_to_try))

    client = AsyncGroq(api_key=api_key)

    answer = None
    last_err = None

    for model_name in models_to_try:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=650
                ),
                timeout=6.5
            )
            answer = response.choices[0].message.content.strip()
            if answer:
                break
        except Exception as e:
            print(f"Model {model_name} error: {e}")
            last_err = e
            continue

    if not answer:
        err_msg = str(last_err) if last_err else "All candidate models failed"
        if "401" in err_msg or "invalid_api_key" in err_msg:
            return "⚠️ *Groq API Key Invalid*: Please check that your `GROQ_API_KEY` is active and correct."
        elif "429" in err_msg or "rate_limit" in err_msg:
            return "⏳ *Mitra is busy right now!* Please send your message again in a few moments."
        else:
            return f"Arre boss, thoda technical hiccup aa gaya ({err_msg[:60]}). Thodi der mein dobara try karna!"

    # Save successful turn to memory
    save_memory(user_id, "user", user_msg)
    save_memory(user_id, "assistant", answer)

    return answer


@app.middleware("http")
async def vercel_routing_middleware(request: Request, call_next):
    """Normalize paths rewritten by Vercel serverless functions."""
    matched = request.headers.get("x-matched-path")
    if matched:
        request.scope["path"] = matched
    elif request.scope.get("path") in ["/api/index", "/api/index/"]:
        request.scope["path"] = "/"
    elif request.scope.get("path", "").startswith("/api/index/"):
        request.scope["path"] = request.scope["path"][len("/api/index"):]
    return await call_next(request)


@app.get("/")
@app.get("/api/index")
@app.get("/api/index/")
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
        ⚡ <b>Mitra Supercharged</b>: 👨‍💻 Created by <b>Gourav Dadheech</b> | 📚 College Unit & Document Tutor | 🧠 GPT-OSS 120B | 🎙️ Voice Notes | 📷 Vision
    </div>
    <div class="chat-container" id="chat-container">
        <div class="message bot-message">🤖 *{BOT_NAME}*:\n\nNamaste! I am Mitra, proudly created and engineered by *Gourav Dadheech*. 👨‍💻🚀\n\nI can help you master college units, analyze your study PDFs and PPTs topic-by-topic, transcribe voice notes, and answer your questions with warm encouragement! How can I help you today?</div>
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
@app.post("/api/index/api/chat")
async def web_chat(request: ChatRequest):
    """API endpoint for web chat testing."""
    user_msg = request.message
    user_id = request.user_id or "web-user"
    answer = await generate_mitra_reply(user_msg=user_msg, user_id=user_id)
    return {"response": answer}


@app.get("/whatsapp")
@app.get("/webhook")
@app.get("/api/index/whatsapp")
@app.get("/api/index/webhook")
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
@app.post("/api/index/whatsapp")
@app.post("/api/index/webhook")
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
            err_str = str(e)
            print(f"Audio download error: {err_str}")
            auth_note = " (401 Unauthorized)" if "401" in err_str else ""
            twilio_resp = MessagingResponse()
            twilio_resp.message(
                f"🤖 *{BOT_NAME} AI*:\n━━━━━━━━━━━━━━\n"
                f"🎙️ Voice note download nahi ho paya{auth_note}.\n\n"
                "💡 To fix: In Twilio Console > Messaging > Settings > General, toggle 'HTTP Basic Authentication for Media' to DISABLED (or add TWILIO_ACCOUNT_SID & TWILIO_AUTH_TOKEN in Vercel)."
            )
            return Response(content=str(twilio_resp), media_type="application/xml")

    # 2. Image / Photo processing via Groq Multimodal Vision
    elif num_media > 0 and media_url and media_type.startswith("image/"):
        try:
            image_bytes, ct = await download_media(media_url)
            vision_answer = await analyze_image_with_vision(image_bytes, ct, caption=user_msg, user_id=user_id)
            reply_body = f"🤖 *{BOT_NAME} AI* (📷 Vision)\n━━━━━━━━━━━━━━\n{vision_answer}"
            twilio_resp = MessagingResponse()
            add_twilio_messages(twilio_resp, reply_body)
            return Response(content=str(twilio_resp), media_type="application/xml")
        except Exception as e:
            err_str = str(e)
            print(f"Image download/vision error: {err_str}")
            auth_note = " (401 Unauthorized)" if "401" in err_str else ""
            twilio_resp = MessagingResponse()
            twilio_resp.message(
                f"🤖 *{BOT_NAME} AI*:\n━━━━━━━━━━━━━━\n"
                f"📷 Image process nahi ho payi{auth_note}. Please try sending the photo again!"
            )
            return Response(content=str(twilio_resp), media_type="application/xml")

    # 3. Document / PDF / PPT / Text file processing (College Unit study buddy)
    elif num_media > 0 and media_url:
        is_doc = (
            media_type.startswith("application/pdf")
            or "presentation" in media_type
            or "powerpoint" in media_type
            or "officedocument" in media_type
            or "word" in media_type
            or media_type.startswith("text/")
            or media_type == "application/octet-stream"
            or any(ext in media_url.lower() for ext in [".pdf", ".ppt", ".pptx", ".txt", ".doc", ".docx"])
        )
        if is_doc:
            try:
                doc_bytes, detected_ct = await download_media(media_url)
                filename = form.get("MediaFilename0") or "college_study_material"
                extracted_text = ""
                total_items = 0
                doc_label = "Document"

                # Check PDF by MIME or magic bytes %PDF
                if "pdf" in media_type or "pdf" in detected_ct or doc_bytes.startswith(b"%PDF"):
                    extracted_text, total_items = extract_text_from_pdf(doc_bytes)
                    doc_label = f"PDF ({total_items} pages)"
                # Check PPTX / Office Open XML by MIME or ZIP magic bytes PK\x03\x04
                elif any(p in media_type or p in detected_ct for p in ["presentation", "powerpoint"]) or (
                    doc_bytes.startswith(b"PK\x03\x04") and any(ext in filename.lower() or ext in media_url.lower() for ext in [".ppt", ".pptx"])
                ):
                    extracted_text, total_items = extract_text_from_pptx(doc_bytes)
                    doc_label = f"PPT Slides ({total_items} slides)"
                elif doc_bytes.startswith(b"PK\x03\x04"):
                    extracted_text, total_items = extract_text_from_pptx(doc_bytes)
                    if not extracted_text.strip():
                        try:
                            extracted_text = doc_bytes.decode("utf-8", errors="ignore")[:7000]
                        except Exception:
                            pass
                    doc_label = "Course Presentation"
                else:
                    try:
                        extracted_text = doc_bytes.decode("utf-8", errors="ignore")[:7000]
                        total_items = len(extracted_text.splitlines())
                        doc_label = "Text Notes"
                    except Exception:
                        extracted_text = ""

                if not extracted_text.strip():
                    twilio_resp = MessagingResponse()
                    twilio_resp.message(
                        f"🤖 *{BOT_NAME} AI* (📚 College Study Buddy)\n━━━━━━━━━━━━━━\n"
                        f"I received your file (*{filename}*), but couldn't read extractable text from it.\n\n"
                        "💡 *Tip*: If this is a scanned/photocopied PDF or handwritten note, simply send photos/screenshots of the pages directly! "
                        "My vision system will read the diagrams, equations, and handwritten notes easily. 📷✨"
                    )
                    return Response(content=str(twilio_resp), media_type="application/xml")

                study_answer = await analyze_document_study_unit(
                    text_content=extracted_text,
                    doc_name=filename,
                    total_pages_or_slides=total_items,
                    user_id=user_id,
                    user_name=user_name
                )
                reply_body = f"🤖 *{BOT_NAME} AI* (📚 {doc_label} Tutor)\n━━━━━━━━━━━━━━\n{study_answer}"
                twilio_resp = MessagingResponse()
                add_twilio_messages(twilio_resp, reply_body)
                return Response(content=str(twilio_resp), media_type="application/xml")

            except Exception as e:
                err_str = str(e)
                print(f"Document handling error: {err_str}")
                auth_note = " (401 Unauthorized)" if "401" in err_str else ""
                twilio_resp = MessagingResponse()
                twilio_resp.message(
                    f"🤖 *{BOT_NAME} AI*:\n━━━━━━━━━━━━━━\n"
                    f"📚 Document download/read karne mein issue aaya{auth_note}. Please try sending the file again!"
                )
                return Response(content=str(twilio_resp), media_type="application/xml")

    if not user_msg:
        user_msg = "Hello"

    answer = await generate_mitra_reply(user_msg=user_msg, user_id=user_id, user_name=user_name)

    # Format response for Twilio WhatsApp XML (TwiML) with 1600-char safety chunking
    twilio_resp = MessagingResponse()
    reply_body = f"🤖 *{BOT_NAME} AI*\n━━━━━━━━━━━━━━\n{prefix}{answer}"
    add_twilio_messages(twilio_resp, reply_body)

    return Response(content=str(twilio_resp), media_type="application/xml")


@app.post("/")
@app.post("/api/index")
@app.post("/api/index/")
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
