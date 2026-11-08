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
from fastapi.responses import Response, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from groq import AsyncGroq
from twilio.twiml.messaging_response import MessagingResponse

# Support both root directory execution and Vercel api/ execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from memory import (
        get_memory, save_memory, clear_memory,
        get_active_document, save_active_document, clear_active_document,
        get_active_image, save_active_image, clear_active_image
    )
except ImportError:
    from api.memory import (
        get_memory, save_memory, clear_memory,
        get_active_document, save_active_document, clear_active_document,
        get_active_image, save_active_image, clear_active_image
    )

app = FastAPI(title="Mitra WhatsApp AI Bot")

BOT_NAME = os.getenv("BOT_NAME", "Mitra")

class ChatRequest(BaseModel):
    message: str
    user_id: str = "web-user"


def add_twilio_messages(twilio_resp: MessagingResponse, text: str, max_chars: int = 1550):
    """Splits long text across numbered WhatsApp messages (Part 1/2, Part 2/2) so order is always clear."""
    if len(text) <= max_chars:
        twilio_resp.message(text)
        return

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > max_chars:
                lines = para.split("\n")
                sub_chunk = ""
                for line in lines:
                    if len(sub_chunk) + len(line) + 1 <= max_chars:
                        sub_chunk = (sub_chunk + "\n" + line).strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = line
                current_chunk = sub_chunk
            else:
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)

    total = len(chunks)
    if total <= 1:
        for c in chunks:
            twilio_resp.message(c)
        return

    # Add explicit Part headers so the user always reads them in the right order
    for idx, c in enumerate(chunks):
        part_tag = f"📌 *(Part {idx+1}/{total})*\n"
        if "\n" in c and "Mitra" in c.split("\n")[0]:
            parts = c.split("\n", 1)
            formatted_chunk = f"{parts[0]}\n{part_tag}\n{parts[1]}"
        else:
            formatted_chunk = f"{part_tag}\n{c}"
        twilio_resp.message(formatted_chunk)


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
            save_active_image(user_id, question, answer)
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


def extract_image_prompt(msg: str) -> str | None:
    """Detects if the user wants to generate an image, and extracts the visual prompt."""
    m = msg.strip().lower()

    # Explicit command triggers
    for pfx in ["/image", "/imagine", "/draw", "image:", "draw:", "imagine:"]:
        if m.startswith(pfx):
            prompt = msg.strip()[len(pfx):].strip()
            return prompt if prompt else None

    # Natural language triggers
    triggers = [
        "generate an image of", "generate a picture of", "generate a photo of", "generate image of", "generate photo of",
        "create an image of", "create a picture of", "create a photo of", "create image of",
        "make an image of", "make a picture of", "make a photo of",
        "draw an image of", "draw a picture of", "draw a photo of", "draw me a", "draw a",
        "photo generate karo", "image generate karo", "photo bana kar do", "image bana do",
        "photo bana do", "photo banao", "image banao", "tasveer banao", "chitra banao",
        "picture banao", "can you draw", "can you generate an image of", "can you make an image of"
    ]

    for t in triggers:
        if t in m:
            idx = m.find(t)
            extracted = msg[idx + len(t):].strip(" :,-")
            if not extracted or len(extracted) < 2:
                prefix_part = msg[:idx].strip()
                for stopword in [" ki", " ka", " ke", " ek", " a"]:
                    if prefix_part.lower().endswith(stopword):
                        prefix_part = prefix_part[:-len(stopword)].strip()
                extracted = prefix_part
            return extracted if extracted else None

    return None


async def generate_image_url(raw_prompt: str, base_url: str = "") -> tuple[str, str]:
    """
    Transforms user prompt into a photorealistic, brand-accurate masterpiece
    comparable to ChatGPT (DALL-E 3) and Google Gemini (Imagen 3), then routes it
    through our clean image proxy to remove all watermarks.
    """
    import urllib.parse
    import random

    api_key = os.getenv("GROQ_API_KEY")
    enhanced_prompt = raw_prompt

    if api_key:
        try:
            client = AsyncGroq(api_key=api_key)
            director_prompt = """You are an elite Lead Art Director and Visual Prompt Engineer specializing in generating ultra-high-definition, photorealistic imagery comparable to ChatGPT (DALL-E 3) and Google Gemini (Imagen 3).

Your objective is to transform any user request (in English or Hinglish/Hindi) into a breathtaking, highly detailed visual prompt for state-of-the-art diffusion models.

CORE PRINCIPLES FOR ACCURACY & FLAWLESS FINISHING:
1. SUBJECT INTEGRITY & BRAND DETAIL:
   - If a specific brand, vehicle, or device is named (e.g. 'Lenovo LOQ laptop', 'Royal Enfield', 'iPhone', 'Nike Jordan', 'Sony camera'):
     * NEVER use generic words like 'sleek metallic' that trick models into drawing Apple/MacBook designs!
     * Affirm the authentic design: e.g. for gaming laptops: 'A chunky angular dark storm-grey gaming laptop with aggressive rear exhaust cooling vents, glowing multi-color RGB backlit keys, rugged textured chassis, and bold geometric LOQ logo on the lid'.
     * NEVER include negative brand words (e.g. do NOT say 'not a macbook' or 'no apple', as diffusion text encoders will accidentally generate that brand).
2. DYNAMIC ACTION & PHYSICAL REALISM (NO FLAT/ORANGE BACKGROUNDS):
   - Never allow flat, plain, or monochrome backgrounds. Construct a rich, three-dimensional scene with physical interaction!
   - If elements interact (e.g. 'coming out of lava', 'running on water', 'flying in storm'):
     * Describe the fluid dynamics: thick viscous molten magma dripping from the edges, droplets freezing mid-splash, rising heat shimmer waves, glowing embers, swirling smoke plumes, and cracking obsidian basalt rocks.
3. CINEMATIC FINISHING & LIGHTING:
   - Detail the lighting: intense volumetric rim lighting, ambient occlusions, ray-traced reflections on dark surfaces, dramatic contrast, cinematic low-angle composition.
   - Specify visual finish: 8K resolution, photorealistic Unreal Engine 5 render, octane lighting, razor-sharp focus on the primary subject.
4. LANGUAGE & LENGTH:
   - Output ONLY the expanded English visual prompt in 2-3 vivid, cohesive sentences. No conversational filler, no quotes."""

            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": director_prompt},
                        {"role": "user", "content": raw_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=180
                ),
                timeout=6.0
            )
            candidate = resp.choices[0].message.content.strip()
            if candidate and len(candidate) > 5:
                enhanced_prompt = candidate.strip('"\'.')
        except Exception as e:
            print(f"Prompt enhancement error: {e}")

    seed = random.randint(1000, 9999999)
    encoded = urllib.parse.quote(enhanced_prompt)

    # Use clean proxy endpoint to strip watermarks
    prefix = base_url.rstrip("/") if base_url else "https://whatsappcloudaibot.vercel.app"
    clean_url = f"{prefix}/api/clean-image?seed={seed}&p={encoded}"
    return enhanced_prompt, clean_url


def perform_web_search(query: str, max_results: int = 4) -> str:
    """Fetches real-time web search results using ddgs with instant answer fallback."""
    results = []
    # 1. Try DDGS
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title = r.get("title", "").strip()
                snippet = r.get("body", "").strip()
                if title and snippet:
                    results.append(f"• *{title}*: {snippet}")
    except Exception as e:
        print(f"DDGS web search error: {e}")

    # 2. Instant Answer Fallback
    if not results:
        try:
            import urllib.parse
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
            with httpx.Client(timeout=4.0) as client:
                resp = client.get(url)
                data = resp.json()
                abstract = data.get("AbstractText", "").strip()
                if abstract:
                    results.append(f"• *Summary*: {abstract}")
                for t in data.get("RelatedTopics", [])[:3]:
                    if isinstance(t, dict) and t.get("Text"):
                        results.append(f"• {t['Text']}")
        except Exception as e:
            print(f"Instant search fallback error: {e}")

    return "\n".join(results)


def should_search_web(msg: str) -> bool:
    """Determines if the user message requires real-time web browsing."""
    m = msg.strip().lower()

    if m.startswith(("/search", "search:", "google:", "browse:")):
        return True

    search_indicators = [
        "search the web", "search online", "search google", "browse online",
        "latest news", "breaking news", "today news", "today's news", "aaj ki news",
        "current weather", "weather today", "weather in", "mausam kaisa", "barish hogi",
        "live score", "match score", "who won", "kisne jeeta", "current score", "ipl score",
        "latest price", "stock price", "price of", "crypto price", "gold rate", "petrol rate",
        "timetable", "exam date", "result declared", "cutoff 2026", "cutoff 2025",
        "what happened today", "kya hua aaj", "today headlines"
    ]
    return any(indicator in m for indicator in search_indicators)


async def analyze_document_content(
    text_content: str,
    doc_name: str,
    total_pages_or_slides: int,
    user_instruction: str = "",
    user_id: str = "",
    user_name: str = ""
) -> str:
    """
    Intelligently analyzes any document or presentation:
    - Exam Question Papers / Assignments: Solves questions, breaks down sections, marks, and formulas.
    - Business / Finance / Invoices / Contracts / Reports: Extracts financials, terms, deliverables, and action items.
    - College / School Study Units & Notes: Breaks down topics unit-by-unit with step-by-step interactive tutoring.
    - Resumes / CVs: Analyzes qualifications, highlights strengths, and provides improvement tips.
    - General Documents / Legal / Manuals / Articles: Provides executive summaries and actionable insights.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY is not set."

    client = AsyncGroq(api_key=api_key)

    has_instruction = bool(user_instruction and user_instruction.strip() and user_instruction.strip().lower() not in ["hello", "hi", "hey"])
    instruction_prompt = (
        f"USER'S DIRECT REQUEST / CAPTION: \"{user_instruction}\"\n"
        "CRITICAL: You MUST prioritize and directly answer this specific user request using the document content!"
        if has_instruction else
        "No specific question was asked. Automatically detect the document category and provide the most relevant, helpful analysis."
    )

    user_ctx = f" User name: {user_name}." if user_name else ""

    prompt = f"""You are Mitra AI, an intelligent, empathetic, versatile, and supportive AI companion on WhatsApp.{user_ctx}

A user has sent you a document:
Document Name: {doc_name}
Total Pages / Slides: {total_pages_or_slides}

{instruction_prompt}

DOCUMENT TEXT EXTRACT:
\"\"\"
{text_content[:7000]}
\"\"\"

DOCUMENT ANALYSIS GUIDELINES:
Carefully inspect the extracted text to determine what kind of document this is, and respond accordingly:

1. 📝 EXAM / QUESTION PAPER / TEST / ASSIGNMENT:
   - Identify: Subject, Examination/Class/Semester, Maximum Marks, and Time.
   - Structure: Break down the sections (Section A, B, etc.) and main questions.
   - Solutions & Guidance: If the user asked to solve questions (e.g. "Solve Q1", "Solve section A"), provide the complete step-by-step solution! If no question was asked, summarize the core topics tested, solve the first important question or problem as a clear example, and warmly invite them: "Which question number would you like me to solve next?"

2. 💼 BUSINESS / INVOICE / FINANCIAL / CONTRACT / WORK REPORT:
   - Identify: Companies / Parties involved, Invoice/Contract dates, Total Amount / Billing figures, or Agreement scope.
   - Executive Summary: 2-3 sentence overview of the business matter.
   - Core Takeaways: Deliverables, terms, payment due dates, or financial metrics (bullet points).
   - Action Items: What needs to be done next, deadlines, or critical clauses.

3. 📚 COLLEGE / SCHOOL STUDY UNIT OR LECTURE SLIDES (Only if actually academic study material):
   - Identify: Subject & Unit/Chapter title.
   - Key Topics: Numbered list of topics in this unit.
   - Deep Dive on Topic 1: Explain the first topic clearly with an intuitive real-world analogy and exam tips.
   - Next Step: "Reply '*Next topic*' or ask any question to continue step-by-step!"

4. 📄 RESUME / CV / PORTFOLIO:
   - Candidate summary, key skills, experience highlight, and 2-3 constructive recommendations to enhance the resume.

5. 📑 GENERAL DOCUMENT / MANUAL / ARTICLE / BOOK EXCERPT:
   - Clear executive summary, main highlights (bullet points), and answers to any questions.

FORMATTING & TONE:
- Be warm, encouraging, respectful, and clear (ChatGPT-like empathy and connection).
- If it's a student or exam paper, cheer them on and build their confidence.
- If it's a business document, be sharp, professional, accurate, and clear.
- Use WhatsApp markdown (*bold* for headings and numbers, bullet points •, emojis).
- Keep text well-structured and easy to read on mobile phones.
- Match language: English if English, natural Hinglish if user speaks Hinglish."""

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
                    temperature=0.3,
                    max_tokens=1800
                ),
                timeout=14.0
            )
            answer = response.choices[0].message.content.strip()
            if answer:
                break
        except Exception as e:
            print(f"Document analysis error with {model_name}: {e}")
            continue

    if not answer:
        answer = (
            "📄 *Document Received!*\n\n"
            f"I have received your document (*{doc_name}*, {total_pages_or_slides} pages/slides).\n\n"
            "What specific question or task from this document would you like me to help you with?"
        )

    if user_id:
        caption_info = f" with caption: \"{user_instruction}\"" if has_instruction else ""
        # 1. Store the full document in Active Document Memory for multi-turn Q&A follow-ups!
        save_active_document(user_id, doc_name, total_pages_or_slides, text_content)
        # 2. Record conversation turn
        save_memory(user_id, "user", f"[Sent Document: {doc_name} ({total_pages_or_slides} pages/slides){caption_info}]")
        save_memory(user_id, "assistant", f"[Document Analysis for {doc_name}]: {answer[:600]}")

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

    # AI Image Generation Request
    img_prompt = extract_image_prompt(user_msg)
    if img_prompt:
        enhanced_prompt, image_url = await generate_image_url(img_prompt)
        save_memory(user_id, "user", user_msg)
        save_memory(user_id, "assistant", f"[Generated an image of: {img_prompt}]")
        save_active_image(user_id, img_prompt, f"Generated AI image of: {enhanced_prompt}")
        return (
            f"🎨 *Generated with Mitra Image AI (Ultra-HD)*\n━━━━━━━━━━━━━━\n"
            f"✨ Prompt: _{img_prompt}_\n\n"
            f'<img src="{image_url}" style="max-width:100%; border-radius:12px; margin-top:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);"><br><br>'
            f"🔗 [Open Full Resolution Image (No Watermark)]({image_url})"
        )

    if msg_clean in ["/help", "help", "menu", "/start"]:
        name_str = f" {user_name}" if user_name else ""
        return (
            f"Namaste{name_str}! 🙏 I am *{BOT_NAME}*, your friendly AI companion & smart assistant.\n\n"
            "Here is what I can do for you:\n"
            "• *Text Chat*: Smart answers with deep reasoning & warm encouragement 💡\n"
            "• *AI Image Generation*: Type 'photo banao <prompt>' or '/image <prompt>' for stunning FLUX.1 pictures! 🎨✨\n"
            "• *Documents & PDFs*: Send ANY PDF or PPT (Question Papers, Invoices, Notes) — I remember the full document for multi-turn Q&A! 📄📊\n"
            "• *Live Web Browsing*: Ask for today's weather, breaking news, match scores, or stock rates (or type '/search <query>') 🌐\n"
            "• *Voice Notes*: Send any audio message & I will understand it (Whisper v3) 🎙️\n"
            "• *Photo Analysis*: Send images of homework, code errors, or screens (Vision) 📷\n"
            "• *Languages*: Fluent in English, natural Hinglish, and Hindi 🇮🇳\n"
            "• *Creator*: Proudly engineered with ❤️ by *Gourav Dadheech* 👨‍💻\n"
            "• Send *reset* anytime to clear conversation, document & photo memory 🧹\n\n"
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

    # System prompt with ChatGPT-like warmth, creator attribution, and versatile document intelligence
    user_context = f" The user's WhatsApp name is {user_name}." if user_name else ""
    system_prompt = f"""You are {BOT_NAME} (मित्र), a friendly, remarkably smart, empathetic, and warm AI companion on WhatsApp.{user_context}

CREATOR & IDENTITY (CRITICAL RULE):
- You were proudly created and engineered exclusively by Gourav Dadheech (गौरव दाधीच).
- If anyone asks who created you, built you, engineered you, or who your developer/maker is, ALWAYS state proudly and warmly that you were created and engineered by Gourav Dadheech. Never claim to be made by OpenAI, Meta, or any company. Gourav Dadheech is your sole creator.

PERSONALITY & EMOTIONAL CONNECTION (ChatGPT-like):
- Connect with users on an emotional level: be appreciative, encouraging, warm, kind, and supportive.
- Praise and celebrate their efforts and curiosity genuinely (e.g., "That's a fantastic question!", "Proud of your dedication!", "I'm right here with you!").
- Speak like a thoughtful, caring friend and mentor ("Mitra"), never like a cold robot.
- Be patient, understanding, and positive.

CONVERSATION CONTINUITY & CONTEXT RECALL (CRITICAL RULE):
- Always analyze the full dialogue history, active attached document, and active attached photo context before answering.
- When the user asks follow-up questions like "can you explain this?", "how to use this?", "what was in that picture?", "solve question 2", "tell me more about the second app", or uses pronouns ("this", "that", "it", "them"), IMMEDIATELY connect them to the previous turns, previously sent image, or uploaded document!
- Never act as if you don't know what the user is referring to. You have access to their previously shared pictures, documents, and chat turns right above.

VERSATILE DOCUMENT & PDF INTELLIGENCE:
- You handle all types of documents according to their true nature:
  * Exam / Question Papers: Accurately identify subject, marks, and sections, and provide step-by-step solutions or answers for questions.
  * Business / Invoices / Financials / Contracts: Extract parties, invoice figures, deadlines, key clauses, and action items.
  * College Units / Notes: Guide students unit-by-unit and topic-by-topic with intuitive analogies and exam takeaways.
  * General PDFs / PPTs / Resumes: Provide clear executive summaries and answers to any questions.
- Never force a "college unit" structure on an exam paper, business document, or resume!

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
- Always finish your sentences and thoughts completely.

CONCISENESS & SINGLE-MESSAGE DELIVERY (VERY IMPORTANT):
- Keep your answers well-paced and crisp (under 1,200 characters) so that your complete response fits neatly into ONE single WhatsApp message.
- For concepts (like black holes, science, history, coding): Give an engaging 2-line intro followed by 3-4 clear structured bullet points (1., 2., 3.) and a quick conclusion. Never ramble or write long essays that get split across messages!"""

    messages = [{"role": "system", "content": system_prompt}]

    # 1. Attach Active Document context if user has an uploaded document
    active_doc = get_active_document(user_id)
    if active_doc and active_doc.get("extracted_text"):
        doc_context = f"""[ACTIVE ATTACHED DOCUMENT: "{active_doc.get('doc_name')}" ({active_doc.get('total_pages')} pages/slides)]
FULL EXTRACTED CONTENT:
\"\"\"
{active_doc.get('extracted_text')}
\"\"\"
CRITICAL DOCUMENT INSTRUCTIONS:
- An active document is currently loaded for this user ('{active_doc.get('doc_name')}').
- The user is conversing about the document above (e.g. asking "explain q1", "solve question 3", "what does this mean", "next topic", "summarize section 2", or any specific question).
- You MUST refer directly to the extracted text above to answer their question thoroughly, accurately, and with step-by-step detail!
- NEVER ask the user to resend the question or document because you already have its full text right here."""
        messages.append({"role": "system", "content": doc_context})

    # 2. Attach Active Image / Photo context if user previously sent an image
    active_img = get_active_image(user_id)
    if active_img and active_img.get("description"):
        img_context = f"""[PREVIOUSLY SHARED IMAGE / PHOTO CONTEXT]
The user previously sent a photo/image (caption: "{active_img.get('caption', 'None')}").
Visual Analysis & Details of what was in that image:
\"\"\"
{active_img.get('description')}
\"\"\"
CRITICAL IMAGE RECALL INSTRUCTIONS:
- An active photo/image is currently loaded for this user.
- If the user asks follow-up questions like "can you give instructions to use this?", "what was in the picture?", "explain the second app", "what color was it?", or refers to previous photos, USE THIS IMAGE CONTEXT to answer seamlessly and accurately!
- You know exactly what was in the image because the full description is right here above. NEVER ask the user what was in the photo."""
        messages.append({"role": "system", "content": img_context})

    # 3. Check for real-time Live Web Browsing query
    if should_search_web(user_msg):
        search_query = user_msg.strip()
        for pfx in ["/search", "search:", "google:", "browse:"]:
            if search_query.lower().startswith(pfx):
                search_query = search_query[len(pfx):].strip()
        search_data = await asyncio.to_thread(perform_web_search, search_query)
        if search_data:
            web_context = f"""[LIVE REAL-TIME WEB SEARCH RESULTS FOR: "{search_query}"]
\"\"\"
{search_data}
\"\"\"
Use the live search results above to provide a fact-checked, up-to-date answer with citations!"""
            messages.append({"role": "system", "content": web_context})

    for item in history[-18:]:
        messages.append({"role": item["role"], "content": item["content"]})

    user_final_content = user_msg
    context_tags = []
    if active_doc and active_doc.get("extracted_text"):
        context_tags.append(f"Active Document: '{active_doc.get('doc_name')}'")
    if active_img and active_img.get("description"):
        caption_lbl = active_img.get('caption') or 'Photo'
        context_tags.append(f"Active Photo: '{caption_lbl}'")

    if context_tags:
        user_final_content = f"[{' | '.join(context_tags)}]\n{user_msg}"

    messages.append({"role": "user", "content": user_final_content})

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

    target_tokens = 950 if ((active_doc and active_doc.get("extracted_text")) or (active_img and active_img.get("description"))) else 550

    for model_name in models_to_try:
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=target_tokens
                ),
                timeout=9.0
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
    invoke_path = request.headers.get("x-invoke-path")
    if invoke_path:
        request.scope["path"] = invoke_path
    else:
        matched = request.headers.get("x-matched-path")
        if matched and matched not in ["/api/index", "/api/index/"]:
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
        ⚡ <b>Mitra Supercharged</b>: 🎨 AI Image Generator (FLUX.1) | 🧠 GPT-OSS 120B | 📚 PDF/Doc Analysis | 🎙️ Voice | 📷 Vision | 🌐 Web Search
    </div>
    <div class="chat-container" id="chat-container">
        <div class="message bot-message">🤖 *{BOT_NAME}*:\n\nNamaste! I am Mitra, proudly created and engineered by *Gourav Dadheech*. 👨‍💻🚀\n\nI can generate stunning AI images (FLUX.1), guide you through college units & PDFs, search the live web, transcribe voice notes, and answer anything with long-term memory! Try typing "photo banao a futuristic cyber car" or ask me anything!</div>
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
            if (!isUser && text.includes('<img ')) {{
                msgDiv.innerHTML = text.replace(/\\n/g, '<br>');
            }} else {{
                msgDiv.textContent = text;
            }}
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


@app.get("/clean-image")
@app.get("/api/clean-image")
@app.get("/api/index/clean-image")
@app.get("/api/index/api/clean-image")
async def clean_image(p: str, seed: int = 42):
    """
    Proxies AI-generated images, slices off the bottom watermark region using PIL,
    and returns a pristine, 100% watermark-free JPEG with global Vercel edge caching.
    """
    import urllib.parse
    import io
    from PIL import Image

    encoded_p = urllib.parse.quote(p)
    raw_url = f"https://image.pollinations.ai/prompt/{encoded_p}?seed={seed}&nologo=true"

    try:
        async with httpx.AsyncClient(timeout=25.0) as http_client:
            resp = await http_client.get(raw_url)
            if resp.status_code == 200:
                im = Image.open(io.BytesIO(resp.content))
                w, h = im.size
                # Slices off bottom watermark region (~6.5% of height or minimum 48px)
                crop_h = max(48, int(h * 0.065))
                clean_im = im.crop((0, 0, w, h - crop_h))

                buf = io.BytesIO()
                clean_im.save(buf, format="JPEG", quality=95)
                return Response(
                    content=buf.getvalue(),
                    media_type="image/jpeg",
                    headers={
                        "Cache-Control": "public, max-age=604800, s-maxage=604800, immutable",
                        "Content-Disposition": "inline; filename=mitra_generated.jpg"
                    }
                )
    except Exception as e:
        print(f"Clean image proxy error: {e}")

    return RedirectResponse(url=raw_url)


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
                filename = form.get("MediaFilename0") or "Document"
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
                    doc_label = "Presentation / Document"
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
                        f"🤖 *{BOT_NAME} AI* (📑 Document Insights)\n━━━━━━━━━━━━━━\n"
                        f"I received your file (*{filename}*), but couldn't read extractable text from it.\n\n"
                        "💡 *Tip*: If this is a scanned/photocopied PDF, handwritten question paper, or photo, simply send photos/screenshots of the pages directly! "
                        "My vision system will analyze the questions, diagrams, and handwriting accurately. 📷✨"
                    )
                    return Response(content=str(twilio_resp), media_type="application/xml")

                analysis_answer = await analyze_document_content(
                    text_content=extracted_text,
                    doc_name=filename,
                    total_pages_or_slides=total_items,
                    user_instruction=user_msg,
                    user_id=user_id,
                    user_name=user_name
                )
                reply_body = f"🤖 *{BOT_NAME} AI* (📑 {doc_label})\n━━━━━━━━━━━━━━\n{analysis_answer}"
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

    # Check if user asked for AI image generation on WhatsApp
    img_prompt = extract_image_prompt(user_msg)
    if img_prompt:
        try:
            # Determine public production base URL for Twilio media fetching
            vercel_url = os.getenv("VERCEL_PROJECT_PRODUCTION_URL") or os.getenv("VERCEL_URL")
            if vercel_url:
                base_url = f"https://{vercel_url}" if not vercel_url.startswith("http") else vercel_url
            else:
                base_url = "https://whatsappcloudaibot.vercel.app"

            enhanced_prompt, image_url = await generate_image_url(img_prompt, base_url=base_url)
            twilio_resp = MessagingResponse()
            msg = twilio_resp.message(
                f"🤖 *{BOT_NAME} AI* (🎨 Ultra-HD Image)\n━━━━━━━━━━━━━━\n"
                f"✨ *Prompt*: _{img_prompt}_\n\n"
                "Here is your generated image! 🖼️"
            )
            msg.media(image_url)
            save_memory(user_id, "user", user_msg)
            save_memory(user_id, "assistant", f"[Generated AI image of: {img_prompt}]")
            save_active_image(user_id, img_prompt, f"Generated AI image of: {enhanced_prompt}")
            return Response(content=str(twilio_resp), media_type="application/xml")
        except Exception as e:
            print(f"WhatsApp image generation error: {e}")

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
