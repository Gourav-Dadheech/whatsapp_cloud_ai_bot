================================================================================
MITRA AI - CLOUD-BASED WHATSAPP AI COMPANION, STUDY MENTOR & CREATIVE SUITE
================================================================================
Creator & Lead Engineer: Gourav Dadheech
Repository: https://github.com/Gourav-Dadheech/whatsapp_cloud_ai_bot
Live Deployment: https://whatsappcloudaibot.vercel.app
Webhook Endpoint: /whatsapp or /webhook

--------------------------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
Mitra (मित्र - Friend) is a supercharged, 24/7 cloud-native WhatsApp AI companion
engineered to deliver lightning-fast responses (~1.1 seconds), multimodal
intelligence (Text, Voice, Photos, Documents, and Image Generation), and personalized
college study mentoring, exam solving, and business document analysis.

Unlike local models that heat up your laptop and consume heavy RAM/battery,
Mitra is completely serverless and hosted on Vercel Fluid Compute. It runs with
zero sleep delays, handles concurrent users simultaneously, and maintains deep
long-term conversation memory with 100% individual user isolation.

--------------------------------------------------------------------------------
2. CORE CAPABILITIES & FEATURES
--------------------------------------------------------------------------------
• 💡 Lightning-Fast Reasoning & Chat:
  - Powered by Groq's high-performance LLM engines (OpenAI GPT-OSS 120B / 20B
    and Qwen 27B fallback).
  - Empathetic, supportive, ChatGPT-like conversational warmth and appreciation.
  - Fluent in English, natural Hinglish, and Devanagari Hindi.

• 🎨 High-Definition AI Image Generation (FLUX.1):
  - Generate ultra-detailed, photorealistic images directly in WhatsApp.
  - Triggers: "photo banao <anything>", "/image <prompt>", "/imagine <prompt>",
    "generate an image of <prompt>".
  - Uses Groq Qwen 27B for vivid cinematic prompt enhancement paired with FLUX.1.
  - Returns direct high-res JPEG image previews directly inside WhatsApp chats.

• 🧠 Active Multi-Turn Recall (Photos & Documents Continuity):
  - Solves the common AI limitation where follow-up questions lose track of previous
    attachments.
  - Active Document Recall: Upload a Question Paper or Notes once; ask follow-up
    questions like "explain q1", "now solve q3", or "what does section B mean"
    and Mitra recalls the exact document without asking you to resend.
  - Active Photo Recall: Send a photo or screenshot once; follow up with "how to
    use this?", "what was in the picture?", or "what is the second app?" and
    Mitra retains the full visual analysis context.

• 🌐 Real-Time Live Web Browsing:
  - Access up-to-the-minute information from the web.
  - Triggers automatically on queries like "latest news", "today weather in...",
    "live score", "stock/crypto price", or explicit commands like "/search <query>".
  - DuckDuckGo live search integration delivers fact-checked real-time answers.

• 💬 Native WhatsApp "typing..." Indicator & Read Receipts:
  - While Mitra thinks, reasons, or generates images, a native WhatsApp "... typing"
    status bubble appears under the contact name on the user's phone.
  - Automatically marks incoming messages with blue ticks (read receipt).
  - Web chat preview displays an authentic WhatsApp animated 3-dot typing bubble
    and green "typing..." header status.

• 🎙️ WhatsApp Voice Note Transcription:
  - Send audio voice notes directly on WhatsApp.
  - Transcribed in ~300ms using Groq Whisper-large-v3-turbo.
  - Mitra understands your spoken voice and replies with full context.

• 📷 Multimodal Vision & Image Recognition:
  - Powered by Groq Qwen 2.5 / 3.8 27B Vision.
  - Recognizes screenshots, computer screens, taskbars (Windows vs. macOS),
    homework equations, receipts, scientific calculators, and everyday objects.

• 📄 Versatile Document Intelligence (PDF, PPT/PPTX, Word, Text):
  - 📝 Exam Question Papers & Assignments: Analyzes marks, sections, questions,
    and provides step-by-step solutions or guidance.
  - 💼 Business Invoices & Contracts: Extracts parties, amounts, GST, payment
    deadlines, deliverables, and action items.
  - 📚 College Study Units: Teaches topics unit-by-unit with real-world analogies
    and exam takeaways.
  - 💬 Direct Caption Priority: If a user sends a document with a caption
    (e.g., "Solve Q3", "Check GST total"), Mitra prioritizes answering that request.

• 🔒 Long-Term Memory with 100% User Isolation:
  - Powered by Upstash Redis Cloud with 30-day persistence.
  - Up to 50 conversation turns remembered per user.
  - Isolated by WhatsApp phone number (`mitra_mem_{user_id}`) so no user's
    conversation ever mixes with another user's chat.

• 🛡️ WhatsApp Safety Chunking:
  - Automatically splits long messages exceeding Twilio's 1600-character limit
    across clean paragraph chunks so no answers get cut off.

--------------------------------------------------------------------------------
3. CONCURRENCY, CAPACITY & USAGE LIMITS
--------------------------------------------------------------------------------
How many users can use Mitra simultaneously?
- Vercel Serverless Architecture:
  * Scales automatically from 0 to 1,000+ concurrent requests.
  * Every incoming webhook is handled in an isolated ephemeral execution context.
  * 50 to 500+ active users can chat simultaneously without server queuing or lag.

API Rate Limits & Free Tiers Breakdown:
• Text Chat & Reasoning (Groq Cloud):
  * Standard tier allows up to 30 requests/minute and ~14,400 requests/day.
  * Safe usage: ~1,000+ full multi-turn conversations daily.
  * Memory is capped at 50 messages per user to preserve fast context windows.

• Voice Notes (Groq Whisper v3):
  * Whisper rate limit: ~2,000 audio minutes per day.
  * Transcribes 30-second voice notes at sub-second speeds.

• Vision Analysis (Groq Qwen Vision):
  * Up to 30 vision requests per minute.
  * Handles hundreds of photo questions per day seamlessly.

• Image Generation (FLUX.1 via Pollinations):
  * 100% free and uncapped public FLUX.1 endpoint.
  * No per-user API key required; generates 1024x1024 high-resolution images.

• Upstash Redis Cloud (Memory Store):
  * Free tier provides 10,000 commands per day and 256MB storage.
  * Easily supports 50-100 daily active users with 30-day chat persistence.

--------------------------------------------------------------------------------
4. ARCHITECTURE & TECH STACK
--------------------------------------------------------------------------------
• Framework: FastAPI (Python 3.12)
• Cloud Hosting: Vercel Serverless Functions (Global Edge / Washington DC)
• AI Inference: Groq Cloud API (LPU hardware acceleration)
• Voice Model: Groq Whisper-large-v3-turbo / Whisper-large-v3
• Vision Model: Groq Qwen/qwen3.8-27b (Multimodal Vision)
• Text LLM: Groq openai/gpt-oss-20b & openai/gpt-oss-120b
• Image Generation: Pollinations AI FLUX.1 with LLM prompt enhancement
• Live Search: DuckDuckGo Search API (ddgs / instant answer fallback)
• Document Parsers: pypdf (PDFs), zipfile/xml.etree (PPTX/Office XML)
• Database & Cache: Upstash Redis REST Cloud (with local JSON fallback)
• Messaging Gateway: Twilio WhatsApp API & Meta WhatsApp Cloud API

--------------------------------------------------------------------------------
5. ENVIRONMENT VARIABLES
--------------------------------------------------------------------------------
GROQ_API_KEY              - Groq Cloud API key for LLM, Whisper, and Vision
GROQ_MODEL                - Preferred model (default: openai/gpt-oss-20b)
BOT_NAME                  - Display name (default: Mitra)
UPSTASH_REDIS_REST_URL    - Upstash Redis REST endpoint for memory
UPSTASH_REDIS_REST_TOKEN  - Upstash Redis REST authorization token
TWILIO_ACCOUNT_SID        - Twilio Account SID (Required to display native WhatsApp '... typing' popup)
TWILIO_AUTH_TOKEN         - Twilio Auth Token (Required to display native WhatsApp '... typing' popup)
MAX_MEMORY_MESSAGES       - History turns retained per user (default: 50)

--------------------------------------------------------------------------------
6. FUTURE ROADMAP & PLANNED INNOVATIONS
--------------------------------------------------------------------------------
[x] Real-time Web Search & Browsing (DuckDuckGo integration)
[x] AI Image Generation with FLUX.1
[x] Active Document & Photo Multi-turn Memory Recall
[ ] Interactive Quiz & Flashcard Mode: Auto-generate flashcards from uploaded study notes
[ ] Audio Voice Reply: Send spoken voice audio back to user using ElevenLabs / edge-tts
[ ] Code Execution Sandbox: Run small Python snippets in real-time to verify math calculations
[ ] WhatsApp Interactive Buttons & List Menus (via Meta Cloud API)

--------------------------------------------------------------------------------
7. CREATOR ATTRIBUTION
--------------------------------------------------------------------------------
Mitra was engineered and designed by Gourav Dadheech.
When asked "Who created you?", "Who engineered you?", or "kisne banaya?",
Mitra proudly attributes its creation to Gourav Dadheech.
================================================================================
