================================================================================
MITRA AI - CLOUD-BASED WHATSAPP AI ASSISTANT & STUDY MENTOR
================================================================================
Creator & Lead Engineer: Gourav Dadheech
Repository: https://github.com/Gourav-Dadheech/whatsapp_cloud_ai_bot
Live Deployment: https://whatsappcloudaibot.vercel.app
Webhook Endpoint: /whatsapp

--------------------------------------------------------------------------------
1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
Mitra (मित्र - Friend) is a supercharged, 24/7 cloud-native WhatsApp AI companion
engineered to deliver lightning-fast responses (~1.1 seconds), multimodal
intelligence (Text, Voice, Photos, and Documents), and personalized college study
mentoring and business document analysis.

Unlike local models that heat up your laptop and consume heavy RAM/battery,
Mitra is completely serverless and hosted in the cloud. It runs with zero sleep
delays, handles multiple users simultaneously, and maintains deep long-term
conversation memory with 100% individual user isolation.

--------------------------------------------------------------------------------
2. CORE CAPABILITIES & FEATURES
--------------------------------------------------------------------------------
• 💡 Lightning-Fast Reasoning & Chat:
  - Powered by Groq's high-performance LLM engines (OpenAI GPT-OSS 120B / 20B
    and Qwen 27B fallback).
  - Empathetic, supportive, ChatGPT-like conversational warmth.
  - Fluent in English, natural Hinglish, and Devanagari Hindi.

• 🎙️ WhatsApp Voice Note Transcription:
  - Send audio voice notes directly on WhatsApp.
  - Transcribed in ~300ms using Groq Whisper-large-v3-turbo.
  - Mitra understands your spoken voice and replies with full context.

• 📷 Multimodal Vision & Image Recognition:
  - Powered by Qwen 2.5 / 3.8 27B Vision.
  - Recognizes screenshots, computer screens, taskbars (Windows vs. macOS),
    homework equations, receipts, and everyday objects.

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
3. ARCHITECTURE & TECH STACK
--------------------------------------------------------------------------------
• Framework: FastAPI (Python 3.12)
• Cloud Hosting: Vercel Serverless Functions (Global Edge / Washington DC)
• AI Inference: Groq Cloud API (LPU hardware acceleration)
• Voice Model: Groq Whisper-large-v3-turbo / Whisper-large-v3
• Vision Model: Groq Qwen/qwen3.8-27b (Multimodal)
• Text LLM: Groq openai/gpt-oss-20b & openai/gpt-oss-120b
• Document Parsers: pypdf (PDFs), zipfile/xml.etree (PPTX/Office XML)
• Database & Cache: Upstash Redis REST Cloud
• Messaging Gateway: Twilio WhatsApp API & Meta WhatsApp Cloud API

--------------------------------------------------------------------------------
4. ENVIRONMENT VARIABLES
--------------------------------------------------------------------------------
GROQ_API_KEY              - Groq Cloud API key for LLM, Whisper, and Vision
GROQ_MODEL                - Preferred model (default: openai/gpt-oss-20b)
BOT_NAME                  - Display name (default: Mitra)
UPSTASH_REDIS_REST_URL    - Upstash Redis REST endpoint for memory
UPSTASH_REDIS_REST_TOKEN  - Upstash Redis REST authorization token
TWILIO_ACCOUNT_SID        - Twilio Account SID (for authenticated media)
TWILIO_AUTH_TOKEN         - Twilio Auth Token (for authenticated media)
MAX_MEMORY_MESSAGES       - History turns retained per user (default: 50)

--------------------------------------------------------------------------------
5. QUICK LOCAL TESTING
--------------------------------------------------------------------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run local development server:
   python main.py

3. Preview web chat:
   http://localhost:8000

4. Webhook endpoint:
   http://localhost:8000/whatsapp

--------------------------------------------------------------------------------
6. CREATOR ATTRIBUTION
--------------------------------------------------------------------------------
Mitra was engineered and designed by Gourav Dadheech.
When asked "Who created you?", "Who engineered you?", or "kisne banaya?",
Mitra proudly attributes its creation to Gourav Dadheech.
================================================================================
