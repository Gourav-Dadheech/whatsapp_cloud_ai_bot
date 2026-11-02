from fastapi import FastAPI, Request
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from groq import Groq
from memory import get_memory, save_memory

app = FastAPI()
client = Groq()

BOT_NAME = "Mitra"

class ChatRequest(BaseModel):
    message: str
    user_id: str

@app.get("/")
async def index(request: Request):
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mitra AI Chat</title>
    <style>
        body { font-family: sans-serif; background-color: #ece5dd; margin: 0; display: flex; flex-direction: column; height: 100vh; }
        .header { background-color: #075e54; color: white; padding: 15px; text-align: center; font-size: 1.2rem; font-weight: bold; }
        .chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 10px; }
        .message { max-width: 75%; padding: 10px 15px; border-radius: 15px; line-height: 1.4; word-wrap: break-word; }
        .user-message { background-color: #dcf8c6; align-self: flex-end; border-bottom-right-radius: 0; }
        .bot-message { background-color: #ffffff; align-self: flex-start; border-bottom-left-radius: 0; box-shadow: 0 1px 1px rgba(0,0,0,0.1); }
        .input-area { background-color: #f0f0f0; padding: 15px; display: flex; gap: 10px; }
        input[type="text"] { flex: 1; padding: 12px; border: none; border-radius: 20px; outline: none; font-size: 1rem;}
        button { background-color: #128c7e; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; font-size: 1rem; font-weight: bold;}
        button:hover { background-color: #075e54; }
    </style>
</head>
<body>
    <div class="header">🤖 Mitra AI</div>
    <div class="chat-container" id="chat-container">
        <div class="message bot-message">Hello! I am Mitra. How can I help you today?</div>
    </div>
    <div class="input-area">
        <input type="text" id="message-input" placeholder="Type a message..." onkeypress="handleKeyPress(event)">
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        // Generate a random web user ID and store it in the browser
        let userId = localStorage.getItem('userId');
        if (!userId) {
            userId = 'web-' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('userId', userId);
        }

        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');

        function addMessage(text, isUser) {
            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            msgDiv.innerText = text;
            chatContainer.appendChild(msgDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const text = messageInput.value.trim();
            if (!text) return;

            addMessage(text, true);
            messageInput.value = '';

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text, user_id: userId })
                });
                
                const data = await response.json();
                addMessage(data.response, false);
            } catch (error) {
                addMessage("Sorry, I'm having trouble connecting to the server.", false);
            }
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.post("/api/chat")
async def web_chat(request: ChatRequest):
    user_msg = request.message
    user_id = request.user_id

    if not user_msg.strip():
        user_msg = "Hello"

    # Load previous memory
    history = get_memory(user_id)

    messages = [
        {
            "role": "system",
            "content": f"""
You are {BOT_NAME}, a friendly, warm, and supportive AI friend.

IMPORTANT RULES:
- Always complete your sentences.
- If your answer may become long, summarize and finish clearly.
- Never end replies mid-sentence.
- If you are running out of space, conclude politely.

Speak naturally and helpfully.
"""
        }
    ]

    for msg in history:
        messages.append(msg)

    messages.append({"role": "user", "content": user_msg})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=250
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = "Arre boss, thoda technical hiccup aa gaya (Groq API error). Thodi der mein try karna!"

    # Save conversation
    save_memory(user_id, "user", user_msg)
    save_memory(user_id, "assistant", answer)

    return {"response": answer}

