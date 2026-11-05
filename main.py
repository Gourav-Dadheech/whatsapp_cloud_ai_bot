import os
import sys
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.index import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting Mitra WhatsApp AI Bot on http://localhost:{port}")
    print(f"👉 Web chat preview: http://localhost:{port}")
    print(f"👉 WhatsApp Webhook: http://localhost:{port}/whatsapp")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

