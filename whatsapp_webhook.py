"""
WhatsApp Webhook - Twilio Integration
Connects your Groq RAG chatbot to WhatsApp via Twilio
"""

import os
from typing import Dict, List
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse  # type: ignore
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")

# Initialize Groq client
groq_client: Groq | None = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Store conversation history per user
conversations: Dict[str, List[Dict[str, str]]] = {}


def get_ai_response(user_message: str, user_id: str) -> str:
    """Get AI response from Groq"""
    if not groq_client:
        return "❌ Error: Groq API key not configured. Please set GROQ_API_KEY in .env file."
    
    # Get or create conversation history for this user
    if user_id not in conversations:
        conversations[user_id] = []
    
    # Add user message to history
    conversations[user_id].append({
        "role": "user",
        "content": user_message
    })
    
    # Keep only last 10 messages to avoid token limits
    if len(conversations[user_id]) > 10:
        conversations[user_id] = conversations[user_id][-10:]
    
    try:
        # Create chat completion with Groq
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast and free model
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful WhatsApp assistant. 
                    Be concise and friendly. Use emojis occasionally.
                    Keep responses short (under 1000 characters) as this is WhatsApp."""
                }
            ] + conversations[user_id],
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message.content or "I couldn't generate a response."
        
        # Add assistant response to history
        conversations[user_id].append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


@app.route("/webhook", methods=["POST"])
def webhook():
    """Handle incoming WhatsApp messages from Twilio"""
    # Get message details
    incoming_msg = request.values.get("Body", "").strip()
    from_number = request.values.get("From", "")
    
    print(f"📱 Message from {from_number}: {incoming_msg}")
    
    # Get AI response
    ai_response = get_ai_response(incoming_msg, from_number)
    
    print(f"🤖 Response: {ai_response}")
    
    # Create Twilio response
    resp = MessagingResponse()
    resp.message(ai_response)
    
    return str(resp)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "groq_configured": bool(GROQ_API_KEY),
        "twilio_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN)
    }


@app.route("/", methods=["GET"])
def home():
    """Home page with setup instructions"""
    return """
    <html>
    <head>
        <title>WhatsApp AI Bot</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #25D366; }
            code { background: #f4f4f4; padding: 2px 8px; border-radius: 4px; }
            pre { background: #f4f4f4; padding: 15px; border-radius: 8px; overflow-x: auto; }
            .step { margin: 20px 0; padding: 15px; border-left: 4px solid #25D366; background: #f9f9f9; }
        </style>
    </head>
    <body>
        <h1>💬 WhatsApp AI Bot</h1>
        <p>Your WhatsApp webhook is running!</p>
        
        <h2>Setup Instructions:</h2>
        
        <div class="step">
            <h3>Step 1: Start ngrok</h3>
            <pre>ngrok http 5001</pre>
            <p>Copy the HTTPS URL (e.g., https://xxxx.ngrok.io)</p>
        </div>
        
        <div class="step">
            <h3>Step 2: Configure Twilio</h3>
            <p>Go to <a href="https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn" target="_blank">Twilio WhatsApp Sandbox</a></p>
            <p>Set the webhook URL to: <code>YOUR_NGROK_URL/webhook</code></p>
        </div>
        
        <div class="step">
            <h3>Step 3: Connect your phone</h3>
            <p>Send the join code to the Twilio WhatsApp number</p>
        </div>
        
        <div class="step">
            <h3>Step 4: Start chatting!</h3>
            <p>Send any message to the Twilio WhatsApp number</p>
        </div>
        
        <p><a href="/health">Check API Status</a></p>
    </body>
    </html>
    """


if __name__ == "__main__":
    print("🚀 Starting WhatsApp AI Bot...")
    print("=" * 50)
    print(f"✅ Groq API: {'Configured' if GROQ_API_KEY else '❌ Not configured'}")
    print(f"✅ Twilio: {'Configured' if TWILIO_ACCOUNT_SID else '❌ Not configured (optional for receiving)'}")
    print("=" * 50)
    print("\n📋 Next steps:")
    print("1. Run: ngrok http 5001")
    print("2. Copy the ngrok HTTPS URL")
    print("3. Go to Twilio Console → WhatsApp Sandbox")
    print("4. Set webhook URL to: YOUR_NGROK_URL/webhook")
    print("5. Send a message to your Twilio WhatsApp number!")
    print("=" * 50)
    
    app.run(host="0.0.0.0", port=5001, debug=True)
