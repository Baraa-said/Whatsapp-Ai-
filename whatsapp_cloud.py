"""
WhatsApp Cloud API Webhook
Connects your Groq RAG chatbot directly to WhatsApp Business via Meta key
"""

import os
import requests
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "BLUE_PANDA")  # Default or set in .env

# Initialize Groq client
groq_client: Groq | None = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Store conversation history per user
# In production, use a database like Redis or PostgreSQL
conversations = {}


def get_ai_response(user_message: str, user_id: str) -> str:
    """Get AI response from Groq"""
    if not groq_client:
        return "❌ Error: Groq API key not configured."

    # Get or create conversation history
    if user_id not in conversations:
        conversations[user_id] = []

    conversations[user_id].append({"role": "user", "content": user_message})

    # Keep context window manageable
    if len(conversations[user_id]) > 10:
        conversations[user_id] = conversations[user_id][-10:]

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful WhatsApp assistant. Keep responses concise.",
                }
            ]
            + conversations[user_id],
            temperature=0.7,
            max_tokens=500,
        )

        assistant_message = (
            response.choices[0].message.content or "I couldn't generate a response."
        )
        conversations[user_id].append(
            {"role": "assistant", "content": assistant_message}
        )
        return assistant_message

    except Exception as e:
        print(f"Error generating AI response: {e}")
        return "Sorry, I encountered an error processing your request."


def send_whatsapp_message(to_number: str, message_body: str):
    """Send message back using Facebook Graph API"""
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        print("❌ Error: Missing WhatsApp credentials in .env")
        return

    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_body},
    }

    try:
        resp = requests.post(url, headers=headers, json=data)
        if resp.status_code != 200:
            print(f"Failed to send message: {resp.text}")
        else:
            print(f"Message sent to {to_number}")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Verify webhook with Facebook"""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            print("WEBHOOK_VERIFIED")
            return challenge, 200
        else:
            return "Verification failed", 403
    return "Hello world", 200


@app.route("/webhook", methods=["POST"])
def webhook_handler():
    """Handle incoming messages"""
    body = request.get_json()

    print(f"Incoming Webhook: {body}")

    try:
        # Check if it's a message from WhatsApp object
        if (
            body
            and "entry" in body
            and body["entry"][0]["changes"][0]["value"]
            and "messages" in body["entry"][0]["changes"][0]["value"]
        ):

            value = body["entry"][0]["changes"][0]["value"]
            message = value["messages"][0]
            from_number = message["from"]  # The user's phone number
            msg_body = message["text"]["body"]  # The message text

            print(f"Message from {from_number}: {msg_body}")

            # Get AI response
            ai_reply = get_ai_response(msg_body, from_number)

            # Send reply
            send_whatsapp_message(from_number, ai_reply)

            return jsonify({"status": "success"}), 200
        else:
            # Possibly a status update or other event
            return jsonify({"status": "ignored"}), 200

    except Exception as e:
        print(f"Error in webhook: {e}")
        return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    print("🚀 Starting WhatsApp Cloud API Bot...")
    app.run(host="0.0.0.0", port=5001, debug=True)
