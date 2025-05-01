import os
import chainlit as cl
from typing import Dict, Optional, List
from pydantic import BaseModel
import google.generativeai as genai
from anthropic import Anthropic
import re
from datetime import datetime
import boto3
from io import StringIO
import uuid

# Configure environment
from dotenv import load_dotenv
load_dotenv()

# Constants
MAX_TOKENS = 2048
TEMPERATURE = 0.3
SAFETY_SETTINGS = {
    "HARASSMENT": "BLOCK_NONE",
    "HATE_SPEECH": "BLOCK_NONE",
    "SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "DANGEROUS_CONTENT": "BLOCK_NONE"
}

# AWS Configuration
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "medical-chatbot-logs")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

class MedicalResponse(BaseModel):
    content: str
    sources: List[str] = []
    warnings: List[str] = []
    confidence: float = 0.0

class ConversationLogger:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=AWS_REGION
        )
    
    def _generate_session_id(self) -> str:
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    async def log_conversation(self, session_id: str, messages: List[dict]):
        try:
            # Create structured log
            log_data = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "messages": messages
            }
            
            # Convert to string
            log_content = StringIO()
            log_content.write(f"=== Medical Chatbot Conversation Log ===\n")
            log_content.write(f"Session ID: {session_id}\n")
            log_content.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for msg in messages:
                role = msg.get("role", "unknown").title()
                content = msg.get("content", "")
                log_content.write(f"{role}: {content}\n\n")
            
            # Upload to S3
            file_key = f"conversations/{session_id}.txt"
            self.s3_client.put_object(
                Bucket=AWS_BUCKET_NAME,
                Key=file_key,
                Body=log_content.getvalue(),
                ContentType='text/plain'
            )
            
            return True
        except Exception as e:
            print(f"Error logging conversation: {str(e)}")
            return False

class MedicalQueryAssistant:
    def __init__(self):
        # Initialize clients
        self.claude = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.gemini = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.logger = ConversationLogger()
        
        # Initialize models
        self.claude_model = "claude-3-sonnet-20240229"
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Medical system prompt
        self.system_prompt = """You are MediAI, an advanced AI medical assistant..."""  # (same as before)

    # ... (keep all existing methods from previous implementation unchanged)

@cl.on_chat_start
async def start_chat():
    assistant = MedicalQueryAssistant()
    cl.user_session.set("assistant", assistant)
    
    # Generate session ID
    session_id = assistant.logger._generate_session_id()
    cl.user_session.set("session_id", session_id)
    
    # Initial welcome message
    welcome_msg = """**Welcome to MediAI** 🩺💡..."""  # (same as before)
    await cl.Message(content=welcome_msg).send()
    
    # Log initial message
    await assistant.logger.log_conversation(
        session_id,
        [{"role": "system", "content": "New session started", "timestamp": datetime.now().isoformat()}]
    )

@cl.on_message
async def handle_message(message: cl.Message):
    assistant = cl.user_session.get("assistant")
    session_id = cl.user_session.get("session_id")
    user_query = message.content
    
    # Show processing indicator
    msg = cl.Message(content="")
    await msg.send()
    
    # Get chat history
    history = cl.user_session.get("history", [])
    
    # Log user query
    user_message = {
        "role": "user",
        "content": user_query,
        "timestamp": datetime.now().isoformat()
    }
    await assistant.logger.log_conversation(session_id, [user_message])
    
    # Get responses from both models (same as before)
    responses = {
        "Claude": await assistant.get_claude_response(user_query, history),
        "Gemini": await assistant.get_gemini_response(user_query, history)
    }
    
    # Validate responses (same as before)
    validated_responses = await assistant.validate_responses(responses)
    
    # Create assistant messages for logging
    assistant_messages = [
        {
            "role": "assistant",
            "model": "Claude",
            "content": validated_responses["Claude"].content,
            "sources": validated_responses["Claude"].sources,
            "warnings": validated_responses["Claude"].warnings,
            "confidence": validated_responses["Claude"].confidence,
            "timestamp": datetime.now().isoformat()
        },
        {
            "role": "assistant",
            "model": "Gemini",
            "content": validated_responses["Gemini"].content,
            "sources": validated_responses["Gemini"].sources,
            "warnings": validated_responses["Gemini"].warnings,
            "confidence": validated_responses["Gemini"].confidence,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    # Log assistant responses
    await assistant.logger.log_conversation(session_id, assistant_messages)
    
    # Update chat history (same as before)
    new_history = history + [user_message] + assistant_messages
    cl.user_session.set("history", new_history)
    
    # Prepare UI elements (same as before)
    # ... (rest of the UI code remains unchanged)

# ... (keep all remaining code the same)

if __name__ == "__main__":
    # Verify AWS credentials
    if not all([os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")]):
        print("Warning: AWS credentials not found. Logging to S3 will be disabled.")
    
    # Run the Chainlit app
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
