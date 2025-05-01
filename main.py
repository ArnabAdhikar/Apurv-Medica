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
    welcome_msg = """**Welcome to Apurv Medica** 🩺💡..."""  # (same as before)
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
    
    elements = []
    
    # Create tabs for each model response
    tabs = [
        cl.Tab(name="Claude", id="claude", label="Claude Sonnet"),
        cl.Tab(name="gemini", id="gemini", label="Gemini Pro"),
        cl.Tab(name="compare", id="compare", label="Compare Responses")
    ]
    
    # Create tab content
    tab_content = []
    
    # Claude response
    claude_res = validated_responses["Claude"]
    claude_content = f"{claude_res.content}\n\n"
    if claude_res.sources:
        claude_content += "**Sources:**\n" + "\n".join(f"- {src}" for src in claude_res.sources)
    if claude_res.warnings:
        claude_content += "\n\n**Notes:**\n" + "\n".join(f"- ⚠️ {warn}" for warn in claude_res.warnings)
    
    tab_content.append(cl.TabContent(id="claude", content=claude_content))
    
    # Gemini response
    gemini_res = validated_responses["Gemini"]
    gemini_content = f"{gemini_res.content}\n\n"
    if gemini_res.sources:
        gemini_content += "**Sources:**\n" + "\n".join(f"- {src}" for src in gemini_res.sources)
    if gemini_res.warnings:
        gemini_content += "\n\n**Notes:**\n" + "\n".join(f"- ⚠️ {warn}" for warn in gemini_res.warnings)
    
    tab_content.append(cl.TabContent(id="gemini", content=gemini_content))
    
    # Comparison view
    comparison_content = """**Response Comparison**\n\n"""
    comparison_content += f"**Confidence Scores:**\n- Claude: {claude_res.confidence:.1%}\n- Gemini: {gemini_res.confidence:.1%}\n\n"
    
    # Highlight differences
    claude_key = set(claude_res.content.split()[:50])  # Compare first 50 words
    gemini_key = set(gemini_res.content.split()[:50])
    differences = claude_key.symmetric_difference(gemini_key)
    
    if differences:
        comparison_content += "**Key Differences:**\n"
        comparison_content += "\n".join(f"- {diff}" for diff in differences if len(diff) > 3)
    
    tab_content.append(cl.TabContent(id="compare", content=comparison_content))
    
    # Create message with tabs
    response_msg = cl.Message(content="", tabs=tabs, tab_content=tab_content)
    
    # Add confidence badges
    elements.append(
        cl.Badge(
            name="confidence",
            label=f"Claude Confidence: {claude_res.confidence:.0%}",
            value=claude_res.confidence,
            color="green" if claude_res.confidence > 0.7 else "yellow"
        )
    )
    elements.append(
        cl.Badge(
            name="confidence",
            label=f"Gemini Confidence: {gemini_res.confidence:.0%}",
            value=gemini_res.confidence,
            color="green" if gemini_res.confidence > 0.7 else "yellow"
        )
    )
    
    # Add disclaimer
    elements.append(
        cl.Notice(
            name="disclaimer",
            content="This information is not medical advice. Consult a healthcare professional.",
            severity="warning"
        )
    )
    
    # Add feedback buttons
    actions = [
        cl.Action(name="helpful", value="yes", label="👍 Helpful"),
        cl.Action(name="helpful", value="no", label="👎 Not Helpful"),
        cl.Action(name="flag", value="flag", label="⚠️ Flag Content")
    ]
    
    response_msg.elements = elements
    response_msg.actions = actions
    
    await response_msg.update()

@cl.action_callback("helpful")
async def on_feedback(action: cl.Action):
    await cl.Message(content=f"Thank you for your feedback!").send()
    # In production: log feedback to database

@cl.action_callback("flag")
async def on_flag(action: cl.Action):
    await cl.Message(content="This response has been flagged for review. Thank you!").send()
    # In production: implement content moderation workflow

if __name__ == "__main__":
    # Verify AWS credentials
    if not all([os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")]):
        print("Warning: AWS credentials not found. Logging to S3 will be disabled.")
    
    # Run the Chainlit app
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
