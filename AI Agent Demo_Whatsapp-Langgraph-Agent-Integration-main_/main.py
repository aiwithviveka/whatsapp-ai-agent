import asyncio
import base64
import os
import tempfile
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent import main
from app.config.config import setup_groq_client
from app.config.logging import setup_logger

load_dotenv()

WAIT_TIME = os.getenv("WAIT_TIME", 1)
LANG = os.getenv("LANGUAGE")

GROQ_CLIENT = setup_groq_client()

logger = setup_logger()

message_buffers = defaultdict(list)
processing_tasks = {}
class Sender(BaseModel):
    id: str
    isUser: bool


class WebhookMessage(BaseModel):
    event: str
    session: str
    body: str
    type: str
    isNewMsg: bool
    sender: Sender
    isGroupMsg: bool


async def transcribe_base64_audio(base64_audio: str) -> str:
    try:
        audio_data = base64.b64decode(base64_audio)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        with open(tmp_file_path, "rb") as audio_file:
            transcription = GROQ_CLIENT.audio.transcriptions.create(
                model="whisper-large-v3", file=audio_file, language=LANG
            )
        return transcription.text
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)


async def process_aggregated_messages(sender_id, session, is_user, is_group):
    try:
        await asyncio.sleep(int(WAIT_TIME))
        messages = message_buffers[sender_id]
        if not messages:
            return
        combined_message = " ".join([msg for msg in messages])
        message_buffers[sender_id] = []
        if sender_id in processing_tasks:
            del processing_tasks[sender_id]
        phone_number = sender_id.split("@")[0]
        logger.info(f"Processing aggregated messages for {sender_id}: {combined_message}")
        agent_response = await main(phone_number, combined_message)
        logger.info(f"Agent response: {agent_response}")
    except Exception as e:
        logger.error(f"Error processing aggregated messages: {str(e)}")
        message_buffers[sender_id] = []
        if sender_id in processing_tasks:
            del processing_tasks[sender_id]
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("WebHook service starting up")
    yield
    logger.info("WebHook service shutting down")


app = FastAPI(title="WPPConnect Message Parser", lifespan=lifespan)


@app.post("/webhook")
async def webhook_handler(data: Dict[str, Any]):
    request_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    logger.info(f"Received webhook request - ID: {request_id}")
    try:
        if (
            data.get("event") == "onmessage"
            and data.get("isNewMsg") == True
            and data.get("type") in ["chat", "list_response", "ptt"]
        ):
            try:
                message_text = data.get("body", "")
                if data.get("type") == "ptt":
                    message_text = await transcribe_base64_audio(message_text)
                message = WebhookMessage(
                    event=data["event"],
                    session=data["session"],
                    body=message_text,
                    type=data["type"],
                    isNewMsg=data["isNewMsg"],
                    sender=Sender(id=data["sender"]["id"], isUser=data["sender"]["isUser"]),
                    isGroupMsg=data["isGroupMsg"],
                )
                sender_id = message.sender.id
                message_buffers[sender_id].append(message.body)
                if sender_id not in processing_tasks:
                    task = asyncio.create_task(
                        process_aggregated_messages(sender_id, message.session, message.sender.isUser, message.isGroupMsg)
                    )
                    processing_tasks[sender_id] = task
                    return {"status": "aggregating", "message": "Message received and being aggregated"}
                else:
                    return {"status": "aggregating", "message": "Message added to existing aggregation window"}
            except Exception as e:
                logger.error(f"Request {request_id} - Error parsing message: {str(e)}")
                raise HTTPException(status_code=422, detail=f"Error parsing message: {str(e)}")
        logger.info(f"Request {request_id} - Message skipped")
        return {"status": "received", "message": "Message received but not processed"}
    except Exception as e:
        logger.error(f"Request {request_id} - Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")


@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}