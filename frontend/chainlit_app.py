import logging
import sys
sys.path.insert(0, './backend')
import chainlit as cl
import requests
from datetime import datetime, timezone
import base64
import json
import asyncio
import uuid
import os
from pathlib import Path
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
API_BASE_URL = "http://127.0.0.1:8001"
        
async def process_query(query: str, user_id: str, input_files: list , collection_user:str):
    """Send query to FastAPI backend and process response"""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{API_BASE_URL}/query", json={
                "query": query + ''.join(str(x[-1]) for x in input_files),
                "data_path": "data",
                "user_id": user_id,
                "input_files": input_files,
                "collection": collection_user,
            }) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    if line.startswith("data:"):
                        payload = line.replace("data:", "", 1).strip()
                        
                        if payload.startswith("{"):
                            try:
                                parsed = json.loads(payload)
                                yield {"type": "final", "data": parsed}
                            except json.JSONDecodeError as e:
                                print("JSON decode failed:", e)
                                print("Line was:", repr(payload))
                                yield {"type": "error", "data": "Failed to parse JSON"}
                        else:
                            # Streaming status text or logs
                            yield {"type": "update", "data": payload}
    finally:
        await response.aclose()

@cl.on_chat_start
async def start():
    """
    On chat start, generate a unique user ID and send a welcome message.
    The welcome message is sent one word at a time, with a short delay between
    each word, to create a typing effect.
    """
    user_id = f"user_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())}"
    cl.user_session.set("user_id", user_id)
    current_chunk = ""
    msg =cl.Message(content="")
    token_list = "Welcome! I'm ready to answer questions about your documents. Please upload your documents before asking questions. " \
    "This chatbot accepts pdf, docx and xlsx documents"

    for token in token_list.split(' '):
        current_chunk = token + " "
        await asyncio.sleep(0.15)
        await msg.stream_token(current_chunk)
    await msg.send()


@cl.on_message
async def main(message: cl.Message):
    """
    Process a user message by handling uploaded documents and generating a response.

    This function is triggered when a message is received. It checks for any 
    attached document files, processes them, and generates a response based on 
    the user's query. The response includes text and may include images sourced 
    from the documents. If no documents are attached, it uses documents from the 
    current session's collection.

    The function streams a welcome message with a typing effect and then processes 
    the uploaded documents or retrieves them from a session. It sets up a folder for 
    image storage, updates the user session, and communicates with a processing 
    query function to handle the user's query. The response, including any source 
    nodes and images, is then sent back to the user.

    :param message: The message object containing the user's query and any attached 
                    document files.
    :type message: cl.Message
    """

    user_id = cl.user_session.get("user_id")
    msg1 = cl.Message(content="")

    # if not message.elements:
    #     await cl.Message(content="No file attached").send()
    #     return
    
    doc_files = [(file.path,file.id,file.url,file.name) for file in message.elements]
    # print(doc_files)
    if len(doc_files) != 0:
        folder_name = doc_files[-1][0].split('/')[-2]
        cl.user_session.set("collection", folder_name)
        folder_path = '/app/.files/'+doc_files[-1][0].split('/')[-2]
        # print("Folder name: ",folder_name, folder_path)

        if not os.path.exists(f'/app/backend/data_images/{folder_name}/'):
            os.mkdir(f'/app/backend/data_images/{folder_name}/')

        input_files = [(fd[0],fd[-1]) for fd in doc_files]
        # print(input_files)

        token_list = "Processing uploaded documents. This can take up some time. "
        for token in token_list.split(' '):
            current_chunk = token + " "
            await asyncio.sleep(0.15)
            await msg1.stream_token(current_chunk)
        await msg1.send()
        
    else:
        input_files = []
        folder_name = cl.user_session.get("collection")

    
    try:
        source_id = ""
        count= 0
        # response_data = await process_query(message.content, user_id, input_files, folder_name)
        #print("response_data", response_data['source_nodes'])
        async for update in process_query(message.content, user_id, input_files, folder_name):
            if update["type"] == "update" and count == 0:
                message = cl.Message(content=update["data"])
                count += 1
                await asyncio.sleep(1)
                await message.send()
            elif update["type"] == "update" and count > 0:
                message.content = update["data"]
                await asyncio.sleep(1)
                await message.update()
            elif update["type"] == "final":
                # print("after process query:",update["data"])
                response_data = update["data"]
                response_dict = response_data["response"]
                # print("<<<<<>>>>>>response_dict", response_dict, response_data["source_nodes"])
                elements = []
                total_text = []
                if "source_nodes" in response_data:
                    for idx, node in enumerate(response_data["source_nodes"]):
               
                        # pdf_name = node['metadata'].get('pdf_name', 'Unknown')
                        page_num = node['metadata'].get('page_num', 'Unknown')
                        actual_doc_name = node['metadata'].get('actual_doc_name', 'Unknown')
                        document_type = node['metadata'].get('document_type', 'Unknown')
                        source_id = f"Source: {actual_doc_name}, Page: {page_num} , Document Type: {document_type}"
                        # print("/////",pdf_name, page_num, source_id)
                        total_text.append(node['text'])
                        if "image_base64" in node['metadata']:
                            elements.append(cl.Image(name=f"source_image_{len(elements)}",
                                                    display="inline",
                                                    content=base64.b64decode(node['metadata']["image_base64"]),
                                                    caption=f"Source: {node['metadata']['actual_doc_name']}, Page: {node['metadata']['page_num']}"))

                if source_id != " ":
                    actions=[cl.Action(name="show_source",
                                        payload={"source_id": source_id, "text": total_text},
                                        label="Click to view source")]
                
                message.content = response_dict
                message.elements = elements
                message.actions = actions
                await message.update()

    except Exception as e:
        await cl.Message(
            content=f"Error: {str(e)}",
            type="error"
        ).send()
    
    

@cl.action_callback("show_source")
async def on_action(action):
    """Handle clicks to show source content"""
    try:
        source_id = action.payload.get("source_id", "Unknown Source")
        text = action.payload.get("text", "No content available")
        
        await cl.Message(
            content=f"📄 {source_id}\n\n{text}",
            author="Source"
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"Error displaying source: {str(e)}",
            type="error"
        ).send()

#if __name__ == "__main__":
#    from chainlit.cli import run_chainlit

#    run_chainlit(__file__)
