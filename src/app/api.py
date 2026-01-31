import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk
import uuid
import json
import traceback
import time
import aiosqlite
import re
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))

from .graph import make_graph 

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangFuse
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
langfuse_client = Langfuse()

# Stockage des sessions interrompues
INTERRUPTED_SESSIONS = {}


def extract_text_content(chunk_content) -> str:
    """
    Extrait le texte d'un content qui peut être str ou list de blocs.
    Gère les cas où le LLM utilise des outils (tool calls) et renvoie une liste.
    """
    if chunk_content is None:
        return ""
    if isinstance(chunk_content, str):
        return chunk_content
    elif isinstance(chunk_content, list):
        text_parts = []
        for block in chunk_content:
            if isinstance(block, dict):
                # Bloc de type "text"
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                # Ignorer les blocs tool_use, tool_result, etc.
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return ""


@app.on_event("startup")
async def startup_event():
    print("🚀 Initialisation du graphe...")
    app.state.db_conn = await aiosqlite.connect("memory.sqlite", check_same_thread=False)
    app.state.checkpointer = AsyncSqliteSaver(conn=app.state.db_conn)
    app.state.graph = await make_graph(checkpointer=app.state.checkpointer)
    print("✅ Graphe initialisé et prêt à recevoir des requêtes")

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.db_conn.close()
    langfuse_client.flush()
    print("API SHUTDOWN")

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            { 
                "id": "agent-superviseur-v5", 
                "object": "model", 
                "created": int(time.time()), 
                "owned_by": "ASI"
            }
        ],
        "object": "list",
    }

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    conversation_id: Optional[str] = None
    temperature: float = 0.7
    stream: bool = False
    history_length: int = 5

def sanitize_message_name(name: str) -> str:
    if name is None:
        return None
    if name == "":
        return "json"
    sanitized = re.sub(r'[\s<|\\/>]+', '_', name)
    sanitized = sanitized[:64]
    sanitized = sanitized.strip('_')
    return f"json_{sanitized}" if sanitized else "json"

def extract_context_info(messages: list) -> dict:
    context_info = {
        "has_document_context": False,
        "document_names": [],
        "context_length": 0
    }
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if "Context from" in content or "Use the following context" in content:
                context_info["has_document_context"] = True
                context_info["context_length"] = len(content)
                if "Context from" in content:
                    matches = re.findall(r'Context from ([^:]+):', content)
                    context_info["document_names"] = matches
    return context_info

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        if req.conversation_id:
            session_id = req.conversation_id
            print(f"🔄 Reprise de la conversation existante : {session_id}")
        else:
            session_id = str(uuid.uuid4())
            print(f"🆕 Nouvelle conversation démarrée : {session_id}")

        if session_id in INTERRUPTED_SESSIONS:
            return await handle_resume_after_interrupt(req, session_id)

        thread_id = f"{session_id}_{int(time.time())}" if req.history_length < 100 else session_id
        config = {"configurable": {"session_id": session_id, "thread_id": thread_id}}

        if not req.messages:
            return JSONResponse({"error": "No messages provided"}, status_code=400)

        context_info = extract_context_info(req.messages)
        if context_info["has_document_context"]:
            print(f"📄 Requête avec contexte documentaire détecté")
            print(f"   Documents: {context_info['document_names']}")
            print(f"   Taille contexte: {context_info['context_length']} caractères")

        try:
            state = await app.state.graph.aget_state(config)
            if state and state.values and "messages" in state.values:
                all_history = state.values["messages"]
                print(f"📜 Historique total disponible: {len(all_history)} messages")
        except Exception as e:
            print(f"⚠️ Impossible de récupérer l'historique: {e}")
        
        input_messages = []
        for i, msg in enumerate(req.messages):
            role = msg.get("role")
            content = msg.get("content")
            name = msg.get("name")
            
            if name:
                sanitized_name = sanitize_message_name(name)
                if sanitized_name != name:
                    print(f"⚠️ Message {i}: 'name' nettoyé: '{name}' -> '{sanitized_name}'")
                name = sanitized_name
            
            if role == 'user':
                message = HumanMessage(content=content)
                if name:
                    message.name = name
                input_messages.append(message)
            elif role == 'assistant':
                message = AIMessage(content=content)
                if name:
                    message.name = name
                input_messages.append(message)
            elif role == 'system':
                message = SystemMessage(content=content)
                if name:
                    message.name = name
                input_messages.append(message)
        
        messages_to_send = input_messages[-req.history_length:] if len(input_messages) > req.history_length else input_messages
        print(f"💬 Messages envoyés au graphe: {len(messages_to_send)} sur {len(input_messages)} reçus (limite: {req.history_length})")
        
        inputs = {"messages": messages_to_send}

        if not req.stream:
            return JSONResponse(
                {"error": "Non-streaming mode not supported with HITL interruptions."},
                status_code=400
            )

        return StreamingResponse(
            stream_with_hitl(inputs, config, session_id, thread_id, req.model, req.messages, context_info),
            media_type="text/event-stream"
        )

    except Exception:
        print(f"❌ Unhandled server error: {traceback.format_exc()}")
        return JSONResponse({"error": "Internal Server Error"}, status_code=500)


async def handle_resume_after_interrupt(req: ChatCompletionRequest, session_id: str):
    session_data = INTERRUPTED_SESSIONS[session_id]
    thread_id = session_data["thread_id"]
    
    print(f"--- Reprise de la session {session_id} / thread {thread_id} ---")
    
    user_response = req.messages[-1]['content'] if req.messages else ""
    print(f"--- Réponse utilisateur : {user_response} ---")
    
    INTERRUPTED_SESSIONS.pop(session_id)
    
    config = {"configurable": {"session_id": session_id, "thread_id": thread_id}}
    resume_input = Command(resume=user_response)
    
    return StreamingResponse(
        stream_resume_to_client(resume_input, config, session_id, thread_id, req.model),
        media_type="text/event-stream"
    )


async def stream_with_hitl(inputs, config, session_id: str, thread_id: str, model_name: str, raw_messages: list, context_info: dict):
    """Stream avec détection et gestion des interruptions HITL + Langfuse."""
    chat_completion_id = f"chatcmpl-{thread_id}"
    created_timestamp = int(time.time())
    
    # Créer un span parent Langfuse pour toute la requête
    with langfuse_client.start_as_current_span(
        name="chat-completion-stream-hitl",
        input={"messages": [{"role": m.get("role"), "content": m.get("content", "")[:100]} for m in raw_messages[-3:]]},
        metadata={
            "model": model_name,
            "stream": True,
            "has_document_context": context_info["has_document_context"],
            "hitl_enabled": True
        }
    ) as rootspan:
        rootspan.update_trace(
            session_id=session_id,
            user_id=session_id
        )
        
        try:
            # Créer le CallbackHandler Langfuse dans le contexte du span
            langfuse_handler = CallbackHandler()
            config_with_langfuse = {
                **config,
                "callbacks": [langfuse_handler],
                "metadata": {"langfuse_session_id": session_id}
            }
            
            has_interruption = False
            has_content = False
            accumulated_content = ""
            
            async for event in app.state.graph.astream_events(inputs, config_with_langfuse, version="v2"):
                event_type = event.get("event")
                event_name = event.get("name", "")
                
                if "interrupt" in str(event).lower():
                    print(f"🔍 Event potentiel d'interruption détecté: {event_type} - {event_name}")
                    print(f"   Data keys: {event.get('data', {}).keys() if isinstance(event.get('data'), dict) else 'N/A'}")
                
                if event_type == "on_chat_model_stream":
                    if "final_answer" in event.get("tags", []):
                        chunk = event["data"].get("chunk")
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            # Utiliser la helper function pour extraire le texte
                            content_to_stream = extract_text_content(chunk.content)
                            if content_to_stream:
                                has_content = True
                                accumulated_content += content_to_stream
                                response_chunk = {
                                    "id": chat_completion_id,
                                    "conversation_id": session_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_timestamp,
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content_to_stream},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(response_chunk)}\n\n"
                                await asyncio.sleep(0.001)
            
            print(f"📊 Stream terminé - Contenu streamé: {has_content}")
            
            # Mettre à jour le span Langfuse avec l'output
            rootspan.update(output={"content": accumulated_content, "has_content": has_content})
            
            if not has_content:
                print("⚠️ Aucun contenu streamé, vérification de l'état pour interruption...")
                try:
                    state = await app.state.graph.aget_state(config)
                    if state and hasattr(state, 'tasks') and state.tasks:
                        print(f"🔍 État contient des tasks: {len(state.tasks)}")
                        for task in state.tasks:
                            if hasattr(task, 'interrupts') and task.interrupts:
                                has_interruption = True
                                interrupt_obj = task.interrupts[0]
                                interrupt_value = interrupt_obj.value if hasattr(interrupt_obj, 'value') else {}
                                
                                INTERRUPTED_SESSIONS[session_id] = {
                                    "thread_id": thread_id,
                                    "timestamp": time.time()
                                }
                                
                                question = interrupt_value.get("question", "Action requise.")
                                options = interrupt_value.get("options", [])
                                
                                formatted_message = f"🔒 {question}\n\n➡️ Répondez par : {' ou '.join(options)}"
                                print(f"--- Envoi du message d'interruption après analyse de l'état ---")
                                
                                # Mettre à jour Langfuse avec l'interruption
                                rootspan.update(output={"interruption": True, "question": question, "options": options})
                                
                                words = formatted_message.split()
                                for i, word in enumerate(words):
                                    content = word if i == 0 else f" {word}"
                                    response_chunk = {
                                        "id": chat_completion_id,
                                        "conversation_id": session_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_timestamp,
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": content},
                                            "finish_reason": None
                                        }]
                                    }
                                    yield f"data: {json.dumps(response_chunk)}\n\n"
                                    await asyncio.sleep(0.02)
                                
                                langfuse_client.flush()
                                return
                except Exception as e:
                    print(f"❌ Erreur lors de la vérification de l'état: {e}")
        
        except Exception as e:
            print(f"❌ Error during graph streaming: {traceback.format_exc()}")
            rootspan.update(output={"error": str(e)})
            error_chunk = {
                "id": chat_completion_id,
                "conversation_id": session_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": "\n⚠️ Désolé, une erreur interne est survenue."},
                    "finish_reason": "error"
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        
        finally:
            end_chunk = {
                "id": chat_completion_id,
                "conversation_id": session_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(end_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            langfuse_client.flush()


async def stream_resume_to_client(resume_input, config, session_id: str, thread_id: str, model_name: str):
    """Stream après reprise d'une interruption HITL + Langfuse."""
    chat_completion_id = f"chatcmpl-{thread_id}"
    created_timestamp = int(time.time())
    
    with langfuse_client.start_as_current_span(
        name="chat-completion-resume-hitl",
        input={"resume_value": str(resume_input)[:100]},
        metadata={
            "model": model_name,
            "stream": True,
            "is_resume": True
        }
    ) as rootspan:
        rootspan.update_trace(
            session_id=session_id,
            user_id=session_id
        )
        
        try:
            print(f"🔄 Démarrage du streaming après reprise pour thread {thread_id}")
            
            langfuse_handler = CallbackHandler()
            config_with_langfuse = {
                **config,
                "callbacks": [langfuse_handler],
                "metadata": {"langfuse_session_id": session_id}
            }
            
            event_count = 0
            streamed_content = False
            accumulated_content = ""
            
            async for event in app.state.graph.astream_events(resume_input, config_with_langfuse, version="v2"):
                event_count += 1
                event_type = event.get("event", "unknown")
                
                if event_type in ["on_chat_model_start", "on_chat_model_stream", "on_chat_model_end"]:
                    tags = event.get("tags", [])
                                
                if event["event"] == "on_chat_model_stream":
                    tags = event.get("tags", [])
                    
                    if "final_answer" in tags:
                        chunk = event["data"].get("chunk")
                        if isinstance(chunk, AIMessageChunk) and chunk.content:
                            # Utiliser la helper function pour extraire le texte
                            content_to_stream = extract_text_content(chunk.content)
                            if content_to_stream:
                                streamed_content = True
                                accumulated_content += content_to_stream
                                response_chunk = {
                                    "id": chat_completion_id,
                                    "conversation_id": session_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_timestamp,
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": content_to_stream},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(response_chunk)}\n\n"
                                await asyncio.sleep(0.001)
            
            print(f"✅ Streaming terminé - {event_count} événements traités, contenu streamé: {streamed_content}")
            rootspan.update(output={"content": accumulated_content, "event_count": event_count})
            
            if not streamed_content:
                print("⚠️ ATTENTION : Aucun contenu n'a été streamé ! Vérifiez les tags 'final_answer' dans vos agents.")
        
        except Exception as e:
            print(f"❌ Error during resume streaming: {traceback.format_exc()}")
            rootspan.update(output={"error": str(e)})
            error_chunk = {
                "id": chat_completion_id,
                "conversation_id": session_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": "\n⚠️ Désolé, une erreur est survenue lors de la reprise."},
                    "finish_reason": "error"
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
        
        finally:
            end_chunk = {
                "id": chat_completion_id,
                "conversation_id": session_id,
                "object": "chat.completion.chunk",
                "created": created_timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(end_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            langfuse_client.flush()