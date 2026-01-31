"""
title: Langgraph stream integration
author: alphonse
description: Integrate langgraph with open webui pipeline
required_open_webui_version: 0.4.3
requirements: none
licence: MIT
"""

import os
import aiohttp
import asyncio
import json
import hashlib
import random
from pydantic import BaseModel, Field
from typing import Union, Generator, Iterator, Optional, AsyncGenerator


class Pipe:
    class Valves(BaseModel):
        API_URL: str = Field(
            default="http://127.0.0.1:8000/v1/chat/completions",
            description="Langgraph API URL (v1/chat/completions)",
        )
        USE_USER_ID_AS_SESSION: bool = Field(
            default=True,
            description="Utiliser l'ID utilisateur comme base pour la session (recommandé)",
        )
        HISTORY_LENGTH: int = Field(
            default=5,
            description="Nombre de messages d'historique à envoyer au LangGraph (défaut: 5)",
        )
        STREAMING_SPEED: float = Field(
            default=0.006,
            description="Vitesse de base du streaming (en secondes par caractère, défaut: 0.006)",
        )
        ENABLE_NATURAL_SPEED: bool = Field(
            default=True,
            description="Activer la vitesse variable naturelle (comme ChatGPT)",
        )

    def __init__(self):
        self.id = "LangGraph stream"
        self.name = "LangGraph stream"
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        )
        # Stockage des conversations par utilisateur
        self.user_conversations = {}

    async def on_startup(self):
        print(f"on_startup: {__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")
        pass

    def _generate_conversation_id(self, user_id: str, messages: list) -> str:
        """
        Génère un ID de conversation unique basé sur l'utilisateur et le contexte.
        Utilise un hash des premiers messages pour identifier une conversation unique.
        """
        # Si c'est le premier message, créer un nouvel ID
        if len(messages) <= 1:
            import uuid

            new_id = str(uuid.uuid4())
            self.user_conversations[user_id] = new_id
            print(f"✨ Nouvelle conversation créée: {new_id}")
            return new_id

        # Sinon, réutiliser l'ID existant pour cet utilisateur
        if user_id in self.user_conversations:
            existing_id = self.user_conversations[user_id]
            print(f"🔄 Conversation existante réutilisée: {existing_id}")
            return existing_id

        # Fallback: créer un hash basé sur le contexte
        context = f"{user_id}_{len(messages)}"
        conversation_id = hashlib.md5(context.encode()).hexdigest()
        self.user_conversations[user_id] = conversation_id
        print(f"🔧 Conversation recréée depuis hash: {conversation_id}")
        return conversation_id

    def _get_char_delay(self, char: str) -> float:
        """
        Calcule le délai pour un caractère selon son type
        """
        if not self.valves.ENABLE_NATURAL_SPEED:
            return self.valves.STREAMING_SPEED

        base_speed = self.valves.STREAMING_SPEED

        if char in [".", "!", "?", ":"]:
            return base_speed * 1
        elif char == ",":
            return base_speed * 0.5
        elif char == " ":
            return base_speed * 0.25
        elif char == "\n":
            return base_speed * 1
        else:
            return base_speed + random.uniform(-0.001, 0.002)

    async def pipe(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __task__=None,
        __model__=None,
        __request__=None,
        __id__: str = None,
        __metadata__: dict = None,
        **kwargs,
    ) -> Union[str, Generator, Iterator, AsyncGenerator]:
        messages = body.get("messages", [])
        model_id = body.get("model", "agent-superviseur-v5")

        # Récupération de l'ID utilisateur
        user_id = None
        if __user__:
            user_id = __user__.get("id") or __user__.get("email")

        if not user_id:
            user_id = "anonymous"

        # Génération ou récupération de l'ID de conversation
        conversation_id = None

        # Tentative 1: Chercher dans les paramètres standards
        conversation_id = (
            __id__
            or (
                __metadata__
                and (__metadata__.get("chat_id") or __metadata__.get("conversation_id"))
            )
            or body.get("conversation_id")
            or body.get("chat_id")
            or kwargs.get("conversation_id")
            or kwargs.get("chat_id")
        )

        # Tentative 2: Générer un ID stable pour cet utilisateur
        if not conversation_id and self.valves.USE_USER_ID_AS_SESSION:
            conversation_id = self._generate_conversation_id(user_id, messages)

        # Tentative 3: Headers HTTP
        if not conversation_id and __request__:
            try:
                headers = dict(getattr(__request__, "headers", {}))
                conversation_id = headers.get("x-chat-id") or headers.get(
                    "x-conversation-id"
                )
            except Exception:
                pass

        # Fallback: ID unique aléatoire
        if not conversation_id:
            import uuid

            conversation_id = str(uuid.uuid4())
            print(f"⚠️ Aucun ID trouvé, génération aléatoire: {conversation_id}")

        # Log récapitulatif
        print(f"\n{'='*60}")
        print(f"👤 Utilisateur: {user_id}")
        print(f"💬 Conversation ID: {conversation_id}")
        print(f"📝 Nombre de messages: {len(messages)}")
        print(f"{'='*60}\n")

        # Préparation du payload pour votre API LangGraph
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "conversation_id": conversation_id,
            "history_length": self.valves.HISTORY_LENGTH,
        }

        headers = {
            "accept": "text/event-stream",
            "Content-Type": "application/json",
        }

        try:
            async def content_generator():
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.post(
                            self.valves.API_URL, json=payload, headers=headers
                        ) as response:
                            response.raise_for_status()
                            
                            async for line in response.content:
                                if line:
                                    decoded_line = line.decode("utf-8")
                                    if decoded_line.startswith("data: "):
                                        json_str = decoded_line[6:]
                                        if json_str.strip() == "[DONE]":
                                            break
                                        try:
                                            chunk = json.loads(json_str)
                                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                                            content = delta.get("content")

                                            if content:
                                                # Streaming caractère par caractère avec délai
                                                for char in content:
                                                    yield char
                                                    # Ajouter le délai selon le type de caractère
                                                    delay = self._get_char_delay(char)
                                                    await asyncio.sleep(delay)

                                        except json.JSONDecodeError:
                                            print(f"Impossible de décoder le JSON : {json_str}")
                                            continue
                    except aiohttp.ClientError as e:
                         error_message = (
                            f"Erreur de connexion à l'API Langgraph à {self.valves.API_URL}: {e}"
                        )
                         print(error_message)
                         yield f"Une erreur est survenue : {error_message}"


            return content_generator()

        except Exception as e:
            error_message = (
                f"Erreur lors de l'initialisation de la requête: {e}"
            )
            print(error_message)
            return f"Une erreur est survenue : {error_message}"

    def get_conversation_id(self, user_id: str) -> Optional[str]:
        """Récupère l'ID de conversation pour un utilisateur donné"""
        return self.user_conversations.get(user_id)

    def reset_conversation(self, user_id: str):
        """Réinitialise la conversation pour un utilisateur"""
        if user_id in self.user_conversations:
            del self.user_conversations[user_id]
            print(f"🔄 Conversation réinitialisée pour l'utilisateur: {user_id}")
