"""Abstractions for working with LLM Providers"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, NamedTuple, Optional

import openai

DEFAULT_MAX_TOKENS = 512
DETERMINISTIC_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 40
CLIENT_FILENAME = "client.json"


class ChatMessageRole(Enum):
    """Types of Chat Message roles."""

    ASSISTANT = 1
    SYSTEM = 2
    USER = 3


class ChatMessage(NamedTuple):
    """Properties of a ChatMessage that are processed by LLMs"""

    role: ChatMessageRole
    content: str


class LLMCompletionRequest(NamedTuple):
    """Properties of API requests sent to LLM providers"""

    messages: List[ChatMessage] = []

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DETERMINISTIC_TEMPERATURE
    top_k: int = DEFAULT_TOP_K
    top_p: float = DEFAULT_TOP_P


class ClientData(NamedTuple):
    """Properties of a LLMClient"""

    client_key: str
    client_params: Dict[str, str]


class LLMClient(ABC):
    """Base class for all LLM Providers Client Classes"""

    @abstractmethod
    def get_completion(self, request: LLMCompletionRequest) -> Optional[str]:
        """Returns LLM response"""
        raise NotImplementedError("get_completion is not implemented")

    @abstractmethod
    def save(self, output_path: str) -> None:
        """Save the client to a directory."""
        raise NotImplementedError("save is not implemented")

    @staticmethod
    def from_path(output_path: str) -> LLMClient:
        """Load an LLMClient from a path"""
        with open(os.path.join(output_path, CLIENT_FILENAME), "r", encoding="utf-8") as file:
            serialized_client = json.load(file)
        client_key = serialized_client["client_key"]
        client_class = CLIENT_KEY_TO_CLIENT[client_key]
        return client_class(**serialized_client["client_params"])


class OpenAIClient(LLMClient):
    """Implementation of LLMClient for OpenAI"""

    CLIENT_KEY = "openai"

    def __init__(self, api_key: str, model_name: str):
        super().__init__()
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model_name = model_name

    def save(self, output_path: str) -> None:
        """Save the client to a directory."""
        with open(os.path.join(output_path, CLIENT_FILENAME), "w", encoding="utf-8") as file:
            json.dump(
                {
                    "client_key": self.CLIENT_KEY,
                    "client_params": {
                        "api_key": self.api_key,
                        "model_name": self.model_name,
                    },
                },
                file,
            )

    def get_completion(self, request: LLMCompletionRequest) -> Optional[str]:
        messages = [ChatMessage(role=ChatMessageRole.SYSTEM, content="You are a helpful assistant.")] + request.messages
        message_list = [{"role": message.role.name.lower(), "content": message.content} for message in messages]
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=message_list,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        # Catch all exceptions for now since OpenAI throws all sorts of exceptions
        except Exception as error:  # pylint: disable=broad-except
            logging.error("[Error] OpenAI API request failed: %s", error)
            return None
        if (
            "choices" in response
            and len(response["choices"]) > 0
            and "message" in response["choices"][0]
            and "content" in response["choices"][0]["message"]
        ):
            return response["choices"][0]["message"]["content"].strip()
        return None


CLIENT_KEY_TO_CLIENT = {
    OpenAIClient.CLIENT_KEY: OpenAIClient,
}
