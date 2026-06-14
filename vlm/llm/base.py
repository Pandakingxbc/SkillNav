"""
Base classes for LLM and VLM clients used by SkillNav Multi-Agent System.

These interfaces are designed to be used by:
- Exploration Agent: High-level strategic decisions
- Memory Agent: Target verification and FP handling
- Safe Agent: Dead zone consultation
"""

import os
import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
from enum import Enum

import numpy as np
import cv2


class MessageRole(Enum):
    """Message roles for chat completion."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class VLMMessage:
    """A message that can contain both text and images for VLM."""
    role: MessageRole
    text: str
    images: Optional[List[np.ndarray]] = None  # List of images as numpy arrays

    def to_dict(self, image_format: str = "base64") -> Dict[str, Any]:
        """Convert to API-compatible dict format."""
        content = []

        # Add images first
        if self.images:
            for img in self.images:
                if image_format == "base64":
                    img_str = image_to_base64(img)
                    content.append({
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{img_str}"
                    })

        # Add text
        content.append({
            "type": "text",
            "text": self.text
        })

        return {
            "role": self.role.value,
            "content": content
        }


@dataclass
class LLMResponse:
    """Response from LLM/VLM."""
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


def image_to_base64(img: np.ndarray, quality: int = 85) -> str:
    """Convert numpy array image to base64 string."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode(".jpg", img, encode_param)
    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(b64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array image."""
    img_bytes = base64.b64decode(b64_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)


class BaseLLMClient(ABC):
    """
    Base class for text-only LLM clients.

    Used by agents for reasoning tasks that don't require vision.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion request.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional model-specific parameters

        Returns:
            LLMResponse with the model's response
        """
        pass

    def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Simple interface for single-turn chat.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            Response text string
        """
        messages = []
        if system_prompt:
            messages.append(Message(MessageRole.SYSTEM, system_prompt))
        messages.append(Message(MessageRole.USER, prompt))

        response = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return response.content

    def json_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Chat that expects JSON response.

        Args:
            prompt: User prompt (should ask for JSON output)
            system_prompt: System instruction
            temperature: Lower for more deterministic JSON
            max_tokens: Maximum response tokens

        Returns:
            Parsed JSON as dict
        """
        import json

        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\n\nRespond only with valid JSON, no additional text."

        response_text = self.simple_chat(
            prompt,
            system_prompt=json_system,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response was: {response_text}")
            return {"error": "JSON parse failed", "raw": response_text}


class BaseVLMClient(BaseLLMClient):
    """
    Base class for Vision-Language Model clients.

    Extends BaseLLMClient with image understanding capabilities.
    Used for tasks requiring visual input (dead zone analysis, object verification).
    """

    @abstractmethod
    def chat_with_images(
        self,
        messages: List[VLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion request with images.

        Args:
            messages: List of messages that may contain images
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional model-specific parameters

        Returns:
            LLMResponse with the model's response
        """
        pass

    def analyze_image(
        self,
        image: np.ndarray,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """
        Simple interface for single image analysis.

        Args:
            image: Image as numpy array (BGR format from OpenCV)
            prompt: Question/instruction about the image
            system_prompt: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            Response text string
        """
        messages = []
        if system_prompt:
            messages.append(VLMMessage(MessageRole.SYSTEM, system_prompt))
        messages.append(VLMMessage(MessageRole.USER, prompt, images=[image]))

        response = self.chat_with_images(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.content

    def analyze_image_json(
        self,
        image: np.ndarray,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Analyze image and return JSON response.

        Args:
            image: Image as numpy array
            prompt: Prompt asking for JSON analysis
            system_prompt: System instruction
            temperature: Lower for deterministic output
            max_tokens: Maximum tokens

        Returns:
            Parsed JSON as dict
        """
        import json

        json_system = (system_prompt or "") + "\n\nRespond only with valid JSON, no additional text."

        response_text = self.analyze_image(
            image,
            prompt,
            system_prompt=json_system,
            temperature=temperature,
            max_tokens=max_tokens
        )

        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response was: {response_text}")
            return {"error": "JSON parse failed", "raw": response_text}
