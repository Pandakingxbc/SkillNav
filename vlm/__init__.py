# VLM module for SkillNav
"""
Vision-Language Model integration for SkillNav Multi-Agent System.

Submodules:
- llm: LLM/VLM clients (Qwen3-VL, DeepSeek)
- itm: Image-Text Matching (BLIP2)

Usage:
    from vlm.llm import get_agent_interface
    interface = get_agent_interface()
"""

from . import llm

# Re-export commonly used items
from .llm import (
    AgentLLMInterface,
    get_agent_interface,
    QwenVLMClient,
    DeepSeekLLMClient,
)

__all__ = [
    "llm",
    "AgentLLMInterface",
    "get_agent_interface",
    "QwenVLMClient",
    "DeepSeekLLMClient",
]
