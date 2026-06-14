# LLM/VLM clients for SkillNav Multi-Agent System
"""
Unified VLM/LLM interface for all SkillNav agents.

Provides:
- QwenVLMClient: Vision-language model (DASHSCOPE_API_KEY)
- DeepSeekLLMClient: Text-only LLM (DEEPSEEK_API_KEY)
- AgentLLMInterface: Unified interface with caching, rate limiting

Usage:
    from vlm.llm import get_agent_interface

    # Get singleton interface
    interface = get_agent_interface()

    # Safe Agent: Dead zone analysis (uses VLM)
    result = interface.analyze_dead_zone(image, position, attempts, heading)

    # Memory Agent: Object verification (uses VLM)
    result = interface.verify_object(image, "toilet", position)

    # Exploration Agent: Strategy planning (uses LLM)
    result = interface.plan_exploration(voronoi_summary, coverage, hotspots, phase, target)
"""

from .base import (
    BaseLLMClient,
    BaseVLMClient,
    Message,
    VLMMessage,
    MessageRole,
    LLMResponse,
    image_to_base64,
    base64_to_image,
)
from .qwen_vlm import QwenVLMClient, get_qwen_vlm_client
from .deepseek_llm import DeepSeekLLMClient, get_deepseek_client
from .agent_interface import (
    AgentLLMInterface,
    AgentType,
    AgentQuery,
    AgentResponse,
    get_agent_interface,
)

__all__ = [
    # Base classes
    "BaseLLMClient",
    "BaseVLMClient",
    "Message",
    "VLMMessage",
    "MessageRole",
    "LLMResponse",
    # Utility functions
    "image_to_base64",
    "base64_to_image",
    # Clients
    "QwenVLMClient",
    "DeepSeekLLMClient",
    "get_qwen_vlm_client",
    "get_deepseek_client",
    # Unified interface
    "AgentLLMInterface",
    "AgentType",
    "AgentQuery",
    "AgentResponse",
    "get_agent_interface",
]
