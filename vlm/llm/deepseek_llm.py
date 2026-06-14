"""
DeepSeek LLM client for text-based reasoning tasks.

This client is used for reasoning tasks in SkillNav that don't require vision:
- Exploration strategy planning (Exploration Agent)
- Target prioritization reasoning (Memory Agent)
- Phase transition decisions (Options-SMDP)

Environment variable required:
    DEEPSEEK_API_KEY: Your DeepSeek API key

Reference:
    https://api-docs.deepseek.com/
    https://api-docs.deepseek.com/api/create-chat-completion
"""

import os
import time
from typing import Optional, List, Dict, Any

from .base import (
    BaseLLMClient,
    Message,
    MessageRole,
    LLMResponse,
)


class DeepSeekLLMClient(BaseLLMClient):
    """
    Client for DeepSeek LLM models.

    Uses OpenAI-compatible API with DeepSeek's base URL.

    Usage:
        client = DeepSeekLLMClient()  # Uses DEEPSEEK_API_KEY from env
        response = client.simple_chat("What is the best exploration strategy?")
    """

    # Available models
    MODELS = {
        "deepseek-chat": "deepseek-chat",  # Default chat model
        "deepseek-reasoner": "deepseek-reasoner",  # Reasoning model
        "deepseek-v4-flash": "deepseek-v4-flash",  # Fast model
        "deepseek-v4-pro": "deepseek-v4-pro",  # Pro model
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize DeepSeek LLM client.

        Args:
            api_key: DeepSeek API key. If None, reads from DEEPSEEK_API_KEY env var
            model: Model name (deepseek-chat, deepseek-reasoner, etc.)
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY not found. Set it in environment or pass api_key parameter."
            )

        super().__init__(
            api_key=api_key,
            model=self.MODELS.get(model, model),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Initialize OpenAI client with DeepSeek base URL
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion request.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            stream: Whether to stream the response
            **kwargs: Additional parameters (top_p, frequency_penalty, etc.)

        Returns:
            LLMResponse with the model's response
        """
        formatted_messages = [msg.to_dict() for msg in messages]

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    **kwargs
                )

                if stream:
                    # Handle streaming response
                    content_chunks = []
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content_chunks.append(chunk.choices[0].delta.content)
                    return LLMResponse(
                        content="".join(content_chunks),
                        finish_reason="stop",
                        raw_response=None,
                    )
                else:
                    return LLMResponse(
                        content=response.choices[0].message.content,
                        finish_reason=response.choices[0].finish_reason,
                        usage={
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        } if response.usage else None,
                        raw_response=response,
                    )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request failed: {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

    # ==================== Agent-specific methods ====================

    def plan_exploration_strategy(
        self,
        voronoi_summary: str,
        exploration_percentage: float,
        semantic_hotspots: List[Dict],
        current_phase: str,
        target_object: str,
    ) -> Dict[str, Any]:
        """
        Plan exploration strategy for Exploration Agent.

        Args:
            voronoi_summary: Summary of current Voronoi topology
            exploration_percentage: Percentage of area explored (0-100)
            semantic_hotspots: List of detected semantic hotspots
            current_phase: Current navigation phase
            target_object: Target object to find

        Returns:
            Dict with keys:
                - recommended_phase: str
                - next_waypoint_id: int (Voronoi node ID)
                - alpha: float (IG weight)
                - beta: float (SR weight)
                - reasoning: str
        """
        prompt = f"""You are planning the exploration strategy for a robot searching for "{target_object}".

Current State:
- Exploration: {exploration_percentage:.1f}% complete
- Current Phase: {current_phase}

Voronoi Topology Summary:
{voronoi_summary}

Semantic Hotspots (potential {target_object} locations):
{self._format_hotspots(semantic_hotspots)}

Available Phases:
1. BROAD_EXPLORATION: Maximize coverage (alpha=0.8, beta=0.2)
2. DIRECTED_SEARCH: Follow semantic cues (alpha=0.3, beta=0.7)
3. TARGET_APPROACH: Go to detected target (alpha=0.1, beta=0.9)
4. VERIFICATION: Confirm target identity (alpha=0.0, beta=1.0)

Respond in JSON format:
{{
    "recommended_phase": "BROAD_EXPLORATION" | "DIRECTED_SEARCH" | "TARGET_APPROACH" | "VERIFICATION",
    "next_waypoint_id": <Voronoi node ID to visit next>,
    "alpha": <IG weight 0-1>,
    "beta": <SR weight 0-1>,
    "reasoning": "Brief explanation of your decision"
}}"""

        system_prompt = """You are an expert navigation planner for autonomous robots.
Make decisions that balance efficient exploration with semantic target finding.
Consider the exploration percentage and hotspot confidence when choosing phases."""

        return self.json_chat(prompt, system_prompt=system_prompt, temperature=0.4)

    def _format_hotspots(self, hotspots: List[Dict]) -> str:
        """Format hotspots for prompt."""
        if not hotspots:
            return "None detected"

        lines = []
        for i, h in enumerate(hotspots[:5]):  # Limit to top 5
            lines.append(
                f"  {i+1}. Node {h.get('node_id', '?')}: "
                f"confidence={h.get('confidence', 0):.2f}, "
                f"position=({h.get('x', 0):.1f}, {h.get('y', 0):.1f})"
            )
        return "\n".join(lines)

    def prioritize_targets(
        self,
        candidates: List[Dict],
        robot_position: tuple,
        target_object: str,
    ) -> Dict[str, Any]:
        """
        Help Memory Agent prioritize candidate targets.

        Args:
            candidates: List of candidate targets with their properties
            robot_position: Current robot (x, y)
            target_object: Target object name

        Returns:
            Dict with ranked target IDs and reasoning
        """
        prompt = f"""You need to help prioritize which "{target_object}" candidate to approach.

Robot Position: ({robot_position[0]:.2f}, {robot_position[1]:.2f})

Candidate Targets:
{self._format_candidates(candidates)}

Consider:
1. Detection confidence and observation count
2. False positive history
3. Distance from robot
4. Semantic value from ITM

Respond in JSON:
{{
    "ranked_ids": [<list of candidate IDs in priority order>],
    "best_target_id": <top priority target ID>,
    "reasoning": "Why this ordering?"
}}"""

        system_prompt = "You are helping a robot decide which detected object to investigate first."

        return self.json_chat(prompt, system_prompt=system_prompt, temperature=0.3)

    def _format_candidates(self, candidates: List[Dict]) -> str:
        """Format candidates for prompt."""
        if not candidates:
            return "None"

        lines = []
        for c in candidates[:10]:  # Limit to 10
            lines.append(
                f"  ID {c.get('id', '?')}: "
                f"conf={c.get('confidence', 0):.2f}, "
                f"obs={c.get('observation_count', 0)}, "
                f"fp={c.get('false_positive_count', 0)}, "
                f"dist={c.get('distance', 0):.2f}m, "
                f"sem_val={c.get('semantic_value', 0):.2f}"
            )
        return "\n".join(lines)

    def decide_phase_transition(
        self,
        current_phase: str,
        phase_history: List[str],
        trigger_events: List[str],
        exploration_state: Dict,
    ) -> Dict[str, Any]:
        """
        Decide whether to transition to a new phase (Options-SMDP).

        Args:
            current_phase: Current navigation phase
            phase_history: Recent phase history
            trigger_events: Events that might trigger transition
            exploration_state: Current exploration metrics

        Returns:
            Dict with transition decision
        """
        prompt = f"""Decide if the robot should transition to a new navigation phase.

Current Phase: {current_phase}
Recent History: {' -> '.join(phase_history[-5:])}

Trigger Events:
{chr(10).join(f'  - {e}' for e in trigger_events)}

Exploration State:
- Coverage: {exploration_state.get('coverage', 0):.1f}%
- Semantic hotspots found: {exploration_state.get('hotspot_count', 0)}
- Current target confidence: {exploration_state.get('target_confidence', 0):.2f}
- Steps since last transition: {exploration_state.get('steps_since_transition', 0)}

Phase Transition Rules:
- BROAD_EXPLORATION → DIRECTED_SEARCH: When semantic hotspot found (>0.6 conf) OR coverage >50%
- DIRECTED_SEARCH → TARGET_APPROACH: When target detected (>0.7 conf)
- TARGET_APPROACH → VERIFICATION: When within 1m of target
- Any → BROAD_EXPLORATION: When current target invalidated

Respond in JSON:
{{
    "should_transition": true | false,
    "new_phase": "<phase name if transitioning>",
    "reasoning": "Why or why not?"
}}"""

        system_prompt = "You are managing phase transitions for a navigation agent using Options-SMDP."

        return self.json_chat(prompt, system_prompt=system_prompt, temperature=0.3)


# Convenience function
def get_deepseek_client(**kwargs) -> DeepSeekLLMClient:
    """Factory function to create DeepSeekLLMClient with default settings."""
    return DeepSeekLLMClient(**kwargs)
