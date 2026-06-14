"""
Unified Agent Interface for SkillNav Multi-Agent System.

This module provides a unified interface that all three agents
(Exploration, Memory, Safe) can use to access LLM/VLM capabilities.

The interface handles:
- Automatic client selection (VLM vs LLM based on task)
- Caching for repeated queries
- Rate limiting and retry logic
- Consistent response formats
"""

import os
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json

import numpy as np

from .qwen_vlm import QwenVLMClient
from .deepseek_llm import DeepSeekLLMClient


class AgentType(Enum):
    """Types of agents in the system."""
    EXPLORATION = "exploration"
    MEMORY = "memory"
    SAFE = "safe"


@dataclass
class AgentQuery:
    """A query from an agent to the LLM/VLM system."""
    agent_type: AgentType
    query_type: str
    params: Dict[str, Any]
    image: Optional[np.ndarray] = None
    priority: int = 1  # Higher = more urgent
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class AgentResponse:
    """Response to an agent query."""
    success: bool
    data: Dict[str, Any]
    latency_ms: float
    cached: bool = False
    error: Optional[str] = None


class AgentLLMInterface:
    """
    Unified LLM/VLM interface for all SkillNav agents.

    This class provides:
    1. Automatic routing to VLM (for image tasks) or LLM (for text tasks)
    2. Caching of repeated queries
    3. Rate limiting to avoid API throttling
    4. Consistent response format across all agents

    Usage:
        interface = AgentLLMInterface()

        # Safe Agent: Dead zone analysis (uses VLM)
        result = interface.analyze_dead_zone(image, position, attempts, heading)

        # Memory Agent: Object verification (uses VLM)
        result = interface.verify_object(image, "toilet", position)

        # Exploration Agent: Strategy planning (uses LLM)
        result = interface.plan_exploration(voronoi_summary, coverage, hotspots, phase, target)
    """

    def __init__(
        self,
        vlm_model: str = "qwen-vl-plus",
        llm_model: str = "deepseek-chat",
        enable_cache: bool = True,
        cache_ttl: float = 300.0,  # 5 minutes
        rate_limit_rpm: int = 60,  # requests per minute
    ):
        """
        Initialize the agent interface.

        Args:
            vlm_model: Qwen VLM model to use
            llm_model: DeepSeek LLM model to use
            enable_cache: Whether to cache responses
            cache_ttl: Cache time-to-live in seconds
            rate_limit_rpm: Rate limit (requests per minute)
        """
        self._vlm_client: Optional[QwenVLMClient] = None
        self._llm_client: Optional[DeepSeekLLMClient] = None

        self.vlm_model = vlm_model
        self.llm_model = llm_model
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.rate_limit_rpm = rate_limit_rpm

        # Cache storage
        self._cache: Dict[str, Tuple[Any, float]] = {}

        # Rate limiting
        self._request_times: List[float] = []
        self._min_interval = 60.0 / rate_limit_rpm

        # Statistics
        self.stats = {
            "vlm_calls": 0,
            "llm_calls": 0,
            "cache_hits": 0,
            "total_latency_ms": 0,
        }

    @property
    def vlm_client(self) -> QwenVLMClient:
        """Lazy initialization of VLM client."""
        if self._vlm_client is None:
            try:
                self._vlm_client = QwenVLMClient(model=self.vlm_model)
            except ValueError as e:
                print(f"Warning: Could not initialize VLM client: {e}")
                print("VLM features will not be available.")
                raise
        return self._vlm_client

    @property
    def llm_client(self) -> DeepSeekLLMClient:
        """Lazy initialization of LLM client."""
        if self._llm_client is None:
            try:
                self._llm_client = DeepSeekLLMClient(model=self.llm_model)
            except ValueError as e:
                print(f"Warning: Could not initialize LLM client: {e}")
                print("LLM features will not be available.")
                raise
        return self._llm_client

    def _check_rate_limit(self):
        """Enforce rate limiting."""
        now = time.time()
        # Remove old request times
        self._request_times = [t for t in self._request_times if now - t < 60.0]

        if len(self._request_times) >= self.rate_limit_rpm:
            # Need to wait
            oldest = self._request_times[0]
            wait_time = 60.0 - (now - oldest)
            if wait_time > 0:
                print(f"Rate limit reached, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)

        self._request_times.append(time.time())

    def _get_cache_key(self, query_type: str, params: Dict, image: Optional[np.ndarray] = None) -> str:
        """Generate cache key for a query."""
        key_data = {
            "type": query_type,
            "params": {k: str(v) for k, v in params.items() if k != "image"},
        }
        if image is not None:
            # Use image hash for cache key
            key_data["image_hash"] = hashlib.md5(image.tobytes()[:1000]).hexdigest()

        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached response if valid."""
        if not self.enable_cache:
            return None

        if cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return data
            else:
                del self._cache[cache_key]
        return None

    def _set_cached(self, cache_key: str, data: Any):
        """Store response in cache."""
        if self.enable_cache:
            self._cache[cache_key] = (data, time.time())

    # ==================== Safe Agent Methods ====================

    def analyze_dead_zone(
        self,
        image: np.ndarray,
        position: Tuple[float, float],
        attempt_count: int,
        heading: float,
    ) -> AgentResponse:
        """
        Analyze a potential dead zone (Safe Agent).

        Args:
            image: Current RGB view
            position: Robot (x, y) position
            attempt_count: Number of failed escape attempts
            heading: Current heading in degrees

        Returns:
            AgentResponse with dead zone analysis
        """
        params = {
            "position": position,
            "attempt_count": attempt_count,
            "heading": heading,
        }

        cache_key = self._get_cache_key("dead_zone", params, image)
        cached = self._get_cached(cache_key)
        if cached:
            return AgentResponse(success=True, data=cached, latency_ms=0, cached=True)

        self._check_rate_limit()
        start_time = time.time()

        try:
            result = self.vlm_client.analyze_dead_zone(
                image, position, attempt_count, heading
            )
            latency = (time.time() - start_time) * 1000

            self.stats["vlm_calls"] += 1
            self.stats["total_latency_ms"] += latency

            self._set_cached(cache_key, result)

            return AgentResponse(success=True, data=result, latency_ms=latency)

        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    # ==================== Memory Agent Methods ====================

    def verify_object(
        self,
        image: np.ndarray,
        target_object: str,
        expected_position: Tuple[float, float],
    ) -> AgentResponse:
        """
        Verify if target object is present (Memory Agent).

        Args:
            image: Current RGB view
            target_object: Name of object to verify
            expected_position: Expected (x, y) position

        Returns:
            AgentResponse with verification result
        """
        params = {
            "target_object": target_object,
            "expected_position": expected_position,
        }

        cache_key = self._get_cache_key("verify_object", params, image)
        cached = self._get_cached(cache_key)
        if cached:
            return AgentResponse(success=True, data=cached, latency_ms=0, cached=True)

        self._check_rate_limit()
        start_time = time.time()

        try:
            result = self.vlm_client.verify_object(
                image, target_object, expected_position
            )
            latency = (time.time() - start_time) * 1000

            self.stats["vlm_calls"] += 1
            self.stats["total_latency_ms"] += latency

            self._set_cached(cache_key, result)

            return AgentResponse(success=True, data=result, latency_ms=latency)

        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def prioritize_targets(
        self,
        candidates: List[Dict],
        robot_position: Tuple[float, float],
        target_object: str,
    ) -> AgentResponse:
        """
        Prioritize candidate targets (Memory Agent, uses LLM).

        Args:
            candidates: List of candidate targets
            robot_position: Current robot position
            target_object: Target object name

        Returns:
            AgentResponse with prioritized targets
        """
        params = {
            "candidates": str(candidates),
            "robot_position": robot_position,
            "target_object": target_object,
        }

        cache_key = self._get_cache_key("prioritize_targets", params)
        cached = self._get_cached(cache_key)
        if cached:
            return AgentResponse(success=True, data=cached, latency_ms=0, cached=True)

        self._check_rate_limit()
        start_time = time.time()

        try:
            result = self.llm_client.prioritize_targets(
                candidates, robot_position, target_object
            )
            latency = (time.time() - start_time) * 1000

            self.stats["llm_calls"] += 1
            self.stats["total_latency_ms"] += latency

            self._set_cached(cache_key, result)

            return AgentResponse(success=True, data=result, latency_ms=latency)

        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    # ==================== Exploration Agent Methods ====================

    def plan_exploration(
        self,
        voronoi_summary: str,
        exploration_percentage: float,
        semantic_hotspots: List[Dict],
        current_phase: str,
        target_object: str,
    ) -> AgentResponse:
        """
        Plan exploration strategy (Exploration Agent, uses LLM).

        Args:
            voronoi_summary: Summary of Voronoi topology
            exploration_percentage: Coverage percentage (0-100)
            semantic_hotspots: Detected hotspots
            current_phase: Current navigation phase
            target_object: Target to find

        Returns:
            AgentResponse with exploration plan
        """
        params = {
            "voronoi_summary": voronoi_summary,
            "exploration_percentage": exploration_percentage,
            "semantic_hotspots": str(semantic_hotspots),
            "current_phase": current_phase,
            "target_object": target_object,
        }

        cache_key = self._get_cache_key("plan_exploration", params)
        cached = self._get_cached(cache_key)
        if cached:
            return AgentResponse(success=True, data=cached, latency_ms=0, cached=True)

        self._check_rate_limit()
        start_time = time.time()

        try:
            result = self.llm_client.plan_exploration_strategy(
                voronoi_summary,
                exploration_percentage,
                semantic_hotspots,
                current_phase,
                target_object,
            )
            latency = (time.time() - start_time) * 1000

            self.stats["llm_calls"] += 1
            self.stats["total_latency_ms"] += latency

            self._set_cached(cache_key, result)

            return AgentResponse(success=True, data=result, latency_ms=latency)

        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def decide_phase_transition(
        self,
        current_phase: str,
        phase_history: List[str],
        trigger_events: List[str],
        exploration_state: Dict,
    ) -> AgentResponse:
        """
        Decide phase transition (Exploration Agent, uses LLM).

        Args:
            current_phase: Current phase
            phase_history: Recent phase history
            trigger_events: Events that might trigger transition
            exploration_state: Current state metrics

        Returns:
            AgentResponse with transition decision
        """
        params = {
            "current_phase": current_phase,
            "phase_history": str(phase_history),
            "trigger_events": str(trigger_events),
            "exploration_state": str(exploration_state),
        }

        cache_key = self._get_cache_key("phase_transition", params)
        cached = self._get_cached(cache_key)
        if cached:
            return AgentResponse(success=True, data=cached, latency_ms=0, cached=True)

        self._check_rate_limit()
        start_time = time.time()

        try:
            result = self.llm_client.decide_phase_transition(
                current_phase, phase_history, trigger_events, exploration_state
            )
            latency = (time.time() - start_time) * 1000

            self.stats["llm_calls"] += 1
            self.stats["total_latency_ms"] += latency

            self._set_cached(cache_key, result)

            return AgentResponse(success=True, data=result, latency_ms=latency)

        except Exception as e:
            return AgentResponse(
                success=False,
                data={},
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    # ==================== Utility Methods ====================

    def describe_scene(self, image: np.ndarray, context: Optional[str] = None) -> str:
        """Get scene description (uses VLM)."""
        self._check_rate_limit()
        self.stats["vlm_calls"] += 1
        return self.vlm_client.describe_scene(image, context)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        avg_latency = (
            self.stats["total_latency_ms"] /
            max(1, self.stats["vlm_calls"] + self.stats["llm_calls"])
        )
        return {
            **self.stats,
            "avg_latency_ms": avg_latency,
            "cache_size": len(self._cache),
        }

    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()


# Global singleton instance
_global_interface: Optional[AgentLLMInterface] = None


def get_agent_interface(**kwargs) -> AgentLLMInterface:
    """
    Get the global agent interface instance.

    Creates the instance on first call, returns existing instance on subsequent calls.
    """
    global _global_interface
    if _global_interface is None:
        _global_interface = AgentLLMInterface(**kwargs)
    return _global_interface
