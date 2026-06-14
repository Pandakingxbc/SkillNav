"""
Qwen3-VL client using DashScope API.

This client is used for vision-language tasks in SkillNav:
- Dead zone analysis (Safe Agent)
- Object verification (Memory Agent)
- Scene understanding (Exploration Agent)

Environment variable required:
    DASHSCOPE_API_KEY: Your DashScope API key from Alibaba Cloud

Reference:
    https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-dashscope
    https://docs.litellm.ai/docs/providers/dashscope
"""

import os
import time
from typing import Optional, List, Dict, Any

import numpy as np

from .base import (
    BaseVLMClient,
    VLMMessage,
    Message,
    MessageRole,
    LLMResponse,
    image_to_base64,
)


class QwenVLMClient(BaseVLMClient):
    """
    Client for Qwen3-VL models via DashScope API.

    Supports both OpenAI-compatible endpoint and native DashScope endpoint.

    Usage:
        client = QwenVLMClient()  # Uses DASHSCOPE_API_KEY from env
        response = client.analyze_image(image, "What do you see?")
    """

    # Available models
    MODELS = {
        "qwen-vl-max": "qwen-vl-max",  # Most capable
        "qwen-vl-plus": "qwen-vl-plus",  # Balanced
        "qwen3-vl": "qwen3-vl-72b",  # Latest Qwen3-VL
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        """
        Initialize Qwen VLM client.

        Args:
            api_key: DashScope API key. If None, reads from DASHSCOPE_API_KEY env var
            model: Model name (qwen-vl-max, qwen-vl-plus, qwen3-vl)
            base_url: API base URL (OpenAI-compatible endpoint)
            timeout: Request timeout in seconds
            max_retries: Number of retries on failure
        """
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY not found. Set it in environment or pass api_key parameter."
            )

        super().__init__(
            api_key=api_key,
            model=self.MODELS.get(model, model),
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Initialize OpenAI client with custom base URL
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
        **kwargs
    ) -> LLMResponse:
        """
        Text-only chat completion.

        For VLM, use chat_with_images() instead for image inputs.
        """
        formatted_messages = [msg.to_dict() for msg in messages]

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

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

    def chat_with_images(
        self,
        messages: List[VLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """
        Chat completion with image inputs.

        Args:
            messages: List of VLMMessage with optional images
            temperature: Sampling temperature
            max_tokens: Max response tokens
            **kwargs: Additional parameters

        Returns:
            LLMResponse with model's analysis
        """
        # Format messages for Qwen VL API
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, VLMMessage):
                formatted_messages.append(self._format_vlm_message(msg))
            else:
                formatted_messages.append(msg.to_dict())

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )

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

    def _format_vlm_message(self, msg: VLMMessage) -> Dict[str, Any]:
        """Format VLMMessage for Qwen VL API."""
        content = []

        # Add images
        if msg.images:
            for img in msg.images:
                img_b64 = image_to_base64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })

        # Add text
        content.append({
            "type": "text",
            "text": msg.text
        })

        return {
            "role": msg.role.value,
            "content": content
        }

    # ==================== Agent-specific methods ====================

    # Target objects for ObjectNav - should NEVER abandon areas containing these
    TARGET_OBJECTS = ["toilet", "bed", "chair", "couch", "sofa", "tv", "television", "potted plant", "plant"]

    def analyze_dead_zone(
        self,
        image: np.ndarray,
        position: tuple,
        attempt_count: int,
        heading: float,
        target_object: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze a potential dead zone for Safe Agent.

        IMPORTANT: This method focuses on detecting SYSTEM BUGS (simulation boundaries,
        map edges, void areas) rather than normal obstacles like furniture.

        Args:
            image: Current RGB view
            position: Robot (x, y) position
            attempt_count: Number of failed escape attempts
            heading: Current heading in degrees
            target_object: Current navigation target (e.g., "toilet")

        Returns:
            Dict with keys:
                - obstacle_type: "system_boundary" | "normal_obstacle" | "target_area" | "unclear"
                - contains_target: bool (if target object is visible)
                - abandon_region: bool
                - confidence: float 0-1
                - suggested_action: str
                - description: str
        """
        prompt = f"""Analyze this robot navigation situation. The robot failed to move forward.

Current situation:
- Position: ({position[0]:.2f}, {position[1]:.2f})
- Failed attempts: {attempt_count}
- Heading: {heading:.1f}°
- Navigation target: {target_object if target_object else "unknown"}

IMPORTANT: Your task is to identify if this is a SYSTEM BUG or just a normal obstacle.

Look for these specific signs of SYSTEM BOUNDARY (bugs):
1. White/gray void areas (missing textures)
2. Abrupt edges where the world ends
3. Black holes or rendering artifacts
4. Unusual geometric patterns that don't look like real rooms

Also check: Do you see any of these TARGET OBJECTS in the image?
- toilet, bed, chair, couch/sofa, tv/television, potted plant

Respond in JSON format:
{{
    "obstacle_type": "system_boundary" | "normal_obstacle" | "target_area" | "unclear",
    "description": "What you see",
    "visible_objects": ["list", "of", "objects", "you", "see"],
    "contains_target": true | false,
    "is_system_bug": true | false,
    "abandon_region": true | false,
    "confidence": 0.0-1.0,
    "reasoning": "Your reasoning",
    "suggested_action": "mark_dead_zone" | "continue_exploring" | "reroute"
}}

CRITICAL RULES:
1. If you see a target object (toilet, bed, chair, couch, tv, plant) → contains_target=true, abandon_region=FALSE
2. If you see normal furniture (bed, table, cabinet) blocking the path → This is NOT a reason to abandon! Set abandon_region=FALSE
3. ONLY set abandon_region=true if you see SYSTEM BOUNDARY signs (void, artifacts, map edge)
4. Normal walls and furniture are NOT system bugs - the robot can navigate around them"""

        system_prompt = """You are a navigation bug detector. Your job is to identify SYSTEM BUGS
(simulation boundaries, void areas, rendering errors) - NOT normal obstacles like furniture.
Normal furniture blocking a path is NOT a reason to abandon an area."""

        return self.analyze_image_json(
            image,
            prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=512
        )

    def verify_object(
        self,
        image: np.ndarray,
        target_object: str,
        expected_position: tuple,
    ) -> Dict[str, Any]:
        """
        Verify if target object is present for Memory Agent.

        Args:
            image: Current RGB view
            target_object: Name of object to verify (e.g., "toilet", "chair")
            expected_position: Expected (x, y) position

        Returns:
            Dict with keys:
                - object_present: bool
                - confidence: float 0-1
                - actual_object: str (what is actually there)
                - description: str
        """
        prompt = f"""Look at this image and determine if there is a "{target_object}" visible.

Please respond in JSON format:
{{
    "object_present": true | false,
    "confidence": 0.0-1.0,
    "actual_object": "what you actually see in the center/foreground",
    "description": "Brief description of the scene"
}}

Be precise - only say object_present=true if you clearly see a {target_object}."""

        system_prompt = f"You are verifying whether a '{target_object}' is present in the robot's view."

        return self.analyze_image_json(
            image,
            prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=256
        )

    def describe_scene(
        self,
        image: np.ndarray,
        context: Optional[str] = None,
    ) -> str:
        """
        Get a brief scene description for Exploration Agent.

        Args:
            image: Current RGB view
            context: Optional context (e.g., "looking for bathroom")

        Returns:
            Scene description string
        """
        prompt = "Briefly describe this indoor scene in 1-2 sentences. Focus on room type and notable objects."
        if context:
            prompt += f"\nContext: {context}"

        system_prompt = "You are a scene description assistant for robot navigation."

        return self.analyze_image(
            image,
            prompt,
            system_prompt=system_prompt,
            temperature=0.5,
            max_tokens=100
        )

    # ==================== Memory Agent Methods ====================

    # Room types for classification
    ROOM_TYPES = ["bathroom", "bedroom", "kitchen", "living_room", "dining_room", "hallway", "office", "unknown"]

    def describe_scene_for_memory(
        self,
        image: np.ndarray,
        position: tuple,
        heading: float,
        target_object: str = "",
        nearby_context: List[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate semantic memory for a topology node.

        Args:
            image: Current RGB view (habitat-to-ros native)
            position: Robot (x, y) position
            heading: Current heading in degrees
            target_object: Current navigation target (e.g., "toilet")
            nearby_context: List of nearby node memories for context

        Returns:
            Dict with keys:
                - description: str (scene description)
                - room_type: str (inferred room type)
                - observed_objects: List[str] (objects seen)
                - confidence: float 0-1
                - reasoning: str
        """
        # Build context from nearby memories
        context_str = ""
        if nearby_context:
            context_str = "\nNearby areas (for spatial context):\n"
            for mem in nearby_context[:3]:  # Max 3 nearby memories
                context_str += f"- Node {mem['node_id']}: {mem['room_type']}"
                if mem.get('objects'):
                    context_str += f" (contains: {', '.join(mem['objects'][:3])})"
                context_str += "\n"

        prompt = f"""Analyze this indoor scene for robot navigation memory.

Current position: ({position[0]:.2f}, {position[1]:.2f})
Heading: {heading:.1f}°
Navigation target: {target_object if target_object else "exploring"}
{context_str}

Please identify:
1. What type of room is this?
2. What objects do you see? (focus on: toilet, bed, chair, couch, sofa, tv, plant, table, sink, refrigerator)
3. Brief scene description

Respond in JSON format:
{{
    "room_type": "bathroom" | "bedroom" | "kitchen" | "living_room" | "dining_room" | "hallway" | "office" | "unknown",
    "observed_objects": ["list", "of", "objects"],
    "description": "Brief scene description",
    "confidence": 0.0-1.0,
    "reasoning": "Your reasoning"
}}"""

        system_prompt = """You are a semantic memory system for robot navigation.
Your job is to identify room types and objects to help the robot remember what it has seen.
Be accurate and concise. Focus on ObjectNav target objects."""

        return self.analyze_image_json(
            image,
            prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=400
        )

    def verify_object_with_context(
        self,
        images: List[np.ndarray],
        target_object: str,
        position: tuple,
        heading: float,
        nearby_context: List[Dict] = None,
        has_cluster_image: bool = False,
    ) -> Dict[str, Any]:
        """
        Verify object presence with nearby memory context and ObjectMap cluster image.

        Args:
            images: List of images [current_rgb, optional_cluster_image]
            target_object: Object to verify (e.g., "toilet")
            position: Robot (x, y) position
            heading: Current heading in degrees
            nearby_context: List of nearby node memories
            has_cluster_image: Whether second image is ObjectMap cluster

        Returns:
            Dict with keys:
                - target_visible: bool
                - target_in_cluster: bool (if cluster image provided)
                - is_false_positive: bool
                - semantic_penalty: float (1.0 = no penalty)
                - verification_confidence: float 0-1
                - verification_reasoning: str
                - description: str
                - room_type: str
                - observed_objects: List[str]
                - update_memory: bool
                - reasoning: str
                - confidence: float
        """
        # Build context from nearby memories
        context_str = ""
        if nearby_context:
            context_str = "\nNearby memories (spatial context):\n"
            for mem in nearby_context[:3]:
                context_str += f"- Node {mem['node_id']}: {mem['room_type']}"
                if mem.get('objects'):
                    context_str += f" (seen: {', '.join(mem['objects'][:3])})"
                context_str += "\n"

        image_desc = "Image 1: Current robot view (habitat RGB)"
        if has_cluster_image:
            image_desc += "\nImage 2: ObjectMap cluster detection (what the system thinks is the target)"

        prompt = f"""VERIFICATION TASK: Is there a "{target_object}" visible?

{image_desc}

Current position: ({position[0]:.2f}, {position[1]:.2f})
Heading: {heading:.1f}°
Target object: {target_object}
{context_str}

Analyze:
1. Do you see a {target_object} in Image 1 (current view)?
2. If Image 2 is provided, does it show a {target_object}?
3. Is there a mismatch (system detected object but it's not a {target_object})?

Respond in JSON format:
{{
    "target_visible": true | false,
    "target_in_cluster": true | false,
    "is_false_positive": true | false,
    "semantic_penalty": 0.0-1.0,
    "verification_confidence": 0.0-1.0,
    "verification_reasoning": "Your verification reasoning",
    "room_type": "bathroom" | "bedroom" | "kitchen" | "living_room" | "hallway" | "unknown",
    "observed_objects": ["list", "of", "objects"],
    "description": "Brief scene description",
    "update_memory": true | false,
    "reasoning": "Overall reasoning",
    "confidence": 0.0-1.0
}}

IMPORTANT:
- semantic_penalty: Set to 1.0 if target is visible or unclear. Set to 0.3-0.7 if confirmed FALSE POSITIVE.
- is_false_positive: true ONLY if system claims to see {target_object} but you clearly see it's NOT there.
- update_memory: true if scene understanding should be stored at this node."""

        system_prompt = f"""You are verifying whether a '{target_object}' is present.
Your job is to detect FALSE POSITIVES in the ObjectMap detection system.
If the system thinks there's a {target_object} but you see something else, report is_false_positive=true.
Be conservative: only mark as false positive if you are confident."""

        # Handle single or multiple images
        if len(images) == 1:
            return self.analyze_image_json(
                images[0],
                prompt,
                system_prompt=system_prompt,
                temperature=0.2,
                max_tokens=500
            )
        else:
            # For multiple images, use chat_with_images
            from .base import VLMMessage, MessageRole
            messages = [
                VLMMessage(
                    role=MessageRole.USER,
                    text=prompt,
                    images=images
                )
            ]
            response = self.chat_with_images(
                messages,
                temperature=0.2,
                max_tokens=500
            )
            # Parse JSON from response
            import json
            import re
            content = response.content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            # Return default if parsing fails
            return {
                "target_visible": False,
                "target_in_cluster": False,
                "is_false_positive": False,
                "semantic_penalty": 1.0,
                "verification_confidence": 0.3,
                "verification_reasoning": "Failed to parse VLM response",
                "room_type": "unknown",
                "observed_objects": [],
                "description": content[:200] if content else "",
                "update_memory": False,
                "reasoning": "Response parsing error",
                "confidence": 0.3
            }


# Convenience function
def get_qwen_vlm_client(**kwargs) -> QwenVLMClient:
    """Factory function to create QwenVLMClient with default settings."""
    return QwenVLMClient(**kwargs)
