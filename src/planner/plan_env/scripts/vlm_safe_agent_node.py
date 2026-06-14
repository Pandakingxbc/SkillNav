#!/usr/bin/env python3
"""
VLM SafeAgent Node for SkillNav

This node handles VLM-based dead zone analysis for the SafeAgent.
It subscribes to dead zone analysis requests from the C++ ExplorationFSM
and uses VLM (Qwen-VL) to analyze whether obstacles are permanent or temporary.

Topics:
- Subscribes: /safe_agent/vlm_request (plan_env/VLMDeadZoneRequest)
- Subscribes: /habitat/camera_rgb (sensor_msgs/Image)
- Publishes: /safe_agent/vlm_response (plan_env/VLMDeadZoneResponse)
"""

import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from plan_env.msg import VLMDeadZoneRequest, VLMDeadZoneResponse

import sys
import os

# Add vlm module to path
# scripts -> plan_env -> planner -> src -> SkillNav (4 levels up)
vlm_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'vlm')
sys.path.insert(0, os.path.abspath(vlm_path))

try:
    from llm.qwen_vlm import QwenVLMClient
    VLM_AVAILABLE = True
except ImportError as e:
    rospy.logwarn(f"[VLMSafeAgent] VLM client not available: {e}")
    VLM_AVAILABLE = False


class VLMSafeAgentNode:
    """
    ROS node for VLM-based dead zone analysis.

    Workflow:
    1. Receive VLMDeadZoneRequest from C++ SafeAgent
    2. Get latest RGB image from camera
    3. Call VLM to analyze the scene
    4. Publish VLMDeadZoneResponse with penalty recommendation

    Optimizations to reduce VLM overhead:
    - Position-based caching: Don't re-analyze same location within cache_ttl
    - Throttling: Limit VLM calls per minute
    - Fast reject: Wait for min_escape_attempts before calling VLM
    """

    def __init__(self):
        rospy.init_node('vlm_safe_agent_node', anonymous=False)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Latest camera image
        self.latest_image = None
        self.image_timestamp = None

        # VLM client (lazy initialization)
        self._vlm_client = None

        # === Optimization: Caching ===
        # Cache format: {(node_id, round(x,1), round(y,1)): (result, timestamp)}
        self.result_cache = {}
        self.cache_ttl = rospy.get_param('~cache_ttl', 30.0)  # Cache valid for 30 seconds

        # === Optimization: Throttling ===
        self.call_timestamps = []  # List of recent VLM call timestamps
        self.max_calls_per_minute = rospy.get_param('~max_calls_per_minute', 10)

        # === Optimization: Min escape attempts ===
        self.min_escape_attempts = rospy.get_param('~min_escape_attempts', 2)  # Don't call VLM until 2 attempts

        # ROS subscribers
        self.request_sub = rospy.Subscriber(
            '/safe_agent/vlm_request',
            VLMDeadZoneRequest,
            self.request_callback,
            queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            '/habitat/camera_rgb',
            Image,
            self.image_callback,
            queue_size=1
        )

        # ROS publisher
        self.response_pub = rospy.Publisher(
            '/safe_agent/vlm_response',
            VLMDeadZoneResponse,
            queue_size=10
        )

        # Parameters
        self.image_max_age = rospy.get_param('~image_max_age', 2.0)  # seconds
        self.vlm_model = rospy.get_param('~vlm_model', 'qwen-vl-plus')

        rospy.loginfo("[VLMSafeAgent] Node initialized, VLM available: %s", VLM_AVAILABLE)
        rospy.loginfo("[VLMSafeAgent] Optimizations: cache_ttl=%.1fs, max_calls/min=%d, min_attempts=%d",
                     self.cache_ttl, self.max_calls_per_minute, self.min_escape_attempts)

    @property
    def vlm_client(self):
        """Lazy initialization of VLM client."""
        if self._vlm_client is None and VLM_AVAILABLE:
            try:
                self._vlm_client = QwenVLMClient(model=self.vlm_model)
                rospy.loginfo("[VLMSafeAgent] VLM client initialized with model: %s", self.vlm_model)
            except Exception as e:
                rospy.logerr("[VLMSafeAgent] Failed to initialize VLM client: %s", e)
        return self._vlm_client

    def image_callback(self, msg: Image):
        """Store latest camera image."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.image_timestamp = msg.header.stamp
        except Exception as e:
            rospy.logwarn("[VLMSafeAgent] Failed to convert image: %s", e)

    def get_cache_key(self, msg: VLMDeadZoneRequest) -> tuple:
        """Generate cache key from request (node_id + rounded position)."""
        return (msg.target_node_id, round(msg.robot_x, 1), round(msg.robot_y, 1))

    def check_cache(self, msg: VLMDeadZoneRequest) -> dict:
        """Check if we have a cached result for this location."""
        key = self.get_cache_key(msg)
        if key in self.result_cache:
            result, timestamp = self.result_cache[key]
            age = (rospy.Time.now() - timestamp).to_sec()
            if age < self.cache_ttl:
                rospy.loginfo("[VLMSafeAgent] Cache hit for node %d (age=%.1fs)", msg.target_node_id, age)
                return result
            else:
                # Cache expired, remove it
                del self.result_cache[key]
        return None

    def update_cache(self, msg: VLMDeadZoneRequest, result: dict):
        """Update cache with new result."""
        key = self.get_cache_key(msg)
        self.result_cache[key] = (result, rospy.Time.now())

        # Clean old cache entries (keep max 100)
        if len(self.result_cache) > 100:
            # Remove oldest entries
            sorted_entries = sorted(self.result_cache.items(), key=lambda x: x[1][1].to_sec())
            for key, _ in sorted_entries[:50]:
                del self.result_cache[key]

    def check_throttle(self) -> bool:
        """Check if we're within the call rate limit."""
        now = rospy.Time.now().to_sec()

        # Remove timestamps older than 1 minute
        self.call_timestamps = [t for t in self.call_timestamps if now - t < 60.0]

        if len(self.call_timestamps) >= self.max_calls_per_minute:
            return False
        return True

    def record_call(self):
        """Record a VLM call timestamp."""
        self.call_timestamps.append(rospy.Time.now().to_sec())

    def request_callback(self, msg: VLMDeadZoneRequest):
        """Handle dead zone analysis request."""
        rospy.loginfo("[VLMSafeAgent] Received analysis request for node %d at (%.2f, %.2f), attempts=%d",
                     msg.target_node_id, msg.robot_x, msg.robot_y, msg.escape_attempt_count)

        # === Optimization 1: Fast reject if not enough attempts ===
        if msg.escape_attempt_count < self.min_escape_attempts:
            rospy.loginfo("[VLMSafeAgent] Only %d attempts (min=%d), skipping VLM call",
                         msg.escape_attempt_count, self.min_escape_attempts)
            self.send_default_response(msg, f"min_attempts_not_met ({msg.escape_attempt_count}/{self.min_escape_attempts})")
            return

        # === Optimization 2: Check cache ===
        cached_result = self.check_cache(msg)
        if cached_result is not None:
            self.send_cached_response(msg, cached_result)
            return

        # === Optimization 3: Check throttle ===
        if not self.check_throttle():
            rospy.logwarn("[VLMSafeAgent] Rate limit exceeded (%d calls/min), skipping",
                         self.max_calls_per_minute)
            self.send_default_response(msg, "rate_limited")
            return

        # Check if we have a recent image
        if self.latest_image is None:
            rospy.logwarn("[VLMSafeAgent] No camera image available, sending default response")
            self.send_default_response(msg, "no_image")
            return

        # Check image age
        if self.image_timestamp is not None:
            image_age = (rospy.Time.now() - self.image_timestamp).to_sec()
            if image_age > self.image_max_age:
                rospy.logwarn("[VLMSafeAgent] Image too old (%.1fs), sending default response", image_age)
                self.send_default_response(msg, "stale_image")
                return

        # Call VLM for analysis
        if self.vlm_client is not None:
            self.analyze_with_vlm(msg)
        else:
            rospy.logwarn("[VLMSafeAgent] VLM client not available, sending default response")
            self.send_default_response(msg, "vlm_unavailable")

    def send_cached_response(self, req: VLMDeadZoneRequest, cached_result: dict):
        """Send response using cached VLM result."""
        response = VLMDeadZoneResponse()
        response.header.stamp = rospy.Time.now()
        response.target_node_id = req.target_node_id
        response.obstacle_type = cached_result.get('obstacle_type', 'unclear')
        response.confidence = cached_result.get('confidence', 0.5)
        response.abandon_region = cached_result.get('abandon_region', False)
        response.suggested_action = cached_result.get('suggested_action', 'continue_exploring')
        response.description = "[CACHED] " + cached_result.get('description', '')
        response.reasoning = cached_result.get('reasoning', '')
        response.penalty_strength = self.determine_penalty_strength(cached_result)

        if response.penalty_strength == "weak":
            response.retry_delay_seconds = 30.0
        else:
            response.retry_delay_seconds = 0.0

        response.abandon_region = (response.penalty_strength == "strong")
        self.response_pub.publish(response)

    def analyze_with_vlm(self, req: VLMDeadZoneRequest):
        """Perform VLM analysis and publish response."""
        try:
            # Record this call for throttling
            self.record_call()

            # Call VLM with target object info
            start_time = rospy.Time.now()
            result = self.vlm_client.analyze_dead_zone(
                image=self.latest_image,
                position=(req.robot_x, req.robot_y),
                attempt_count=req.escape_attempt_count,
                heading=np.degrees(req.robot_yaw),
                target_object=getattr(req, 'target_description', '')
            )
            latency = (rospy.Time.now() - start_time).to_sec()

            rospy.loginfo("[VLMSafeAgent] VLM analysis complete in %.2fs: %s", latency, result)

            # Cache the result for future use
            self.update_cache(req, result)

            # Build response
            response = VLMDeadZoneResponse()
            response.header.stamp = rospy.Time.now()
            response.target_node_id = req.target_node_id
            response.obstacle_type = result.get('obstacle_type', 'unclear')
            response.confidence = result.get('confidence', 0.5)
            response.abandon_region = result.get('abandon_region', False)
            response.suggested_action = result.get('suggested_action', 'continue_exploring')
            response.description = result.get('description', '')
            response.reasoning = result.get('reasoning', '')

            # Determine penalty strength based on VLM judgment
            # NEW: Pass the full result for target detection
            response.penalty_strength = self.determine_penalty_strength(result)

            # Set retry delay for temporary obstacles
            if response.penalty_strength == "weak":
                response.retry_delay_seconds = 30.0
            else:
                response.retry_delay_seconds = 0.0

            # Override abandon_region based on our logic
            # Only abandon if it's truly a system bug
            response.abandon_region = (response.penalty_strength == "strong")

            # Publish response
            self.response_pub.publish(response)

        except Exception as e:
            rospy.logerr("[VLMSafeAgent] VLM analysis failed: %s", e)
            self.send_default_response(req, f"vlm_error: {e}")

    # Target objects for ObjectNav - areas containing these should NEVER be abandoned
    TARGET_OBJECTS = ["toilet", "bed", "chair", "couch", "sofa", "tv", "television", "potted plant", "plant"]

    def determine_penalty_strength(self, vlm_result: dict) -> str:
        """
        Determine penalty strength based on VLM analysis.

        NEW LOGIC: Only penalize for SYSTEM BUGS, not normal obstacles.

        Returns:
            "strong": ONLY for system_boundary (simulation bugs)
            "none": for everything else (normal obstacles, target areas)
        """
        obstacle_type = vlm_result.get('obstacle_type', 'unclear')
        confidence = vlm_result.get('confidence', 0.5)
        contains_target = vlm_result.get('contains_target', False)
        is_system_bug = vlm_result.get('is_system_bug', False)
        visible_objects = vlm_result.get('visible_objects', [])
        description = vlm_result.get('description', '').lower()

        # RULE 1: If target object is visible, NEVER penalize
        if contains_target:
            rospy.loginfo("[VLMSafeAgent] Target object detected! No penalty applied.")
            return "none"

        # RULE 2: Check if any target object mentioned in description or visible_objects
        for target in self.TARGET_OBJECTS:
            if target in description or target in str(visible_objects).lower():
                rospy.loginfo("[VLMSafeAgent] Potential target '%s' in view! No penalty applied.", target)
                return "none"

        # RULE 3: Only penalize for confirmed SYSTEM BUGS
        if is_system_bug and obstacle_type == 'system_boundary' and confidence > 0.8:
            rospy.logwarn("[VLMSafeAgent] System boundary detected! Applying strong penalty.")
            return "strong"

        # RULE 4: For normal obstacles (furniture, walls), NO penalty
        # The robot can navigate around them using normal path planning
        if obstacle_type in ['normal_obstacle', 'permanent', 'temporary']:
            rospy.loginfo("[VLMSafeAgent] Normal obstacle detected. No penalty (robot can navigate around).")
            return "none"

        # Default: No penalty (conservative approach)
        rospy.loginfo("[VLMSafeAgent] Unclear situation. No penalty applied (conservative).")
        return "none"

    def send_default_response(self, req: VLMDeadZoneRequest, reason: str):
        """Send default response when VLM is unavailable."""
        response = VLMDeadZoneResponse()
        response.header.stamp = rospy.Time.now()
        response.target_node_id = req.target_node_id
        response.obstacle_type = "unclear"
        response.confidence = 0.3
        response.abandon_region = False
        response.suggested_action = "continue_trying"
        response.penalty_strength = "none"
        response.description = f"VLM analysis skipped: {reason}"
        response.reasoning = "Using default conservative response"
        response.retry_delay_seconds = 0.0

        self.response_pub.publish(response)
        rospy.loginfo("[VLMSafeAgent] Sent default response: %s", reason)

    def run(self):
        """Run the node."""
        rospy.loginfo("[VLMSafeAgent] Node running...")
        rospy.spin()


def main():
    try:
        node = VLMSafeAgentNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
