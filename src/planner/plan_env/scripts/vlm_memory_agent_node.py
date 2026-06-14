#!/usr/bin/env python3
"""
VLM Memory Agent Node for SkillNav — single-shot verification with 3 annotated frames.

Design (2026-05-19, modeled after InfoNav vlm_approach_verifier.py):
- Cache last N annotated RGB frames from /habitat/camera_rgb (already drawn with
  red/green bboxes by get_object_utils.py at the Habitat side).
- On REQUEST_VERIFY, pick top-3 most recent frames in cache and send them in a
  single VLM call along with an InfoNav-style prompt that:
    * names target visual characteristics
    * names common confusion classes for the 6 HM3D categories
    * asks for a 3-level decision: CONFIRM / UNCERTAIN / REJECT + CONFIDENCE
- No multi-distance voting. One call decides.

Topics:
- Subscribes: /memory_agent/vlm_request (plan_env/VLMMemoryRequest)
- Subscribes: /habitat/camera_rgb (sensor_msgs/Image) — already annotated
- Publishes:  /memory_agent/vlm_response (plan_env/VLMMemoryResponse)
"""

import os
import re
import sys
import base64
from collections import deque
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from plan_env.msg import VLMMemoryRequest, VLMMemoryResponse, MultipleMasksWithConfidence

# Make vlm.* importable for the OpenAI-compatible DashScope client
_VLM_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'vlm')
)
if _VLM_PATH not in sys.path:
    sys.path.insert(0, _VLM_PATH)

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ============================================================
#   Object characteristics + common confusions (HM3D 6 classes)
#   Lifted from InfoNav vlm_approach_verifier.py (Apache 2.0 sibling)
# ============================================================

_CHARACTERISTICS = {
    "toilet":       "Porcelain bowl with seat/lid, water tank behind. Usually in bathroom.",
    "bed":          "Mattress for sleeping, usually with bedding/sheets/pillows.",
    "couch":        "Upholstered multi-seat furniture (2-3+ persons) with cushions and backrest.",
    "sofa":         "Same as couch. Upholstered multi-seat furniture with cushions.",
    "chair":        "Single-seat furniture for ONE person. Includes dining chair, armchair, office chair.",
    "tv":           "Screen device mounted or on stand. IGNORE screen content, judge by hardware shape only.",
    "tv_monitor":   "Screen device mounted or on stand. IGNORE screen content, judge by hardware shape only.",
    "plant":        ("ANY plant or plant-like decoration. Includes live plants, dried flowers, "
                     "artificial plants, decorative branches in vase. Be lenient!"),
    "potted plant": ("ANY plant or plant-like decoration. Includes live plants, dried flowers, "
                     "artificial plants, decorative branches in vase. Be lenient!"),
}

_CONFUSIONS = {
    "chair":         ("Often confused with: couch (multi-seat), toilet (has tank), stool (no backrest), "
                      "potted plant, bench. CHAIR is SINGLE-seat for ONE person with backrest. "
                      "Key: vs couch (seats multiple), vs toilet (porcelain bowl + tank), vs stool (no backrest)."),
    "couch":         ("Often confused with: bed (for sleeping), bench (no cushions), chair (single-seat), "
                      "dining table. COUCH is LARGE multi-seat furniture (2-3+ persons) with soft cushions "
                      "and backrest. Key: seats multiple people side by side, has armrests, vs chair (single person)."),
    "sofa":          ("Same as couch. Often confused with: bed, bench, chair, dining table. "
                      "Multi-seat upholstered furniture."),
    "toilet":        ("Often confused with: chair (no tank), bench, potted plant (white pot), sink (has faucet). "
                      "TOILET has porcelain bowl with seat/lid and water tank behind. In bathroom. "
                      "Key: has water tank, vs sink (has faucet), vs chair (no porcelain bowl)."),
    "tv":            ("Often confused with: laptop (has keyboard), computer monitor (small, on desk), "
                      "picture frame (static art), window (sees outdoors). TV is a standalone screen device "
                      "mounted on wall or on a stand. IGNORE screen content."),
    "tv_monitor":    ("Often confused with: laptop, computer monitor, picture frame, window. "
                      "Standalone screen device. IGNORE screen content."),
    "bed":           ("Often confused with: couch (has armrests), dining table, bench (no mattress). "
                      "BED has mattress with bedding/sheets, designed for sleeping/lying down."),
    "plant":         ("Often confused with: lamp (tall standing), teddy bear, toilet (white). "
                      "Includes live plants, dried flowers, artificial plants. Be lenient!"),
    "potted plant":  ("Often confused with: lamp (tall standing), teddy bear, toilet (white). "
                      "Includes live plants, dried flowers, artificial plants. Be lenient!"),
}

_HIGH_FP_CATEGORIES = {"couch", "sofa", "chair", "toilet", "tv", "tv_monitor"}


def _normalize_category(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    s = s.replace("_", " ").strip()
    if s == "tv monitor":
        return "tv_monitor"
    return s


def _display_name(category: str) -> str:
    return category.replace("_", " ")


def _get_characteristic(category: str) -> str:
    key = category if category in _CHARACTERISTICS else category.replace(" ", "")
    return _CHARACTERISTICS.get(key,
        _CHARACTERISTICS.get(category, f"A {_display_name(category)}."))


def _get_confusion(category: str) -> str:
    key = category if category in _CONFUSIONS else category.replace(" ", "")
    return _CONFUSIONS.get(key,
        _CONFUSIONS.get(category,
            "Be lenient - CONFIRM if the object reasonably matches the target category."))


def _build_prompt(category: str, num_images: int) -> str:
    cat = _normalize_category(category)
    display = _display_name(cat)
    chars = _get_characteristic(cat)
    confs = _get_confusion(cat)
    image_desc = ("this image" if num_images <= 1
                  else f"these {num_images} images taken during approach (red bbox = target, green = similar)")

    strictness = ""
    if cat in _HIGH_FP_CATEGORIES:
        strictness = (f'\n**Note**: "{display}" is often confused with similar objects. '
                      f'Verify carefully.\n')

    return f"""A robot is navigating indoors to find a "{display}". Examine {image_desc} and determine if the target object is present.

**IMPORTANT**: Make your judgment based ONLY on what you see in the images. The detector that drew the bounding boxes may have made a false positive — do NOT assume the target is present just because a red box is shown.
{strictness}
**Target**: {display} — {chars}
**Common confusions**: {confs}

**Rules**:
1. **Independent Judgment**: Judge purely from visual evidence. Boxes are hints, not proof. Verify the object inside each box matches the target category.
2. **Reachability**: REJECT if the object is outside a window, a reflection in a mirror, or behind a glass barrier.
3. **Multiple Objects**: There may be multiple "{display}" objects in the scene — CONFIRM if ANY one of them is clearly the target.
4. **Partial/Occluded View**: CONFIRM if visible features are sufficient to identify it. If too little is visible to judge, choose UNCERTAIN.
5. **When Uncertain**: If the visual evidence is ambiguous, choose UNCERTAIN. Do NOT guess.

Reply in EXACTLY this format:
DECISION: CONFIRM / UNCERTAIN / REJECT
CONFIDENCE: 0.0-1.0
REASON: Brief explanation based on visual features you observed"""


def _parse_response(text: str) -> Tuple[int, float, str]:
    """Return (verification_result_enum, confidence, reason).
    enum: VERIFY_FALSE_POSITIVE=0, VERIFY_UNCERTAIN=1, VERIFY_TRUE_POSITIVE=2.
    """
    low = text.lower()
    m_dec  = re.search(r'decision\s*[:\-]\s*(confirm|uncertain|reject|yes|no)', low)
    m_conf = re.search(r'confidence\s*[:\-]\s*([0-9]*\.?[0-9]+)', low)
    m_reas = re.search(r'reason\s*[:\-]\s*(.+?)(?:\n|$)', text, flags=re.IGNORECASE | re.DOTALL)

    if m_dec:
        d = m_dec.group(1)
        if d in ('confirm', 'yes'):
            decision = VLMMemoryResponse.VERIFY_TRUE_POSITIVE
        elif d in ('reject', 'no'):
            decision = VLMMemoryResponse.VERIFY_FALSE_POSITIVE
        else:
            decision = VLMMemoryResponse.VERIFY_UNCERTAIN
    else:
        # keyword fallback
        has_confirm = bool(re.search(r'\bconfirm', low))
        has_reject  = bool(re.search(r'\breject', low))
        if has_confirm and not has_reject:
            decision = VLMMemoryResponse.VERIFY_TRUE_POSITIVE
        elif has_reject and not has_confirm:
            decision = VLMMemoryResponse.VERIFY_FALSE_POSITIVE
        else:
            decision = VLMMemoryResponse.VERIFY_UNCERTAIN

    if m_conf:
        try:
            conf = max(0.0, min(1.0, float(m_conf.group(1))))
        except ValueError:
            conf = 0.5
    else:
        conf = 0.8 if decision != VLMMemoryResponse.VERIFY_UNCERTAIN else 0.5

    reason = m_reas.group(1).strip() if m_reas else text[:200].strip()
    return decision, conf, reason


def _image_to_base64(img_rgb: np.ndarray, quality: int = 85) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('ascii')


# ============================================================
#   Node
# ============================================================

class VLMMemoryAgentNode:

    REQUEST_CREATE = 0
    REQUEST_UPDATE = 1
    REQUEST_VERIFY = 2

    def __init__(self):
        rospy.init_node('vlm_memory_agent_node', anonymous=False)
        self.bridge = CvBridge()

        # Params
        self.image_max_age   = rospy.get_param('~image_max_age',   2.0)
        self.frame_cache_size = rospy.get_param('~frame_cache_size', 20)
        self.frame_max_age    = rospy.get_param('~frame_max_age',    6.0)
        self.num_frames_to_send = rospy.get_param('~num_frames_to_send', 3)
        self.frame_min_gap_sec  = rospy.get_param('~frame_min_gap_sec', 0.3)
        self.model_name      = rospy.get_param('~vlm_model', 'qwen-vl-plus')
        self.api_timeout     = rospy.get_param('~api_timeout', 30.0)
        self.api_max_retries = rospy.get_param('~api_max_retries', 1)

        # Annotated RGB cache: deque of (timestamp_sec, image_rgb)
        self.frame_cache = deque(maxlen=self.frame_cache_size)

        # Detector-side target-detection events. Populated from
        # /detector/clouds_with_scores when label_indices contains 0 (target).
        # The frame selector prefers RGB frames whose timestamp is within
        # `target_detection_window` of one of these events, so VLM is fed
        # frames that actually carry a target bbox — not random "agent looking
        # at empty wall" frames the cluster has wandered off to.
        self.target_detection_ts = deque(maxlen=50)
        self.target_detection_window = rospy.get_param(
            '~target_detection_window', 0.5)
        self.min_target_frames = rospy.get_param('~min_target_frames', 2)

        # OpenAI-compatible DashScope client
        self.client = None
        if _OPENAI_AVAILABLE:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if api_key:
                try:
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        timeout=self.api_timeout,
                    )
                    rospy.loginfo("[VLMMemoryAgent] OpenAI/DashScope client ready (model=%s)",
                                  self.model_name)
                except Exception as e:
                    rospy.logerr("[VLMMemoryAgent] OpenAI client init failed: %s", e)
            else:
                rospy.logerr("[VLMMemoryAgent] DASHSCOPE_API_KEY not set; VLM verification disabled")
        else:
            rospy.logerr("[VLMMemoryAgent] openai package not available; VLM verification disabled")

        # ROS I/O
        self.request_sub = rospy.Subscriber(
            '/memory_agent/vlm_request', VLMMemoryRequest,
            self.request_callback, queue_size=1)
        self.rgb_sub = rospy.Subscriber(
            '/habitat/camera_rgb', Image,
            self.rgb_callback, queue_size=2)
        # Detector output — used solely to learn WHEN the target object was
        # actually drawn into a frame. Per habitat_evaluation.py the publish
        # of /detector/clouds_with_scores happens in the same tick as the
        # /habitat/camera_rgb publish, so timestamp proxy via rospy.Time.now()
        # is good enough for matching.
        self.det_sub = rospy.Subscriber(
            '/detector/clouds_with_scores', MultipleMasksWithConfidence,
            self.detection_callback, queue_size=2)
        self.response_pub = rospy.Publisher(
            '/memory_agent/vlm_response', VLMMemoryResponse, queue_size=10)

        rospy.loginfo("[VLMMemoryAgent] Initialized (frame_cache=%d, send=%d, max_age=%.1fs)",
                      self.frame_cache_size, self.num_frames_to_send, self.frame_max_age)

    # ---------------- RGB cache ----------------

    def rgb_callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            rospy.logwarn_throttle(5.0, "[VLMMemoryAgent] RGB convert failed: %s", e)
            return
        ts = msg.header.stamp.to_sec()
        # Throttle: only keep frames at least frame_min_gap_sec apart
        if self.frame_cache and (ts - self.frame_cache[-1][0]) < self.frame_min_gap_sec:
            return
        self.frame_cache.append((ts, img.copy()))

    def detection_callback(self, msg: MultipleMasksWithConfidence):
        """Record the timestamp of any frame in which the detector saw the target
        (label_indices entry == 0). Used by _select_recent_frames to prefer
        frames that actually carry a target bbox."""
        if 0 in msg.label_indices:
            self.target_detection_ts.append(rospy.Time.now().to_sec())

    def _select_recent_frames(self, n: int) -> List[np.ndarray]:
        """Select up to n annotated RGB frames for VLM verification, with a
        two-tier policy:
          (1) Prefer frames whose timestamp is within `target_detection_window`
              of any recorded target-detection event. Require >= min_target_frames
              such "target-confirmed" frames; otherwise we can't trust the input
              and fall back.
          (2) Fallback: most recent frames in cache, regardless of detection.
        """
        if not self.frame_cache:
            return []
        now = rospy.Time.now().to_sec()
        fresh = [(t, im) for (t, im) in self.frame_cache
                 if (now - t) <= self.frame_max_age]
        if not fresh:
            fresh = list(self.frame_cache)

        # Tier (1): target-confirmed frames
        if self.target_detection_ts:
            ts_arr = list(self.target_detection_ts)
            win = self.target_detection_window
            confirmed = [(t, im) for (t, im) in fresh
                         if any(abs(t - dt) <= win for dt in ts_arr)]
            if len(confirmed) >= self.min_target_frames:
                confirmed.sort(key=lambda p: p[0], reverse=True)
                top = [im for (_, im) in confirmed[:n]]
                rospy.loginfo(
                    "[VLMMemoryAgent] picked %d target-confirmed frames "
                    "(cache=%d, target events=%d)",
                    len(top), len(fresh), len(ts_arr))
                return top
            rospy.logwarn(
                "[VLMMemoryAgent] only %d target-confirmed frames available "
                "(need %d), falling back to most-recent",
                len(confirmed), self.min_target_frames)

        # Tier (2): fallback to most recent
        fresh.sort(key=lambda p: p[0], reverse=True)
        return [im for (_, im) in fresh[:n]]

    # ---------------- Request handling ----------------

    def request_callback(self, msg: VLMMemoryRequest):
        req_str = {0: "CREATE", 1: "UPDATE", 2: "VERIFY"}.get(msg.request_type, "UNKNOWN")
        rospy.loginfo("[VLMMemoryAgent] %s request: node=%d target='%s' at (%.2f, %.2f)",
                      req_str, msg.target_node_id, msg.target_object, msg.robot_x, msg.robot_y)

        if msg.request_type == self.REQUEST_VERIFY:
            self.process_verification(msg)
        else:
            # CREATE/UPDATE: not used in current flow; respond with default to keep
            # the C++ async queue accounting consistent.
            self.send_default_response(msg, "create_update_not_supported_in_single_shot_mode")

    def process_verification(self, req: VLMMemoryRequest):
        t_start = rospy.Time.now()

        if self.client is None:
            self.send_default_response(req, "vlm_client_unavailable")
            return

        frames = self._select_recent_frames(self.num_frames_to_send)
        if not frames:
            rospy.logwarn("[VLMMemoryAgent] No annotated frames in cache, cannot verify")
            self.send_default_response(req, "no_frames_available")
            return

        if not req.target_object:
            rospy.logwarn("[VLMMemoryAgent] Empty target_object in request, defaulting to UNCERTAIN")
            self.send_default_response(req, "empty_target_object")
            return

        # Build payload: N annotated images + InfoNav-style prompt
        prompt = _build_prompt(req.target_object, len(frames))
        content = []
        for img in frames:
            try:
                content.append({"type": "image_url",
                                "image_url": {"url": _image_to_base64(img)}})
            except Exception as e:
                rospy.logwarn("[VLMMemoryAgent] Image encode failed: %s", e)
        content.append({"type": "text", "text": prompt})

        rospy.loginfo("[VLMMemoryAgent] Calling VLM (%s) with %d annotated frames for target='%s'",
                      self.model_name, len(frames), req.target_object)

        decision = VLMMemoryResponse.VERIFY_UNCERTAIN
        confidence = 0.5
        reason = ""
        response_text = ""

        last_err = None
        for attempt in range(1 + self.api_max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    extra_body={'enable_thinking': False},
                )
                response_text = completion.choices[0].message.content or ""
                decision, confidence, reason = _parse_response(response_text)
                last_err = None
                break
            except Exception as e:
                last_err = e
                rospy.logwarn("[VLMMemoryAgent] VLM call attempt %d failed: %s", attempt + 1, e)

        latency = (rospy.Time.now() - t_start).to_sec()
        if last_err is not None:
            rospy.logerr("[VLMMemoryAgent] VLM failed after retries: %s (latency=%.2fs)",
                         last_err, latency)
            self.send_default_response(req, f"vlm_api_error: {last_err}")
            return

        dec_name = {VLMMemoryResponse.VERIFY_FALSE_POSITIVE: "REJECT",
                    VLMMemoryResponse.VERIFY_UNCERTAIN:      "UNCERTAIN",
                    VLMMemoryResponse.VERIFY_TRUE_POSITIVE:  "CONFIRM"}.get(decision, "?")
        rospy.loginfo("[VLMMemoryAgent] decision=%s conf=%.2f latency=%.2fs reason=%s",
                      dec_name, confidence, latency, reason[:200])

        # Confidence adjustment: REJECT → strong penalty, CONFIRM → boost, UNCERTAIN → noop
        if decision == VLMMemoryResponse.VERIFY_FALSE_POSITIVE:
            conf_adj = 0.4
            is_fp = True
            sem_penalty = 0.5
        elif decision == VLMMemoryResponse.VERIFY_TRUE_POSITIVE:
            conf_adj = 1.3
            is_fp = False
            sem_penalty = 1.0
        else:
            conf_adj = 1.0
            is_fp = False
            sem_penalty = 1.0

        resp = VLMMemoryResponse()
        resp.header.stamp = rospy.Time.now()
        resp.target_node_id = req.target_node_id
        resp.request_type = req.request_type
        resp.scene_description = ""
        resp.inferred_room_type = "unknown"
        resp.observed_objects = []
        resp.confidence = confidence
        resp.target_visible = (decision == VLMMemoryResponse.VERIFY_TRUE_POSITIVE)
        resp.target_in_cluster = resp.target_visible
        resp.verification_confidence = confidence
        resp.verification_reasoning = reason[:500]
        resp.verification_result = decision
        resp.confidence_adjustment = conf_adj
        resp.update_memory = False  # single-shot verifier does not update scene memory
        resp.is_false_positive = is_fp
        resp.semantic_penalty = sem_penalty
        resp.reasoning = reason[:500]
        resp.processing_time = latency
        self.response_pub.publish(resp)

    def send_default_response(self, req: VLMMemoryRequest, reason: str):
        resp = VLMMemoryResponse()
        resp.header.stamp = rospy.Time.now()
        resp.target_node_id = req.target_node_id
        resp.request_type = req.request_type
        resp.scene_description = ""
        resp.inferred_room_type = "unknown"
        resp.observed_objects = []
        resp.confidence = 0.0
        resp.target_visible = False
        resp.target_in_cluster = False
        resp.verification_confidence = 0.0
        resp.verification_reasoning = reason
        resp.verification_result = VLMMemoryResponse.VERIFY_UNCERTAIN
        resp.confidence_adjustment = 1.0
        resp.update_memory = False
        resp.is_false_positive = False
        resp.semantic_penalty = 1.0
        resp.reasoning = f"default: {reason}"
        resp.processing_time = 0.0
        self.response_pub.publish(resp)
        rospy.logwarn("[VLMMemoryAgent] Sent default UNCERTAIN response: %s", reason)

    def run(self):
        rospy.loginfo("[VLMMemoryAgent] Running...")
        rospy.spin()


def main():
    try:
        VLMMemoryAgentNode().run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
