#!/usr/bin/env python3
"""
Strategic LLM bridge for SkillNav.

Subscribes:  /strategic_agent/llm_request  (plan_env/StrategicLLMRequest)
Publishes:   /strategic_agent/llm_response (plan_env/StrategicLLMResponse)

For each request, calls DeepSeek (model defaulting to ``deepseek-chat`` —
the platform's latest non-reasoning model) and posts back a JSON-parsed
StrategicLLMResponse. The C++ StrategicAgent then drains its AsyncResultBuffer
and applies the new fusion weights to MultiValueMapManager.

API key is read from DEEPSEEK_API_KEY at startup. If the env var is missing
or the API call fails, the bridge falls back to a deterministic rule that
matches the C++ side's rule_based stub, so episodes still run.

Parameters
----------
~model          : str, DeepSeek model id  (default: ``deepseek-chat``)
~base_url       : str, API base URL       (default: ``https://api.deepseek.com``)
~temperature    : float                   (default: 0.3)
~timeout_sec    : float, per-call HTTP timeout (default: 8.0)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import rospy
from plan_env.msg import StrategicLLMRequest, StrategicLLMResponse


SYSTEM_PROMPT = """You plan strategy for a zero-shot ObjectNav robot. Each call, pick:

phase: 0=BROAD_EXPLORATION, 1=DIRECTED_SEARCH, 2=TARGET_APPROACH, 3=VERIFICATION
w_sr, w_ig: ValueMap weights for semantic-relevance vs info-gain (sum≈1.0)

Heuristics (use context; do not blindly apply):
- coverage<0.30                         → phase 0, w≈(0.3,0.7)
- candidate_count≥1 and fp_count low    → phase 2, w≈(0.8,0.2)
- coverage>0.70 and candidate_count==0  → phase 1, w≈(0.6,0.4) (semantic push)
- otherwise (mid coverage, no candidate)→ phase 1, w≈(0.5,0.5)
- many recent FPs or stuck events       → bias w_ig up by ~0.1

target_node_id: -1 unless you have a specific Voronoi node id (you don't, yet).
expected_semantics: 2–4 visual cues the robot should expect to see for target_object.
confidence: be honest — 0.4–0.7 when context is thin, higher when signals align.

Reply JSON only, no markdown:
{"phase":int,"w_sr":float,"w_ig":float,"target_node_id":int,"expected_semantics":[str],"confidence":float,"reasoning":string}
"""


class StrategicAgentBridge:
    def __init__(self) -> None:
        rospy.init_node('strategic_agent_node', anonymous=False)

        api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
        self.client = None
        if not api_key:
            rospy.logwarn(
                "[StrategicAgentLLM] DEEPSEEK_API_KEY not set — running in "
                "rule-fallback mode (no real LLM calls).")
        else:
            try:
                from openai import OpenAI
            except ImportError as e:
                rospy.logerr(
                    "[StrategicAgentLLM] `openai` package not installed: %s "
                    "— rule fallback only.", e)
            else:
                base_url = rospy.get_param('~base_url',
                                           'https://api.deepseek.com')
                self.client = OpenAI(api_key=api_key, base_url=base_url)

        self.model = rospy.get_param('~model', 'deepseek-chat')
        self.temperature = float(rospy.get_param('~temperature', 0.3))
        self.timeout_sec = float(rospy.get_param('~timeout_sec', 8.0))

        # I/O
        self.pub = rospy.Publisher('/strategic_agent/llm_response',
                                   StrategicLLMResponse, queue_size=10)
        self.sub = rospy.Subscriber('/strategic_agent/llm_request',
                                    StrategicLLMRequest, self._on_request,
                                    queue_size=10)

        rospy.loginfo(
            "[StrategicAgentLLM] Bridge online. model=%s temp=%.2f "
            "(client=%s)",
            self.model, self.temperature,
            "real" if self.client else "fallback-only")

    # ----- ROS callback ----------------------------------------------------

    def _on_request(self, req: StrategicLLMRequest) -> None:
        t0 = time.time()
        parsed: Dict[str, Any]
        prompt_tokens = 0
        completion_tokens = 0

        if self.client is None:
            parsed = self._rule_fallback(req)
        else:
            try:
                parsed, prompt_tokens, completion_tokens = self._call_llm(req)
            except Exception as e:
                rospy.logwarn(
                    "[StrategicAgentLLM] LLM call failed (%s) — falling back",
                    e)
                parsed = self._rule_fallback(req)

        latency = time.time() - t0

        resp = StrategicLLMResponse()
        resp.header.stamp = rospy.Time.now()
        resp.phase = int(parsed.get('phase', 1))
        resp.w_sr = float(parsed.get('w_sr', 0.5))
        resp.w_ig = float(parsed.get('w_ig', 0.5))
        # Normalise weights — guards against LLM drift.
        s = max(1e-6, resp.w_sr + resp.w_ig)
        resp.w_sr /= s
        resp.w_ig /= s
        resp.target_node_id = int(parsed.get('target_node_id', -1))
        resp.expected_semantics = [str(x) for x in
                                   parsed.get('expected_semantics', [])]
        resp.confidence = float(parsed.get('confidence', 0.5))
        resp.reasoning = str(parsed.get('reasoning', ''))[:200]
        resp.latency_seconds = latency
        resp.prompt_tokens = prompt_tokens
        resp.completion_tokens = completion_tokens
        self.pub.publish(resp)

        rospy.loginfo(
            "[StrategicAgentLLM] reply phase=%d w=(%.2f,%.2f) conf=%.2f "
            "latency=%.2fs tok=(%d,%d) reason=%r",
            resp.phase, resp.w_sr, resp.w_ig, resp.confidence,
            latency, prompt_tokens, completion_tokens, resp.reasoning)

    # ----- prompt + LLM ----------------------------------------------------

    def _build_user_prompt(self, req: StrategicLLMRequest) -> str:
        return (
            "Navigation context:\n"
            f"- target_object: {req.target_object or 'unknown'}\n"
            f"- coverage_estimate: {req.coverage_estimate:.2f}\n"
            f"- current_phase: {req.current_phase}\n"
            f"- candidate_count: {req.candidate_count}\n"
            f"- fp_count: {req.fp_count}\n"
            f"- stuck_count: {req.stuck_count}\n"
            "\nReturn JSON only.\n"
        )

    def _call_llm(self, req: StrategicLLMRequest):
        user_msg = self._build_user_prompt(req)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            timeout=self.timeout_sec,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        usage = getattr(resp, 'usage', None)
        ptok = getattr(usage, 'prompt_tokens', 0) if usage else 0
        ctok = getattr(usage, 'completion_tokens', 0) if usage else 0
        return parsed, ptok, ctok

    # ----- fallback --------------------------------------------------------

    def _rule_fallback(self, req: StrategicLLMRequest) -> Dict[str, Any]:
        """Mirror the C++ rule controller exactly so toggling
        enable_weight_override=true with the bridge offline keeps the same
        behaviour as the legacy path."""
        if req.coverage_estimate < 0.30:
            return {"phase": 0, "w_sr": 0.3, "w_ig": 0.7, "target_node_id": -1,
                    "expected_semantics": [], "confidence": 1.0,
                    "reasoning": "rule-fallback: broad"}
        if req.coverage_estimate > 0.70:
            return {"phase": 2, "w_sr": 0.8, "w_ig": 0.2, "target_node_id": -1,
                    "expected_semantics": [], "confidence": 1.0,
                    "reasoning": "rule-fallback: approach"}
        return {"phase": 1, "w_sr": 0.5, "w_ig": 0.5, "target_node_id": -1,
                "expected_semantics": [], "confidence": 1.0,
                "reasoning": "rule-fallback: directed"}


def main() -> None:
    StrategicAgentBridge()
    rospy.spin()


if __name__ == '__main__':
    main()
