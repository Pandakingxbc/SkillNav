#!/usr/bin/env python3
"""
Test DeepSeek LLM latency for decision-making tasks.

Tests various prompt complexities to measure inference latency.
"""

import os
import sys
import time
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_deepseek_latency(api_key: str, prompt: str, system_prompt: str = "", max_tokens: int = 200) -> tuple:
    """
    Test DeepSeek LLM latency.

    Returns: (latency_ms, response_text, token_count)
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        latency = (time.time() - start) * 1000
        result = response.choices[0].message.content if response.choices else ""
        tokens = response.usage.total_tokens if response.usage else 0
        return latency, result, tokens
    except Exception as e:
        return -1, str(e), 0


def main():
    print("=" * 70)
    print("    DEEPSEEK LLM LATENCY TEST")
    print("=" * 70)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    # Test cases with different complexities
    test_cases = [
        {
            "name": "Simple Decision",
            "system": "You are a navigation decision assistant.",
            "prompt": """Obstacle detected ahead. Type: wall. Confidence: 0.9.
Should the robot abandon this area?
Reply JSON: {"abandon": true/false, "reason": "brief"}""",
            "max_tokens": 100,
        },
        {
            "name": "Medium Decision (with history)",
            "system": "You are a navigation decision assistant for a mobile robot.",
            "prompt": """## Current Perception
- Position: (2.5, 3.1)
- Obstacle type: permanent (wall)
- VLM confidence: 0.85

## History
- This area: 3 failed attempts
- Last attempt: 30s ago
- Adjacent areas: Node5 clear, Node7 blocked

## Decision needed
Choose action: ABANDON_REGION, RETRY_LATER, TRY_ALTERNATIVE, FORCE_THROUGH

Reply JSON:
{"decision": "...", "weight": 0.0-1.0, "reason": "brief"}""",
            "max_tokens": 150,
        },
        {
            "name": "Complex Decision (full context)",
            "system": """You are a navigation decision assistant. Analyze the situation and make optimal decisions for exploration efficiency.""",
            "prompt": """## Current VLM Perception
- Position: (2.5, 3.1), Heading: 45°
- Obstacle type: temporary (closed door)
- VLM confidence: 0.75
- Description: "A closed wooden door blocking the hallway"

## Historical Records
- This region attempts: 3 (all failed)
- Last attempt: 30 seconds ago
- Historical VLM results: ["temporary", "temporary"]
- Adjacent nodes: [Node5: clear, Node7: blocked, Node9: unexplored]

## Exploration Status
- Total progress: 65%
- Remaining frontiers: 8
- Current frontier rank: 3/8 (medium priority)
- Time budget remaining: 120 seconds

## Decision Options
1. ABANDON_REGION: Significantly reduce weight, focus elsewhere
2. RETRY_LATER: Temporarily reduce weight, recover after 60s
3. TRY_ALTERNATIVE: Approach from different direction via Node5
4. FORCE_THROUGH: Continue trying (door might open)
5. MARK_FOR_REVIEW: Flag for human intervention

Provide decision with reasoning:
{
    "decision": "...",
    "confidence": 0.0-1.0,
    "weight_adjustment": 0.0-1.0,
    "recovery_time_seconds": null or number,
    "alternative_path": null or "via NodeX",
    "reasoning": "..."
}""",
            "max_tokens": 300,
        },
        {
            "name": "Batch Analysis (3 frontiers)",
            "system": "You are a navigation planner. Analyze multiple frontiers and rank them.",
            "prompt": """Analyze these 3 frontiers and rank by priority:

Frontier 1: distance=2.5m, info_gain=high, past_fails=0
Frontier 2: distance=1.2m, info_gain=medium, past_fails=2, VLM="temporary obstacle"
Frontier 3: distance=4.0m, info_gain=high, past_fails=1

Reply JSON:
{"ranking": [1,3,2], "weights": [1.0, 0.8, 0.4], "reasoning": "brief"}""",
            "max_tokens": 200,
        },
    ]

    runs_per_test = 3
    results = {}

    for test in test_cases:
        name = test["name"]
        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print(f"{'='*70}")
        print(f"Max tokens: {test['max_tokens']}")

        latencies = []
        tokens_list = []

        for run in range(runs_per_test):
            latency, result, tokens = test_deepseek_latency(
                api_key,
                test["prompt"],
                test["system"],
                test["max_tokens"]
            )

            if latency > 0:
                latencies.append(latency)
                tokens_list.append(tokens)
                preview = result[:60].replace('\n', ' ') + "..." if len(result) > 60 else result.replace('\n', ' ')
                print(f"  Run {run+1}: {latency:.0f}ms, {tokens} tokens")
                print(f"          {preview}")
            else:
                print(f"  Run {run+1}: ERROR - {result[:50]}")

        if latencies:
            results[name] = {
                "mean": statistics.mean(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies),
                "avg_tokens": statistics.mean(tokens_list),
            }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Test Case':<30} {'Mean (ms)':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'Tokens':<10}")
    print("-" * 82)

    for name, r in results.items():
        print(f"{name:<30} {r['mean']:<12.0f} {r['std']:<10.0f} {r['min']:<10.0f} {r['max']:<10.0f} {r['avg_tokens']:<10.0f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS FOR NAVIGATION DECISION")
    print("=" * 70)

    if "Simple Decision" in results:
        simple = results["Simple Decision"]["mean"]
        print(f"\nSimple decision latency: {simple:.0f}ms")
        if simple < 1000:
            print("  ✅ Suitable for frequent decisions")
        else:
            print("  ⚠️ May need optimization for frequent use")

    if "Complex Decision (full context)" in results:
        complex_lat = results["Complex Decision (full context)"]["mean"]
        print(f"\nComplex decision latency: {complex_lat:.0f}ms")
        if complex_lat < 3000:
            print("  ✅ Acceptable for async decision pipeline")
        else:
            print("  ⚠️ Consider simplifying prompts")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
