#!/usr/bin/env python3
"""
Test DeepSeek LLM latency with structured JSON output.

Compares normal output vs JSON mode.
"""

import os
import sys
import time
import json
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_deepseek(api_key: str, prompt: str, system_prompt: str,
                  max_tokens: int, use_json_mode: bool = False) -> tuple:
    """
    Test DeepSeek LLM latency.

    Returns: (latency_ms, response_text, token_count, is_valid_json)
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

    kwargs = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    start = time.time()
    try:
        response = client.chat.completions.create(**kwargs)
        latency = (time.time() - start) * 1000
        result = response.choices[0].message.content if response.choices else ""
        tokens = response.usage.total_tokens if response.usage else 0

        # Check if valid JSON
        is_valid = False
        try:
            json.loads(result)
            is_valid = True
        except:
            pass

        return latency, result, tokens, is_valid
    except Exception as e:
        return -1, str(e), 0, False


def main():
    print("=" * 70)
    print("    DEEPSEEK STRUCTURED OUTPUT LATENCY TEST")
    print("=" * 70)

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        return

    # Test prompts - same content, different output modes
    test_configs = [
        {
            "name": "Simple - Normal",
            "json_mode": False,
            "system": "You are a navigation assistant. Always respond in JSON format.",
            "prompt": """Obstacle: wall, confidence: 0.9
Decide: abandon this area?
Reply: {"abandon": true/false, "reason": "brief"}""",
            "max_tokens": 80,
        },
        {
            "name": "Simple - JSON Mode",
            "json_mode": True,
            "system": "You are a navigation assistant. Respond in JSON.",
            "prompt": """Obstacle: wall, confidence: 0.9
Decide: abandon this area?
Return JSON with keys: abandon (bool), reason (string)""",
            "max_tokens": 80,
        },
        {
            "name": "Medium - Normal",
            "json_mode": False,
            "system": "You are a navigation decision assistant.",
            "prompt": """Context:
- Position: (2.5, 3.1)
- Obstacle: permanent wall
- Confidence: 0.85
- Past failures: 3
- Adjacent: Node5=clear, Node7=blocked

Decide action: ABANDON, RETRY, ALTERNATIVE, FORCE
Reply JSON: {"decision": "...", "weight": 0-1, "reason": "brief"}""",
            "max_tokens": 120,
        },
        {
            "name": "Medium - JSON Mode",
            "json_mode": True,
            "system": "You are a navigation decision assistant.",
            "prompt": """Context:
- Position: (2.5, 3.1)
- Obstacle: permanent wall
- Confidence: 0.85
- Past failures: 3
- Adjacent: Node5=clear, Node7=blocked

Decide action from: ABANDON, RETRY, ALTERNATIVE, FORCE
Return JSON with: decision (string), weight (float 0-1), reason (string)""",
            "max_tokens": 120,
        },
        {
            "name": "Complex - Normal",
            "json_mode": False,
            "system": "You are a navigation decision assistant.",
            "prompt": """## Perception
- Obstacle: temporary (closed door)
- Confidence: 0.75

## History
- Attempts: 3 failed
- VLM history: ["temporary", "temporary"]
- Adjacent: Node5=clear, Node7=blocked

## Status
- Progress: 65%
- Frontiers: 8 remaining
- Time: 120s left

Options: ABANDON, RETRY_60s, ALTERNATIVE_via_Node5, FORCE

Reply JSON:
{"decision": "...", "confidence": 0-1, "weight": 0-1, "recovery_sec": null/number, "path": null/"via NodeX", "reason": "brief"}""",
            "max_tokens": 200,
        },
        {
            "name": "Complex - JSON Mode",
            "json_mode": True,
            "system": "You are a navigation decision assistant.",
            "prompt": """## Perception
- Obstacle: temporary (closed door)
- Confidence: 0.75

## History
- Attempts: 3 failed
- VLM history: ["temporary", "temporary"]
- Adjacent: Node5=clear, Node7=blocked

## Status
- Progress: 65%
- Frontiers: 8 remaining
- Time: 120s left

Options: ABANDON, RETRY_60s, ALTERNATIVE_via_Node5, FORCE

Return JSON with keys:
- decision: string (one of options)
- confidence: float 0-1
- weight: float 0-1
- recovery_sec: int or null
- path: string or null
- reason: string (brief)""",
            "max_tokens": 200,
        },
    ]

    runs = 3
    results = {}

    for config in test_configs:
        name = config["name"]
        mode = "JSON Mode" if config["json_mode"] else "Normal"

        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print(f"{'='*70}")

        latencies = []
        tokens_list = []
        valid_count = 0

        for run in range(runs):
            latency, result, tokens, is_valid = test_deepseek(
                api_key,
                config["prompt"],
                config["system"],
                config["max_tokens"],
                config["json_mode"]
            )

            if latency > 0:
                latencies.append(latency)
                tokens_list.append(tokens)
                if is_valid:
                    valid_count += 1

                # Parse and show result
                preview = result[:70].replace('\n', ' ')
                valid_str = "✅" if is_valid else "❌"
                print(f"  Run {run+1}: {latency:.0f}ms, {tokens} tokens {valid_str}")
                print(f"          {preview}...")
            else:
                print(f"  Run {run+1}: ERROR - {result[:50]}")

        if latencies:
            results[name] = {
                "mean": statistics.mean(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies),
                "tokens": statistics.mean(tokens_list),
                "valid_rate": valid_count / len(latencies) * 100,
                "json_mode": config["json_mode"],
            }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Normal vs JSON Mode")
    print("=" * 70)

    print(f"\n{'Test':<25} {'Mode':<12} {'Mean(ms)':<10} {'Std':<8} {'Tokens':<8} {'Valid%':<8}")
    print("-" * 75)

    for name, r in results.items():
        mode = "JSON" if r["json_mode"] else "Normal"
        print(f"{name:<25} {mode:<12} {r['mean']:<10.0f} {r['std']:<8.0f} {r['tokens']:<8.0f} {r['valid_rate']:<8.0f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Normal vs JSON Mode")
    print("=" * 70)

    comparisons = [
        ("Simple - Normal", "Simple - JSON Mode"),
        ("Medium - Normal", "Medium - JSON Mode"),
        ("Complex - Normal", "Complex - JSON Mode"),
    ]

    for normal_name, json_name in comparisons:
        if normal_name in results and json_name in results:
            normal = results[normal_name]
            json_r = results[json_name]
            diff = ((json_r["mean"] - normal["mean"]) / normal["mean"]) * 100

            print(f"\n{normal_name.split(' -')[0]}:")
            print(f"  Normal:    {normal['mean']:.0f}ms, {normal['tokens']:.0f} tokens, {normal['valid_rate']:.0f}% valid")
            print(f"  JSON Mode: {json_r['mean']:.0f}ms, {json_r['tokens']:.0f} tokens, {json_r['valid_rate']:.0f}% valid")
            print(f"  Difference: {diff:+.0f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
