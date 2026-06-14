#!/usr/bin/env python3
"""
VLM Latency Test for Safe Agent Decision Making.

This script tests the inference latency of Qwen VLM and DeepSeek LLM
to help decide whether Safe Agent should use sync or async VLM calls.

Usage:
    python test_vlm_latency.py [--image PATH] [--runs N]

Environment:
    DASHSCOPE_API_KEY: Required for Qwen VLM
    DEEPSEEK_API_KEY: Required for DeepSeek LLM
"""

import os
import sys
import time
import argparse
import statistics
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlm.llm import QwenVLMClient, DeepSeekLLMClient, get_agent_interface


def load_test_image(image_path: str) -> np.ndarray:
    """Load test image as numpy array."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert('RGB')
        return np.array(img)
    except ImportError:
        print("PIL not available, using OpenCV")
        import cv2
        img = cv2.imread(image_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_synthetic_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic test image if no real image available."""
    # Create a simple gradient image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(255 * x / width),      # R: gradient left-right
                int(255 * y / height),     # G: gradient top-bottom
                128                         # B: constant
            ]
    return img


def test_vlm_dead_zone_analysis(
    client: QwenVLMClient,
    image: np.ndarray,
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Test VLM latency for dead zone analysis (Safe Agent use case).

    This is the primary VLM call that Safe Agent would make.
    """
    latencies = []
    results = []

    print(f"\n{'='*60}")
    print("Testing: VLM Dead Zone Analysis (Safe Agent)")
    print(f"{'='*60}")

    for i in range(num_runs):
        position = (1.5 + i * 0.1, 2.3)  # Slightly vary position
        attempt_count = 3
        heading = 45.0 + i * 10

        start = time.time()
        try:
            result = client.analyze_dead_zone(
                image=image,
                position=position,
                attempt_count=attempt_count,
                heading=heading
            )
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            results.append(result)
            print(f"  Run {i+1}/{num_runs}: {latency:.0f}ms - {result.get('obstacle_type', 'N/A')}")
        except Exception as e:
            print(f"  Run {i+1}/{num_runs}: ERROR - {e}")
            latencies.append(float('inf'))

    valid_latencies = [l for l in latencies if l != float('inf')]

    return {
        "task": "dead_zone_analysis",
        "num_runs": num_runs,
        "latencies_ms": latencies,
        "mean_ms": statistics.mean(valid_latencies) if valid_latencies else None,
        "std_ms": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else None,
        "min_ms": min(valid_latencies) if valid_latencies else None,
        "max_ms": max(valid_latencies) if valid_latencies else None,
        "success_rate": len(valid_latencies) / num_runs,
        "sample_result": results[0] if results else None
    }


def test_vlm_object_verification(
    client: QwenVLMClient,
    image: np.ndarray,
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Test VLM latency for object verification (Memory Agent use case).
    """
    latencies = []
    results = []

    print(f"\n{'='*60}")
    print("Testing: VLM Object Verification (Memory Agent)")
    print(f"{'='*60}")

    target_objects = ["chair", "toilet", "bed", "table", "sofa"]

    for i in range(num_runs):
        target = target_objects[i % len(target_objects)]
        position = (2.0, 3.0)

        start = time.time()
        try:
            result = client.verify_object(
                image=image,
                target_object=target,
                expected_position=position
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            results.append(result)
            print(f"  Run {i+1}/{num_runs}: {latency:.0f}ms - '{target}' present={result.get('object_present', 'N/A')}")
        except Exception as e:
            print(f"  Run {i+1}/{num_runs}: ERROR - {e}")
            latencies.append(float('inf'))

    valid_latencies = [l for l in latencies if l != float('inf')]

    return {
        "task": "object_verification",
        "num_runs": num_runs,
        "latencies_ms": latencies,
        "mean_ms": statistics.mean(valid_latencies) if valid_latencies else None,
        "std_ms": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else None,
        "min_ms": min(valid_latencies) if valid_latencies else None,
        "max_ms": max(valid_latencies) if valid_latencies else None,
        "success_rate": len(valid_latencies) / num_runs,
        "sample_result": results[0] if results else None
    }


def test_vlm_scene_description(
    client: QwenVLMClient,
    image: np.ndarray,
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Test VLM latency for scene description (Exploration Agent use case).
    """
    latencies = []
    results = []

    print(f"\n{'='*60}")
    print("Testing: VLM Scene Description (Exploration Agent)")
    print(f"{'='*60}")

    for i in range(num_runs):
        context = f"looking for bathroom" if i % 2 == 0 else None

        start = time.time()
        try:
            result = client.describe_scene(image=image, context=context)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            results.append(result)
            desc_preview = result[:50] + "..." if len(result) > 50 else result
            print(f"  Run {i+1}/{num_runs}: {latency:.0f}ms - '{desc_preview}'")
        except Exception as e:
            print(f"  Run {i+1}/{num_runs}: ERROR - {e}")
            latencies.append(float('inf'))

    valid_latencies = [l for l in latencies if l != float('inf')]

    return {
        "task": "scene_description",
        "num_runs": num_runs,
        "latencies_ms": latencies,
        "mean_ms": statistics.mean(valid_latencies) if valid_latencies else None,
        "std_ms": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else None,
        "min_ms": min(valid_latencies) if valid_latencies else None,
        "max_ms": max(valid_latencies) if valid_latencies else None,
        "success_rate": len(valid_latencies) / num_runs,
        "sample_result": results[0] if results else None
    }


def test_llm_exploration_planning(
    client: DeepSeekLLMClient,
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Test LLM latency for exploration planning (Exploration Agent use case).
    """
    latencies = []
    results = []

    print(f"\n{'='*60}")
    print("Testing: LLM Exploration Planning (Exploration Agent)")
    print(f"{'='*60}")

    for i in range(num_runs):
        voronoi_summary = f"15 nodes, {20 + i*5}% explored, 3 frontier-adjacent nodes"
        exploration_pct = 20.0 + i * 10
        hotspots = [{"node_id": 5, "confidence": 0.6 + i*0.05, "x": 2.0, "y": 3.0}]

        start = time.time()
        try:
            result = client.plan_exploration_strategy(
                voronoi_summary=voronoi_summary,
                exploration_percentage=exploration_pct,
                semantic_hotspots=hotspots,
                current_phase="BROAD_EXPLORATION",
                target_object="chair"
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            results.append(result)
            phase = result.get('recommended_phase', 'N/A')
            print(f"  Run {i+1}/{num_runs}: {latency:.0f}ms - phase={phase}")
        except Exception as e:
            print(f"  Run {i+1}/{num_runs}: ERROR - {e}")
            latencies.append(float('inf'))

    valid_latencies = [l for l in latencies if l != float('inf')]

    return {
        "task": "exploration_planning",
        "num_runs": num_runs,
        "latencies_ms": latencies,
        "mean_ms": statistics.mean(valid_latencies) if valid_latencies else None,
        "std_ms": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else None,
        "min_ms": min(valid_latencies) if valid_latencies else None,
        "max_ms": max(valid_latencies) if valid_latencies else None,
        "success_rate": len(valid_latencies) / num_runs,
        "sample_result": results[0] if results else None
    }


def test_llm_phase_transition(
    client: DeepSeekLLMClient,
    num_runs: int = 5
) -> Dict[str, Any]:
    """
    Test LLM latency for phase transition decision.
    """
    latencies = []
    results = []

    print(f"\n{'='*60}")
    print("Testing: LLM Phase Transition Decision")
    print(f"{'='*60}")

    for i in range(num_runs):
        current_phase = ["BROAD_EXPLORATION", "DIRECTED_SEARCH", "TARGET_APPROACH"][i % 3]

        start = time.time()
        try:
            result = client.decide_phase_transition(
                current_phase=current_phase,
                phase_history=["BROAD_EXPLORATION", current_phase],
                trigger_events=["semantic_hotspot_detected", f"coverage_{30+i*10}%"],
                exploration_state={
                    "coverage": 30.0 + i * 10,
                    "hotspot_count": 2,
                    "target_confidence": 0.5 + i * 0.1,
                    "steps_since_transition": 50 + i * 20
                }
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            results.append(result)
            should_transition = result.get('should_transition', 'N/A')
            print(f"  Run {i+1}/{num_runs}: {latency:.0f}ms - transition={should_transition}")
        except Exception as e:
            print(f"  Run {i+1}/{num_runs}: ERROR - {e}")
            latencies.append(float('inf'))

    valid_latencies = [l for l in latencies if l != float('inf')]

    return {
        "task": "phase_transition",
        "num_runs": num_runs,
        "latencies_ms": latencies,
        "mean_ms": statistics.mean(valid_latencies) if valid_latencies else None,
        "std_ms": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else None,
        "min_ms": min(valid_latencies) if valid_latencies else None,
        "max_ms": max(valid_latencies) if valid_latencies else None,
        "success_rate": len(valid_latencies) / num_runs,
        "sample_result": results[0] if results else None
    }


def print_summary(all_results: List[Dict[str, Any]]):
    """Print summary of all latency tests."""
    print("\n")
    print("=" * 70)
    print("                        LATENCY TEST SUMMARY")
    print("=" * 70)
    print(f"{'Task':<25} {'Mean (ms)':<12} {'Std':<10} {'Min':<10} {'Max':<10} {'Success'}")
    print("-" * 70)

    for r in all_results:
        task = r['task']
        mean = f"{r['mean_ms']:.0f}" if r['mean_ms'] else "N/A"
        std = f"{r['std_ms']:.0f}" if r['std_ms'] else "N/A"
        min_v = f"{r['min_ms']:.0f}" if r['min_ms'] else "N/A"
        max_v = f"{r['max_ms']:.0f}" if r['max_ms'] else "N/A"
        success = f"{r['success_rate']*100:.0f}%"
        print(f"{task:<25} {mean:<12} {std:<10} {min_v:<10} {max_v:<10} {success}")

    print("=" * 70)

    # Analysis for Safe Agent decision
    print("\n" + "=" * 70)
    print("                    ANALYSIS FOR SAFE AGENT")
    print("=" * 70)

    dead_zone_result = next((r for r in all_results if r['task'] == 'dead_zone_analysis'), None)

    if dead_zone_result and dead_zone_result['mean_ms']:
        mean_latency = dead_zone_result['mean_ms']

        print(f"\nDead Zone Analysis Mean Latency: {mean_latency:.0f}ms")
        print()

        if mean_latency < 100:
            print("✅ RECOMMENDATION: SYNCHRONOUS call is feasible")
            print("   - Latency < 100ms allows blocking call without major impact")
            print("   - Safe Agent can wait for VLM result before continuing")
        elif mean_latency < 500:
            print("⚠️  RECOMMENDATION: HYBRID approach suggested")
            print("   - Latency 100-500ms is borderline")
            print("   - Consider: trigger async, use cached result on next stuck detection")
            print("   - Safe Agent continues escape attempts while waiting")
        else:
            print("❌ RECOMMENDATION: ASYNCHRONOUS call required")
            print(f"   - Latency {mean_latency:.0f}ms too high for blocking call")
            print("   - Safe Agent MUST NOT wait for VLM result")
            print("   - Pattern: detect → trigger async → continue escape → check result later")

        print()
        print("Implementation implications:")
        if mean_latency >= 500:
            print("  1. Safe Agent publishes VLM request to ROS topic")
            print("  2. Python VLM node processes asynchronously")
            print("  3. Memory Agent receives result via callback")
            print("  4. Safe Agent checks Voronoi node values (already updated)")
        else:
            print("  1. Safe Agent can call VLM service directly")
            print("  2. Block for result (acceptable latency)")
            print("  3. Update Voronoi immediately")
    else:
        print("❓ Could not analyze - Dead Zone test failed")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test VLM/LLM latency for Safe Agent")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--runs", type=int, default=5, help="Number of test runs per task")
    parser.add_argument("--vlm-only", action="store_true", help="Only test VLM (skip LLM)")
    parser.add_argument("--llm-only", action="store_true", help="Only test LLM (skip VLM)")
    args = parser.parse_args()

    print("=" * 70)
    print("        VLM/LLM LATENCY TEST FOR SKILLNAV SAFE AGENT")
    print("=" * 70)

    # Check API keys
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")

    print(f"\nAPI Key Status:")
    print(f"  DASHSCOPE_API_KEY: {'✓ Found' if dashscope_key else '✗ Missing'}")
    print(f"  DEEPSEEK_API_KEY:  {'✓ Found' if deepseek_key else '✗ Missing'}")

    # Load or create test image
    if args.image and os.path.exists(args.image):
        print(f"\nLoading test image: {args.image}")
        image = load_test_image(args.image)
    else:
        # Try to find a good test image
        default_images = [
            "/home/yangz/Nav/SkillNav/habitat-lab/docs/images/habitat-lab-demo-images/habitat-lab-demo.png",
            "/home/yangz/Nav/SkillNav/habitat-lab/docs/images/quickstart-images/quickstart.png",
        ]
        image = None
        for img_path in default_images:
            if os.path.exists(img_path):
                print(f"\nLoading default test image: {img_path}")
                image = load_test_image(img_path)
                break

        if image is None:
            print("\nNo test image found, creating synthetic image")
            image = create_synthetic_image()

    print(f"Image shape: {image.shape}")

    all_results = []

    # Test VLM
    if not args.llm_only and dashscope_key:
        print("\n" + "=" * 70)
        print("                    QWEN VLM TESTS")
        print("=" * 70)

        try:
            vlm_client = QwenVLMClient()
            print("VLM client initialized successfully")

            # Test 1: Dead zone analysis (Safe Agent primary use case)
            result = test_vlm_dead_zone_analysis(vlm_client, image, args.runs)
            all_results.append(result)

            # Test 2: Object verification (Memory Agent)
            result = test_vlm_object_verification(vlm_client, image, args.runs)
            all_results.append(result)

            # Test 3: Scene description (Exploration Agent)
            result = test_vlm_scene_description(vlm_client, image, args.runs)
            all_results.append(result)

        except Exception as e:
            print(f"VLM initialization failed: {e}")
    elif not dashscope_key:
        print("\n⚠️  Skipping VLM tests - DASHSCOPE_API_KEY not set")

    # Test LLM
    if not args.vlm_only and deepseek_key:
        print("\n" + "=" * 70)
        print("                    DEEPSEEK LLM TESTS")
        print("=" * 70)

        try:
            llm_client = DeepSeekLLMClient()
            print("LLM client initialized successfully")

            # Test 1: Exploration planning
            result = test_llm_exploration_planning(llm_client, args.runs)
            all_results.append(result)

            # Test 2: Phase transition
            result = test_llm_phase_transition(llm_client, args.runs)
            all_results.append(result)

        except Exception as e:
            print(f"LLM initialization failed: {e}")
    elif not deepseek_key:
        print("\n⚠️  Skipping LLM tests - DEEPSEEK_API_KEY not set")

    # Print summary
    if all_results:
        print_summary(all_results)
    else:
        print("\n❌ No tests completed. Check API keys and try again.")

    return all_results


if __name__ == "__main__":
    main()
