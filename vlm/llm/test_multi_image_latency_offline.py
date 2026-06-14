#!/usr/bin/env python3
"""
Test VLM latency with multiple images (1, 2, 3, 4 images).

Uses pre-saved ROS images from docs/vlm_test_results/ros_images/
"""

import os
import sys
import time
import base64
import statistics
from pathlib import Path
from io import BytesIO
from typing import List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_image(path: str) -> np.ndarray:
    """Load image as numpy array."""
    return np.array(Image.open(path).convert('RGB'))


def encode_image(image: np.ndarray, quality: int = 90) -> str:
    """Encode image to base64 JPEG."""
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_image_size_kb(image: np.ndarray, quality: int = 90) -> float:
    """Get JPEG size in KB."""
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    return len(buffer.getvalue()) / 1024


def test_vlm_multi_image(images: List[np.ndarray], api_key: str) -> tuple:
    """
    Test VLM with multiple images.

    Returns: (latency_ms, result_text)
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=120.0,
    )

    # Build content with multiple images
    content = []
    for i, img in enumerate(images):
        b64_image = encode_image(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
        })

    # Prompt adapted for multiple images
    if len(images) == 1:
        prompt = """Analyze this navigation view. Is the path ahead blocked?

Reply in JSON:
{
    "blocked": true/false,
    "obstacle_type": "wall" | "furniture" | "door" | "clear" | "unclear",
    "confidence": 0.0-1.0,
    "description": "brief description"
}"""
    else:
        prompt = f"""Analyze these {len(images)} navigation views from different positions/angles.

For each image, determine if the path is blocked.

Reply in JSON:
{{
    "overall_assessment": "passable" | "blocked" | "partially_blocked",
    "images": [
        {{"image_id": 1, "blocked": true/false, "obstacle_type": "...", "confidence": 0.0-1.0}},
        ...
    ],
    "recommendation": "proceed" | "reroute" | "abandon_area",
    "reasoning": "brief explanation"
}}"""

    content.append({"type": "text", "text": prompt})

    start = time.time()
    try:
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {"role": "system", "content": "You are a robot navigation assistant analyzing multiple viewpoints."},
                {"role": "user", "content": content}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        latency = (time.time() - start) * 1000
        result = response.choices[0].message.content if response.choices else ""
        return latency, result
    except Exception as e:
        return -1, str(e)


def main():
    print("=" * 70)
    print("    MULTI-IMAGE VLM LATENCY TEST (Offline)")
    print("=" * 70)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set")
        return

    # Load pre-saved images (resize 480 versions)
    image_dir = "/home/yangz/Nav/SkillNav/docs/vlm_test_results/ros_images"
    image_paths = [
        f"{image_dir}/ros_rgb_resize480_1.jpg",
        f"{image_dir}/ros_rgb_resize480_2.jpg",
        f"{image_dir}/ros_rgb_resize480_3.jpg",
    ]

    print("\n1. Loading pre-saved ROS images...")
    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = load_image(path)
            images.append(img)
            h, w = img.shape[:2]
            kb = get_image_size_kb(img)
            print(f"   Loaded: {path}")
            print(f"           {w}x{h}, {kb:.1f}KB")

    if len(images) < 3:
        print("ERROR: Need at least 3 images for testing")
        return

    # Duplicate last image for 4-image test
    images.append(images[-1].copy())
    print(f"\n   Total images available: {len(images)}")

    # Test with different number of images
    print("\n" + "=" * 70)
    print("2. TESTING VLM LATENCY WITH 1, 2, 3, 4 IMAGES")
    print("=" * 70)

    runs_per_config = 2  # 2 runs each to get average
    results = {}

    for num_images in [1, 2, 3, 4]:
        test_images = images[:num_images]
        total_kb = sum(get_image_size_kb(img) for img in test_images)

        print(f"\n   {num_images} image(s) (total {total_kb:.1f}KB):")

        latencies = []
        for run in range(runs_per_config):
            latency, result = test_vlm_multi_image(test_images, api_key)
            if latency > 0:
                latencies.append(latency)
                print(f"      Run {run+1}: {latency:.0f}ms")
            else:
                print(f"      Run {run+1}: ERROR - {result[:80]}")

        if latencies:
            results[num_images] = {
                'total_kb': total_kb,
                'mean': statistics.mean(latencies),
                'min': min(latencies),
                'max': max(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            }

    # Summary
    print("\n" + "=" * 70)
    print("3. SUMMARY: LATENCY vs NUMBER OF IMAGES")
    print("=" * 70)

    print(f"\n{'Images':<10} {'Total KB':<12} {'Mean (ms)':<12} {'Min':<10} {'Max':<10} {'Per Image':<12}")
    print("-" * 70)

    baseline = results.get(1, {}).get('mean', 1)

    for num, r in sorted(results.items()):
        per_image = r['mean'] / num
        diff = ((r['mean'] - baseline) / baseline * 100)
        print(f"{num:<10} {r['total_kb']:<12.1f} {r['mean']:<12.0f} {r['min']:<10.0f} {r['max']:<10.0f} {per_image:<12.0f} ({diff:+.0f}% vs 1img)")

    # Analysis
    print("\n" + "=" * 70)
    print("4. ANALYSIS")
    print("=" * 70)

    if 1 in results and 2 in results:
        scale_2 = results[2]['mean'] / results[1]['mean']
        print(f"\n   1→2 images: {scale_2:.2f}x latency increase")

    if 2 in results and 3 in results:
        scale_3 = results[3]['mean'] / results[2]['mean']
        print(f"   2→3 images: {scale_3:.2f}x latency increase")

    if 3 in results and 4 in results:
        scale_4 = results[4]['mean'] / results[3]['mean']
        print(f"   3→4 images: {scale_4:.2f}x latency increase")

    if 1 in results and 4 in results:
        total_scale = results[4]['mean'] / results[1]['mean']
        print(f"\n   1→4 images total: {total_scale:.2f}x latency increase")
        print(f"   Single image baseline: {results[1]['mean']:.0f}ms")
        print(f"   Four images: {results[4]['mean']:.0f}ms")

    # Recommendation
    print("\n" + "=" * 70)
    print("5. RECOMMENDATION")
    print("=" * 70)

    if results:
        single_lat = results.get(1, {}).get('mean', 0)
        if single_lat > 0:
            if single_lat < 2000:
                print(f"\n   ✅ Single image latency ({single_lat:.0f}ms) is acceptable for synchronous calls")
            else:
                print(f"\n   ⚠️ Single image latency ({single_lat:.0f}ms) may require async architecture")

        four_lat = results.get(4, {}).get('mean', 0)
        if four_lat > 0:
            if four_lat < 5000:
                print(f"   ✅ Four images latency ({four_lat:.0f}ms) is feasible for batch analysis")
            else:
                print(f"   ⚠️ Four images latency ({four_lat:.0f}ms) - consider limiting to 1-2 images")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
