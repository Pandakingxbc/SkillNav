#!/usr/bin/env python3
"""
Multi-run latency test to get reliable statistics.
Tests each image size 3 times to account for API variance.
"""

import os
import sys
import time
import base64
import statistics
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))


def resize_image(image: np.ndarray, max_dim: int) -> np.ndarray:
    pil_img = Image.fromarray(image)
    w, h = pil_img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img)


def test_vlm(image: np.ndarray, api_key: str) -> float:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60.0,
    )

    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=90)
    b64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    prompt = """Analyze: is this area passable or blocked?
Reply JSON: {"passable": true/false, "reason": "brief"}"""

    start = time.time()
    response = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            {"type": "text", "text": prompt}
        ]}],
        temperature=0.3,
        max_tokens=100,
    )
    return (time.time() - start) * 1000


def main():
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set")
        return

    image_path = "/home/yangz/Nav/SkillNav/habitat-lab/docs/images/habitat-lab-demo-images/habitat-lab-demo.png"
    original = load_image(image_path)

    configs = [
        ("Original", original),
        ("Resize 640", resize_image(original, 640)),
        ("Resize 480", resize_image(original, 480)),
        ("Resize 320", resize_image(original, 320)),
        ("Resize 240", resize_image(original, 240)),
    ]

    runs = 3
    print(f"\nTesting each config {runs} times...\n")

    results = {}
    for name, img in configs:
        h, w = img.shape[:2]
        pil_img = Image.fromarray(img)
        buffer = BytesIO()
        pil_img.save(buffer, format='JPEG', quality=90)
        kb = len(buffer.getvalue()) / 1024

        latencies = []
        print(f"{name} ({w}x{h}, {kb:.1f}KB):")
        for i in range(runs):
            try:
                lat = test_vlm(img, api_key)
                latencies.append(lat)
                print(f"  Run {i+1}: {lat:.0f}ms")
            except Exception as e:
                print(f"  Run {i+1}: ERROR - {e}")

        if latencies:
            results[name] = {
                'size': f'{w}x{h}',
                'kb': kb,
                'mean': statistics.mean(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'min': min(latencies),
                'max': max(latencies),
            }
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY (sorted by mean latency)")
    print("=" * 70)
    print(f"{'Config':<15} {'Size':<12} {'KB':<8} {'Mean':<10} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'])
    baseline = sorted_results[-1][1]['mean']  # Original is usually largest

    for name, r in sorted_results:
        diff = (r['mean'] - baseline) / baseline * 100
        print(f"{name:<15} {r['size']:<12} {r['kb']:<8.1f} {r['mean']:<10.0f} {r['std']:<8.0f} {r['min']:<8.0f} {r['max']:<8.0f} ({diff:+.0f}%)")


if __name__ == "__main__":
    main()
