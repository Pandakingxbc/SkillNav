#!/usr/bin/env python3
"""
Test VLM latency with multiple images (1, 2, 3, 4 images).

This tests how latency scales when sending multiple viewpoints to the VLM,
which is useful for navigation decisions that might need multiple perspectives.
"""

import os
import sys
import time
import base64
import statistics
from pathlib import Path
from io import BytesIO
from typing import List, Optional

import numpy as np
from PIL import Image

# ROS imports
import rospy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ROSImageCapture:
    """Capture images from ROS topic."""

    def __init__(self, topic: str = "/habitat/camera_rgb"):
        self.topic = topic
        self.bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.image_count = 0

        rospy.init_node('vlm_multi_image_test', anonymous=True)
        self.sub = rospy.Subscriber(topic, ROSImage, self.callback)
        print(f"Subscribed to {topic}")

    def callback(self, msg: ROSImage):
        """ROS image callback."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.latest_image = np.array(cv_image)
            self.image_count += 1
        except Exception as e:
            print(f"Error converting image: {e}")

    def wait_for_image(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Wait for an image to arrive."""
        start = time.time()
        while self.latest_image is None:
            if time.time() - start > timeout:
                print(f"Timeout waiting for image from {self.topic}")
                return None
            rospy.sleep(0.1)
        return self.latest_image.copy()

    def capture_multiple(self, count: int, interval: float = 0.5) -> List[np.ndarray]:
        """Capture multiple different images."""
        images = []
        for i in range(count):
            rospy.sleep(interval)
            if self.latest_image is not None:
                images.append(self.latest_image.copy())
                print(f"  Captured image {i+1}/{count}: {self.latest_image.shape}")
        return images


def resize_image(image: np.ndarray, max_dim: int = 480) -> np.ndarray:
    """Resize image keeping aspect ratio."""
    pil_img = Image.fromarray(image)
    w, h = pil_img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img)


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
        timeout=120.0,  # Longer timeout for multiple images
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
    print("    MULTI-IMAGE VLM LATENCY TEST")
    print("=" * 70)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set")
        return

    # Capture images from ROS
    print("\n1. Capturing images from ROS topic...")
    capture = ROSImageCapture("/habitat/camera_rgb")

    print("   Waiting for first image...")
    first_image = capture.wait_for_image(timeout=15.0)

    if first_image is None:
        print("ERROR: Could not capture image from ROS topic")
        print("Make sure Habitat simulation is running!")
        return

    print(f"   First image: {first_image.shape}")

    # Capture multiple images (we need 4 different images)
    print("\n   Capturing additional images (need 4 total)...")
    images = [first_image]
    more_images = capture.capture_multiple(3, interval=1.0)
    images.extend(more_images)

    print(f"\n   Total images captured: {len(images)}")

    if len(images) < 4:
        print("WARNING: Could not capture 4 images, will reuse some")
        while len(images) < 4:
            images.append(images[-1].copy())

    # Resize all images to 480 (recommended configuration)
    print("\n2. Preprocessing images (resize to max 480)...")
    resized_images = [resize_image(img, 480) for img in images]

    for i, img in enumerate(resized_images):
        h, w = img.shape[:2]
        kb = get_image_size_kb(img)
        print(f"   Image {i+1}: {w}x{h}, {kb:.1f}KB")

    # Test with different number of images
    print("\n" + "=" * 70)
    print("3. TESTING VLM LATENCY WITH 1, 2, 3, 4 IMAGES")
    print("=" * 70)

    runs_per_config = 2  # 2 runs each to get average
    results = {}

    for num_images in [1, 2, 3, 4]:
        test_images = resized_images[:num_images]
        total_kb = sum(get_image_size_kb(img) for img in test_images)

        print(f"\n   {num_images} image(s) (total {total_kb:.1f}KB):")

        latencies = []
        for run in range(runs_per_config):
            latency, result = test_vlm_multi_image(test_images, api_key)
            if latency > 0:
                latencies.append(latency)
                preview = result[:50] + "..." if len(result) > 50 else result
                print(f"      Run {run+1}: {latency:.0f}ms")
            else:
                print(f"      Run {run+1}: ERROR - {result[:50]}")

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
    print("4. SUMMARY: LATENCY vs NUMBER OF IMAGES")
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
    print("5. ANALYSIS")
    print("=" * 70)

    if 1 in results and 2 in results:
        scale_2 = results[2]['mean'] / results[1]['mean']
        print(f"\n   1→2 images: {scale_2:.2f}x latency increase")

    if 1 in results and 4 in results:
        scale_4 = results[4]['mean'] / results[1]['mean']
        print(f"   1→4 images: {scale_4:.2f}x latency increase")

    if 1 in results:
        print(f"\n   Single image baseline: {results[1]['mean']:.0f}ms")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
