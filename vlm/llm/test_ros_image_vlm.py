#!/usr/bin/env python3
"""
Test VLM latency with real RGB images from ROS topic.

Subscribes to /habitat/camera_rgb, captures images, and tests VLM latency.
"""

import os
import sys
import time
import base64
import statistics
from pathlib import Path
from io import BytesIO
from typing import Optional

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

        rospy.init_node('vlm_test_node', anonymous=True)
        self.sub = rospy.Subscriber(topic, ROSImage, self.callback)
        print(f"Subscribed to {topic}")

    def callback(self, msg: ROSImage):
        """ROS image callback."""
        try:
            # Convert ROS Image to numpy array
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

    def capture_multiple(self, count: int, interval: float = 0.5) -> list:
        """Capture multiple different images."""
        images = []
        for i in range(count):
            rospy.sleep(interval)
            if self.latest_image is not None:
                images.append(self.latest_image.copy())
                print(f"  Captured image {i+1}/{count}: {self.latest_image.shape}")
        return images


def resize_image(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize image keeping aspect ratio."""
    pil_img = Image.fromarray(image)
    w, h = pil_img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img)


def get_image_info(image: np.ndarray, quality: int = 90) -> dict:
    """Get image size info."""
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    jpeg_bytes = len(buffer.getvalue())
    return {
        'dimensions': f'{w}x{h}',
        'jpeg_kb': jpeg_bytes / 1024,
    }


def test_vlm(image: np.ndarray, api_key: str) -> tuple:
    """Test VLM with image, return (latency_ms, result)."""
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

    # Structured prompt for dead zone analysis
    prompt = """Analyze this navigation view. Is the path ahead blocked?

Reply in JSON:
{
    "blocked": true/false,
    "obstacle_type": "wall" | "furniture" | "door" | "clear" | "unclear",
    "confidence": 0.0-1.0,
    "description": "brief description"
}"""

    start = time.time()
    try:
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {"role": "system", "content": "You are a robot navigation assistant."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": prompt}
                ]}
            ],
            temperature=0.3,
            max_tokens=150,
        )
        latency = (time.time() - start) * 1000
        result = response.choices[0].message.content if response.choices else ""
        return latency, result
    except Exception as e:
        return -1, str(e)


def save_test_images(images: list, output_dir: str):
    """Save captured images for reference."""
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(images):
        # Save original
        orig_path = f"{output_dir}/ros_rgb_original_{i+1}.jpg"
        Image.fromarray(img).save(orig_path, quality=95)

        # Save resized versions
        for max_dim in [480, 320]:
            resized = resize_image(img, max_dim)
            path = f"{output_dir}/ros_rgb_resize{max_dim}_{i+1}.jpg"
            Image.fromarray(resized).save(path, quality=90)

    print(f"Images saved to {output_dir}/")


def main():
    print("=" * 70)
    print("    VLM LATENCY TEST WITH REAL ROS IMAGES")
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

    # Capture a few more images
    print("\n   Capturing additional images...")
    images = [first_image]
    more_images = capture.capture_multiple(2, interval=1.0)
    images.extend(more_images)

    print(f"\n   Total images captured: {len(images)}")

    # Save images for reference
    output_dir = "/home/yangz/Nav/SkillNav/docs/vlm_test_results/ros_images"
    save_test_images(images, output_dir)

    # Test VLM with different sizes
    print("\n" + "=" * 70)
    print("2. TESTING VLM LATENCY")
    print("=" * 70)

    # Use first image for testing
    test_image = images[0]
    h, w = test_image.shape[:2]
    print(f"\n   Test image size: {w}x{h}")

    configs = [
        ("Original", test_image),
        ("Resize 480", resize_image(test_image, 480)),
        ("Resize 320", resize_image(test_image, 320)),
        ("Resize 240", resize_image(test_image, 240)),
    ]

    results = {}
    runs_per_config = 3

    for name, img in configs:
        info = get_image_info(img)
        print(f"\n   {name} ({info['dimensions']}, {info['jpeg_kb']:.1f}KB):")

        latencies = []
        for i in range(runs_per_config):
            latency, result = test_vlm(img, api_key)
            if latency > 0:
                latencies.append(latency)
                preview = result[:40] + "..." if len(result) > 40 else result
                print(f"      Run {i+1}: {latency:.0f}ms - {preview}")
            else:
                print(f"      Run {i+1}: ERROR - {result}")

        if latencies:
            results[name] = {
                'dimensions': info['dimensions'],
                'kb': info['jpeg_kb'],
                'mean': statistics.mean(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'min': min(latencies),
                'max': max(latencies),
            }

    # Summary
    print("\n" + "=" * 70)
    print("3. SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<15} {'Size':<12} {'KB':<8} {'Mean':<10} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 70)

    if results:
        baseline = results.get('Original', {}).get('mean', 1)
        for name, r in sorted(results.items(), key=lambda x: x[1]['mean']):
            diff = ((r['mean'] - baseline) / baseline * 100) if baseline else 0
            print(f"{name:<15} {r['dimensions']:<12} {r['kb']:<8.1f} {r['mean']:<10.0f} {r['std']:<8.0f} {r['min']:<8.0f} {r['max']:<8.0f} ({diff:+.0f}%)")

    print("\n" + "=" * 70)
    print(f"Test images saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
