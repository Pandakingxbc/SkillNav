#!/usr/bin/env python3
"""
Image Preprocessing Comparison for VLM.

Compares:
1. Original image
2. Resized (scaled down, keeps full FOV)
3. Cropped (center crop, loses FOV)
4. Different JPEG qualities

Outputs comparison images and tests VLM latency.
"""

import os
import sys
import time
import base64
from pathlib import Path
from io import BytesIO
from typing import Dict, Any, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_image(path: str) -> np.ndarray:
    """Load image as numpy array."""
    return np.array(Image.open(path).convert('RGB'))


def create_comparison_grid(
    images: List[Tuple[str, np.ndarray, Dict]],
    output_path: str,
    cols: int = 3
) -> str:
    """Create a comparison grid of images with labels."""

    # Calculate grid dimensions
    n = len(images)
    rows = (n + cols - 1) // cols

    # Find max dimensions
    max_h = max(img.shape[0] for _, img, _ in images)
    max_w = max(img.shape[1] for _, img, _ in images)

    # Add space for labels
    label_height = 80
    padding = 10

    # Create canvas
    canvas_w = cols * (max_w + padding) + padding
    canvas_h = rows * (max_h + label_height + padding) + padding

    canvas = Image.new('RGB', (canvas_w, canvas_h), color='white')
    draw = ImageDraw.Draw(canvas)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()
        font_small = font

    for i, (name, img, info) in enumerate(images):
        row = i // cols
        col = i % cols

        x = padding + col * (max_w + padding)
        y = padding + row * (max_h + label_height + padding)

        # Paste image
        pil_img = Image.fromarray(img)
        canvas.paste(pil_img, (x, y))

        # Draw border
        draw.rectangle(
            [x-1, y-1, x + img.shape[1], y + img.shape[0]],
            outline='black',
            width=2
        )

        # Draw label background
        label_y = y + img.shape[0] + 5
        draw.rectangle(
            [x, label_y, x + max_w, label_y + label_height - 10],
            fill='#f0f0f0'
        )

        # Draw labels
        draw.text((x + 5, label_y + 2), name, fill='black', font=font)

        # Draw info
        info_lines = [
            f"Size: {info.get('dimensions', 'N/A')}",
            f"JPEG: {info.get('jpeg_kb', 0):.1f}KB, Base64: {info.get('base64_kb', 0):.1f}KB",
            f"Latency: {info.get('latency_ms', 'N/A')}ms" if info.get('latency_ms') else "Latency: pending..."
        ]

        for j, line in enumerate(info_lines):
            draw.text((x + 5, label_y + 20 + j * 15), line, fill='#333333', font=font_small)

    canvas.save(output_path)
    return output_path


def resize_image(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize image keeping aspect ratio (preserves full FOV)."""
    pil_img = Image.fromarray(image)
    w, h = pil_img.size

    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    return np.array(pil_img)


def center_crop(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Center crop image (LOSES peripheral FOV)."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # If target is larger than image, just return resized
    if target_w >= w and target_h >= h:
        return image

    # Calculate crop region
    start_x = max(0, (w - target_w) // 2)
    start_y = max(0, (h - target_h) // 2)

    cropped = image[start_y:start_y + target_h, start_x:start_x + target_w]
    return cropped


def get_image_info(image: np.ndarray, quality: int = 90) -> Dict[str, Any]:
    """Get image size info."""
    h, w = image.shape[:2]

    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    jpeg_bytes = len(buffer.getvalue())
    base64_bytes = len(base64.b64encode(buffer.getvalue()))

    return {
        'dimensions': f'{w}x{h}',
        'pixels': w * h,
        'jpeg_kb': jpeg_bytes / 1024,
        'base64_kb': base64_bytes / 1024,
    }


def test_vlm_latency(image: np.ndarray, api_key: str, description: str = "") -> float:
    """Test VLM latency with given image."""
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60.0,
    )

    # Encode image
    pil_img = Image.fromarray(image)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=90)
    b64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # The structured prompt (proven to be faster due to constrained output)
    prompt = """Analyze this navigation situation where a robot has failed to move forward.

Please respond in JSON format:
{
    "obstacle_type": "permanent" | "temporary" | "system_boundary" | "unclear",
    "abandon_region": true | false,
    "confidence": 0.0-1.0,
    "suggested_action": "mark_dead_zone" | "retry_later" | "reroute"
}"""

    start = time.time()
    try:
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {"role": "system", "content": "You are a navigation analysis assistant."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": prompt}
                ]}
            ],
            temperature=0.3,
            max_tokens=200,
        )
        latency = (time.time() - start) * 1000
        print(f"    {description}: {latency:.0f}ms")
        return latency
    except Exception as e:
        print(f"    {description}: ERROR - {e}")
        return -1


def main():
    print("=" * 70)
    print("        IMAGE PREPROCESSING COMPARISON FOR VLM")
    print("=" * 70)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("WARNING: DASHSCOPE_API_KEY not set, will skip VLM tests")

    # Load original image
    image_path = "/home/yangz/Nav/SkillNav/habitat-lab/docs/images/habitat-lab-demo-images/habitat-lab-demo.png"
    print(f"\nLoading: {image_path}")
    original = load_image(image_path)
    print(f"Original size: {original.shape[1]}x{original.shape[0]}")

    # Prepare different versions
    images_to_compare = []

    # 1. Original
    info = get_image_info(original)
    images_to_compare.append(("1. Original", original, info))

    # 2. Resized versions (keeps full FOV)
    for max_dim in [640, 480, 320]:
        resized = resize_image(original, max_dim)
        info = get_image_info(resized)
        images_to_compare.append((f"2. Resize max={max_dim}", resized, info))

    # 3. Center crop versions (LOSES FOV - for comparison)
    for size in [(640, 480), (480, 360), (320, 240)]:
        cropped = center_crop(original, size)
        info = get_image_info(cropped)
        info['note'] = 'LOSES peripheral vision!'
        images_to_compare.append((f"3. Crop {size[0]}x{size[1]}", cropped, info))

    # 4. Different JPEG qualities (on resized 480)
    resized_480 = resize_image(original, 480)
    for quality in [90, 70, 50]:
        info = get_image_info(resized_480, quality=quality)
        info['quality'] = quality
        # Store same image but different quality info
        images_to_compare.append((f"4. Resize 480 Q{quality}", resized_480.copy(), info))

    # Print comparison table
    print("\n" + "=" * 70)
    print("IMAGE SIZE COMPARISON")
    print("=" * 70)
    print(f"{'Name':<25} {'Dimensions':<12} {'JPEG KB':<10} {'Base64 KB':<12}")
    print("-" * 70)

    for name, img, info in images_to_compare:
        print(f"{name:<25} {info['dimensions']:<12} {info['jpeg_kb']:<10.1f} {info['base64_kb']:<12.1f}")

    # Create comparison grid (first 9 images)
    output_dir = "/home/yangz/Nav/SkillNav/vlm/llm/test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save individual images for detailed viewing
    print("\n" + "=" * 70)
    print("SAVING INDIVIDUAL IMAGES")
    print("=" * 70)

    saved_paths = []
    for name, img, info in images_to_compare[:9]:
        safe_name = name.replace(" ", "_").replace("=", "").replace(".", "_")
        path = f"{output_dir}/{safe_name}.jpg"
        Image.fromarray(img).save(path, quality=90)
        saved_paths.append(path)
        print(f"  Saved: {path}")

    # Create comparison grid
    grid_path = f"{output_dir}/comparison_grid.jpg"
    create_comparison_grid(images_to_compare[:9], grid_path, cols=3)
    print(f"\n  Comparison grid: {grid_path}")

    # Test VLM latency if API key available
    if api_key:
        print("\n" + "=" * 70)
        print("VLM LATENCY TESTS")
        print("=" * 70)

        # Test key configurations
        test_configs = [
            ("Original (1414x1872)", original),
            ("Resize max=640", resize_image(original, 640)),
            ("Resize max=480", resize_image(original, 480)),
            ("Resize max=320", resize_image(original, 320)),
            ("Crop 640x480", center_crop(original, (640, 480))),
            ("Crop 320x240", center_crop(original, (320, 240))),
        ]

        results = []
        for desc, img in test_configs:
            latency = test_vlm_latency(img, api_key, desc)
            info = get_image_info(img)
            info['latency_ms'] = latency if latency > 0 else None
            results.append((desc, img, info))

        # Print results table
        print("\n" + "=" * 70)
        print("LATENCY COMPARISON RESULTS")
        print("=" * 70)
        print(f"{'Config':<25} {'Size':<12} {'KB':<8} {'Latency':<10} {'vs Original'}")
        print("-" * 70)

        baseline = results[0][2].get('latency_ms', 0) if results else 0

        for name, img, info in results:
            latency = info.get('latency_ms')
            if latency and latency > 0:
                diff = ((latency - baseline) / baseline * 100) if baseline else 0
                diff_str = f"{diff:+.0f}%" if baseline else "N/A"
                print(f"{name:<25} {info['dimensions']:<12} {info['jpeg_kb']:<8.1f} {latency:<10.0f}ms {diff_str}")
            else:
                print(f"{name:<25} {info['dimensions']:<12} {info['jpeg_kb']:<8.1f} {'ERROR':<10}")

        # Create final comparison with latency info
        final_grid_path = f"{output_dir}/comparison_with_latency.jpg"
        create_comparison_grid(results, final_grid_path, cols=3)
        print(f"\n  Final comparison grid: {final_grid_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: RESIZE vs CROP")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                     RESIZE (推荐) vs CROP                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RESIZE (缩放):                                                     │
│  ✅ 保留完整视野 (Full FOV)                                         │
│  ✅ 所有物体仍然可见                                                 │
│  ✅ 适合导航场景理解                                                 │
│  ⚠️ 细节减少，但VLM通常不需要像素级细节                              │
│                                                                     │
│  CROP (裁剪):                                                       │
│  ❌ 丢失周边视野 (Peripheral FOV lost)                              │
│  ❌ 可能裁掉重要障碍物                                               │
│  ❌ 不适合导航决策                                                   │
│  ✅ 保留中心区域细节                                                 │
│                                                                     │
│  结论: 对于Safe Agent死区分析，应该使用 RESIZE 而不是 CROP          │
│        推荐配置: resize max_dim=480, JPEG quality=90                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

    print(f"\n所有输出图片保存在: {output_dir}/")
    print("请查看 comparison_grid.jpg 对比不同处理方式的效果")


if __name__ == "__main__":
    main()
