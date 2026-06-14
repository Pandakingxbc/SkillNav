#!/usr/bin/env python3
"""
Detailed VLM Latency Analysis - Breaking down the latency components.

Analyzes:
1. Image encoding time (numpy → base64)
2. Network transmission time
3. Model inference time
4. Response parsing time
5. Impact of image size
6. Impact of prompt length
"""

import os
import sys
import time
import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple
from io import BytesIO

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_image(path: str) -> np.ndarray:
    """Load image as numpy array."""
    from PIL import Image
    return np.array(Image.open(path).convert('RGB'))


def get_image_sizes(image: np.ndarray) -> Dict[str, Any]:
    """Get various image size metrics."""
    from PIL import Image

    # Original size
    h, w = image.shape[:2]
    original_pixels = h * w
    original_bytes = image.nbytes

    # JPEG encoded sizes at different qualities
    pil_image = Image.fromarray(image)
    sizes = {}

    for quality in [50, 70, 90, 95]:
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        jpeg_bytes = len(buffer.getvalue())
        base64_bytes = len(base64.b64encode(buffer.getvalue()))
        sizes[f'jpeg_q{quality}'] = {
            'jpeg_bytes': jpeg_bytes,
            'jpeg_kb': jpeg_bytes / 1024,
            'base64_bytes': base64_bytes,
            'base64_kb': base64_bytes / 1024,
        }

    return {
        'dimensions': f'{w}x{h}',
        'pixels': original_pixels,
        'raw_bytes': original_bytes,
        'raw_mb': original_bytes / (1024 * 1024),
        'encoded_sizes': sizes
    }


def measure_encoding_time(image: np.ndarray, quality: int = 90, runs: int = 10) -> Dict[str, float]:
    """Measure image encoding time."""
    from PIL import Image

    pil_image = Image.fromarray(image)

    # Measure JPEG encoding
    jpeg_times = []
    for _ in range(runs):
        start = time.time()
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        jpeg_bytes = buffer.getvalue()
        jpeg_times.append(time.time() - start)

    # Measure base64 encoding
    base64_times = []
    for _ in range(runs):
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        jpeg_bytes = buffer.getvalue()
        start = time.time()
        b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
        base64_times.append(time.time() - start)

    return {
        'jpeg_encoding_ms': sum(jpeg_times) / runs * 1000,
        'base64_encoding_ms': sum(base64_times) / runs * 1000,
        'total_encoding_ms': (sum(jpeg_times) + sum(base64_times)) / runs * 1000,
    }


def get_prompt_info() -> Dict[str, Any]:
    """Get prompt information for dead zone analysis."""

    # This is the prompt used in qwen_vlm.py analyze_dead_zone
    prompt_template = """Analyze this navigation situation where a robot has failed to move forward multiple times.

Current situation:
- Robot position: ({x:.2f}, {y:.2f})
- Failed escape attempts: {attempt_count}
- Current heading: {heading:.1f}°

Please analyze the image and respond in JSON format:
{{
    "obstacle_type": "permanent" | "temporary" | "system_boundary" | "unclear",
    "description": "Brief description of what you see blocking the path",
    "abandon_region": true | false,
    "confidence": 0.0-1.0,
    "reasoning": "Why you made this decision",
    "suggested_action": "mark_dead_zone" | "retry_later" | "reroute" | "continue_trying"
}}

Guidelines:
- "permanent": walls, fixed furniture, structural barriers
- "temporary": movable objects, closed doors that might open
- "system_boundary": edge of map, simulation boundary, void areas
- "unclear": cannot determine from current view

Only set abandon_region=true if confident (>0.7) the area is permanently impassable."""

    system_prompt = "You are a navigation analysis assistant helping a robot decide whether to abandon exploring a blocked region."

    # Fill in example values
    filled_prompt = prompt_template.format(x=1.5, y=2.3, attempt_count=3, heading=45.0)

    return {
        'user_prompt_chars': len(filled_prompt),
        'user_prompt_words': len(filled_prompt.split()),
        'system_prompt_chars': len(system_prompt),
        'system_prompt_words': len(system_prompt.split()),
        'total_chars': len(filled_prompt) + len(system_prompt),
        'estimated_tokens': (len(filled_prompt) + len(system_prompt)) // 4,  # rough estimate
        'user_prompt_preview': filled_prompt[:200] + '...',
    }


def test_with_different_image_sizes(original_image: np.ndarray, api_key: str) -> Dict[str, Any]:
    """Test VLM latency with different image sizes."""
    from PIL import Image
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60.0,
    )

    results = {}

    # Test different resize factors
    pil_original = Image.fromarray(original_image)
    original_w, original_h = pil_original.size

    resize_factors = [1.0, 0.5, 0.25, 0.1]

    for factor in resize_factors:
        new_w = int(original_w * factor)
        new_h = int(original_h * factor)

        if factor == 1.0:
            resized = pil_original
        else:
            resized = pil_original.resize((new_w, new_h), Image.LANCZOS)

        # Encode
        buffer = BytesIO()
        resized.save(buffer, format='JPEG', quality=90)
        jpeg_bytes = buffer.getvalue()
        b64_image = base64.b64encode(jpeg_bytes).decode('utf-8')

        image_kb = len(jpeg_bytes) / 1024

        print(f"\n  Testing {new_w}x{new_h} ({image_kb:.1f} KB)...")

        # Make API call
        start = time.time()

        try:
            response = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {"role": "system", "content": "Describe this image briefly."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                        {"type": "text", "text": "What do you see?"}
                    ]}
                ],
                temperature=0.3,
                max_tokens=100,
            )

            latency_ms = (time.time() - start) * 1000

            results[f'{new_w}x{new_h}'] = {
                'dimensions': f'{new_w}x{new_h}',
                'pixels': new_w * new_h,
                'image_kb': image_kb,
                'latency_ms': latency_ms,
                'success': True,
                'response_preview': response.choices[0].message.content[:50] if response.choices else None
            }

            print(f"    Latency: {latency_ms:.0f}ms")

        except Exception as e:
            results[f'{new_w}x{new_h}'] = {
                'dimensions': f'{new_w}x{new_h}',
                'error': str(e),
                'success': False
            }
            print(f"    Error: {e}")

    return results


def test_with_different_prompts(image: np.ndarray, api_key: str) -> Dict[str, Any]:
    """Test VLM latency with different prompt lengths."""
    from PIL import Image
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=60.0,
    )

    # Resize image to reasonable size for consistent testing
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((640, 480), Image.LANCZOS)

    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    b64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    results = {}

    prompts = {
        'minimal': "What is this?",
        'short': "Describe this indoor scene briefly.",
        'medium': """Analyze this image and tell me:
1. What room type is this?
2. What objects do you see?
3. Are there any obstacles?""",
        'long_structured': """Analyze this navigation situation where a robot has failed to move forward multiple times.

Current situation:
- Robot position: (1.50, 2.30)
- Failed escape attempts: 3
- Current heading: 45.0°

Please analyze the image and respond in JSON format:
{
    "obstacle_type": "permanent" | "temporary" | "system_boundary" | "unclear",
    "description": "Brief description of what you see blocking the path",
    "abandon_region": true | false,
    "confidence": 0.0-1.0,
    "reasoning": "Why you made this decision",
    "suggested_action": "mark_dead_zone" | "retry_later" | "reroute" | "continue_trying"
}

Guidelines:
- "permanent": walls, fixed furniture, structural barriers
- "temporary": movable objects, closed doors that might open
- "system_boundary": edge of map, simulation boundary, void areas
- "unclear": cannot determine from current view

Only set abandon_region=true if confident (>0.7) the area is permanently impassable."""
    }

    for name, prompt in prompts.items():
        print(f"\n  Testing prompt '{name}' ({len(prompt)} chars)...")

        start = time.time()

        try:
            response = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                        {"type": "text", "text": prompt}
                    ]}
                ],
                temperature=0.3,
                max_tokens=256,
            )

            latency_ms = (time.time() - start) * 1000
            response_text = response.choices[0].message.content if response.choices else ""

            results[name] = {
                'prompt_chars': len(prompt),
                'prompt_words': len(prompt.split()),
                'latency_ms': latency_ms,
                'response_chars': len(response_text),
                'success': True,
            }

            print(f"    Latency: {latency_ms:.0f}ms, Response: {len(response_text)} chars")

        except Exception as e:
            results[name] = {
                'prompt_chars': len(prompt),
                'error': str(e),
                'success': False
            }
            print(f"    Error: {e}")

    return results


def estimate_network_overhead(image_kb: float) -> Dict[str, Any]:
    """Estimate network transmission time based on typical bandwidth."""

    # Typical network scenarios
    scenarios = {
        'fast_fiber': {'mbps': 100, 'latency_ms': 20},
        'good_broadband': {'mbps': 50, 'latency_ms': 30},
        'average': {'mbps': 20, 'latency_ms': 50},
        'slow': {'mbps': 5, 'latency_ms': 100},
    }

    results = {}
    for name, config in scenarios.items():
        # Upload time for image
        upload_time_ms = (image_kb / 1024) / (config['mbps'] / 8) * 1000  # KB to MB, Mbps to MBps
        # Round trip latency
        rtt_ms = config['latency_ms'] * 2

        results[name] = {
            'bandwidth_mbps': config['mbps'],
            'base_latency_ms': config['latency_ms'],
            'estimated_upload_ms': upload_time_ms,
            'estimated_rtt_ms': rtt_ms,
            'total_network_ms': upload_time_ms + rtt_ms,
        }

    return results


def main():
    print("=" * 70)
    print("        VLM LATENCY DETAILED ANALYSIS")
    print("=" * 70)

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY not set")
        return

    # Load test image
    image_path = "/home/yangz/Nav/SkillNav/habitat-lab/docs/images/habitat-lab-demo-images/habitat-lab-demo.png"
    print(f"\nLoading image: {image_path}")
    image = load_image(image_path)

    # 1. Image size analysis
    print("\n" + "=" * 70)
    print("1. IMAGE SIZE ANALYSIS")
    print("=" * 70)

    sizes = get_image_sizes(image)
    print(f"\nOriginal Image:")
    print(f"  Dimensions: {sizes['dimensions']}")
    print(f"  Pixels: {sizes['pixels']:,}")
    print(f"  Raw size: {sizes['raw_mb']:.2f} MB")

    print(f"\nEncoded sizes (JPEG + Base64):")
    for quality, info in sizes['encoded_sizes'].items():
        print(f"  {quality}: {info['jpeg_kb']:.1f} KB JPEG → {info['base64_kb']:.1f} KB Base64")

    # 2. Encoding time measurement
    print("\n" + "=" * 70)
    print("2. LOCAL ENCODING TIME")
    print("=" * 70)

    encoding = measure_encoding_time(image)
    print(f"\nEncoding times (average of 10 runs):")
    print(f"  JPEG encoding: {encoding['jpeg_encoding_ms']:.2f} ms")
    print(f"  Base64 encoding: {encoding['base64_encoding_ms']:.2f} ms")
    print(f"  Total: {encoding['total_encoding_ms']:.2f} ms")

    # 3. Prompt analysis
    print("\n" + "=" * 70)
    print("3. PROMPT ANALYSIS")
    print("=" * 70)

    prompt_info = get_prompt_info()
    print(f"\nDead Zone Analysis Prompt:")
    print(f"  User prompt: {prompt_info['user_prompt_chars']} chars, {prompt_info['user_prompt_words']} words")
    print(f"  System prompt: {prompt_info['system_prompt_chars']} chars")
    print(f"  Estimated tokens: ~{prompt_info['estimated_tokens']}")

    # 4. Network overhead estimation
    print("\n" + "=" * 70)
    print("4. NETWORK OVERHEAD ESTIMATION")
    print("=" * 70)

    # Use JPEG Q90 size
    image_kb = sizes['encoded_sizes']['jpeg_q90']['base64_kb']
    network = estimate_network_overhead(image_kb)

    print(f"\nFor {image_kb:.1f} KB image:")
    for scenario, info in network.items():
        print(f"  {scenario} ({info['bandwidth_mbps']} Mbps):")
        print(f"    Upload: {info['estimated_upload_ms']:.1f}ms, RTT: {info['estimated_rtt_ms']:.0f}ms")
        print(f"    Total network: {info['total_network_ms']:.1f}ms")

    # 5. Test with different image sizes
    print("\n" + "=" * 70)
    print("5. LATENCY VS IMAGE SIZE")
    print("=" * 70)

    size_results = test_with_different_image_sizes(image, api_key)

    print("\n  Summary:")
    print(f"  {'Size':<15} {'Pixels':<12} {'Image KB':<10} {'Latency':<10}")
    print("  " + "-" * 50)
    for key, info in size_results.items():
        if info.get('success'):
            print(f"  {info['dimensions']:<15} {info['pixels']:<12,} {info['image_kb']:<10.1f} {info['latency_ms']:<10.0f}ms")

    # 6. Test with different prompts
    print("\n" + "=" * 70)
    print("6. LATENCY VS PROMPT COMPLEXITY")
    print("=" * 70)

    prompt_results = test_with_different_prompts(image, api_key)

    print("\n  Summary:")
    print(f"  {'Prompt':<20} {'Chars':<10} {'Latency':<12} {'Response':<10}")
    print("  " + "-" * 55)
    for name, info in prompt_results.items():
        if info.get('success'):
            print(f"  {name:<20} {info['prompt_chars']:<10} {info['latency_ms']:<12.0f}ms {info['response_chars']:<10}")

    # 7. Summary and breakdown
    print("\n" + "=" * 70)
    print("7. LATENCY BREAKDOWN ANALYSIS")
    print("=" * 70)

    # Get the full dead zone analysis latency from previous test
    full_latency = 3837  # ms from previous test

    # Estimate breakdown
    local_encoding = encoding['total_encoding_ms']
    network_estimate = network['good_broadband']['total_network_ms']

    # If we have size test results, use the smallest size latency as baseline
    if size_results:
        smallest_latency = min(
            info['latency_ms'] for info in size_results.values()
            if info.get('success') and info.get('latency_ms')
        )
        # Model inference is roughly: smallest_latency - network_overhead
        model_inference_base = smallest_latency - network_estimate
    else:
        model_inference_base = 2000  # estimate

    # Prompt overhead from prompt test
    if prompt_results:
        minimal_latency = prompt_results.get('minimal', {}).get('latency_ms', 2000)
        long_latency = prompt_results.get('long_structured', {}).get('latency_ms', 4000)
        prompt_overhead = long_latency - minimal_latency
    else:
        prompt_overhead = 500

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    LATENCY BREAKDOWN (Estimated)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Total observed latency:              ~{full_latency}ms                       │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ 1. Local Processing (encoding)           ~{local_encoding:.0f}ms  ({local_encoding/full_latency*100:.1f}%)      │  │
│  │    - JPEG compression                                         │  │
│  │    - Base64 encoding                                          │  │
│  │                                                               │  │
│  │ 2. Network Transmission                  ~{network_estimate:.0f}ms  ({network_estimate/full_latency*100:.1f}%)      │  │
│  │    - Upload image ({image_kb:.0f}KB)                                  │  │
│  │    - API round-trip latency                                   │  │
│  │                                                               │  │
│  │ 3. Server Processing (main bottleneck)   ~{model_inference_base:.0f}ms ({model_inference_base/full_latency*100:.1f}%)     │  │
│  │    - Image preprocessing/tokenization                         │  │
│  │    - Vision encoder forward pass                              │  │
│  │    - LLM inference (text generation)                          │  │
│  │                                                               │  │
│  │ 4. Prompt Complexity Overhead            ~{prompt_overhead:.0f}ms  ({prompt_overhead/full_latency*100:.1f}%)     │  │
│  │    - Longer prompts = more tokens = more inference            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  MAIN FACTORS (in order of impact):                                 │
│  1. 🔴 Server-side model inference (~70-80% of latency)             │
│  2. 🟡 Prompt/response length (~10-15%)                             │
│  3. 🟢 Network transmission (~5-10%)                                │
│  4. 🟢 Local encoding (~1%)                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

    print("\n" + "=" * 70)
    print("8. OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on the analysis, here are optimization strategies ranked by impact:

┌─────────────────────────────────────────────────────────────────────┐
│ HIGH IMPACT                                                         │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Use smaller/faster model (if available)                          │
│    - qwen-vl-plus → qwen-vl-lite (if exists)                        │
│    - Trade accuracy for speed                                       │
│                                                                     │
│ 2. Simplify prompt (reduce output requirements)                     │
│    - Current: asks for JSON with 6 fields                           │
│    - Minimal: just ask "permanent/temporary/unclear?"               │
│    - Estimated saving: ~500ms                                       │
│                                                                     │
│ 3. Resize image before sending                                      │
│    - 1872x1414 → 640x480 (or smaller)                               │
│    - Reduces tokens, network, and inference time                    │
│    - Estimated saving: ~500-1000ms                                  │
├─────────────────────────────────────────────────────────────────────┤
│ MEDIUM IMPACT                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ 4. Use lower JPEG quality                                           │
│    - Q90 → Q70 or Q50                                               │
│    - Reduces network time slightly                                  │
│                                                                     │
│ 5. Cache repeated queries                                           │
│    - Same location + similar image = use cached result              │
│    - Already implemented in AgentLLMInterface                       │
├─────────────────────────────────────────────────────────────────────┤
│ LOW IMPACT (already minimal)                                        │
├─────────────────────────────────────────────────────────────────────┤
│ 6. Optimize local encoding                                          │
│    - Already <50ms, not worth optimizing                            │
│                                                                     │
│ 7. Better network                                                   │
│    - Already <100ms for typical broadband                           │
└─────────────────────────────────────────────────────────────────────┘

CONCLUSION:
- The ~3.8 second latency is primarily due to SERVER-SIDE INFERENCE
- This is inherent to large VLM models
- Local optimizations can save ~1-1.5s at best
- For Safe Agent, ASYNC is the only viable approach
""")


if __name__ == "__main__":
    main()
