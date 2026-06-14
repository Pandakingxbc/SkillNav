#!/usr/bin/env python3
"""
BLIP2-ITM Client for Multi-ValueMap System

Connects to BLIP2-ITM service at http://localhost:12182
Computes ITM scores for multiple prompts.

Author: Zager-Zhang
"""

import rospy
import requests
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Float64MultiArray, String
from cv_bridge import CvBridge


class BLIP2ITMClient:
    """
    ROS node that connects to BLIP2-ITM HTTP service.

    Subscribes to:
    - /camera/rgb/image_raw (sensor_msgs/Image): Input image

    Publishes:
    - /blip2_itm/scores (Float64MultiArray): ITM scores for 3 prompts
    - /blip2_itm/status (String): Status messages
    """

    def __init__(self):
        rospy.init_node('blip2_itm_client', anonymous=False)

        # BLIP2-ITM service URL
        self.blip_url = rospy.get_param('~blip_url', 'http://localhost:12182')
        self.endpoint = f"{self.blip_url}/blip2itm"  # Correct BLIP2 endpoint

        # Load prompts from parameter server
        # Try to load from YAML config first, fallback to defaults
        try:
            prompt_config = rospy.get_param('/blip2_itm/prompts', None)
            if prompt_config and isinstance(prompt_config, list):
                self.prompts = [p['text'] for p in prompt_config]
                rospy.loginfo(f"[BLIP2ITMClient] Loaded {len(self.prompts)} prompts from YAML config")
            else:
                raise KeyError("Prompts not found in config")
        except:
            # Fallback to default prompts
            self.prompts = [
                "dining chair around wooden table in kitchen",
                "open doorway to dining room, kitchen entrance with table visible",
                "unexplored room entrance, new doorway"
            ]
            rospy.logwarn("[BLIP2ITMClient] Using default prompts")

        rospy.loginfo(f"[BLIP2ITMClient] Initialized with {len(self.prompts)} prompts")
        rospy.loginfo(f"[BLIP2ITMClient] Service URL: {self.endpoint}")

        # CV Bridge
        self.bridge = CvBridge()

        # Publishers
        self.scores_pub = rospy.Publisher('/blip2_itm/scores', Float64MultiArray, queue_size=10)
        self.status_pub = rospy.Publisher('/blip2_itm/status', String, queue_size=10)

        # Subscriber
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', ROSImage, self.image_callback)

        # Test service availability
        self.test_service()

        rospy.loginfo("[BLIP2ITMClient] Ready to process images")

    def test_service(self):
        """Test if BLIP2-ITM service is available."""
        try:
            # Create a dummy image
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            response = self.compute_itm_score(dummy_image, "test prompt")

            if response is not None:
                rospy.loginfo(f"[BLIP2ITMClient] ✓ Service is available (test score: {response:.3f})")
                self.status_pub.publish(String(data="Service Available"))
                return True
            else:
                rospy.logwarn("[BLIP2ITMClient] ✗ Service test failed")
                self.status_pub.publish(String(data="Service Unavailable"))
                return False
        except Exception as e:
            rospy.logerr(f"[BLIP2ITMClient] Service test error: {e}")
            self.status_pub.publish(String(data=f"Error: {e}"))
            return False

    def encode_image(self, image):
        """
        Encode image to base64 for HTTP transmission.

        Args:
            image: numpy array (H, W, 3) in RGB format

        Returns:
            Base64 encoded string
        """
        # Convert to PIL Image
        if image.shape[2] == 3:
            pil_image = Image.fromarray(image)
        else:
            rospy.logwarn("[BLIP2ITMClient] Unexpected image channels")
            return None

        # Encode to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_str

    def compute_itm_score(self, image, text_prompt, timeout=5.0):
        """
        Compute ITM score for single prompt.

        Args:
            image: numpy array (H, W, 3) in RGB
            text_prompt: text prompt string
            timeout: request timeout in seconds

        Returns:
            ITM score [0.0, 1.0] or None on failure
        """
        try:
            # Encode image
            image_base64 = self.encode_image(image)
            if image_base64 is None:
                return None

            # Prepare request (BLIP2 uses 'txt' not 'text')
            payload = {
                "image": image_base64,
                "txt": text_prompt  # BLIP2 API uses 'txt'
            }

            # Send request
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=timeout
            )

            # Check response
            if response.status_code == 200:
                data = response.json()
                # BLIP2 returns 'itm score' with a space
                score = data.get('itm score', data.get('response', 0.0))
                return float(score)
            else:
                rospy.logwarn(f"[BLIP2ITMClient] HTTP {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
            rospy.logwarn_throttle(5.0, "[BLIP2ITMClient] Request timeout")
            return None
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[BLIP2ITMClient] Error: {e}")
            return None

    def compute_all_scores(self, image):
        """
        Compute ITM scores for all prompts.

        Args:
            image: numpy array (H, W, 3) in RGB

        Returns:
            List of scores [score_DT, score_SN, score_BE]
        """
        scores = []

        for i, prompt in enumerate(self.prompts):
            score = self.compute_itm_score(image, prompt)

            if score is not None:
                scores.append(score)
                rospy.logdebug(f"[BLIP2ITMClient] Prompt {i}: {score:.3f}")
            else:
                # Use default score on failure
                scores.append(0.0)
                rospy.logwarn(f"[BLIP2ITMClient] Failed to compute score for prompt {i}")

        return scores

    def image_callback(self, msg):
        """
        ROS image callback.

        Args:
            msg: sensor_msgs/Image
        """
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Compute ITM scores for all prompts
            start_time = rospy.Time.now()
            scores = self.compute_all_scores(cv_image)
            elapsed = (rospy.Time.now() - start_time).to_sec()

            # Publish scores
            scores_msg = Float64MultiArray()
            scores_msg.data = scores
            self.scores_pub.publish(scores_msg)

            rospy.loginfo(f"[BLIP2ITMClient] Scores: {[f'{s:.3f}' for s in scores]} | Time: {elapsed:.2f}s")

            # Publish status
            status = f"OK | Scores: [{scores[0]:.2f}, {scores[1]:.2f}, {scores[2]:.2f}]"
            self.status_pub.publish(String(data=status))

        except Exception as e:
            rospy.logerr(f"[BLIP2ITMClient] Image callback error: {e}")
            self.status_pub.publish(String(data=f"Error: {e}"))

    def run(self):
        """Main loop."""
        rospy.spin()


if __name__ == '__main__':
    try:
        client = BLIP2ITMClient()
        client.run()
    except rospy.ROSInterruptException:
        pass
