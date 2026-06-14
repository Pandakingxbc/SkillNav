import cv2
import numpy as np
from vlm.itm.blip2itm import BLIP2ITMClient

itmclient = BLIP2ITMClient(port=12182)

def get_itm_message(rgb_image, label):
    txt = f"Is there a {label} in the image?"
    cosine = itmclient.cosine(rgb_image, txt)
    itm_score = itmclient.itm_score(rgb_image, txt)
    return cosine, itm_score

def get_itm_message_cosine(rgb_image, label, room):
    if room != "everywhere":
        txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        txt = f"Seems like there is a {label} ahead?"
    cosine = itmclient.cosine(rgb_image, txt)
    return cosine


def get_itm_message_cosines_dual(rgb_image, label, room):
    """Compute two ITM cosine similarities for the MultiValueMap fusion system.

    Returns
    -------
    (sr_cosine, ig_cosine)
        sr_cosine: semantic-relevance score (target-direct evidence; reuses
                   the existing room+label prompt design)
        ig_cosine: information-gain score (exploration-oriented; favors
                   doorways/hallways heading toward the target room)
    """
    # Semantic Relevance — same as legacy get_itm_message_cosine
    if room != "everywhere":
        sr_txt = f"Seems like there is a {room} or a {label} ahead?"
    else:
        sr_txt = f"Seems like there is a {label} ahead?"
    sr_cosine = itmclient.cosine(rgb_image, sr_txt)

    # Information Gain — corridor / doorway leading toward target area
    if room != "everywhere":
        ig_txt = f"An open doorway or hallway leading toward a {room} or unexplored rooms"
    else:
        ig_txt = "An open doorway or hallway with multiple unexplored rooms ahead"
    ig_cosine = itmclient.cosine(rgb_image, ig_txt)

    return sr_cosine, ig_cosine