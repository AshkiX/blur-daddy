"""Blur Daddy - Fast, accurate face blurring for images and videos."""

__version__ = "0.3.0"

from blur_daddy.api import BlurDaddy, blur
from blur_daddy.models import BlurResult, Detection, DetectionResult, Face

__all__ = [
    "BlurDaddy",
    "blur",
    "BlurResult",
    "Detection",
    "DetectionResult",
    "Face",
]
