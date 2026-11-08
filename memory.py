import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.memory import (
    get_memory, save_memory, clear_memory,
    get_active_document, save_active_document, clear_active_document,
    get_active_image, save_active_image, clear_active_image
)

__all__ = [
    "get_memory", "save_memory", "clear_memory",
    "get_active_document", "save_active_document", "clear_active_document",
    "get_active_image", "save_active_image", "clear_active_image"
]
