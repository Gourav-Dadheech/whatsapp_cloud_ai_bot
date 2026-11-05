import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.memory import get_memory, save_memory, clear_memory

__all__ = ["get_memory", "save_memory", "clear_memory"]
