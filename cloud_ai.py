import os
import sys

# Ensure api directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.cloud_ai import ask_ai, get_groq_client

__all__ = ["ask_ai", "get_groq_client"]
