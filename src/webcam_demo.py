#!/usr/bin/env python3
"""Entry point for the enhanced Facial Expression Recognition demo.

Usage:
    fer_face_env\Scripts\python src\webcam_demo.py
"""

import sys
import os

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import main

if __name__ == "__main__":
    main()
