"""Lets every script run from any folder: puts the repo root on sys.path and makes it the working directory."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
