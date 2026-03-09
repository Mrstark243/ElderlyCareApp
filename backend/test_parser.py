from app.services.ai_engine import ai_engine
import os
import sys

# Mock logger to see debug output
import logging
logging.basicConfig(level=logging.DEBUG)

def callback(p, msg):
    print(f"CALLBACK -> Progress: {p}, Msg: {msg}")

print("Starting simulation test...")
cwd = os.getcwd()
cmd = ["python", "simulate_download.py"]

try:
    ai_engine._run_command_with_progress(cmd, cwd=cwd, progress_callback=callback, start_percent=10, end_percent=30)
except Exception as e:
    print(f"Error: {e}")
