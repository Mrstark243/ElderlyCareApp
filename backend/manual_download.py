import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("--- Manual Model Download Script ---")
print("Checking network and downloading model...", flush=True)

try:
    from TTS.api import TTS
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    print(f"Downloading model: {model_name}")
    print("Please watch this terminal for progress bars...", flush=True)
    
    # This should trigger the standard TQDM bar in the user's terminal
    tts = TTS(model_name=model_name).to(device)
    
    print("\nSUCCESS: Model downloaded and loaded successfully!")

except ImportError:
    print("Error: TTS library not found. Run 'pip install TTS'")
except Exception as e:
    print(f"\nERROR: Download failed: {e}")
    import traceback
    traceback.print_exc()
