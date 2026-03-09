import argparse
import os
import sys

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

# Auto-agree to Coqui TTS license to prevent blocking prompts
os.environ["COQUI_TOS_AGREED"] = "1"

print("Status: Initializing TTS...", flush=True)

# Try to import TTS, handle failure gracefully
try:
    from TTS.api import TTS
except ImportError:
    print("Error: Coqui TTS not installed. Please install it or use a fallback.", flush=True)
    sys.exit(1)

import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--speaker_wav", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="tts_models/multilingual/multi-dataset/xtts_v2")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    
    if device == "cpu":
        print("Status: Warning - Running on CPU (Slow)...", flush=True)
    
    # Initialize TTS
    try:
        print("Status: Loading/Downloading TTS Model (~2GB)...", flush=True)
        # This might trigger a download
        tts = TTS(model_name=args.model_name).to(device)
        print("Status: Model Loaded.", flush=True)
    except Exception as e:
        print(f"Failed to load TTS model: {e}", flush=True)
        sys.exit(1)
        
    # Generate
    try:
        print("Status: Generating Audio...", flush=True)
        tts.tts_to_file(text=args.text, speaker_wav=args.speaker_wav, language="en", file_path=args.out_path)
        print(f"Generated audio at {args.out_path}", flush=True)
    except Exception as e:
        print(f"Generation failed: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
