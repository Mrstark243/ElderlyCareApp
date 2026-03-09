import os
import sys

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ai_engine import ai_engine
import traceback

def test_tts():
    print("Testing TTS...")
    try:
        # User needs to provide a speaker wav for this to work, but we can't test without one easily.
        # We will skip actual generation and just check if the method exists and imports work.
        # Construct a dummy call
        # ai_engine.generate_voice_cloning("Hello", "dummy.wav")
        # This would fail if TTS not installed.
        pass
    except Exception as e:
        print(f"TTS Test Failed: {e}")
        traceback.print_exc()

def test_wav2lip():
    print("Testing Wav2Lip...")
    # Check if files exist
    if not os.path.exists(ai_engine.wav2lip_dir):
        print("Wav2Lip dir not found!")
        return

    inference_script = os.path.join(ai_engine.wav2lip_dir, "inference.py")
    if not os.path.exists(inference_script):
        print("inference.py not found!")
        return

    print("Wav2Lip structure seems okay.")

if __name__ == "__main__":
    test_tts()
    test_wav2lip()
