import os
import replicate
from typing import Optional
from app.core.config import settings
import time

class AIService:
    def __init__(self):
        self.api_token = settings.REPLICATE_API_TOKEN

    async def generate_speech(self, text: str, speaker_wav_path: str) -> str:
        """
        Generates speech from text using a reference audio file for voice cloning.
        Uses Replicate's XTTS-v2 model.
        """
        # Reliable fallback URL
        MOCK_AUDIO_URL = "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3"

        if not self.api_token:
            print("WARNING: REPLICATE_API_TOKEN not set in settings. Returning mock audio.")
            return MOCK_AUDIO_URL

        print(f"Cloning voice from {speaker_wav_path} for text: {text}")

        try:
            # Check if paths are local files
            if os.path.exists(speaker_wav_path):
                 input_audio = open(speaker_wav_path, "rb")
            else:
                 input_audio = speaker_wav_path

            # XTTS-v2 Model on Replicate
            # Latest known valid hash as of late 2023/2024
            model_id = "lucataco/xtts-v2:211d152e590755db4a4b9f7fcdf276c344b3f545"
            
            output = replicate.run(
                model_id,
                input={
                    "text": text,
                    "speaker_wav": input_audio,
                    "language": "en"
                }
            )
            
            print(f"XTTS Output: {output}")
            return output

        except Exception as e:
            # Catch Auth/Billing errors specifically to enable Simulation Mode cleanly
            if "401" in str(e) or "402" in str(e) or "Invalid token" in str(e):
                print(f"Simulation Mode: Replicate API limit reached or invalid token ({e}). Returning sample audio.")
                return MOCK_AUDIO_URL
            
            print(f"XTTS Error (Replicate): {e}")
            return MOCK_AUDIO_URL

    async def generate_lip_sync_video(self, image_url: str, audio_url: str) -> str:
        """
        Generates a lip-synced video using Replicate's Sadtalker model.
        (Previously used Wav2Lip, but switched due to model availability issues).
        """
        # Reliable fallback URL (Big Buck Bunny)
        MOCK_VIDEO_URL = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

        # Check for token validity immediately to avoid waiting for API error
        if not self.api_token or self.api_token == "your_token_here":
            print("Simulation Mode: REPLICATE_API_TOKEN missing. Returning sample video immediately.")
            return MOCK_VIDEO_URL

        print(f"Generating video for image: {image_url} and audio: {audio_url}")
        
        # Retry Logic for Rate Limits (429)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                input_args = {
                    "still": True,
                    "preprocess": "full",
                    "enhancer": "gfpgan"
                }
                
                # Handle Inputs (File handles or URLs)
                # Sadtalker uses 'source_image' and 'driven_audio'
                if os.path.exists(image_url):
                     input_args['source_image'] = open(image_url, "rb")
                else:
                     input_args['source_image'] = image_url
                
                if isinstance(audio_url, str) and (audio_url.startswith("http") or audio_url.startswith("https")):
                     input_args['driven_audio'] = audio_url
                elif os.path.exists(audio_url):
                     input_args['driven_audio'] = open(audio_url, "rb")
                else:
                     input_args['driven_audio'] = audio_url

                # CJWBW/Sadtalker: Latest Version Hash
                model_id = "cjwbw/sadtalker:a519cc0cfebaaeade068b23899165a11ec76aaa1d2b313d40d214f204ec957a3"

                output = replicate.run(
                    model_id,
                    input=input_args
                )
                
                print(f"Replicate output: {output}")
                return output

            except Exception as e:
                # Check for Rate Limit error
                if "429" in str(e) or "throttled" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        print(f"Rate limited. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                
                # Catch Auth/Billing errors specifically
                if "401" in str(e) or "402" in str(e) or "Invalid token" in str(e):
                    print(f"Simulation Mode: Replicate API limit reached or invalid token ({e}). Returning sample video.")
                    return MOCK_VIDEO_URL

                print(f"Replicate Error (Video): {e}")
                print("Falling back to mock video due to API error.")
                return MOCK_VIDEO_URL

ai_service = AIService()

