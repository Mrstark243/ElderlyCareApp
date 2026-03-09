import subprocess
import os
import uuid
from typing import Optional
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class AIEngine:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.backend_dir = os.path.dirname(os.path.dirname(self.base_dir))
        self.ai_models_dir = os.path.join(self.backend_dir, "ai_models")
        self.wav2lip_dir = os.path.join(self.ai_models_dir, "Wav2Lip")
        self.output_dir = os.path.join(self.backend_dir, "uploads", "ai_generated")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check for TTS script availability
        self.tts_script = os.path.join(self.backend_dir, "app", "scripts", "run_tts.py")
        if not os.path.exists(self.tts_script):
            logger.warning("TTS script not found!")
            
    def is_tts_available(self) -> bool:
        """Checks if TTS is installed and available."""
        # Simple check: try to run the script with --help
        try:
             subprocess.run(["python", self.tts_script, "--help"], check=True, capture_output=True)
             return True
        except Exception:
             return False

    def _run_command_with_progress(self, cmd: list, cwd: str = None, progress_callback=None, start_percent=0, end_percent=100):
        """
        Runs a command and estimates progress, handling \r for tqdm-style progress bars.
        """
        # We need to read binary (or unbuffered text) to catch \r
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, # We still want text decoding
            bufsize=1, # Line buffered
            universal_newlines=True
        )
        
        logger.info(f"Started process: {' '.join(cmd)}")
        
        full_output = []
        buffer = ""
        while True:
            # Read one character at a time to catch \r
            char = process.stdout.read(1)
            
            if not char and process.poll() is not None:
                break
            
            if not char:
                continue
            
            # Append to full output for debugging
            full_output.append(char)
            
            if char == '\r' or char == '\n':
                line = buffer.strip()
                buffer = "" # Reset buffer
                
                if not line: continue
                
                logger.debug(f"CMD Output: {line}")
                
                # Check for tqdm-like progress (e.g. " 50%|...| 100M/200M")
                if "%" in line and "|" in line:
                    try:
                        # Extract percentage
                        parts = line.split('%')[0].split()
                        if parts:
                            p_val = int(parts[-1])
                            
                            # Extract Size Info (e.g., 100M/1.0G)
                            # Looking for pattern: number[unit]/number[unit]
                            import re
                            # Regex for size pattern (e.g., 1.2M/2.0G, 100k/10M)
                            size_match = re.search(r'(\d+(?:\.\d+)?[a-zA-Z]+/\d+(?:\.\d+)?[a-zA-Z]+)', line)
                            size_info = size_match.group(1) if size_match else ""
                            
                            # Map p_val (0-100) to (start_percent-end_percent)
                            relative_p = (p_val / 100.0) * (end_percent - start_percent)
                            current_p = start_percent + relative_p
                            
                            msg = f"Downloading: {p_val}%"
                            if size_info:
                                msg += f" ({size_info})"
                            
                            # Just "Processing" if it's not a download (heuristic: if 'Downloading' in line or just generic)
                            if "Downloading" not in line and not size_info:
                                 msg = f"Processing: {p_val}%"
                            
                            if progress_callback:
                                progress_callback(int(current_p), msg)
                    except Exception as e:
                        # logger.debug(f"Parsing error: {e}")
                        pass

                # Fallback: specific log messages
                if "Status: Initializing TTS" in line:
                    if progress_callback: progress_callback(5, "Initializing AI Engine...")
                elif "Status: Warning - Running on CPU" in line:
                    # Don't change progress, but maybe update message?
                    if progress_callback: progress_callback(int(current_p), "Running on CPU (Slow)...")
                elif "Status: Loading/Downloading TTS Model" in line:
                    # Give a hint this is a big file
                    if progress_callback: progress_callback(10, "Starting Model Download (~2GB)...")
                elif "Status: Model Loaded" in line:
                    if progress_callback: progress_callback(25, "Model Ready")
                elif "Status: Generating Audio" in line:
                    if progress_callback: progress_callback(28, "Synthesizing Speech...")
                
            else:
                buffer += char
        
        rc = process.poll()
        if rc != 0:
            # Reconstruct full output
            full_log = "".join(full_output)
            logger.error(f"Command failed. Log: {full_log}")
            raise RuntimeError(f"Command failed with return code {rc}. Last Output: {full_log[-500:]}")
            
        return

    def generate_voice_cloning(self, text: str, speaker_wav: str, progress_callback=None) -> str:
        """
        Runs TTS generation.
        """
        output_filename = f"tts_{uuid.uuid4()}.wav"
        output_path = os.path.join(self.output_dir, output_filename)
        
        script_path = os.path.join(self.backend_dir, "app", "scripts", "run_tts.py")
        
        cmd = [
            "python", script_path,
            "--text", text,
            "--speaker_wav", speaker_wav,
            "--out_path", output_path
        ]
        
        if progress_callback:
            progress_callback(5, "Starting Voice Cloning...")
            
        self._run_command_with_progress(cmd, progress_callback=progress_callback, start_percent=5, end_percent=30)
        
        if progress_callback:
            progress_callback(30, "Voice Cloning Complete")

        if not os.path.exists(output_path):
            raise RuntimeError("TTS Output file was not created.")
            
        return output_path

    def detect_face(self, image_path: str) -> Optional[list]:
        """
        Detects face using OpenCV Haar Cascade.
        Returns [y1, y2, x1, x2] (top, bottom, left, right) or None.
        """
        try:
            if not hasattr(cv2, 'data'):
                logger.warning("CV2 data missing, skipping face detection")
                return None
                
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            if not os.path.exists(cascade_path):
                logger.warning("Haar cascade not found")
                return None
                
            face_cascade = cv2.CascadeClassifier(cascade_path)
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                logger.info("No face detected by OpenCV")
                return None
            
            # Select largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Add padding to ensure chin/mouth is included
            h_pad = int(h * 0.1) # 10% padding vertical
            w_pad = int(w * 0.05) # 5% padding horizontal
            
            y1 = max(0, y - h_pad)
            y2 = min(img.shape[0], y + h + h_pad)
            x1 = max(0, x - w_pad)
            x2 = min(img.shape[1], x + w + w_pad)
            
            logger.info(f"Face box with padding: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
            return [y1, y2, x1, x2]
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return None

    def sync_lips(self, video_path: str, audio_path: str, progress_callback=None) -> str:
        """
        Runs Wav2Lip.
        """
        output_filename = f"synced_{uuid.uuid4()}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        
        script_path = os.path.join(self.wav2lip_dir, "inference.py")
        checkpoint_path = os.path.join(self.wav2lip_dir, "checkpoints", "wav2lip_gan.pth")
        
        if not os.path.exists(checkpoint_path):
             checkpoint_path = os.path.join(self.wav2lip_dir, "checkpoints", "wav2lip.pth")
             
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Wav2Lip checkpoint not found at {checkpoint_path}")

        cmd = [
            "python", script_path,
            "--checkpoint_path", checkpoint_path,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--pads", "0", "0", "0", "0",
            "--resize_factor", "1"
        ]
        
        # Attempt explicit face detection
        box = self.detect_face(video_path)
        if box:
            # Wav2Lip args.box expects: top bottom left right
            # We return: y1(top), y2(bottom), x1(left), x2(right)
            cmd.extend(["--box", str(box[0]), str(box[1]), str(box[2]), str(box[3])])
        
        if progress_callback:
            progress_callback(35, "Starting Lip Sync...")

        self._run_command_with_progress(cmd, cwd=self.wav2lip_dir, progress_callback=progress_callback, start_percent=35, end_percent=95)

        if not os.path.exists(output_path):
             raise RuntimeError("Wav2Lip output file was not created.")

        # POST-PROCESSING: Resize for Android Compatibility
        # Android decoders often fail with weird resolutions (e.g. 1836x1376).
        # We force it to a standard width (e.g. 720) and even height.
        
        resized_output_path = output_path.replace(".mp4", "_resized.mp4")
        
        resize_cmd = [
            "ffmpeg", "-y",
            "-i", output_path,
            "-vf", "scale=720:-2", # Scale width to 720, height auto (must be even)
            "-c:v", "libx264",
            "-profile:v", "baseline", # High compatibility profile
            "-level", "3.0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            resized_output_path
        ]
        
        try:
            logger.info("Resizing video for Android compatibility...")
            # We use a simple run here since it's fast
            subprocess.run(resize_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Replace original with resized
            if os.path.exists(resized_output_path):
                os.remove(output_path)
                os.rename(resized_output_path, output_path)
                logger.info("Video resized successfully.")
        except Exception as e:
            logger.warning(f"Failed to resize video (ffmpeg might be missing): {e}")

        return output_path

ai_engine = AIEngine()
