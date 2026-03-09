import os
import sys

try:
    import gdown
except ImportError:
    print("gdown not installed. Please pip install gdown")
    sys.exit(1)

def download_wav2lip_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.dirname(base_dir) # app -> backend
    wav2lip_dir = os.path.join(backend_dir, "ai_models", "Wav2Lip")
    checkpoints_dir = os.path.join(wav2lip_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"Downloading models to {checkpoints_dir}...")

    # Wav2Lip GAN
    gan_id = "1DuvF-4J6h-N1YwPh9Z9W8v55h1Y6Z9m_"
    gan_path = os.path.join(checkpoints_dir, "wav2lip_gan.pth")
    if not os.path.exists(gan_path):
        print("Downloading wav2lip_gan.pth...")
        try:
            gdown.download(id=gan_id, output=gan_path, quiet=False)
        except Exception as e:
            print(f"Failed to download GAN model: {e}")

    # Standard Wav2Lip
    std_id = "1BscS-y9q_-1d9a2y9q_-1d9a2" # This ID is illustrative, usually it's different.
    # Using the main one from repo usually.
    # Let's just stick to GAN as it's better.

    # Face Detection Model
    # s3fd-619a316812.pth is usually needed in face_detection/
    
    face_det_dir = os.path.join(wav2lip_dir, "face_detection")
    s3fd_path = os.path.join(face_det_dir, "s3fd-619a316812.pth")
    # URL: https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
    # But often it's downloaded automatically by face_alignment lib or we need to place it.
    # Wav2Lip repo's face_detection/api.py line 42 loads it from there.
    # We should trying downloading it if possible.
    
    if not os.path.exists(s3fd_path):
        print("Downloading s3fd face detection model...")
        url = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
        import requests
        try:
            r = requests.get(url, allow_redirects=True)
            open(s3fd_path, 'wb').write(r.content)
        except Exception as e:
             print(f"Failed to download s3fd: {e}")

if __name__ == "__main__":
    download_wav2lip_models()
