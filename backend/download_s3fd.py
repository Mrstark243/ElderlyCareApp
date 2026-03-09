import os
import requests
import sys

# URL for the weights
URL = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
TARGET_DIR = os.path.join(os.getcwd(), "ai_models", "Wav2Lip", "face_detection", "detection", "sfd")
TARGET_FILE = os.path.join(TARGET_DIR, "s3fd-619a316812.pth")
# Also duplicate simple name just in case
TARGET_FILE_SIMPLE = os.path.join(TARGET_DIR, "s3fd.pth")

def download_file():
    print(f"Checking target directory: {TARGET_DIR}")
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory does not exist: {TARGET_DIR}")
        return

    print(f"Downloading s3fd weights from {URL}...")
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()
        
        with open(TARGET_FILE, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Download complete: {TARGET_FILE}")
        
        # Copy to simple name too
        import shutil
        shutil.copy(TARGET_FILE, TARGET_FILE_SIMPLE)
        print(f"Copied to: {TARGET_FILE_SIMPLE}")
        
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_file()
