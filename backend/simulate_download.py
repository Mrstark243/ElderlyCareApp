import time
import sys
import random

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("Status: Loading/Downloading TTS Model (~2GB)...", flush=True)
time.sleep(1)

total_size = 2048 # MB
current_size = 0

# Simulate tqdm format: 
#  10%|█         | 204.8M/2.05G [00:10<01:30, 19.8MB/s]
# We will print to stderr as tqdm usually does, but our Popen merges it to stdout.

chars = ["|", "/", "-", "\\"]

while current_size < total_size:
    current_size += 20
    percent = int((current_size / total_size) * 100)
    
    # Create valid tqdm-like string
    downloaded = f"{current_size}M"
    total = "2.0G"
    bar = "=" * (percent // 5) + ">" + " " * ((100 - percent) // 5)
    
    # Carriage return to overwrite line
    output = f"\r{percent}%|{bar}| {downloaded}/{total} [00:xx<00:xx, 20.0MB/s]"
    
    sys.stdout.write(output)
    sys.stdout.flush()
    time.sleep(0.1)

print("\r\nStatus: Model Loaded.", flush=True)
