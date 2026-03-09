import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    print(f"IP: {ip}")
except Exception as e:
    print(f"Error: {e}")
    # Fallback
    print(f"Fallback IP: {socket.gethostbyname(socket.gethostname())}")
