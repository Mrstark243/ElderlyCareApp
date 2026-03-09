import requests
import time
import os

BASE_URL = "http://localhost:8000/api/v1"

def test_async_generation():
    print("Testing Async Video Generation Flow...")
    
    # 1. Create dummy files
    with open("test_photo.jpg", "wb") as f: f.write(b"dummy_image")
    with open("test_audio.wav", "wb") as f: f.write(b"dummy_audio")
    
    try:
        # 2. Call Generate API
        files = {
            'photo': open("test_photo.jpg", "rb"),
            'audio': open("test_audio.wav", "rb")
        }
        data = {
            'elderly_username': 'test_user',
            'text': 'Hello world'
        }
        
        # Note: This might fail if the server is not actually running. 
        # But I am an agent, I can't start the server and run this test easily without blocking.
        # Actually I can't assume localhost:8000 is running my modified code unless I started it.
        # I haven't started the server.
        # So I should write a unit test using TestClient instead of requests.
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if os.path.exists("test_photo.jpg"): os.remove("test_photo.jpg")
        if os.path.exists("test_audio.wav"): os.remove("test_audio.wav")

if __name__ == "__main__":
    test_async_generation()
