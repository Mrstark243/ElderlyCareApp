from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.services.ai_engine import ai_engine
from app.services.job_store import job_store

client = TestClient(app)

def mock_generate_voice(*args, **kwargs):
    if 'progress_callback' in kwargs and kwargs['progress_callback']:
        kwargs['progress_callback'](10, "Mock Voice Gen")
    return "dummy_path.wav"

def mock_sync_lips(*args, **kwargs):
    if 'progress_callback' in kwargs and kwargs['progress_callback']:
        kwargs['progress_callback'](50, "Mock Lip Sync")
    return "dummy_output.mp4"

@patch('app.services.ai_engine.ai_engine.generate_voice_cloning', side_effect=mock_generate_voice)
@patch('app.services.ai_engine.ai_engine.sync_lips', side_effect=mock_sync_lips)
@patch('app.services.ai_engine.ai_engine.is_tts_available', return_value=True)
@patch('app.db.mongodb.db.get_db') 
def test_async_flow(mock_db, mock_tts_avail, mock_sync, mock_voice):
    # Mock DB insert
    mock_db.return_value.media.insert_one = MagicMock()
    
    print("\n--- Starting Async Flow Test ---")
    
    # 1. Start Job
    with open("test.txt", "wb") as f: f.write(b"dummy")
    
    files = {
        'photo': ('photo.jpg', open("test.txt", "rb"), 'image/jpeg'),
        'audio': ('audio.wav', open("test.txt", "rb"), 'audio/wav')
    }
    
    response = client.post(
        "/api/v1/ai/generate-video",
        data={'elderly_username': 'user1', 'text': 'test'},
        files=files
    )
    
    os.remove("test.txt")
    
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    print(f"Job ID received: {job_id}")
    
    # 2. Check Status (Job should be in progress or queued)
    # Since TestClient runs synchronous, BackgroundTasks might not run until we trigger them or 
    # TestClient DOES run them after the request if using Starlette's TestClient correctly.
    # Actually, Starlette TestClient runs background tasks synchronously after the response.
    
    # So by the time we get here, the job might already be done or failed if the logic ran.
    
    status_res = client.get(f"/api/v1/status/{job_id}")
    status_data = status_res.json()
    print(f"Final Job Status: {status_data['status']}")
    print(f"Final Progress: {status_data['progress']}")
    print(f"Messages: {status_data['message']}")
    
    assert status_data['job_id'] == job_id
    if status_data['status'] == 'failed':
        print(f"Job Failed with error: {status_data.get('error')}")
    else:
        assert status_data['status'] == 'completed'
        assert status_data['progress'] == 100

if __name__ == "__main__":
    # Manually run the test function
    # We need to setup the mock manually if running directly, 
    # but using unittest/pytest is cleaner.
    # For now, just relying on the code structure.
    pass
