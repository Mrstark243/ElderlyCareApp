from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import Optional
from app.services.ai_engine import ai_engine
from app.services.job_store import job_store
import shutil
import os
import uuid
import traceback
from datetime import datetime
from app.db.mongodb import db

router = APIRouter()

# Helper function for background processing
def process_video_generation(job_id: str, elderly_username: str, text: Optional[str], temp_photo_path: str, temp_audio_path: str):
    logger = ai_engine.logger # Reuse logger
    
    def update_progress(p, msg):
        job_store.update_job(job_id, progress=p, message=msg)

    try:
        job_store.update_job(job_id, status="processing", progress=5, message="Starting process...")
        
        # 2. Voice Cloning (if text is provided)
        audio_for_sync = temp_audio_path
        temp_generated_audio_path = None
        
        if text and text.strip():
            # Check TTS availability
            if not ai_engine.is_tts_available():
                raise Exception("TTS Service unavailable.")
            
            job_store.update_job(job_id, message="Generating Voice...")
            temp_generated_audio_path = ai_engine.generate_voice_cloning(
                text, 
                temp_audio_path, 
                progress_callback=update_progress
            )
            audio_for_sync = temp_generated_audio_path
            job_store.update_job(job_id, progress=35, message="Voice Generated")

        # 3. Lip Sync
        job_store.update_job(job_id, message="Syncing Lips...")
        # We pass the photo as the "video_path" (face) to Wav2Lip
        output_path = ai_engine.sync_lips(
            temp_photo_path, 
            audio_for_sync,
            progress_callback=update_progress
        )
        
        # 4. Save to DB (Media Gallery)
        filename = os.path.basename(output_path)
        file_url = f"/static/ai_generated/{filename}"
        
        # We need to run DB operations in a loop since this is a sync function running in a thread
        # But pymongo is thread-safe, or we can use the sync client if db handles it.
        # Wait, 'db.get_db()' typically returns a motor async client. 
        # We cannot await async calls easily in a sync background task without a loop.
        # It's better to just skip the DB insert here OR run it in a new event loop.
        # Actually, let's keep it simple: just update the job result. 
        # The Frontend can then save it or we provide a separate 'finalize' step.
        # OR better: use `asyncio.run` for the DB part if it's isolated.
        
        # Since the original code used 'await db.get_db().media.insert_one', it implies async DB.
        # We are in a sync background function.
        # Valid approach: Store the result in the job. The frontend can redirect the user to the gallery, 
        # but the specific DB record won't be there unless we insert it.
        
        # Alternative: Make 'process_video_generation' async and add it to BackgroundTasks?
        # FastAPI BackgroundTasks can accept async functions!
        # So let's make this function async.
        pass # Placeholder to switch to async def below
        
    except Exception as e:
        traceback.print_exc()
        job_store.update_job(job_id, status="failed", error=str(e), message="Processing Failed")
        return

    # Success (handled in async wrapper below)

async def async_process_video_generation(job_id: str, elderly_username: str, text: Optional[str], temp_photo_path: str, temp_audio_path: str):
    temp_generated_audio_path = None
    try:
        job_store.update_job(job_id, status="processing", progress=1, message="Initializing...")
        
        # Wrap sync blocking calls in run_in_executor to avoid blocking the event loop
        import asyncio
        from functools import partial
        
        loop = asyncio.get_event_loop()
        
        def update_progress(p, msg):
            # multiple updates might conflict if not careful, but for a dictionary it's atomic enough for this
            job_store.update_job(job_id, progress=p, message=msg)

        # 2. Voice Cloning
        audio_for_sync = temp_audio_path
        
        if text and text.strip():
             job_store.update_job(job_id, message="Cloning Voice...", progress=5)
             # run sync function in thread
             temp_generated_audio_path = await loop.run_in_executor(
                 None, 
                 partial(ai_engine.generate_voice_cloning, text, temp_audio_path, progress_callback=update_progress)
             )
             audio_for_sync = temp_generated_audio_path

        # 3. Lip Sync
        job_store.update_job(job_id, message="Syncing Lips...", progress=35)
        output_path = await loop.run_in_executor(
            None,
            partial(ai_engine.sync_lips, temp_photo_path, audio_for_sync, progress_callback=update_progress)
        )
        
        # 4. DB Insert
        # WE DO NOT INSERT INTO DB HERE ANYMORE
        # The frontend will call /approve-video to do that.
        
        filename = os.path.basename(output_path)
        file_url = f"/static/ai_generated/{filename}"
        
        job_store.update_job(
            job_id, 
            status="completed", 
            progress=100, 
            message="Completed", 
            result={"video_url": file_url}
        )
        
    except Exception as e:
        traceback.print_exc()
        job_store.update_job(job_id, status="failed", error=str(e), message=f"Error: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(temp_photo_path):
            try: os.remove(temp_photo_path)
            except: pass
        if os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except: pass
        if temp_generated_audio_path and os.path.exists(temp_generated_audio_path):
            try: os.remove(temp_generated_audio_path)
            except: pass

@router.post("/generate-video")
async def generate_video(
    background_tasks: BackgroundTasks,
    elderly_username: str = Form(...),
    text: Optional[str] = Form(None),
    photo: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    # 1. Save inputs synchronously (fast)
    job_id = str(uuid.uuid4())
    job_store.create_job(job_id)
    
    # Create temp files
    # Create temp files
    temp_photo_path = os.path.abspath(f"uploads/temp_{job_id}_photo.jpg")
    temp_audio_path = os.path.abspath(f"uploads/temp_{job_id}_audio.wav")
    
    with open(temp_photo_path, "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    # Queue background task
    background_tasks.add_task(
        async_process_video_generation, 
        job_id, 
        elderly_username, 
        text, 
        temp_photo_path, 
        temp_audio_path
    )
    
    return {"message": "Video generation started", "job_id": job_id}

@router.post("/voice-cloning")
async def generate_voice_cloning(
    text: str = Form(...),
    speaker_wav: UploadFile = File(...)
):
    # This remains sync/simple for now as per user request to focus on video, 
    # but could be updated similarly.
    if not ai_engine.is_tts_available():
        raise HTTPException(status_code=503, detail="TTS Service unavailable (missing dependencies).")

    temp_wav_path = f"uploads/temp_{uuid.uuid4()}.wav"
    with open(temp_wav_path, "wb") as buffer:
        shutil.copyfileobj(speaker_wav.file, buffer)
        
    try:
        output_path = ai_engine.generate_voice_cloning(text, temp_wav_path)
        filename = os.path.basename(output_path)
        return {"message": "Voice generated successfully", "audio_url": f"/static/ai_generated/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

@router.post("/lip-sync")
async def sync_lips(
    video: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    # Keeping this simple for now too
    temp_video_path = f"uploads/temp_{uuid.uuid4()}.mp4"
    temp_audio_path = f"uploads/temp_{uuid.uuid4()}.wav"
    
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
        
    try:
        output_path = ai_engine.sync_lips(temp_video_path, temp_audio_path)
        filename = os.path.basename(output_path)
        return {"message": "Lip sync completed", "video_url": f"/static/ai_generated/{filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@router.post("/approve-video")
async def approve_video(
    elderly_username: str = Form(...),
    video_url: str = Form(...),
    description: Optional[str] = Form(None)
):
    try:
        if not description:
            description = "AI Generated Video"
            
        media_doc = {
            "elderly_username": elderly_username,
            "type": "cloned_video",
            "url": video_url,
            "description": description,
            "created_at": datetime.utcnow()
        }
        
        await db.get_db().media.insert_one(media_doc)
        return {"message": "Video approved and sent to elderly gallery."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


