from fastapi import APIRouter, HTTPException, Depends, Form, File, UploadFile
from typing import List, Optional
from app.models.media import MediaCreate, MediaResponse
from app.db.mongodb import db
from app.api.endpoints.auth import get_current_user
from datetime import datetime

router = APIRouter()

@router.post("/", response_model=MediaResponse)
async def upload_media(
    elderly_username: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if current_user["role"] != "caretaker":
         raise HTTPException(status_code=403, detail="Only caretakers can upload media")
    
    # Verify elderly user exists
    elderly = await db.get_db().users.find_one({"username": elderly_username, "role": "elderly"})
    if not elderly:
        raise HTTPException(status_code=404, detail="Elderly user not found")

    UPLOAD_DIR = "uploads"
    import os
    import shutil
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Save file
    safe_filename = f"{datetime.utcnow().timestamp()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Construct URL (Localhost for now)
    # In production this would be S3.
    # For local expo, we need the machine IP (settings.API_URL or similar implied).
    # We will store the relative path or a constructed URL.
    # Let's store relative path, and frontend/backend can append base URL.
    # actually, let's store /static/filename if we mount static.
    # For simplicity, we'll return a full URL if possible, or just the filename?
    # Let's simple return the filename and have a static mount.
    
    file_url = f"/static/{safe_filename}" 

    media_doc = {
        "elderly_username": elderly_username,
        "type": "photo",
        "url": file_url,
        "description": description,
        "created_at": datetime.utcnow()
    }
    
    new_media = await db.get_db().media.insert_one(media_doc)
    created = await db.get_db().media.find_one({"_id": new_media.inserted_id})
    return created

@router.get("/", response_model=List[MediaResponse])
async def get_gallery(elderly_username: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    target_user = None
    
    if current_user["role"] == "elderly":
        target_user = current_user["username"]
    elif current_user["role"] == "caretaker":
        # Caretaker must specify which elderly user they want to see, OR see all their uploads?
        # The prompt says "visible on elderly".
        # If caretaker sends 'elderly_username', show that implementation.
        if elderly_username:
            target_user = elderly_username
        else:
             # Just return all for now or empty? Let's return all for this caretaker's uploads?
             # But media doesn't store 'uploaded_by'.
             # Better to require 'elderly_username' or return empty.
             if not elderly_username:
                 return []
    
    if not target_user:
        return []

    media = await db.get_db().media.find({"elderly_username": target_user}).to_list(100)
    return media

@router.get("/list", response_model=List[MediaResponse])
async def list_media(elderly_username: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    # Duplicate of get_gallery, keeping for backward compatibility if needed, but routing to same logic
    return await get_gallery(elderly_username, current_user)

@router.delete("/{media_id}")
async def delete_media(media_id: str, current_user: dict = Depends(get_current_user)):
    from bson import ObjectId
    import os

    try:
        oid = ObjectId(media_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid media ID")

    media_item = await db.get_db().media.find_one({"_id": oid})
    if not media_item:
        raise HTTPException(status_code=404, detail="Media not found")

    # Permission check: User must be the owner (elderly) or a caretaker
    if current_user["role"] == "elderly":
        if media_item["elderly_username"] != current_user["username"]:
             raise HTTPException(status_code=403, detail="Not authorized to delete this media")
    elif current_user["role"] == "caretaker":
        # Caretakers can delete any media for now, or check relationship?
        # Assuming caretaker has full access to managed elderly's data
        pass 
    else:
         raise HTTPException(status_code=403, detail="Not authorized")

    # Delete file from filesystem
    # The URL is stored as "/static/filename" in this specific implementation (see upload_media)
    # We need to resolve this back to the file system path "uploads/filename"
    # Note: upload_media implementation has: file_url = f"/static/{safe_filename}"
    
    if "url" in media_item:
        url_path = media_item["url"]
        if url_path.startswith("/static/"):
            filename = url_path.replace("/static/", "")
            file_path = os.path.join("uploads", filename)
            if os.path.exists(file_path):
                os.remove(file_path)
    
    # Delete from DB
    await db.get_db().media.delete_one({"_id": oid})
    
    return {"message": "Media deleted successfully"}
