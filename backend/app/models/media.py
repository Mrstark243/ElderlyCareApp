from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId
from app.models.user import PyObjectId

class MediaBase(BaseModel):
    elderly_username: str
    type: str # "photo", "video", "cloned_video"
    url: str
    description: Optional[str] = None

class MediaCreate(MediaBase):
    pass

class MediaResponse(MediaBase):
    id: Optional[PyObjectId] = Field(alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        simple_json = True
        json_encoders = {ObjectId: str, datetime: str}
