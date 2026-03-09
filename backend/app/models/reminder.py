from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId
from app.models.user import PyObjectId

class ReminderBase(BaseModel):
    title: str
    description: Optional[str] = None
    time: datetime
    assigned_to: str # Username of elderly
    audio_alert_url: Optional[str] = None
    status: str = "pending" # pending, completed

class ReminderCreate(ReminderBase):
    pass

class ReminderInDB(ReminderBase):
    created_by: str # Username of caretaker
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ReminderResponse(ReminderBase):
    id: Optional[PyObjectId] = Field(alias="_id")
    created_by: str

    class Config:
        simple_json = True
        json_encoders = {ObjectId: str, datetime: str}
