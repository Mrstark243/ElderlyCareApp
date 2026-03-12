from pydantic import BaseModel, Field
from typing import Optional
from bson import ObjectId
from app.models.user import PyObjectId

class GeoFenceBase(BaseModel):
    elderly_username: str
    latitude: float
    longitude: float
    radius: float # in meters
    is_active: bool = True
    caretaker_username: Optional[str] = None

class GeoFenceCreate(GeoFenceBase):
    pass

class GeoFenceInDB(GeoFenceBase):
    pass

class GeoFenceResponse(GeoFenceBase):
    id: Optional[PyObjectId] = Field(alias="_id")

    class Config:
        simple_json = True
        json_encoders = {ObjectId: str}
