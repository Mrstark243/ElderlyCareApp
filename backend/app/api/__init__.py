from fastapi import APIRouter
from app.api.endpoints import auth, reminders, geofence, media, ai

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(reminders.router, prefix="/reminders", tags=["reminders"])
api_router.include_router(geofence.router, prefix="/geofence", tags=["geofence"])
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])
