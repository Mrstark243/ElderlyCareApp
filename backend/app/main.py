from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api import api_router
from app.core.config import settings
from app.db.mongodb import db
import os
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.connect()
    yield
    db.close()

app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION, lifespan=lifespan)

from app.api.endpoints import auth, ai, media, status, reminders

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/static", StaticFiles(directory="uploads"), name="static")

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
# app.include_router(elderly.router, prefix="/api/v1/elderly", tags=["elderly"])
# app.include_router(caretaker.router, prefix="/api/v1/caretaker", tags=["caretaker"])
app.include_router(ai.router, prefix="/api/v1/ai", tags=["ai"])
app.include_router(media.router, prefix="/api/v1/media", tags=["media"])
app.include_router(status.router, prefix="/api/v1/status", tags=["status"])
app.include_router(reminders.router, prefix="/api/v1/reminders", tags=["reminders"])

@app.get("/")
async def root():
    return {"message": "Elderly Care App API is running"}
