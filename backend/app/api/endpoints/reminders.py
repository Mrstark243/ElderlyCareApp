from fastapi import APIRouter, HTTPException, status, Depends
from typing import List
from app.models.reminder import ReminderCreate, ReminderResponse, ReminderInDB
from app.models.user import UserResponse
from app.db.mongodb import db
from app.api.endpoints.auth import get_current_user
from bson import ObjectId

router = APIRouter()

@router.post("/", response_model=ReminderResponse)
async def create_reminder(reminder: ReminderCreate, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "caretaker":
        raise HTTPException(status_code=403, detail="Only caretakers can create reminders")
    
    # Verify assigned_to exists
    elderly = await db.get_db().users.find_one({"username": reminder.assigned_to, "role": "elderly"})
    if not elderly:
        raise HTTPException(status_code=404, detail="Elderly user not found")

    reminder_in_db = ReminderInDB(**reminder.dict(), created_by=current_user["username"])
    new_reminder = await db.get_db().reminders.insert_one(reminder_in_db.dict())
    created_reminder = await db.get_db().reminders.find_one({"_id": new_reminder.inserted_id})
    return created_reminder

@router.get("/", response_model=List[ReminderResponse])
async def get_reminders(current_user: dict = Depends(get_current_user)):
    # If caretaker, show reminders created by them
    # If elderly, show reminders assigned to them
    if current_user["role"] == "caretaker":
        reminders = await db.get_db().reminders.find({"created_by": current_user["username"]}).to_list(100)
    else:
        reminders = await db.get_db().reminders.find({"assigned_to": current_user["username"]}).to_list(100)
    return reminders
