from fastapi import APIRouter, HTTPException, Depends
from app.models.geofence import GeoFenceCreate, GeoFenceResponse, GeoFenceInDB
from app.db.mongodb import db
from app.api.endpoints.auth import get_current_user
from math import radians, cos, sin, asin, sqrt
import requests

router = APIRouter()

# Haversine formula to calculate distance
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 * 1000 # Radius of earth in meters
    return c * r


@router.post("/", response_model=GeoFenceResponse)
async def create_geofence(geofence: GeoFenceCreate, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "caretaker":
        raise HTTPException(status_code=403, detail="Only caretakers can set geofences")
    
    # Check if a geofence already exists for this elderly user, if so update it or just replace
    existing = await db.get_db().geofences.find_one({"elderly_username": geofence.elderly_username})
    if existing:
        await db.get_db().geofences.delete_one({"_id": existing["_id"]})

    # Inject the caretaker username so we know who to alert
    geofence_dict = geofence.dict()
    geofence_dict["caretaker_username"] = current_user["username"]

    new_geofence = await db.get_db().geofences.insert_one(geofence_dict)
    created = await db.get_db().geofences.find_one({"_id": new_geofence.inserted_id})
    return created

@router.post("/check-location")
async def check_location(location: dict, current_user: dict = Depends(get_current_user)):
    # location = {"latitude": float, "longitude": float}
    # Logic: Find geofence for this user -> Check distance -> If out, return alert: True
    geofence = await db.get_db().geofences.find_one({"elderly_username": current_user["username"]})
    if not geofence:
        return {"alert": False, "message": "No geofence set"}
    
    dist = haversine(location["longitude"], location["latitude"], geofence["longitude"], geofence["latitude"])
    if dist > geofence["radius"]:
        # Alert the caretaker
        caretaker_username = geofence.get("caretaker_username")
        if caretaker_username:
            caretaker = await db.get_db().users.find_one({"username": caretaker_username})
            if caretaker and caretaker.get("expo_push_token"):
                push_token = caretaker["expo_push_token"]
                message = {
                    "to": push_token,
                    "title": "📍 Geofence Alert",
                    "body": f"{current_user['username']} has left their safe zone! (Currently {int(dist)}m away from center)",
                    "sound": "default"
                }
                try:
                    requests.post("https://exp.host/--/api/v2/push/send", json=message)
                except Exception as e:
                    print(f"Failed to send push notification: {e}")

        return {"alert": True, "distance_outside": dist - geofence["radius"], "message": "User is outside the geofence!"}
    
    return {"alert": False, "message": "User is within safe zone"}
