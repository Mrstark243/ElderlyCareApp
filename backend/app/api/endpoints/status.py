from fastapi import APIRouter, HTTPException
from app.services.job_store import job_store

router = APIRouter()

@router.get("/{job_id}")
async def get_job_status(job_id: str):
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
