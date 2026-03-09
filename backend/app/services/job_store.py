from typing import Dict, Any, Optional
import datetime

# Simple in-memory job store
# In a production app, use Redis or a Database
class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, job_id: str):
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Queued...",
            "created_at": datetime.datetime.utcnow(),
            "result": None,
            "error": None
        }

    def update_job(self, job_id: str, status: str = None, progress: int = None, message: str = None, result: Any = None, error: str = None):
        if job_id in self._jobs:
            if status:
                self._jobs[job_id]["status"] = status
            if progress is not None:
                self._jobs[job_id]["progress"] = progress
            if message:
                self._jobs[job_id]["message"] = message
            if result:
                self._jobs[job_id]["result"] = result
            if error:
                self._jobs[job_id]["error"] = error
                self._jobs[job_id]["status"] = "failed"
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

job_store = JobStore()
