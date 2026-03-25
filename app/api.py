from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from app.engine import generate_assessment
from app.utils import save_temp_file
from app.config import API_KEY
import uuid

app = FastAPI(title="AI Question Generator API")

SESSIONS = {}

# ---------------- AUTH ----------------
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------- ENDPOINT ----------------
@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    difficulty: str = Form("Medium"),
    mcq: int = Form(5),
    short: int = Form(5),
    long: int = Form(2),
    x_api_key: str = Header(...)
):
    verify_api_key(x_api_key)

    session_id = str(uuid.uuid4())

    file_path = save_temp_file(file.filename, await file.read())

    data = generate_assessment(file_path, difficulty, mcq, short, long)

    SESSIONS[session_id] = data

    return {
        "session_id": session_id,
        "questions": data
    }


@app.get("/health")
def health():
    return {"status": "running"}