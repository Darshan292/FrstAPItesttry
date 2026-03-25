from pydantic import BaseModel

class GenerateResponse(BaseModel):
    session_id: str
    questions: dict