from fastapi import APIRouter
from pydantic import BaseModel
from src.controller.investigation_controller import process_question

router = APIRouter(prefix="/investigation", tags=["Investigation"])

class QuestionRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_investigation(request: QuestionRequest):
    return await process_question(request.question)