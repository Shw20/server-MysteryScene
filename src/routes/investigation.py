from fastapi import APIRouter
from pydantic import BaseModel
from src.controller.investigation_controller import (
    get_case_start,
    get_progress,
    process_question,
    submit_solution,
)

router = APIRouter(prefix="/investigation", tags=["Investigation"])


class QuestionRequest(BaseModel):
    """사용자가 채팅창에서 보내는 질문/행동 요청."""

    question: str
    session_id: str | None = None
    current_level: int | None = None


class StartRequest(BaseModel):
    """새 사건 시작과 진행도 조회에 공통으로 쓰는 세션 요청."""

    session_id: str | None = None


class SubmissionRequest(BaseModel):
    """사건 종결 보고서 제출 요청."""

    answer: str
    session_id: str | None = None
    culprit: str | None = None
    motive: str | None = None
    method: str | None = None


@router.post("/start")
async def start_investigation(request: StartRequest):
    """사건 소개와 초기 진행 상태를 내려준다."""
    return await get_case_start(request.session_id)


@router.post("/progress")
async def progress_investigation(request: StartRequest):
    """현재 세션의 진행도와 확보한 단서 목록을 내려준다."""
    return await get_progress(request.session_id)


@router.post("/ask")
async def ask_investigation(request: QuestionRequest):
    """질문/행동 입력을 컨트롤러에 넘겨 조사관 답변을 생성한다."""
    return await process_question(request.question, request.session_id)


@router.post("/submit")
async def submit_investigation(request: SubmissionRequest):
    """사용자 정답 후보를 채점하고, 해결 시 랭크와 해설을 내려준다."""
    return await submit_solution(
        answer=request.answer,
        session_id=request.session_id,
        culprit=request.culprit,
        motive=request.motive,
        method=request.method,
    )
