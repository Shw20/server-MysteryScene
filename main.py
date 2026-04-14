from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI()

# 프론트엔드 레포지토리가 다르므로 CORS 허용 설정 필수
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시에는 프론트엔드 주소만 허용하도록 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str
    current_level: int = 1

@app.get("/")
def read_root():
    return {"status": "Backend Server is Running!"}

@app.post("/chat")
async def chat(request: ChatRequest):
    # 여기에 실제 RAG 로직과 OpenAI API 호출이 들어갈 예정입니다.
    # 지금은 테스트용 반환값만 설정합니다.
    return {
        "found": True, 
        "content": f"백엔드에서 수신됨: {request.user_input} (레벨 {request.current_level})"
    }