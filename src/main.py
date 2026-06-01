import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.routes import investigation


app = FastAPI(
    title="Crime Scene AI",
    description="RAG based mystery game API",
)

# 배포 도메인은 CORS_ALLOW_ORIGINS 환경변수에 쉼표로 추가한다.
# 로컬 개발에서는 Vite/정적 미리보기 포트를 기본 허용한다.
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,"
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 단서 이미지는 개발용 web_preview와 배포용 web_demo가 함께 사용한다.
assets_dir = Path("web_preview/assets")
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

# 사건 시작/질문/진행도/정답 제출 API를 등록한다.
app.include_router(investigation.router)


@app.get("/")
def demo_page():
    """배포용 데모 페이지를 루트 URL에서 바로 보여준다."""
    demo_index = Path("web_demo/index.html")
    if demo_index.exists():
        return FileResponse(demo_index)
    return {"status": "ok", "message": "Backend is running"}


@app.get("/health")
def health_check():
    """배포 플랫폼과 프론트가 서버 생존 여부를 확인하는 엔드포인트."""
    return {"status": "ok", "message": "Backend is running"}


if __name__ == "__main__":
    # 로컬에서 python src/main.py로 실행할 때만 reload 모드로 띄운다.
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
