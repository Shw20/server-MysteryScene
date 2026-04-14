from fastapi import FastAPI
from src.routes import investigation
import uvicorn

app = FastAPI(title="Crime Scene AI", description="RAG 기반 추리 게임 API")

# 라우터 등록
app.include_router(investigation.router)

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend is running"}

if __name__ == "__main__":
    # 위치에 상관없이 실행될 수 있도록 설정
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)