from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

# 데이터 로드 함수
def load_evidence():
    with open("database/evidence.json", "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    evidences = load_evidence()
    user_q = request.question
    
    # 가짜 검색 로직: 질문에 포함된 단어가 증거에 있는지 확인
    found_contents = []
    for ev in evidences:
        # 간단한 키워드 매칭 테스트
        keywords = ["A", "시계", "혈흔", "C", "연구실"]
        for kw in keywords:
            if kw in user_q and kw in ev['content']:
                found_contents.append(ev['content'])
    
    if not found_contents:
        answer = f"'{user_q}'에 대한 직접적인 단서는 아직 발견되지 않았습니다. 다른 질문을 해보세요."
    else:
        context = " ".join(list(set(found_contents)))
        answer = f"조사 결과: {context}"

    return {"answer": answer}