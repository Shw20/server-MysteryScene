from fastapi import FastAPI
from pydantic import BaseModel
import json
import os

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

# 1. 에러가 났던 함수 정의 (반드시 파일 상단 혹은 호출부 이전에 위치)
def load_evidence():
    # 파일 경로를 정확하게 지정 (폴더 구조에 맞춰 확인 필요)
    file_path = "database/evidence.json"
    
    if not os.path.exists(file_path):
        print(f"에러: {file_path} 파일을 찾을 수 없습니다.")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON 로드 에러: {e}")
        return []

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        evidences = load_evidence()
        # 사용자 질문에서 공백 제거 (비교 정확도 향상)
        user_q = request.question.replace(" ", "")
        
        found_contents = []
        
        for ev in evidences:
            # keywords 필드에서 단어 하나씩 꺼내서 비교
            keywords = ev.get("keywords", [])
            for kw in keywords:
                # 키워드가 사용자 질문에 포함되어 있는지 확인
                if kw in user_q:
                    found_contents.append(ev['content'])
                    break # 증거 하나당 키워드 하나만 걸려도 통과
        
        if not found_contents:
            return {
                "answer": f"'{request.question}'에 대한 단서는 발견되지 않았습니다. (힌트: 시계, 알리바이, CCTV 등으로 질문해보세요)",
                "status": "no_clue"
            }
        
        # 중복 제거 후 답변 생성
        unique_results = list(set(found_contents))
        answer = f"조사 결과: {' '.join(unique_results)}"
        
        return {"answer": answer, "status": "success"}

    except Exception as e:
        # 에러 발생 시 상세 내용을 반환하도록 수정
        return {"answer": f"서버 내부 에러 발생: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)