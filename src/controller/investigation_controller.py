from src.database.db_handler import load_evidence

async def process_question(question: str):
    try:
        evidences = load_evidence()
        user_q = question.replace(" ", "")
        
        found_contents = []
        
        for ev in evidences:
            keywords = ev.get("keywords", [])
            for kw in keywords:
                if kw in user_q:
                    found_contents.append({
                        "text": ev['content'],
                        "level": ev.get("level", 1)
                    })
                    break
        
        if not found_contents:
            return {
                "answer": f"'{question}'에 대한 단서는 발견되지 않았습니다.",
                "status": "no_clue",
                "level": 1
            }
        
        # 첫 번째 찾은 증거를 대표로 반환 (추후 정교화 가능)
        best_clue = found_contents[0]
        return {
            "answer": f"조사 결과: {best_clue['text']}",
            "status": "success",
            "level": best_clue['level']
        }
    except Exception as e:
        return {"answer": f"컨트롤러 에러: {str(e)}", "status": "error", "level": 1}