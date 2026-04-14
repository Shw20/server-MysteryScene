import json
import os

def load_evidence():
    # 현재 파일 위치 기준이 아니라 프로젝트 루트 기준으로 경로 설정
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_path, "database", "evidence.json")
    
    if not os.path.exists(file_path):
        print(f"🚨 에러: {file_path}를 찾을 수 없습니다.")
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ JSON 로드 에러: {e}")
        return []