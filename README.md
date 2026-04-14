# MysteryScene_server

.env 파일 생성

# 가상환경 생성 및 활성화

```
python -m venv .venv
source .venv/Scripts/activate
```

# 필수 패키지 설치

```
pip install fastapi uvicorn openai python-dotenv numpy
```

# 실행

```
uvicorn main:app --reload
```

http://127.0.0.1:8000/docs
