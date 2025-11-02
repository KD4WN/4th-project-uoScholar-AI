# ---- Runtime (single stage; 간단/빠름) ----
FROM python:3.11-slim

# 시스템 기본 패키지(필요 최소)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements가 있다면 먼저 복사/설치해서 캐시 극대화
COPY requirements-chatbot.txt /app/
RUN pip install --no-cache-dir -r requirements-chatbot.txt

# SentenceTransformer 모델 미리 다운로드 (런타임 시 다운로드 방지)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sroberta-multitask')"

# 소스 복사
COPY . /app

# 서비스 포트(원하면 바꿀 수 있음)
ENV PORT=9000
# 앱 모듈 경로: 기본 main:app (예: main.py 안의 app 객체). 필요하면 배포 전에 APP_MODULE=... 로 바꾸면 됨.
 ENV APP_MODULE=src.uosai.chat.chatbot:app

EXPOSE 9000

# uvicorn으로 실행 (FastAPI/Starlette/ASGI 호환)
CMD ["sh", "-c", "python -m uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT}"]
