# HealMatrix AI — Dockerfile
# Terminal 1 — Docker container chalao (agar already build hai)
#docker compose up -d
#docker ps

# Terminal 2 — public link banao
#ssh -R 80:localhost:7860 nokey@localhost.run


FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip uv

RUN uv pip install --system \
    torch==2.4.1+cpu \
    torchvision==0.19.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system \
    groq==1.5.0 \
    gradio==6.19.0 \
    deepface==0.0.100 \
    tf-keras \
    opencv-python-headless \
    sentence-transformers==5.6.0 \
    faiss-cpu \
    transformers \
    twilio==9.10.9 \
    googlemaps==4.10.0 \
    gtts==2.5.4 \
    langchain-community \
    langchain-text-splitters \
    numpy==1.26.4 \
    pandas \
    pillow

COPY main.py .
COPY config.py .
COPY agi_engine.py .
COPY voice_input.py .
COPY sentiment_analysis.py .
COPY therapist_finder.py .
COPY crisis_detection.py .
COPY emotion_detection.py .
COPY pose_detection.py .
COPY rag_system.py .
COPY build_knowledge_base.py .

COPY checkpoints/ ./checkpoints/

RUN mkdir -p data/chat_logs data/emotions data/crisis_alerts \
    data/session data/rag_vectorstore data/knowledge_base

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:7860 || exit 1

CMD ["python", "main.py"]
