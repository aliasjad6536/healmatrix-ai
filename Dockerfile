# HealMatrix AI — Dockerfile
# Build: docker build -t healmatrix-ai .
# Run:   docker run -p 7860:7860 healmatrix-ai

FROM python:3.11-slim

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
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system \
    groq \
    gradio \
    deepface \
    tf-keras \
    opencv-python-headless \
    sentence-transformers \
    faiss-cpu \
    transformers \
    twilio \
    googlemaps \
    gtts \
    langchain-community \
    langchain-text-splitters \
    numpy \
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

# here i creta data dir
RUN mkdir -p data/chat_logs data/emotions data/crisis_alerts \
    data/session data/rag_vectorstore data/knowledge_base

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
    CMD curl -f http://localhost:7860 || exit 1

CMD ["python", "main.py"]