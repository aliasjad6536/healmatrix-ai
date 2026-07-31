# 🧠 HealMatrix AI

A multimodal mental health support system combining facial emotion detection, posture analysis, retrieval-augmented conversational support, and crisis detection into a single assistant. Built as a Final Year Project. 🎓

---

## 📋 Overview

HealMatrix AI observes a user's facial expression 😊, posture 🧍, and message sentiment 💬, retrieves relevant therapeutic guidance from a knowledge base 📚, and generates an empathetic, context-aware response through an LLM-based reasoning engine 🤖. It also detects crisis language 🚨 and can alert a designated emergency contact via WhatsApp/voice call 📞.

---

## 🔄 System Workflow

1. **Input Capture** 📥 — User sends a text message, optionally with a webcam snapshot (facial emotion) and/or voice note (transcribed via Whisper).
2. **Facial Emotion Analysis** 😐➡️😢 — `emotion_detection.py` runs FER (fast path) and falls back to DeepFace when confidence is low, returning a detected emotion + confidence score.
3. **Posture Analysis** 🧍➡️🧎 — `pose_detection.py` uses MediaPipe landmarks + the trained MobileNetV2 classifier to detect confident / tense / slouched / neutral posture.
4. **Sentiment Analysis** 💬 — `sentiment_analysis.py` classifies the text as positive / negative / neutral using a RoBERTa sentiment model (with keyword fallback).
5. **Crisis Detection** 🚨 — `crisis_detection.py` scans the message for high/medium/low severity crisis language and flags it immediately if found.
6. **Knowledge Retrieval (RAG)** 📚🔍 — `rag_system.py` embeds the query with a fine-tuned BGE model, searches a FAISS vector index of therapeutic knowledge, and retrieves the most relevant passages.
7. **AGI Reasoning & Decision** 🧠⚙️ — `agi_engine.py` combines emotion + posture + sentiment + crisis history + RAG context to decide the best action: REASSURE 💙 / GUIDE 📚 / ESCALATE 🚨 / REFER_THERAPIST 🏥 / ASSESS 🔍 / MOTIVATE ⚡.
8. **Response Generation** ✍️ — The decision + context is sent to Llama 3.3 70B (via Groq API), which generates an empathetic, therapist-style response, with crisis hotlines automatically prepended if needed.
9. **Emergency Alerting** ☎️ — For high-severity crisis messages, `backend.py` triggers a WhatsApp alert and/or automated voice call to the emergency contact via Twilio.
10. **Therapist Referral** 🗺️ — If needed, `therapist_finder.py` searches Google Maps for nearby therapists/clinics and returns contact cards.

---

## 🧩 Core Components

### 1. 😊 Emotion Detection
- **Model:** EfficientNet-B2 (fine-tuned), with FER + DeepFace as the production fallback pipeline
- **Dataset:** FER-2013 (enhanced), 7 classes — angry 😠, disgust 🤢, fear 😨, happy 😄, sad 😢, surprise 😲, neutral 😐
- **Training:** dropout regularization, augmentation (rotation, color jitter, random erasing), validation-based early stopping
- **Result:** ~69% test accuracy (weighted F1 ~0.69)

### 2. 🧍 Pose / Posture Detection
- **Model:** MobileNetV2 (fine-tuned), 4 classes — confident 💪, tense 😬, slouched 😔, neutral 😐
- **Dataset:** MPII Human Pose, with labels derived from a geometric rule applied to MediaPipe landmarks (shoulder tilt, forward-head offset, torso lean, head height)
- **Result:** ~51% test accuracy (2x+ better than the 25% random-chance baseline)
- **Known limitation:** ground truth is a geometric-rule proxy, not human-annotated; "neutral" class needs further threshold tuning

### 3. 📚 RAG (Retrieval-Augmented Generation)
- **Embedding model:** BAAI/bge-small-en-v1.5 (base + fine-tuned)
- **Vector store:** FAISS
- **Knowledge base:** mental health counseling conversation datasets
- **Result:** Recall@4 ~75%, MRR ~0.59

### 4. 🤖 AGI Reasoning Engine
- **LLM:** Llama 3.3 70B via Groq API
- **Decision states:** REASSURE 💙 / GUIDE 📚 / ESCALATE 🚨 / REFER_THERAPIST 🏥 / ASSESS 🔍 / MOTIVATE ⚡
- Combines all multimodal signals to select and generate the most appropriate therapeutic response

### 5. 🚨 Crisis Detection
- Keyword/pattern-based severity classification (none / low / medium / high)
- Automatic hotline injection into responses ☎️
- WhatsApp/voice emergency alerts via Twilio for high-severity cases

### 6. 🛠️ Supporting Modules
- 💬 Sentiment analysis (cardiffnlp/twitter-roberta-base-sentiment-latest + keyword fallback)
- 🎙️ Voice input transcription (Groq Whisper)
- 🗺️ Therapist finder (Google Maps Places API)
- 🌐 WhatsApp webhook backend (Flask + Twilio + ngrok)

---

## 📊 Evaluation Summary

| Model | Metric | Score |
|---|---|---|
| 😊 Emotion Detection (EfficientNet-B2) | Accuracy | ~69% |
| 😊 Emotion Detection (EfficientNet-B2) | F1 (weighted) | ~0.69 |
| 🧍 Pose Detection (MobileNetV2) | Accuracy | ~51% |
| 📚 RAG Retrieval (BGE) | Recall@4 | ~75% |
| 📚 RAG Retrieval (BGE) | MRR | ~0.59 |

Full details, confusion matrices, and per-class breakdowns are in `evaluation_summary.csv`, `emotion_confusion_matrix.png`, and `pose_confusion_matrix.png`. 📈

---

## 🧰 Tech Stack

- **ML/DL:** PyTorch, torchvision, EfficientNet-PyTorch, MediaPipe, scikit-learn
- **NLP/Retrieval:** sentence-transformers, FAISS, Hugging Face Transformers
- **LLM:** Groq API (Llama 3.3 70B, Whisper)
- **Backend:** Flask, Twilio API, Google Maps API
- **Data:** Hugging Face Datasets (FER-2013, MPII Human Pose, mental health counseling conversations)

---

## 📁 Project Structure

healmatrix-ai/ - agi_engine.py (LLM-based reasoning and response generation) - backend.py (Flask server + WhatsApp webhook) - crisis_detection.py (Crisis severity classification) - emotion_detection.py (Production emotion inference: FER/DeepFace) - emotion_finetuning.py (Emotion model training script) - pose_detection.py (Production posture inference) - pose_finetuning.py (Pose model training script) - rag_system.py (RAG retrieval logic) - rag_finetuning.py (BGE embedding fine-tuning) - sentiment_analysis.py (Text sentiment classification) - therapist_finder.py (Google Maps therapist search) - voice_input.py (Whisper-based transcription) - config.py (Environment-based configuration) - healmatrix-ai.ipynb (Training/evaluation notebook) - evaluation_summary.csv (Final metrics) - requirements.txt

---

## ⚙️ Setup

**1. Clone the repository:**
```bash
git clone https://github.com/aliasjad6536/healmatrix-ai.git
cd healmatrix-ai
```

**2. Create a virtual environment and install dependencies:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Create a `.env` file with the required API keys:** GROQ_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, TWILIO_WHATSAPP_NUMBER, EMERGENCY_CONTACT, EMERGENCY_WHATSAPP, GOOGLE_MAPS_API_KEY, NGROK_AUTHTOKEN

**4. Run the backend:**
```bash
python backend.py
```

---

## 🧪 Training and Evaluation

Model training and evaluation are run through `healmatrix-ai.ipynb` (designed for Kaggle/Colab GPU environments). It covers dataset loading, fine-tuning for emotion and pose models, RAG embedding evaluation, and generation of the confusion matrices and summary metrics included in this repository.

---

## ⚠️ Known Limitations

- Emotion detection accuracy (~69%) is in line with published results on FER-2013-style benchmarks for single-model CNNs, limited by low image resolution and inherent label ambiguity (e.g. fear/surprise, sad/neutral).
- Pose detection ground truth is derived from a geometric rule rather than human-annotated labels, so accuracy reflects self-consistency with that rule rather than validated real-world posture judgment.
- RAG fine-tuned vs. pretrained comparisons require the fine-tuned checkpoint to be present locally; without it, evaluation falls back to the pretrained model for both sides.

---

## 📄 License

Add your license here (e.g., MIT). 📝

---

⭐ If this project helped you, consider giving it a star!
