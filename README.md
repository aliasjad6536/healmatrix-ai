# HealMatrix AI

HealMatrix AI is a multimodal mental health support system that combines facial emotion detection, posture analysis, retrieval augmented conversation, and crisis detection into a single assistant. It was built as a Final Year Project.

## Overview

The system looks at a user's facial expression, posture, and the sentiment of what they type, then pulls relevant therapeutic guidance from a knowledge base and generates an empathetic, context aware response through a language model based reasoning engine. It can also recognize language that suggests a mental health crisis and alert a designated emergency contact through WhatsApp or an automated phone call.

## How the system works

When a user sends a message, they can optionally include a webcam snapshot for facial emotion analysis and a voice note that gets transcribed automatically. The message and any accompanying image go through several stages.

First, facial emotion is analyzed. The emotion detection module runs a lightweight FER model as its first pass and falls back to DeepFace when that first pass isn't confident enough, returning a detected emotion along with a confidence score.

Second, posture is analyzed. The system uses MediaPipe to detect body landmarks in the image and a fine tuned MobileNetV2 classifier to categorize the posture as confident, tense, slouched, or neutral.

Third, the text itself is analyzed for sentiment, classifying it as positive, negative, or neutral using a RoBERTa based sentiment model, with a simple keyword based fallback if that model isn't available.

Fourth, the message is scanned for crisis language. If high, medium, or low severity crisis indicators are found, the system flags this immediately so it can be handled with priority.

Fifth, the system retrieves relevant knowledge. The user's message is embedded using a fine tuned BGE embedding model and used to search a FAISS vector index built from mental health counseling material, pulling back the most relevant passages.

Sixth, all of this information, the detected emotion, posture, sentiment, crisis history, and retrieved knowledge, is passed into the reasoning engine, which decides what kind of response the user actually needs. The options are to reassure, to guide them through a coping technique, to escalate because of a crisis, to refer them to a therapist, to ask a clarifying question, or to offer motivation.

Seventh, that decision and its supporting context are sent to a large language model, currently Llama 3.3 70B through the Groq API, which writes the actual response in an empathetic, therapist like tone. If a crisis was detected, hotline information is automatically added to the response.

Eighth, if the crisis is severe, the backend automatically sends a WhatsApp alert and can place an automated phone call to a designated emergency contact through Twilio.

Finally, if the user needs professional help, the system can search Google Maps for nearby therapists and clinics and present that information as simple contact cards.

## Core components

**Emotion detection** uses a fine tuned EfficientNet B2 model trained on an enhanced version of the FER 2013 dataset across seven emotion classes: angry, disgust, fear, happy, sad, surprise, and neutral. Training included dropout, image augmentation such as rotation, color jitter, and random erasing, and validation based early stopping. The model currently reaches about sixty nine percent test accuracy, with a weighted F1 score of about 0.69.

**Posture detection** uses a fine tuned MobileNetV2 model trained on the MPII Human Pose dataset, with four posture classes: confident, tense, slouched, and neutral. Since MPII doesn't come with posture labels directly, labels are derived from a geometric rule applied to MediaPipe detected landmarks, looking at shoulder tilt, forward head position, torso lean, and head height. The model currently reaches about fifty one percent test accuracy, more than double the twenty five percent random chance baseline for four classes. It's worth being upfront that the ground truth here is a geometric rule rather than human annotated labels, so this measures how well the model learned that rule rather than validated real world posture judgment, and the neutral class in particular still needs better threshold tuning.

**Retrieval augmented generation** uses the BAAI bge small English model, both in its base form and a fine tuned version, together with a FAISS vector index built over mental health counseling conversation data. Retrieval quality currently sits around seventy five percent recall at four results, with a mean reciprocal rank of about 0.59.

**The reasoning engine** runs on Llama 3.3 70B through Groq and chooses between reassuring the user, guiding them through a technique, escalating a crisis, referring them to a therapist, asking a clarifying question, or offering motivation, based on everything the system has learned about their current state.

**Crisis detection** classifies messages into none, low, medium, or high severity based on language patterns, automatically adds hotline information to responses when needed, and can trigger WhatsApp or voice alerts to an emergency contact through Twilio for high severity cases.

**Supporting modules** round out the system: sentiment analysis using a RoBERTa based model with a keyword fallback, voice transcription using Groq's Whisper model, a therapist finder built on the Google Maps Places API, and a Flask based backend that handles WhatsApp webhooks through Twilio and ngrok.

## Evaluation summary

Emotion detection with EfficientNet B2 reaches about sixty nine percent accuracy with a weighted F1 score of about 0.69. Posture detection with MobileNetV2 reaches about fifty one percent accuracy. RAG retrieval with the BGE model reaches about seventy five percent recall at four results with a mean reciprocal rank of about 0.59.

Full evaluation details, confusion matrices, and per class breakdowns are available in evaluation_summary.csv, emotion_confusion_matrix.png, and pose_confusion_matrix.png.

## Technology used

The machine learning side is built on PyTorch, torchvision, EfficientNet PyTorch, MediaPipe, and scikit learn. Retrieval and natural language processing rely on sentence transformers, FAISS, and Hugging Face Transformers. The language model side uses the Groq API for both Llama 3.3 70B and Whisper. The backend runs on Flask together with the Twilio API and Google Maps API. Training data comes from Hugging Face Datasets, specifically FER 2013, MPII Human Pose, and a mental health counseling conversation dataset.

## Project structure

The repository contains the reasoning engine in agi_engine.py, the Flask backend and WhatsApp webhook in backend.py, crisis severity classification in crisis_detection.py, the production emotion inference pipeline in emotion_detection.py, the emotion model training script in emotion_finetuning.py, the production posture inference pipeline in pose_detection.py, the posture model training script in pose_finetuning.py, the RAG retrieval logic in rag_system.py, the BGE embedding fine tuning script in rag_finetuning.py, text sentiment classification in sentiment_analysis.py, the Google Maps based therapist search in therapist_finder.py, Whisper based transcription in voice_input.py, environment based configuration in config.py, the training and evaluation notebook healmatrix-ai.ipynb, and the final evaluation metrics in evaluation_summary.csv.

## Setup

Start by cloning the repository and moving into the project folder.

```bash
git clone https://github.com/aliasjad6536/healmatrix-ai.git
cd healmatrix-ai
```

Then create a virtual environment and install the dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Next, create a .env file in the project root with the required API keys: GROQ_API_KEY, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER, TWILIO_WHATSAPP_NUMBER, EMERGENCY_CONTACT, EMERGENCY_WHATSAPP, GOOGLE_MAPS_API_KEY, and NGROK_AUTHTOKEN.

Finally, run the backend.

```bash
python backend.py
```

## Training and evaluation

Model training and evaluation are done through healmatrix-ai.ipynb, which is designed to run in a Kaggle or Colab GPU environment. It covers loading the datasets, fine tuning the emotion and posture models, evaluating the RAG embeddings, and generating the confusion matrices and summary metrics included in this repository.

## Known limitations

The emotion detection accuracy of around sixty nine percent is in line with what's typically published for single model CNNs on FER 2013 style benchmarks, and is limited by the low resolution of the images and genuine ambiguity between some classes, particularly fear and surprise, and sad and neutral.

The posture detection ground truth is derived from a geometric rule rather than human annotated labels, so its accuracy reflects how consistently the model reproduces that rule rather than validated agreement with real world posture judgments.

The comparison between the fine tuned and pretrained RAG models depends on the fine tuned checkpoint actually being present locally. Without it, the evaluation quietly falls back to using the pretrained model for both sides of the comparison.

## License

Add your license here, for example MIT.
