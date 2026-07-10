"""
HealMatrix AI — Voice Input Module
Speech-to-Text using Groq Whisper API (whisper-large-v3-turbo)
No extra packages needed — uses groq which is already installed.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

try:
    from config import GROQ_API_KEY, GROQ_WHISPER_MODEL
except ImportError:
    GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
    GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text using Groq Whisper.
    Supports: .mp3, .wav, .m4a, .webm, .ogg
    """
    if not audio_path or not Path(audio_path).exists():
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(Path(audio_path).name, f.read()),
                model=GROQ_WHISPER_MODEL,
                language="en",
                response_format="text",
            )
        result = str(transcription).strip()
        print(f"  Transcribed: {result[:80]}")
        return result
    except Exception as e:
        print(f"    Whisper failed: {e}")
        return f"[Transcription error: {e}]"


def save_gradio_audio(audio_data) -> Optional[str]:
    """Save Gradio audio input to a temp file and return its path."""
    if audio_data is None:
        return None
    try:
        if isinstance(audio_data, str):
            return audio_data
        if isinstance(audio_data, tuple):
            import numpy as np
            import wave

            sample_rate, audio_array = audio_data
            if audio_array is None or len(audio_array) == 0:
                return None
            audio_array = audio_array.astype(np.int16)
            data_dir = Path(__file__).parent / "data"
            data_dir.mkdir(exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav", dir=str(data_dir))
            with wave.open(tmp.name, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_array.tobytes())
            return tmp.name
    except Exception as e:
        print(f"   Audio save error: {e}")
        return None