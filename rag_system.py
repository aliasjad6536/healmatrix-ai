"""
HealMatrix AI — RAG System (v2 — BGE + Reranker pipeline)

=============================================================================
UPGRADED RAG PIPELINE
=============================================================================
Knowledge Base Files (data/knowledge_base/)
        |
        v
RecursiveCharacterTextSplitter   (chunk_size=500, overlap=100)
        |
        v
BAAI/bge-small-en-v1.5           (embedding model — better than MiniLM)
        |
        v
FAISS                            (vector search — top 10 candidates)
        |
        v
cross-encoder/ms-marco-MiniLM-L-6-v2   (reranker — picks best 3)
        |
        v
Groq LLaMA 3.3 70B               (final therapeutic response)

=============================================================================
WHAT CHANGED FROM v1
=============================================================================
- Embeddings: all-MiniLM-L6-v2  ->  BAAI/bge-small-en-v1.5  (better retrieval)
- Splitting : text.split("\\n\\n")  ->  RecursiveCharacterTextSplitter
- NEW       : Cross-Encoder reranker re-scores the top-10 and keeps top-3
- Retrieval : fetch 10 candidates, rerank, send only the best 3 to the LLM
- KB        : auto-loads .txt/.json from data/knowledge_base/, then chunks them
- Perf      : embedder, reranker and FAISS index each load ONCE (singletons)

All models are CPU-friendly and require no GPU.
"""

import os
import json
from pathlib import Path
from typing import List, Optional
import threading

try:
    from config import GROQ_API_KEY, GROQ_MODEL  #
except ImportError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


EMBED_MODEL    = "BAAI/bge-small-en-v1.5"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval settings
FETCH_K        = 10   # candidates pulled from FAISS
FINAL_K        = 3    # best chunks after reranking, sent to the LLM
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 100

BASE_DIR        = Path.cwd()
KB_DIR          = BASE_DIR / "data" / "knowledge_base"
VECTORSTORE_DIR = BASE_DIR / "data" / "rag_vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)


_BUILTIN_KB: List[str] = [
    "Cognitive Behavioural Therapy (CBT) teaches that thoughts, feelings, and behaviours are interconnected. Changing unhelpful thoughts changes how you feel and act.",
    "Core CBT techniques: thought records, cognitive restructuring, behavioural activation, and exposure hierarchies.",
    "An automatic negative thought (ANT) is an instant, involuntary thought that is usually distorted. Common ANTs include catastrophising, mind-reading, and all-or-nothing thinking.",
    "Anxiety management: diaphragmatic breathing, progressive muscle relaxation (PMR), 5-4-3-2-1 grounding, and mindfulness meditation.",
    "4-7-8 breathing: inhale through nose for 4 seconds, hold for 7 seconds, exhale through mouth for 8 seconds. Repeat 3-4 times to calm the nervous system.",
    "Generalised Anxiety Disorder (GAD) involves chronic, excessive worry about many topics. CBT and mindfulness-based therapies are first-line treatments.",
    "Panic attacks peak within 10 minutes and are not dangerous, even though they feel terrifying. Slow breathing and grounding prevent escalation.",
    "Behavioural activation (BA) is an evidence-based treatment for depression. It involves scheduling and doing pleasurable or meaningful activities to break the inactivity-low mood cycle.",
    "For depression: establish a regular sleep-wake schedule, exercise 30+ minutes most days, maintain social connections, and eat balanced meals.",
    "Depression distorts thinking — common patterns include hopelessness, self-blame, and believing things will never improve. These are symptoms, not facts.",
    "Mindfulness means paying non-judgmental attention to the present moment. Regular practice (even 5-10 min/day) measurably reduces stress, anxiety, and depression.",
    "Body scan: lie down and slowly move attention from your feet to the top of your head, noticing sensations without judgement.",
    "STOP mindfulness technique: Stop, Take a breath, Observe your thoughts and feelings, Proceed with awareness.",
    "Sleep hygiene: consistent bedtime and wake time, no screens 1 hour before bed, cool dark room, no caffeine after 2pm, use the bed only for sleep.",
    "Stimulus control therapy: get out of bed if unable to sleep after 20 minutes. Only return when sleepy. Builds association between bed and sleep.",
    "PTSD symptoms: re-experiencing (flashbacks, nightmares), avoidance, negative cognitions, and hyperarousal. Trauma-focused CBT and EMDR are most effective.",
    "Grounding techniques help during flashbacks: name 5 objects in the room, hold something cold, focus on your feet on the floor.",
    "Crisis safety planning: identify personal warning signs, list coping strategies, name trusted contacts, and remove or secure means of self-harm.",
    "Active listening for someone in crisis: stay calm, reflect their feelings, ask directly about thoughts of suicide, do not leave them alone.",
    "Healthy communication: use 'I feel...' statements rather than 'You always...'. Active listening means reflecting back what you heard before responding.",
    "Setting limits is self-care. Clearly state your needs and what you will or won't accept, calmly and without apologising.",
    "DBT (Dialectical Behaviour Therapy) teaches four skill modules: Mindfulness, Distress Tolerance, Emotional Regulation, and Interpersonal Effectiveness.",
    "TIPP for emotional regulation: Temperature (cold water on face), Intense exercise, Paced breathing, Progressive muscle relaxation.",
    "ACT (Acceptance and Commitment Therapy): accept difficult inner experiences, defuse from unhelpful thoughts, and commit to actions guided by personal values.",
    "Self-care pillars: quality sleep, nutritious food, regular movement, social connection, creative expression, time in nature, and activities that feel meaningful.",
    "Journalling for mental health: write freely for 10 minutes about thoughts and feelings without editing. Helps process emotions and spot patterns.",
]

# Knowledge base loading + chunking
def _load_raw_documents() -> List[str]:
    """Load raw .txt / .json documents from data/knowledge_base/."""
    docs: List[str] = []
    if KB_DIR.exists():
        for f in sorted(KB_DIR.iterdir()):
            try:
                if f.suffix == ".txt":
                    docs.append(f.read_text(encoding="utf-8"))
                elif f.suffix == ".json":
                    data = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        docs.extend([str(item) for item in data if item])
                    elif isinstance(data, dict):
                        docs.extend([str(v) for v in data.values() if v])
            except Exception:
                pass
    return docs


def _chunk_documents(raw_docs: List[str]) -> List[str]:
    """
    Split raw documents into clean overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter (better than text.split("\\n\\n")).
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: List[str] = []
        for doc in raw_docs:
            chunks.extend(splitter.split_text(doc))
        return [c.strip() for c in chunks if c.strip()]
    except Exception:
        chunks = []
        for doc in raw_docs:
            chunks.extend([p.strip() for p in doc.split("\n\n") if p.strip()])
        return chunks

# Singletons
#
_index    = None
_embedder = None
_reranker = None
_kb_texts: List[str] = []
_lock = threading.Lock()


def _get_kb() -> List[str]:
    """Return chunked KB text (from disk if present, else built-in)."""
    global _kb_texts
    if not _kb_texts:
        raw = _load_raw_documents()
        if raw:
            _kb_texts = _chunk_documents(raw)
        if not _kb_texts:
            _kb_texts = _BUILTIN_KB
    return _kb_texts


def _get_embedder():
    """Load the BGE embedding model once."""
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer
                _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _get_reranker():
    """Load the cross-encoder reranker once."""
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _embed(texts: List[str]):
    """
    Encode texts with BGE. BGE recommends a query instruction prefix for
    retrieval queries to boost accuracy; documents are encoded as-is.
    """
    import numpy as np
    embedder = _get_embedder()
    emb = embedder.encode(texts, convert_to_numpy=True,
                          normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


def _ensure_index() -> bool:
    """Build or load the FAISS index. Returns True on success."""
    global _index
    if _index is not None:
        return True
    try:
        import faiss
        import numpy as np

        kb = _get_kb()
        index_file = VECTORSTORE_DIR / "kb_bge.index"
        meta_file  = VECTORSTORE_DIR / "kb_bge_size.txt"

        stored_size = int(meta_file.read_text()) if meta_file.exists() else -1

        if index_file.exists() and stored_size == len(kb):
            _index = faiss.read_index(str(index_file))
        else:
            emb = _embed(kb)
            # Inner product on normalised vectors = cosine similarity
            _index = faiss.IndexFlatIP(emb.shape[1])
            _index.add(emb)
            faiss.write_index(_index, str(index_file))
            meta_file.write_text(str(len(kb)))
        return True
    except Exception as e:
        print(f"   RAG index build failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Retrieval: FAISS search -> Cross-Encoder rerank -> best chunks
# ---------------------------------------------------------------------------
def get_relevant_context(question: str, k: int = FINAL_K) -> List[str]:
    """
    Two-stage retrieval:
      1. BGE + FAISS pull FETCH_K (10) candidate chunks.
      2. Cross-encoder reranker re-scores them; keep the best k (3).
    """
    kb = _get_kb()

    if not _ensure_index() or _index is None:
        return kb[:k]

    try:
        q_prefixed = "Represent this sentence for searching relevant passages: " + question
        q_vec = _embed([q_prefixed])
        scores, idx = _index.search(q_vec, min(FETCH_K, len(kb)))
        candidates = [kb[i] for i in idx[0] if 0 <= i < len(kb)]

        if not candidates:
            return kb[:k]

        #  Rerank with cross-encoder 
        try:
            reranker = _get_reranker()
            pairs = [[question, c] for c in candidates]
            rerank_scores = reranker.predict(pairs)
            ranked = sorted(zip(candidates, rerank_scores),
                            key=lambda x: x[1], reverse=True)
            return [c for c, _ in ranked[:k]]
        except Exception:
            # If reranker fails, fall back to FAISS order
            return candidates[:k]

    except Exception:
        return kb[:k]


# Improved therapist system prompt
_SYSTEM = (
    "You are Dr. Emily Hartman, a compassionate, trauma-informed mental health "
    "therapist with expertise in CBT, DBT, ACT, and mindfulness-based approaches.\n\n"
    "Guidelines:\n"
    "• Use the provided therapeutic context ONLY when it is relevant to the user's message.\n"
    "• If the context is not relevant, rely on your own therapeutic knowledge — do not force it.\n"
    "• Never invent facts, studies, or statistics (avoid hallucination).\n"
    "• Reflect the person's feelings BEFORE offering guidance.\n"
    "• Stay warm, empathetic, and non-judgmental at all times.\n"
    "• Apply CBT, DBT, ACT, and mindfulness principles where helpful.\n"
    "• Keep replies conversational — 150 to 250 words.\n"
    "• Never diagnose; recommend professional help when appropriate.\n"
    "• If any crisis or self-harm signal appears, immediately include crisis "
    "hotlines: 988 (US) and 0800-00-002 (Pakistan)."
)


def query_with_rag(question: str, k: int = FINAL_K) -> str:
    """Retrieve reranked context and generate a therapist response via Groq."""
    context_chunks = get_relevant_context(question, k=k)
    context_block  = "\n".join(f"• {c}" for c in context_chunks)

    user_msg = (
        f"Relevant therapeutic knowledge (use only if relevant):\n{context_block}\n\n"
        f"Patient message: {question}"
        if context_block else question
    )

    try:
        from groq import Groq
        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=500,
            temperature=0.72,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            "I'm here for you and I want to support you through this. "
            "I'm experiencing a brief technical difficulty right now.\n\n"
            f"Technical detail: {e}\n\n"
            "If you're in crisis, please reach out immediately:\n"
            "**988** (US) | **0800-00-002** (Pakistan)"
        )

try:
    _ensure_index()
    print("   RAG vectorstore   warmed up (BGE-small + FAISS + reranker)")
except Exception:
    pass