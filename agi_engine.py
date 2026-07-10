"""
HealMatrix AI — AGI Reasoning Engine
Multi-step therapeutic agent using LangChain.
Dynamically decides: escalate / guide / reassure / refer therapist
based on user emotional state, crisis level, and conversation history.
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    from config import GROQ_API_KEY, GROQ_MODEL
except ImportError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   = "llama-3.3-70b-versatile"

# AGI Decision States
AGI_STATES = {
    "REASSURE":       "User needs emotional validation and comfort",
    "GUIDE":          "User needs CBT/DBT technique guidance",
    "ESCALATE":       "User is in crisis — trigger emergency alert",
    "REFER_THERAPIST":"User needs professional help referral",
    "ASSESS":         "Gather more information about user state",
    "MOTIVATE":       "User needs motivational support",
}

# System prompt for AGI agent
_AGI_SYSTEM = """You are HealMatrix AGI — an advanced emotional wellness intelligence system.
You are Dr. Emily Hartman, a compassionate trauma-informed therapist with expertise in CBT, DBT, ACT, and mindfulness.

Your AGI reasoning process:
1. ASSESS the user's emotional state from their message
2. IDENTIFY the primary need: comfort, guidance, crisis help, or professional referral
3. SELECT the best therapeutic approach
4. RESPOND with empathy FIRST, then guidance

Decision framework:
- If crisis signals detected → ESCALATE (include hotlines immediately)
- If severe distress → REASSURE + suggest breathing techniques
- If seeking understanding → GUIDE with CBT/DBT tools
- If recurring/chronic issues → REFER_THERAPIST
- If mild stress → MOTIVATE + self-care suggestions

Style rules:
• Always reflect feelings BEFORE offering advice
• Keep responses 150-250 words
• Use therapeutic language, not clinical jargon
• Never diagnose — always recommend professional help when appropriate
• For ANY crisis signal: include 988 (US) and 0800-00-002 (Pakistan)

Current emotional context will be provided. Use it to personalize your response."""


def _build_context_block(
    emotion: Optional[str] = None,
    sentiment: Optional[str] = None,
    posture: Optional[str] = None,
    crisis_history: Optional[List] = None,
) -> str:
    """Build multimodal emotional context string for AGI reasoning."""
    parts = []
    if emotion and emotion != "neutral":
        parts.append(f"Facial emotion detected: {emotion}")
    if sentiment and sentiment != "neutral":
        parts.append(f"Text sentiment: {sentiment}")
    if posture and posture != "neutral":
        parts.append(f"Body language: {posture}")
    if crisis_history:
        recent = [c["severity"] for c in crisis_history[-3:]]
        if recent:
            parts.append(f"Recent crisis alerts: {', '.join(recent)}")
    if not parts:
        return ""
    return "Multimodal emotional context:\n" + "\n".join(f"• {p}" for p in parts)


def decide_action(
    message: str,
    severity: str,
    emotion: Optional[str] = None,
    sentiment: Optional[str] = None,
    posture: Optional[str] = None,
    crisis_history: Optional[List] = None,
) -> str:
    """
    AGI multi-step decision: determine the best action for this user state.
    Returns one of: ESCALATE, REASSURE, GUIDE, REFER_THERAPIST, MOTIVATE, ASSESS
    """
    # Rule-based fast path for crisis
    if severity == "high":
        return "ESCALATE"
    if severity == "medium":
        return "REASSURE"

    # Multimodal signals
    neg_signals = 0
    if emotion in ["sad", "fear", "angry", "disgust"]:
        neg_signals += 1
    if sentiment == "negative":
        neg_signals += 1
    if posture in ["slouched", "closed", "tense"]:
        neg_signals += 1

    # Recurring crisis history
    if crisis_history and len(crisis_history) >= 3:
        return "REFER_THERAPIST"

    if neg_signals >= 2:
        return "REASSURE"
    if neg_signals == 1:
        return "GUIDE"

    # Check message keywords
    msg_lower = message.lower()
    if any(w in msg_lower for w in ["therapist","professional","doctor","clinic","hospital"]):
        return "REFER_THERAPIST"
    if any(w in msg_lower for w in ["how","what","explain","technique","help me","teach"]):
        return "GUIDE"
    if any(w in msg_lower for w in ["feel","feeling","today","lately","always","never"]):
        return "ASSESS"

    return "MOTIVATE"


def agi_query(
    message: str,
    severity: str,
    conversation_history: Optional[List[Dict]] = None,
    emotion: Optional[str] = None,
    sentiment: Optional[str] = None,
    posture: Optional[str] = None,
    crisis_history: Optional[List] = None,
    rag_context: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Main AGI reasoning function.
    Combines multimodal signals + RAG context + conversation history
    to generate the most appropriate therapeutic response.

    Returns
    -------
    (response_text, action_taken)
    """
    action = decide_action(message, severity, emotion, sentiment, posture, crisis_history)
    context_block = _build_context_block(emotion, sentiment, posture, crisis_history)

    # Build action-specific instruction
    action_instructions = {
        "ESCALATE":        "IMMEDIATELY provide crisis hotlines. Be calm, warm, direct. Do not delay hotline info.",
        "REASSURE":        "Start by validating their feelings deeply. Then offer one grounding technique.",
        "GUIDE":           "Teach one specific CBT or DBT technique relevant to their message. Be practical.",
        "REFER_THERAPIST": "Gently suggest professional help. Explain why it would benefit them specifically.",
        "MOTIVATE":        "Offer encouragement and one small actionable step they can take today.",
        "ASSESS":          "Ask one gentle clarifying question to better understand their emotional state.",
    }

    action_instruction = action_instructions.get(action, "Respond with empathy and support.")

    # Build full system prompt
    system = _AGI_SYSTEM
    if context_block:
        system += f"\n\n{context_block}"
    if rag_context:
        system += f"\n\nRelevant therapeutic knowledge:\n{rag_context}"
    system += f"\n\nAGI Decision: {action} — {action_instructions[action]}"

    # Build messages
    messages = [{"role": "system", "content": system}]

    # Add conversation history (last 8 turns)
    if conversation_history:
        for msg in conversation_history[-8:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)})

    messages.append({"role": "user", "content": message})

    try:
        from groq import Groq
        client   = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=500,
            temperature=0.72,
        )
        return response.choices[0].message.content.strip(), action

    except Exception as e:
        fallback = (
            "I'm here for you and I want to support you through this. "
            "I'm experiencing a brief technical difficulty.\n\n"
            f"If you're in crisis: **988** (US) | **0800-00-002** (Pakistan)\n\n"
            f"Technical detail: {e}"
        )
        return fallback, action


def get_action_badge(action: str) -> str:
    """Return HTML badge showing AGI decision."""
    colors = {
        "ESCALATE":        "#fb7185",
        "REASSURE":        "#63b3ed",
        "GUIDE":           "#2dd4bf",
        "REFER_THERAPIST": "#a78bfa",
        "MOTIVATE":        "#34d399",
        "ASSESS":          "#fbbf24",
    }
    icons = {
        "ESCALATE":        "🚨",
        "REASSURE":        "💙",
        "GUIDE":           "📚",
        "REFER_THERAPIST": "🏥",
        "MOTIVATE":        "⚡",
        "ASSESS":          "🔍",
    }
    color = colors.get(action, "#94a3b8")
    icon  = icons.get(action, "🤖")
    desc  = AGI_STATES.get(action, action)
    return (
        f'<span style="background:rgba(99,179,237,0.08);border:1px solid {color}44;'
        f'color:{color};padding:0.2rem 0.8rem;border-radius:20px;font-size:0.78rem">'
        f'{icon} AGI: {action}</span>'
        f'<span style="color:#64748b;font-size:0.75rem;margin-left:0.5rem">{desc}</span>'
    )