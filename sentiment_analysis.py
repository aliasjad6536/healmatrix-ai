"""
HealMatrix AI — Sentiment Analysis Module
Uses cardiffnlp/twitter-roberta-base-sentiment-latest (transformers).
Falls back to keyword-based analysis if model unavailable.
"""

from typing import Dict, Tuple

_LABEL_MAP = {
    "POSITIVE":"positive","NEGATIVE":"negative","NEUTRAL":"neutral",
    "LABEL_0":"negative","LABEL_1":"neutral","LABEL_2":"positive",
    "POS":"positive","NEG":"negative","NEU":"neutral",
}
_INSIGHTS = {
    "positive":("#34d399"," Your message reflects a **positive emotional tone**.","Keep nurturing what brings you this energy!"),
    "neutral": ("#63b3ed"," Your message reflects a **neutral emotional tone**.","You seem balanced. A great state for reflection and clear thinking."),
    "negative":("#fb7185"," Your message reflects a **negative emotional tone**.","It sounds like you may be going through something difficult. You don't have to face it alone."),
}
_POS = ["happy","great","good","wonderful","excited","grateful","love","joy","better","hope","calm","peaceful"]
_NEG = ["sad","depressed","anxious","stressed","hopeless","worthless","empty","tired","alone","scared","angry","hurt","pain"]

_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from transformers import pipeline
            _pipeline = pipeline("sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=3)
            print("   Sentiment pipeline loaded")
        except Exception as e:
            print(f"    Sentiment fallback ({e})")
            _pipeline = "fallback"
    return _pipeline

def _keyword(text):
    t=text.lower()
    pos=sum(1 for w in _POS if w in t)
    neg=sum(1 for w in _NEG if w in t)
    if neg>pos: return "negative", min(0.6+neg*0.05,0.95)
    if pos>neg: return "positive", min(0.6+pos*0.05,0.95)
    return "neutral", 0.6

def analyze_sentiment(text: str) -> Dict:
    if not text or not text.strip():
        return {"sentiment":"neutral","confidence":0.0,"result_text":"No text provided.","color":"#63b3ed"}
    pipe=_get_pipeline()
    sentiment="neutral"; confidence=0.6; all_scores={}
    if pipe=="fallback" or pipe is None:
        sentiment,confidence=_keyword(text)
        all_scores={sentiment:confidence}
    else:
        try:
            preds=sorted(pipe(text[:512])[0],key=lambda x:x["score"],reverse=True)
            top=preds[0]
            sentiment=_LABEL_MAP.get(top["label"].upper(),"neutral")
            confidence=float(top["score"])
            all_scores={_LABEL_MAP.get(p["label"].upper(),p["label"].lower()):round(p["score"],3) for p in preds}
        except Exception:
            sentiment,confidence=_keyword(text)
            all_scores={sentiment:confidence}
    color,desc,tip=_INSIGHTS.get(sentiment,_INSIGHTS["neutral"])
    bar_lines=[]
    for lbl,score in sorted(all_scores.items(),key=lambda x:x[1],reverse=True):
        pct=score*100; bar="█"*int(pct/5)+"░"*(20-int(pct/5))
        bar_lines.append(f"  {lbl.capitalize():10s} [{bar}] {pct:.1f}%")
    result_text="\n".join([
        f" Sentiment     : **{sentiment.capitalize()}**",
        f" Confidence    : {confidence*100:.1f}%","",desc,"",
        "Score Breakdown:",*bar_lines,"",f"Tip: {tip}",
    ])
    return {"sentiment":sentiment,"confidence":confidence,"all_scores":all_scores,
            "description":desc,"tip":tip,"color":color,"result_text":result_text}

def get_sentiment_badge_html(sentiment: str, confidence: float) -> str:
    colors={"positive":"#34d399","negative":"#fb7185","neutral":"#63b3ed"}
    icons={"positive":"😊","negative":"😔","neutral":"😐"}
    color=colors.get(sentiment,"#94a3b8"); icon=icons.get(sentiment,"😐")
    return (f'<span style="background:rgba(99,179,237,0.1);border:1px solid {color}33;'
            f'color:{color};padding:0.2rem 0.7rem;border-radius:20px;font-size:0.8rem">'
            f'{icon} {sentiment.capitalize()} ({confidence*100:.0f}%)</span>')