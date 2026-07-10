"""
HealMatrix AI - Large Knowledge Base Builder
============================================
Downloads and builds a large mental health corpus for RAG.

Target: 1-5 GB of mental health text

Sources:
1. HuggingFace datasets (fine-tuning)
2. NIMH articles (nimh.nih.gov)
3. WHO mental health docs
4. NHS mental health guides
5. PubMed open-access articles
6. CBT/DBT text resources
7. Psychology Today articles
8. MentalHealth.gov resources

Usage:
    python build_knowledge_base.py
"""

import os
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

print("=" * 60)
print("  HealMatrix AI - Knowledge Base Builder")
print("=" * 60)

# Folders
KB  = Path("data/knowledge_base")
RAW = Path("data/raw_downloads")
DS  = Path("hf_datasets")

for p in [KB, RAW, DS/"rag", DS/"emotion"]:
    p.mkdir(parents=True, exist_ok=True)

stats = {}

# PART 1: HuggingFace Fine-tuning Datasets
print("PART 1: HuggingFace Datasets (Fine-tuning)")

from datasets import load_dataset

# 1a. Mental Health Counseling
print("\n[1/3] Mental Health Counseling Conversations...")
try:
    ds1 = load_dataset(
        "Amod/mental_health_counseling_conversations",
        cache_dir=str(DS/"rag")
    )
    count = len(ds1["train"])
    print(f"  {count:,} Q-A pairs")
    stats["counseling"] = count

    # Save as KB documents too
    out = KB / "counseling_conversations.txt"
    with open(out, "w", encoding="utf-8") as f:
        for item in ds1["train"]:
            q = item.get("Context") or item.get("context") or ""
            a = item.get("Response") or item.get("response") or ""
            if q and a:
                f.write(f"Patient: {q}\nTherapist: {a}\n\n")
    size = out.stat().st_size / 1024 / 1024
    print(f"   Saved: {out.name} ({size:.1f} MB)")
except Exception as e:
    print(f"  ❌ {e}")
    stats["counseling"] = 0

# 1b. MentalChat16K
print("\n[2/3] MentalChat16K...")
try:
    ds2 = load_dataset(
        "ShenLab/MentalChat16K",
        cache_dir=str(DS/"rag")
    )
    count = len(ds2["train"])
    print(f"  {count:,} conversations")
    stats["mentalchat"] = count

    out = KB / "mentalchat16k.txt"
    with open(out, "w", encoding="utf-8") as f:
        for item in ds2["train"]:
            q = str(item.get("question") or item.get("input") or "")
            a = str(item.get("answer") or item.get("output") or "")
            if q and a:
                f.write(f"Q: {q}\nA: {a}\n\n")
    size = out.stat().st_size / 1024 / 1024
    print(f" Saved: {out.name} ({size:.1f} MB)")
except Exception as e:
    print(f"  ❌ {e}")

# 1c. Counsel Chat
print("\n[3/3] CounselChat (Real therapist Q-A)...")
try:
    ds3 = load_dataset(
        "nbertagnolli/counsel-chat",
        cache_dir=str(DS/"rag")
    )
    count = len(ds3["train"])
    print(f"  {count:,} therapist answers")
    stats["counselchat"] = count

    out = KB / "counsel_chat.txt"
    with open(out, "w", encoding="utf-8") as f:
        for item in ds3["train"]:
            q = str(item.get("questionText") or item.get("question") or "")
            a = str(item.get("answerText") or item.get("answer") or "")
            topic = str(item.get("topic") or "")
            if q and a:
                f.write(f"Topic: {topic}\nQuestion: {q}\nAnswer: {a}\n\n")
    size = out.stat().st_size / 1024 / 1024
    print(f"  Saved: {out.name} ({size:.1f} MB)")
except Exception as e:
    print(f"  ❌ {e}")

# PART 2: NIMH Articles (National Institute of Mental Health)
print("\n" + "="*60)
print("PART 2: NIMH Mental Health Articles")
print("="*60)

NIMH_TOPICS = [
    ("anxiety-disorders", "https://www.nimh.nih.gov/health/topics/anxiety-disorders"),
    ("depression", "https://www.nimh.nih.gov/health/topics/depression"),
    ("bipolar-disorder", "https://www.nimh.nih.gov/health/topics/bipolar-disorder"),
    ("ptsd", "https://www.nimh.nih.gov/health/topics/post-traumatic-stress-disorder-ptsd"),
    ("schizophrenia", "https://www.nimh.nih.gov/health/topics/schizophrenia"),
    ("ocd", "https://www.nimh.nih.gov/health/topics/obsessive-compulsive-disorder-ocd"),
    ("eating-disorders", "https://www.nimh.nih.gov/health/topics/eating-disorders"),
    ("adhd", "https://www.nimh.nih.gov/health/topics/attention-deficit-hyperactivity-disorder-adhd"),
    ("autism", "https://www.nimh.nih.gov/health/topics/autism-spectrum-disorder-asd"),
    ("borderline", "https://www.nimh.nih.gov/health/topics/borderline-personality-disorder"),
    ("suicide", "https://www.nimh.nih.gov/health/topics/suicide-prevention"),
    ("panic", "https://www.nimh.nih.gov/health/topics/panic-disorder"),
    ("social-anxiety", "https://www.nimh.nih.gov/health/topics/social-anxiety-disorder"),
    ("sleep", "https://www.nimh.nih.gov/health/topics/sleep-disorders"),
    ("stress", "https://www.nimh.nih.gov/health/publications/stress"),
    ("mental-illness-basics", "https://www.nimh.nih.gov/health/topics/mental-illness"),
    ("psychotherapy", "https://www.nimh.nih.gov/health/topics/psychotherapies"),
    ("medications", "https://www.nimh.nih.gov/health/topics/mental-health-medications"),
    ("brain-stimulation", "https://www.nimh.nih.gov/health/topics/brain-stimulation-therapies"),
    ("children-mh", "https://www.nimh.nih.gov/health/topics/child-and-adolescent-mental-health"),
]

headers = {
    "User-Agent": "Mozilla/5.0 (Research Bot for Academic Project)"
}

nimh_file = KB / "nimh_mental_health_articles.txt"
nimh_count = 0

with open(nimh_file, "w", encoding="utf-8") as f:
    for name, url in tqdm(NIMH_TOPICS, desc="  NIMH articles"):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")

                # Remove nav, header, footer
                for tag in soup(["nav","header","footer","script","style"]):
                    tag.decompose()

                # Get main content
                main = soup.find("main") or soup.find("article") or soup.body
                if main:
                    text = main.get_text(separator="\n", strip=True)
                    # Clean up
                    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
                    clean = "\n".join(lines)
                    if len(clean) > 200:
                        f.write(f"\n\n=== NIMH: {name.upper()} ===\n")
                        f.write(f"Source: {url}\n\n")
                        f.write(clean)
                        f.write("\n")
                        nimh_count += 1
            time.sleep(1)  # Be polite
        except Exception as e:
            pass

size = nimh_file.stat().st_size / 1024 / 1024
print(f"  {nimh_count} articles saved → {nimh_file.name} ({size:.1f} MB)")
stats["nimh"] = nimh_count

# PART 3: WHO Mental Health Resources
print("\n" + "="*60)
print("PART 3: WHO Mental Health Resources")
print("="*60)

WHO_PAGES = [
    ("mental-health-overview", "https://www.who.int/news-room/fact-sheets/detail/mental-disorders"),
    ("depression-who", "https://www.who.int/news-room/fact-sheets/detail/depression"),
    ("anxiety-who", "https://www.who.int/news-room/fact-sheets/detail/anxiety-disorders"),
    ("suicide-who", "https://www.who.int/news-room/fact-sheets/detail/suicide"),
    ("schizophrenia-who", "https://www.who.int/news-room/fact-sheets/detail/schizophrenia"),
    ("bipolar-who", "https://www.who.int/news-room/fact-sheets/detail/bipolar-disorder"),
    ("dementia-who", "https://www.who.int/news-room/fact-sheets/detail/dementia"),
    ("ptsd-who", "https://www.who.int/news-room/fact-sheets/detail/post-traumatic-stress-disorder"),
    ("eating-who", "https://www.who.int/news-room/fact-sheets/detail/eating-disorders"),
    ("autism-who", "https://www.who.int/news-room/fact-sheets/detail/autism-spectrum-disorders"),
    ("adhd-who", "https://www.who.int/news-room/fact-sheets/detail/attention-deficit-hyperactivity-disorder"),
    ("child-mh-who", "https://www.who.int/news-room/fact-sheets/detail/adolescent-mental-health"),
    ("workplace-mh", "https://www.who.int/news-room/fact-sheets/detail/mental-health-at-work"),
    ("mh-action-plan", "https://www.who.int/publications/i/item/9789240031029"),
]

who_file = KB / "who_mental_health_resources.txt"
who_count = 0

with open(who_file, "w", encoding="utf-8") as f:
    for name, url in tqdm(WHO_PAGES, desc="  WHO articles"):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["nav","header","footer","script","style"]):
                    tag.decompose()
                main = soup.find("main") or soup.find("article") or soup.body
                if main:
                    text = main.get_text(separator="\n", strip=True)
                    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
                    clean = "\n".join(lines)
                    if len(clean) > 200:
                        f.write(f"\n\n=== WHO: {name.upper()} ===\n")
                        f.write(f"Source: {url}\n\n")
                        f.write(clean)
                        who_count += 1
            time.sleep(1)
        except Exception:
            pass

size = who_file.stat().st_size / 1024 / 1024
print(f"  ✅ {who_count} WHO articles → {who_file.name} ({size:.1f} MB)")
stats["who"] = who_count

# PART 4: NHS Mental Health Guides
print("\n" + "="*60)
print("PART 4: NHS Mental Health Guides")
print("="*60)

NHS_PAGES = [
    ("anxiety", "https://www.nhs.uk/mental-health/conditions/generalised-anxiety-disorder/overview/"),
    ("depression", "https://www.nhs.uk/mental-health/conditions/depression-in-adults/overview/"),
    ("ptsd", "https://www.nhs.uk/mental-health/conditions/post-traumatic-stress-disorder-ptsd/overview/"),
    ("ocd", "https://www.nhs.uk/mental-health/conditions/obsessive-compulsive-disorder-ocd/overview/"),
    ("bipolar", "https://www.nhs.uk/mental-health/conditions/bipolar-disorder/overview/"),
    ("schizophrenia", "https://www.nhs.uk/mental-health/conditions/schizophrenia/overview/"),
    ("phobias", "https://www.nhs.uk/mental-health/conditions/phobias/overview/"),
    ("panic", "https://www.nhs.uk/mental-health/conditions/panic-disorder/"),
    ("eating", "https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/feelings-and-symptoms/"),
    ("sleep", "https://www.nhs.uk/every-mind-matters/mental-health-issues/sleep/"),
    ("stress", "https://www.nhs.uk/every-mind-matters/mental-health-issues/stress/"),
    ("low-mood", "https://www.nhs.uk/every-mind-matters/mental-health-issues/low-mood-and-depression/"),
    ("cbt-nhs", "https://www.nhs.uk/mental-health/talking-therapies-medicine-treatments/talking-therapies-and-counselling/cognitive-behavioural-therapy-cbt/overview/"),
    ("mindfulness", "https://www.nhs.uk/mental-health/self-help/tips-and-support/mindfulness/"),
    ("self-harm", "https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/behaviours/self-harm/overview/"),
    ("suicide-help", "https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/behaviours/help-for-suicidal-thoughts/"),
    ("grief", "https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/feelings-and-symptoms/grief-bereavement-loss/"),
    ("loneliness", "https://www.nhs.uk/every-mind-matters/lifestyles-and-wellbeing/loneliness/"),
    ("anger", "https://www.nhs.uk/mental-health/feelings-symptoms-behaviours/feelings-and-symptoms/anger/"),
    ("trauma", "https://www.nhs.uk/mental-health/conditions/post-traumatic-stress-disorder-ptsd/treatment/"),
]

nhs_file = KB / "nhs_mental_health_guides.txt"
nhs_count = 0

with open(nhs_file, "w", encoding="utf-8") as f:
    for name, url in tqdm(NHS_PAGES, desc="  NHS guides"):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["nav","header","footer","script","style","aside"]):
                    tag.decompose()
                main = soup.find("main") or soup.find("article") or soup.body
                if main:
                    text = main.get_text(separator="\n", strip=True)
                    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
                    clean = "\n".join(lines)
                    if len(clean) > 200:
                        f.write(f"\n\n=== NHS: {name.upper()} ===\n")
                        f.write(f"Source: {url}\n\n")
                        f.write(clean)
                        nhs_count += 1
            time.sleep(1)
        except Exception:
            pass

size = nhs_file.stat().st_size / 1024 / 1024
print(f"   {nhs_count} NHS guides → {nhs_file.name} ({size:.1f} MB)")
stats["nhs"] = nhs_count

# PART 5: CBT & DBT Knowledge Base (Hardcoded Expert Content)
print("\n" + "="*60)
print("PART 5: CBT & DBT Comprehensive Knowledge Base")
print("="*60)

CBT_DBT_CONTENT = """
=== COGNITIVE BEHAVIOURAL THERAPY (CBT) COMPREHENSIVE GUIDE ===

WHAT IS CBT?
Cognitive Behavioural Therapy (CBT) is a structured, evidence-based psychotherapy
that addresses the relationship between thoughts, feelings, and behaviours.
Developed by Aaron Beck in the 1960s, CBT is the most widely researched form
of psychological therapy, with strong evidence for depression, anxiety, PTSD,
OCD, eating disorders, and many other conditions.

CORE CBT PRINCIPLES:
1. Cognitive Model: Psychological problems are partly maintained by unhelpful
   ways of thinking and learned patterns of unhelpful behaviour.
2. Thoughts influence feelings: What we think affects how we feel and behave.
3. Behaviour affects mood: What we do affects how we think and feel.
4. Change is possible: People can learn better ways of thinking and behaving.
5. Collaborative empiricism: Therapist and client work together scientifically.

CBT THOUGHT RECORDS:
A thought record is a structured way to examine and challenge unhelpful thoughts.
Columns: Situation | Thoughts | Emotions | Evidence For | Evidence Against | Balanced Thought
Example: "I failed my exam" | "I am stupid" | Shame 90% |
Evidence For: I got a low grade.
Evidence Against: I passed other exams. I was unwell this week. One exam doesn't define intelligence.
Balanced Thought: "I struggled with this exam, but that doesn't mean I'm stupid."

COGNITIVE DISTORTIONS (Automatic Negative Thoughts):
1. All-or-nothing thinking: "If I'm not perfect, I'm a failure."
2. Catastrophising: "This headache means I have a brain tumour."
3. Mind reading: "They didn't reply because they hate me."
4. Fortune telling: "I know things will go wrong."
5. Emotional reasoning: "I feel stupid, so I must be stupid."
6. Personalisation: "The accident happened because I'm unlucky."
7. Should statements: "I should always be productive."
8. Labelling: "I'm a loser" instead of "I made a mistake."
9. Mental filter: Focusing only on negatives.
10. Disqualifying positives: "That doesn't count."
11. Magnification/Minimisation: Blowing up negatives, shrinking positives.
12. Overgeneralisation: "I always mess up."

BEHAVIOURAL ACTIVATION (BA):
Behavioural activation is used for depression. It involves scheduling activities
that provide a sense of pleasure, achievement, or social connection.
Steps: 1. Monitor current activities and mood.
2. Identify activities that previously gave pleasure or meaning.
3. Schedule these activities even when you don't feel motivated.
4. Start very small (5-10 minutes if needed).
5. Review and increase gradually.

EXPOSURE THERAPY:
Used for anxiety, phobias, PTSD, and OCD.
Principle: Avoidance maintains fear. Gradual exposure reduces it.
Types:
- In vivo exposure: Real-life situations.
- Imaginal exposure: Imagining feared situations.
- Interoceptive exposure: Inducing feared body sensations.
- Virtual reality exposure: Using VR technology.
Steps: Build a fear hierarchy (0-100). Start with the least feared. Work up gradually.

RELAXATION TECHNIQUES IN CBT:
1. Progressive Muscle Relaxation (PMR): Tense and release muscle groups.
2. Diaphragmatic breathing: Slow belly breathing to calm the nervous system.
3. 4-7-8 breathing: Inhale 4s, hold 7s, exhale 8s.
4. Box breathing: Inhale 4s, hold 4s, exhale 4s, hold 4s.
5. Grounding: 5-4-3-2-1 sensory technique.

=== DIALECTICAL BEHAVIOUR THERAPY (DBT) COMPREHENSIVE GUIDE ===

WHAT IS DBT?
Dialectical Behaviour Therapy was developed by Marsha Linehan in the late 1980s.
Originally designed for borderline personality disorder, it is now used for
self-harm, suicidal ideation, eating disorders, PTSD, and substance use.
The word dialectical means balancing opposites — accepting yourself as you are
while also working to change.

FOUR DBT SKILL MODULES:

MODULE 1 — MINDFULNESS:
Core mindfulness skills:
- Observe: Notice without reacting. Watch thoughts like clouds passing.
- Describe: Put words to experience without judging.
- Participate: Fully engage in the present moment.
Mindfulness stances:
- Non-judgmental: Describe facts, not evaluations.
- One-mindfully: Do one thing at a time with full attention.
- Effectively: Focus on what works, not what's "right."

MODULE 2 — DISTRESS TOLERANCE:
Crisis survival skills (when you cannot fix the situation right now):
- TIPP: Temperature (cold water on face), Intense exercise, Paced breathing,
  Progressive muscle relaxation.
- ACCEPTS: Activities, Contributing, Comparisons, Emotions (opposite),
  Pushing away, Thoughts (other), Sensations.
- Self-soothe with five senses: Sight, hearing, smell, taste, touch.
- IMPROVE the moment: Imagery, Meaning, Prayer, Relaxation, One thing,
  brief Vacation, Encouragement.
Radical Acceptance: Completely accepting reality as it is, not as you wish it were.
This does not mean approving or giving up — it means stopping the fight against facts.

MODULE 3 — EMOTIONAL REGULATION:
Understanding emotions:
- Name the emotion (reduces its intensity).
- Identify the event that triggered it.
- Identify vulnerability factors.
- Notice the urge (what the emotion wants you to do).
- Identify the effect on behaviour.
Changing emotions:
- Opposite Action: Do the opposite of what the emotion urges.
  Fear urges avoidance → approach the situation gradually.
  Shame urges hiding → share with a trusted person.
  Anger urges attack → be kind or step away.
- Check the Facts: Is my emotion justified by the facts?
- Problem solving: Change the situation that triggers the emotion.
- ABC PLEASE: Accumulate positives, Build mastery, Cope ahead, treat
  PhysicaL illness, Eat balanced, Avoid substances, Sleep well, Exercise.

MODULE 4 — INTERPERSONAL EFFECTIVENESS:
DEAR MAN (Getting what you want):
- Describe the situation factually.
- Express your feelings and opinions.
- Assert what you want or don't want.
- Reinforce why it benefits the other person.
- stay Mindful of your goals.
- Appear confident (even if you don't feel it).
- Negotiate and be willing to give to get.
GIVE (Maintaining relationships):
- Gentle (no attacks or threats).
- Interested (listen, ask questions).
- Validate (acknowledge feelings).
- Easy manner (light tone).
FAST (Self-respect):
- Fair to yourself and others.
- no Apologies for existing or having needs.
- Stick to your values.
- Truthful.

=== MINDFULNESS-BASED INTERVENTIONS ===

MINDFULNESS-BASED STRESS REDUCTION (MBSR):
Developed by Jon Kabat-Zinn at the University of Massachusetts.
8-week programme including:
- Body scan meditation: Systematic attention to sensations throughout the body.
- Sitting meditation: Breath, body, sounds, thoughts, open awareness.
- Mindful movement: Gentle yoga and walking meditation.
- Informal practices: Mindful eating, mindful daily activities.
Evidence: Reduces stress, anxiety, depression, and chronic pain.

MINDFULNESS-BASED COGNITIVE THERAPY (MBCT):
Combines MBSR with CBT. Specifically designed to prevent relapse of depression.
Key skill: Recognising the early warning signs of depression and responding
with awareness rather than autopilot.
Decentring: Seeing thoughts as mental events, not facts.
"Thoughts are not facts, even when they feel true."

=== ACCEPTANCE AND COMMITMENT THERAPY (ACT) ===

SIX CORE PROCESSES:
1. Acceptance: Making room for difficult thoughts and feelings.
2. Cognitive Defusion: Stepping back from thoughts.
   "I am having the thought that I am worthless" vs "I am worthless."
3. Present Moment Awareness: Connecting with now.
4. Self-as-Context: The observing self — you are not your thoughts.
5. Values: What matters most to you in life.
6. Committed Action: Taking action aligned with values.

ACT METAPHORS:
- Passengers on the bus: You are the driver. Difficult thoughts are passengers.
  They can't stop the bus — only you can.
- Tug of war with a monster: Dropping the rope (acceptance) vs fighting.
- Leaves on a stream: Watching thoughts float by without grabbing them.

=== CRISIS SUPPORT PROTOCOLS ===

SAFE MESSAGING FOR CRISIS:
- Ask directly: "Are you thinking about suicide?" (does NOT increase risk)
- Listen without judgement.
- Do not promise confidentiality.
- Involve professional help.
- Remove access to means where possible.
- Create a safety plan together.

SAFETY PLAN COMPONENTS:
1. Personal warning signs that a crisis is developing.
2. Internal coping strategies (things I can do alone).
3. People and social settings that provide distraction.
4. People I can ask for help.
5. Professionals and agencies I can contact in a crisis.
6. Making the environment safe.

CRISIS RESOURCES:
- US: 988 Suicide and Crisis Lifeline (call or text 988)
- UK: Samaritans 116 123
- Pakistan: Umang helpline 0317-4288665 / 0800-00-002
- Australia: Lifeline 13 11 14
- Canada: Crisis Services Canada 1-833-456-4566

=== ANXIETY DISORDERS — CLINICAL OVERVIEW ===

GENERALISED ANXIETY DISORDER (GAD):
Core feature: Excessive, uncontrollable worry about many topics.
Physical symptoms: Muscle tension, fatigue, concentration problems,
irritability, sleep disturbance.
CBT approach: Worry postponement, challenging worry beliefs,
relaxation training, problem solving.
Medication: SSRIs, SNRIs, buspirone. Benzodiazepines short-term only.

PANIC DISORDER:
Core feature: Recurrent unexpected panic attacks + fear of future attacks.
Panic attack symptoms: Heart pounding, shortness of breath, dizziness,
chest pain, numbness, feeling of unreality, fear of dying or going crazy.
CBT approach: Psychoeducation (panic is not dangerous), interoceptive
exposure, cognitive restructuring of catastrophic misinterpretations.
Key message: Panic attacks feel life-threatening but are not. The body's
alarm system has been triggered incorrectly.

SOCIAL ANXIETY DISORDER:
Core feature: Marked fear of social or performance situations.
Safety behaviours maintain anxiety (e.g., avoiding eye contact, rehearsing).
CBT approach: Dropping safety behaviours, video feedback, behavioural experiments.

POST-TRAUMATIC STRESS DISORDER (PTSD):
Symptoms: Re-experiencing (flashbacks, nightmares), avoidance, negative
cognitions and mood, hyperarousal.
First-line treatments: Trauma-focused CBT, EMDR (Eye Movement Desensitisation
and Reprocessing), prolonged exposure, cognitive processing therapy.
EMDR: Bilateral stimulation while processing traumatic memories reduces distress.

=== DEPRESSION — CLINICAL OVERVIEW ===

DIAGNOSTIC CRITERIA:
5+ symptoms for 2+ weeks including depressed mood or loss of interest:
Depressed mood, loss of interest, weight change, sleep change,
psychomotor changes, fatigue, worthlessness, concentration problems,
suicidal ideation.

NEUROBIOLOGICAL BASIS:
Involves serotonin, norepinephrine, dopamine systems.
HPA axis dysregulation (cortisol).
Neuroplasticity changes in hippocampus.
Inflammatory markers elevated in depression.

TREATMENT:
Mild: CBT, behavioural activation, exercise, sleep hygiene.
Moderate-severe: CBT + antidepressant medication.
Medication: SSRIs (first line), SNRIs, tricyclics, MAOIs.
Physical: ECT for severe/treatment-resistant. TMS. Ketamine.

=== MEDICATION OVERVIEW ===

SSRI (Selective Serotonin Reuptake Inhibitors):
Examples: Fluoxetine, Sertraline, Escitalopram, Paroxetine.
Used for: Depression, anxiety, OCD, PTSD.
Side effects: Nausea (temporary), sexual dysfunction, sleep changes.
Onset: 2-6 weeks for full effect.

SNRI (Serotonin-Norepinephrine Reuptake Inhibitors):
Examples: Venlafaxine, Duloxetine.
Used for: Depression, anxiety, chronic pain.

ANTIPSYCHOTICS:
Typical: Haloperidol, Chlorpromazine.
Atypical: Risperidone, Olanzapine, Quetiapine, Aripiprazole.
Used for: Schizophrenia, bipolar disorder, depression augmentation.

MOOD STABILISERS:
Lithium (gold standard for bipolar), Valproate, Lamotrigine, Carbamazepine.

=== SLEEP AND MENTAL HEALTH ===

COGNITIVE BEHAVIOURAL THERAPY FOR INSOMNIA (CBT-I):
First-line treatment for chronic insomnia (more effective than medication).
Components:
1. Sleep restriction: Limit time in bed to actual sleep time initially.
2. Stimulus control: Bed only for sleep and sex. Get up if not sleeping.
3. Sleep hygiene education.
4. Relaxation training.
5. Cognitive restructuring of unhelpful sleep beliefs.

SLEEP HYGIENE GUIDELINES:
- Consistent sleep and wake times (even weekends).
- Cool, dark, quiet bedroom.
- No screens 60 minutes before bed.
- No caffeine after 2pm.
- Regular daytime exercise (not within 3 hours of bed).
- Wind-down routine: bath, reading, relaxation.
- Avoid lying awake in bed more than 20 minutes.

=== TRAUMA-INFORMED CARE ===

PRINCIPLES OF TRAUMA-INFORMED CARE:
1. Safety: Ensuring physical and emotional safety.
2. Trustworthiness: Clear and consistent communication.
3. Choice: Providing options and respecting decisions.
4. Collaboration: Sharing power and decision-making.
5. Empowerment: Building on strengths.
6. Cultural sensitivity: Recognising cultural, historical, and gender issues.

ACE (ADVERSE CHILDHOOD EXPERIENCES):
Research shows ACEs (abuse, neglect, household dysfunction) increase risk
of mental illness, substance use, and physical health problems in adulthood.
The more ACEs, the higher the risk. However, resilience and protective factors
can buffer these effects significantly.

=== SELF-CARE AND WELLNESS STRATEGIES ===

THE 5 WAYS TO WELLBEING (NHS England):
1. Connect: Build and maintain relationships.
2. Be active: Exercise improves mood, energy, and sleep.
3. Take notice: Be curious and aware of the world around you.
4. Keep learning: Try something new or rediscover an old interest.
5. Give: Do something nice for someone. Volunteer.

EXERCISE AND MENTAL HEALTH:
30 minutes of moderate exercise 5x per week is as effective as antidepressants
for mild-moderate depression. Exercise increases BDNF (brain-derived neurotrophic
factor), which supports neuroplasticity and mood regulation.

NUTRITION AND MENTAL HEALTH:
Mediterranean diet associated with lower depression risk.
Omega-3 fatty acids: Anti-inflammatory, support mood regulation.
Gut-brain axis: 90% of serotonin produced in the gut.
Avoid: Excessive sugar, alcohol, ultra-processed foods.

SOCIAL CONNECTION:
Loneliness is as harmful to health as smoking 15 cigarettes per day.
Social connection is the strongest predictor of wellbeing and longevity.
Quality of relationships matters more than quantity.
"""

cbt_file = KB / "cbt_dbt_comprehensive_guide.txt"
with open(cbt_file, "w", encoding="utf-8") as f:
    f.write(CBT_DBT_CONTENT)
size = cbt_file.stat().st_size / 1024 / 1024
print(f"  CBT/DBT guide saved → {cbt_file.name} ({size:.2f} MB)")
stats["cbt_dbt"] = len(CBT_DBT_CONTENT)

# PART 6: MentalHealth.gov Content
print("\n" + "="*60)
print("PART 6: MentalHealth.gov Articles")
print("="*60)

MENTALHEALTHGOV = [
    "https://www.mentalhealth.gov/basics/what-is-mental-health",
    "https://www.mentalhealth.gov/get-help/immediate-help",
    "https://www.mentalhealth.gov/basics/mental-health-myths-facts",
    "https://www.mentalhealth.gov/talk/friends-family-members",
    "https://www.mentalhealth.gov/get-help/veterans",
]

mhgov_file = KB / "mentalhealth_gov_articles.txt"
mhgov_count = 0
with open(mhgov_file, "w", encoding="utf-8") as f:
    for url in tqdm(MENTALHEALTHGOV, desc="  MentalHealth.gov"):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["nav","header","footer","script","style"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40]
                clean = "\n".join(lines)
                if len(clean) > 100:
                    f.write(f"\n\n=== {url} ===\n\n{clean}\n")
                    mhgov_count += 1
            time.sleep(1)
        except Exception:
            pass

size = mhgov_file.stat().st_size / 1024 / 1024
print(f"   {mhgov_count} pages → {mhgov_file.name} ({size:.2f} MB)")

# PART 7: PubMed Open Access Mental Health Abstracts
print("\n" + "="*60)
print("PART 7: PubMed Mental Health Abstracts")
print("="*60)

PUBMED_SEARCHES = [
    "cognitive+behavioral+therapy+depression",
    "mindfulness+anxiety+treatment",
    "DBT+borderline+personality+disorder",
    "PTSD+trauma+therapy",
    "mental+health+intervention",
    "suicide+prevention+therapy",
    "anxiety+disorder+CBT+treatment",
    "depression+psychotherapy+outcome",
]

pubmed_file = KB / "pubmed_mental_health_abstracts.txt"
abstract_count = 0

with open(pubmed_file, "w", encoding="utf-8") as f:
    f.write("=== PUBMED MENTAL HEALTH RESEARCH ABSTRACTS ===\n\n")
    for search in tqdm(PUBMED_SEARCHES, desc="  PubMed searches"):
        try:
            search_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&term={search}&retmax=20&retmode=json"
            )
            resp = requests.get(search_url, timeout=10)
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            for pmid in ids[:10]:
                fetch_url = (
                    f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                    f"?db=pubmed&id={pmid}&rettype=abstract&retmode=text"
                )
                r2 = requests.get(fetch_url, timeout=10)
                if r2.status_code == 200 and len(r2.text) > 100:
                    f.write(f"\n--- PMID: {pmid} ---\n")
                    f.write(r2.text)
                    f.write("\n")
                    abstract_count += 1
                time.sleep(0.4)
        except Exception:
            pass

size = pubmed_file.stat().st_size / 1024 / 1024
print(f"  {abstract_count} abstracts → {pubmed_file.name} ({size:.2f} MB)")
stats["pubmed"] = abstract_count

# FINAL REPORT
print("\n" + "="*60)
print("  KNOWLEDGE BASE BUILD COMPLETE")
print("="*60)

total_size = 0
file_list = []
for f in sorted(KB.iterdir()):
    if f.is_file():
        sz = f.stat().st_size / 1024 / 1024
        total_size += sz
        file_list.append((f.name, sz))
        print(f"  {f.name:50s} {sz:6.2f} MB")

print(f"\n  TOTAL KB SIZE: {total_size:.2f} MB ({total_size/1024:.3f} GB)")
print(f"\n  Files in data/knowledge_base/: {len(file_list)}")

report = {
    "total_size_mb": round(total_size, 2),
    "total_size_gb": round(total_size/1024, 4),
    "files": {name: round(sz, 2) for name, sz in file_list},
    "stats": stats,
}
with open("data/kb_build_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("""
Next Steps:
  1. python rag_finetuning.py     (fine-tune BGE on counseling data)
  2. python emotion_finetuning.py (fine-tune on FER dataset)
  3. python main.py               (run the app — RAG uses KB folder)
""")
