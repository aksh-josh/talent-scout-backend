"""
Outreach Agent with multi-model fallback.
2 turns to minimize API usage.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = [
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

QUESTIONS = [
    (
        "opening",
        "Hi {first_name}! I came across your profile — you're a {title} with {exp} years "
        "of experience in {top_skills}. We have a {jd_title} opportunity that looks like "
        "a strong fit. Are you open to hearing more, and what's your current situation?",
    ),
    (
        "alignment",
        "The role involves {responsibilities}. Your notice period is {availability} — "
        "does the timeline and this kind of work align with what you're looking for next?",
    ),
]


def _build_system_prompt(candidate: dict, jd_parsed: dict) -> str:
    exp = candidate.get("experience_years", 3)
    if exp <= 2:
        persona = "You are junior and eager. You show enthusiasm and ask questions."
    elif exp <= 5:
        persona = "You are mid-level and selective. Open but want to understand growth opportunities."
    elif exp <= 8:
        persona = "You are senior and thoughtful. You evaluate roles against your current situation."
    else:
        persona = "You are very experienced. Not desperate — you care about ownership and impact."

    return f"""You are {candidate['name']}, a {candidate['title']} with {exp} years experience.
Skills: {', '.join(candidate.get('skills', [])[:5])}.
Location: {candidate.get('location', 'India')}. Availability: {candidate.get('availability', '2 weeks')}.
{persona}
Respond naturally in 2-3 sentences. Stay in character. Never say you are an AI."""


def _fill_question(template: str, candidate: dict, jd_parsed: dict) -> str:
    return template.format(
        first_name=candidate["name"].split()[0],
        title=candidate.get("title", ""),
        exp=candidate.get("experience_years", ""),
        top_skills=", ".join(candidate.get("skills", [])[:3]),
        jd_title=jd_parsed.get("title", "the role"),
        responsibilities="; ".join(jd_parsed.get("key_responsibilities", ["drive impact"])[:2]),
        availability=candidate.get("availability", "2 weeks"),
    )


def _call_with_fallback(messages: list) -> str:
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model, max_tokens=200, messages=messages,
            )
            print(f"[outreach] Used model: {model}")
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err or "limit" in err.lower():
                print(f"[outreach] Model {model} rate limited, trying next...")
                continue
            raise
    return "Thank you for reaching out. I'd be interested in learning more about this opportunity."


def simulate_conversation(candidate: dict, jd_parsed: dict) -> list[dict]:
    system_prompt = _build_system_prompt(candidate, jd_parsed)
    conversation_history = []
    turns = []

    for turn_key, q_template in QUESTIONS:
        recruiter_msg = _fill_question(q_template, candidate, jd_parsed)

        messages = [{"role": "system", "content": system_prompt}]
        for past in conversation_history:
            messages.append({"role": "user", "content": past["recruiter"]})
            messages.append({"role": "assistant", "content": past["response"]})
        messages.append({"role": "user", "content": recruiter_msg})

        candidate_response = _call_with_fallback(messages)
        turns.append({"turn": turn_key, "recruiter": recruiter_msg, "candidate": candidate_response})
        conversation_history.append({"recruiter": recruiter_msg, "response": candidate_response})

    return turns