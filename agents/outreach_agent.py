"""
Outreach Agent v2
-----------------
Reduced to 2 turns (was 4) to cut API calls by 50%.
Covers: interest check + role alignment (most signal-rich turns).
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Only 2 turns — covers the most important signals
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
        persona = "You are junior and eager. You ask questions and show enthusiasm."
    elif exp <= 5:
        persona = "You are mid-level and selective. You're open but want to understand growth."
    elif exp <= 8:
        persona = "You are senior and thoughtful. You evaluate roles against your current situation."
    else:
        persona = "You are very experienced. You are not desperate — you ask about ownership and impact."

    return f"""You are {candidate['name']}, a {candidate['title']} with {exp} years experience.
Skills: {', '.join(candidate.get('skills', [])[:5])}.
Location: {candidate.get('location', 'India')}. Availability: {candidate.get('availability', '2 weeks')}.
{persona}
Respond naturally in 2-3 sentences. Stay in character. Do NOT say you are an AI."""


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


def simulate_conversation(candidate: dict, jd_parsed: dict) -> list[dict]:
    system_prompt = _build_system_prompt(candidate, jd_parsed)
    conversation_history = []
    turns = []

    for turn_key, q_template in QUESTIONS:
        recruiter_msg = _fill_question(q_template, candidate, jd_parsed)

        groq_messages = [{"role": "system", "content": system_prompt}]
        for past in conversation_history:
            groq_messages.append({"role": "user", "content": past["recruiter"]})
            groq_messages.append({"role": "assistant", "content": past["response"]})
        groq_messages.append({"role": "user", "content": recruiter_msg})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=200,
            messages=groq_messages,
        )
        candidate_response = response.choices[0].message.content.strip()

        turns.append({"turn": turn_key, "recruiter": recruiter_msg, "candidate": candidate_response})
        conversation_history.append({"recruiter": recruiter_msg, "response": candidate_response})

    return turns