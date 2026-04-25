"""
Outreach Agent
--------------
Simulates a realistic recruiter → candidate conversation using Claude.
The agent asks 4 targeted questions and gets persona-consistent responses
from a simulated candidate.

Conversation structure:
  1. Opening / interest check
  2. Role alignment
  3. Motivations & expectations
  4. Logistics (notice period, remote, relocation)
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Recruiter question templates ──────────────────────────────────────────
QUESTIONS = [
    (
        "opening",
        "Hi {first_name}! I came across your profile — you're a {title} with {exp} years of "
        "experience and skills in {top_skills}. We have a {jd_title} opportunity that looks "
        "like a strong match. Are you open to a quick conversation about it?",
    ),
    (
        "alignment",
        "Great! The role involves {responsibilities}. Based on your background, how well does "
        "this align with what you're looking for in your next move?",
    ),
    (
        "motivation",
        "What's motivating you to explore new opportunities right now? And what would make "
        "a role truly exciting for you at this stage of your career?",
    ),
    (
        "logistics",
        "One last thing — your current notice period is listed as {availability}. "
        "The role is {remote_info}. Does that timeline and setup work for you?",
    ),
]


def _build_candidate_system_prompt(candidate: dict, jd_parsed: dict) -> str:
    """
    Build a detailed system prompt for the simulated candidate persona.
    """
    first_name = candidate["name"].split()[0]
    skills_str = ", ".join(candidate.get("skills", [])[:5])
    
    # Determine personality variation based on experience
    exp = candidate.get("experience_years", 3)
    if exp <= 2:
        persona_hint = "You are junior and keen to grow. You ask questions eagerly."
    elif exp <= 5:
        persona_hint = "You are mid-level and selective. You're interested but want to understand growth opportunities."
    elif exp <= 8:
        persona_hint = "You are senior and value impact. You are thoughtful and want to understand the company's product direction."
    else:
        persona_hint = "You are a seasoned professional. You're not desperate — you evaluate roles carefully against your current comp and ownership."

    return f"""You are {candidate['name']}, a {candidate['title']} with {exp} years of experience.
Your core skills: {skills_str}.
Location: {candidate.get('location', 'India')}.
Availability: {candidate.get('availability', '2 weeks notice')}.
Open to remote: {candidate.get('open_to_remote', True)}.

{persona_hint}

When responding to the recruiter:
- Speak naturally, conversationally. 2-4 sentences max per reply.
- Be realistic — show genuine enthusiasm OR measured interest, not robotic positivity.
- Occasionally mention a specific skill or past project that's relevant.
- If you have concerns, voice them briefly (e.g., "I'd want to understand the team size" or "Is there equity involved?")
- Do NOT say you are an AI. Stay completely in character as {first_name}.
"""


def _fill_question(template: str, candidate: dict, jd_parsed: dict) -> str:
    """Fill question template with candidate and JD data."""
    first_name = candidate["name"].split()[0]
    top_skills = ", ".join(candidate.get("skills", [])[:3])
    responsibilities = "; ".join(jd_parsed.get("key_responsibilities", ["drive impact"])[:2])
    
    remote_info = "fully remote" if candidate.get("open_to_remote") else "hybrid / in-office"
    
    return template.format(
        first_name=first_name,
        title=candidate.get("title", ""),
        exp=candidate.get("experience_years", ""),
        top_skills=top_skills,
        jd_title=jd_parsed.get("title", "the role"),
        responsibilities=responsibilities,
        availability=candidate.get("availability", "2 weeks"),
        remote_info=remote_info,
    )


def simulate_conversation(candidate: dict, jd_parsed: dict) -> list[dict]:
    """
    Simulate a 4-turn recruiter ↔ candidate conversation.

    Args:
        candidate: Enriched candidate dict (with match_score etc.)
        jd_parsed: Structured JD from jd_parser.py

    Returns:
        List of {"recruiter": str, "candidate": str, "turn": str} dicts
    """
    system_prompt = _build_candidate_system_prompt(candidate, jd_parsed)
    conversation_history = []  # OpenAI-style message history for multi-turn
    turns = []

    for turn_key, q_template in QUESTIONS:
        recruiter_msg = _fill_question(q_template, candidate, jd_parsed)

        # Build messages with system prompt included for Groq
        groq_messages = [{"role": "system", "content": system_prompt}]
        for past_turn in conversation_history:
            groq_messages.append({"role": "user", "content": past_turn["recruiter"]})
            groq_messages.append({"role": "assistant", "content": past_turn["candidate_response"]})
        groq_messages.append({"role": "user", "content": recruiter_msg})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            messages=groq_messages,
        )

        candidate_response = response.choices[0].message.content.strip()

        turns.append({
            "turn": turn_key,
            "recruiter": recruiter_msg,
            "candidate": candidate_response,
        })

        conversation_history.append({
            "recruiter": recruiter_msg,
            "candidate_response": candidate_response,
        })

    return turns