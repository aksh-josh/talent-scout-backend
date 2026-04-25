"""
Interest Scorer
---------------
Analyses a simulated recruiter-candidate conversation and scores the
candidate's genuine interest level on a 0–100 scale.

Also returns:
  - availability (extracted/confirmed)
  - positive_signals (list of enthusiasm indicators)
  - concerns (hesitations, red flags)
  - recommendation (strong yes / yes / maybe / no)
  - interest_reasoning (1-2 sentence explanation)
"""

import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        lines.append(f"Recruiter: {t['recruiter']}")
        lines.append(f"Candidate: {t['candidate']}")
        lines.append("")
    return "\n".join(lines)


def score_interest(conversation: list[dict], candidate: dict) -> dict:
    """
    Score a candidate's interest level from conversation.

    Args:
        conversation: List of turn dicts from outreach_agent.simulate_conversation()
        candidate: Candidate dict (for context)

    Returns:
        dict with interest_score, availability, positive_signals, concerns,
        recommendation, interest_reasoning
    """
    convo_text = _format_conversation(conversation)

    prompt = f"""You are an expert talent acquisition analyst. Analyse this recruiter-candidate conversation and evaluate the candidate's genuine interest level.

Candidate: {candidate['name']} | {candidate['title']} | {candidate.get('experience_years')} yrs exp

Conversation:
\"\"\"
{convo_text}
\"\"\"

Return ONLY valid JSON — no markdown, no explanation:
{{
  "interest_score": <integer 0-100 — 0=actively disinterested, 50=neutral, 100=very excited>,
  "availability": "<Immediate | 1 week | 2 weeks | 1 month | 3 months | Not available>",
  "positive_signals": ["list of 2-5 specific positive signals from their responses"],
  "concerns": ["list of 0-3 hesitations, objections, or red flags — empty list if none"],
  "recommendation": "<strong yes | yes | maybe | no>",
  "interest_reasoning": "1-2 sentence explanation of the score"
}}

Scoring guide:
- 80-100: Enthusiastic, asks follow-up questions, mentions why this role fits them
- 60-79: Positive, engaged, open — minor hesitations
- 40-59: Neutral or mixed — needs more convincing
- 20-39: Lukewarm, deflects, mentions competing offers or concerns
- 0-19: Uninterested, passive, declines or strongly hesitates"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback — return neutral score
        result = {
            "interest_score": 50,
            "availability": candidate.get("availability", "2 weeks"),
            "positive_signals": ["Engaged in conversation"],
            "concerns": [],
            "recommendation": "maybe",
            "interest_reasoning": "Could not parse detailed analysis. Manual review recommended.",
        }

    # Ensure types
    result["interest_score"] = int(result.get("interest_score", 50))
    result["positive_signals"] = result.get("positive_signals", [])
    result["concerns"] = result.get("concerns", [])

    return result