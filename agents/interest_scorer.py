"""
Interest Scorer with multi-model fallback.
"""

import json
import os
import re
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


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        lines.append(f"Recruiter: {t['recruiter']}")
        lines.append(f"Candidate: {t['candidate']}")
        lines.append("")
    return "\n".join(lines)


def score_interest(conversation: list[dict], candidate: dict) -> dict:
    convo_text = _format_conversation(conversation)

    prompt = f"""Analyse this recruiter-candidate conversation and evaluate the candidate's genuine interest.

Candidate: {candidate['name']} | {candidate['title']} | {candidate.get('experience_years')} yrs

Conversation:
\"\"\"{convo_text}\"\"\"

Return ONLY valid JSON:
{{
  "interest_score": <integer 0-100>,
  "availability": "<Immediate | 1 week | 2 weeks | 1 month | Not available>",
  "positive_signals": ["2-4 specific positive signals"],
  "concerns": ["0-2 hesitations or red flags"],
  "recommendation": "<strong yes | yes | maybe | no>",
  "interest_reasoning": "1-2 sentence explanation"
}}

Scoring: 80-100=enthusiastic, 60-79=positive, 40-59=neutral, 20-39=lukewarm, 0-19=uninterested"""

    last_error = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            result["interest_score"] = int(result.get("interest_score", 50))
            print(f"[interest_scorer] Used model: {model}")
            return result
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err or "limit" in err.lower():
                print(f"[interest_scorer] Model {model} rate limited, trying next...")
                last_error = e
                continue
            # Non-rate-limit error — return fallback
            break

    return {
        "interest_score": 55,
        "availability": candidate.get("availability", "2 weeks"),
        "positive_signals": ["Engaged in conversation"],
        "concerns": [],
        "recommendation": "maybe",
        "interest_reasoning": "Auto-scored due to API limits. Manual review recommended.",
    }