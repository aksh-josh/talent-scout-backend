"""
JD Parser Agent
--------------
Parses a raw Job Description string into structured data using Claude.
Returns a dict with: title, required_skills, nice_to_have_skills,
experience_years_min, role_type, key_responsibilities, summary.
"""

import json
import os
import re

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def parse_jd(jd_text: str) -> dict:
    """
    Parse a Job Description into structured JSON using Claude.

    Args:
        jd_text: Raw job description string (any format)

    Returns:
        dict with structured JD fields

    Raises:
        ValueError: If LLM response cannot be parsed as JSON
    """
    prompt = f"""You are a precise job description parser. Extract structured information from the job description below.

Return ONLY valid JSON — no markdown, no explanation, no backticks.

Use this exact schema:
{{
  "title": "string — job title",
  "required_skills": ["list", "of", "must-have", "technical", "skills"],
  "nice_to_have_skills": ["list", "of", "preferred", "skills"],
  "experience_years_min": 0,
  "experience_years_max": 10,
  "role_type": "technical | managerial | hybrid",
  "seniority": "junior | mid | senior | lead | principal",
  "key_responsibilities": ["3-5 main responsibilities as short phrases"],
  "industry_context": "string — domain/industry if mentioned",
  "summary": "2 sentence summary of the ideal candidate"
}}

Job Description:
\"\"\"
{jd_text}
\"\"\"

Return ONLY the JSON object. Nothing else."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw}")

    # Normalise types
    parsed["experience_years_min"] = int(parsed.get("experience_years_min", 0))
    parsed["experience_years_max"] = int(parsed.get("experience_years_max", 20))
    parsed["required_skills"] = parsed.get("required_skills", [])
    parsed["nice_to_have_skills"] = parsed.get("nice_to_have_skills", [])
    parsed["key_responsibilities"] = parsed.get("key_responsibilities", [])

    return parsed