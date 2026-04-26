"""
JD Parser Agent with multi-model fallback.
Tries models in order until one succeeds.
"""

import json
import os
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Model fallback chain — if one hits rate limit, try next
MODELS = [
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


def parse_jd(jd_text: str) -> dict:
    prompt = f"""You are a precise job description parser. Extract structured information.

Return ONLY valid JSON — no markdown, no backticks, no explanation.

Schema:
{{
  "title": "job title",
  "required_skills": ["list", "of", "must-have", "skills"],
  "nice_to_have_skills": ["list", "of", "preferred", "skills"],
  "experience_years_min": 0,
  "experience_years_max": 10,
  "role_type": "technical | managerial | hybrid",
  "seniority": "junior | mid | senior | lead | principal",
  "key_responsibilities": ["3-5 main responsibilities"],
  "industry_context": "domain/industry if mentioned",
  "summary": "2 sentence summary of the ideal candidate"
}}

Job Description:
\"\"\"{jd_text}\"\"\"

Return ONLY the JSON object."""

    last_error = None
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1200,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            parsed["experience_years_min"] = int(parsed.get("experience_years_min", 0))
            parsed["experience_years_max"] = int(parsed.get("experience_years_max", 20))
            parsed["required_skills"] = parsed.get("required_skills", [])
            parsed["nice_to_have_skills"] = parsed.get("nice_to_have_skills", [])
            parsed["key_responsibilities"] = parsed.get("key_responsibilities", [])
            print(f"[jd_parser] Used model: {model}")
            return parsed
        except Exception as e:
            err = str(e)
            if "rate" in err.lower() or "429" in err or "limit" in err.lower():
                print(f"[jd_parser] Model {model} rate limited, trying next...")
                last_error = e
                continue
            raise ValueError(f"JD parsing failed: {e}")

    raise ValueError(f"All models rate limited. Try again in a few minutes. Last error: {last_error}")