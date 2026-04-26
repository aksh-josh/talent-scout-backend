# TalentScout AI — Backend

> AI-Powered Talent Scouting & Engagement Agent  
> Built for **Catalyst Hackathon by Deccan AI** · April 2026  
> By **Akshat Joshi**

[![Railway](https://img.shields.io/badge/Deployed-Railway-blueviolet)](https://web-production-c301c.up.railway.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)

---

## What This Does

A 4-agent AI pipeline that takes a Job Description and returns a ranked candidate shortlist scored on two dimensions:

| Dimension | Weight | How |
|---|---|---|
| **Match Score** | 55% | Semantic search + skill overlap + experience fit |
| **Interest Score** | 45% | Analysis of simulated recruiter conversation |
| **Combined Score** | Final | Weighted combination of both |

---

## Agent Pipeline

```
JD Text Input
    ↓
[Agent 1] JD Parser         → Extracts: skills, experience, role type, responsibilities
    ↓
[Agent 2] Candidate Matcher → ChromaDB semantic search + skill overlap + experience scoring
    ↓
[Agent 3] Outreach Agent    → Simulates 2-turn recruiter ↔ candidate conversation
    ↓
[Agent 4] Interest Scorer   → Scores conversation for enthusiasm, availability, fit
    ↓
Ranked Shortlist (Match + Interest + Combined scores with explainability)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| LLM | Groq API (llama-3.1-8b-instant + 3 fallback models) |
| Vector Search | ChromaDB with built-in onnxruntime embeddings |
| Language | Python 3.11 |
| Deployment | Railway.app (free tier) |

---

## Key Features

- **Multi-model fallback chain** — 4 Groq models in sequence. If one hits rate limits, auto-switches to the next
- **Request caching** — MD5 hash of JD text → cached response. Same JD never hits the API twice
- **Graceful degradation** — If all models rate-limited, returns match scores without crashing
- **Explainable results** — Every candidate includes match reasons and exact skill overlaps

---

## Local Setup

### Prerequisites
- Python 3.11+
- Groq API key (free at console.groq.com)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/aksh-josh/talent-scout-backend.git
cd talent-scout-backend

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env file
echo "GROQ_API_KEY=your_key_here" > .env

# 6. Generate candidate data (run once)
python generate_candidates.py

# 7. Start the server
uvicorn main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## API Reference

### `POST /analyze`
Full pipeline: JD → match → outreach → shortlist

**Request:**
```json
{
  "jd_text": "We are looking for a Senior Python Engineer...",
  "top_k": 10,
  "outreach_top_n": 3
}
```

**Response:**
```json
{
  "jd_parsed": {
    "title": "Senior Python Engineer",
    "required_skills": ["Python", "FastAPI"],
    "experience_years_min": 4,
    "seniority": "senior"
  },
  "shortlist": [
    {
      "name": "Priya Sharma",
      "title": "Senior Backend Developer",
      "match_score": 87.4,
      "interest_score": 82,
      "combined_score": 85.0,
      "recommendation": "strong yes",
      "match_reasons": ["Matched required skills: Python, FastAPI"],
      "skill_overlap": ["Python", "FastAPI"],
      "conversation": [...],
      "interest_analysis": {
        "positive_signals": ["Expressed enthusiasm for the role"],
        "concerns": [],
        "interest_reasoning": "Candidate showed strong interest..."
      }
    }
  ],
  "total_candidates_scanned": 100,
  "outreach_conducted": 3,
  "cached": false
}
```

### `GET /health`
Returns system status.

### `GET /candidates`
Returns paginated candidate list (for debugging).

---

## Scoring Logic

```
Match Score  = cosine_similarity(0-60) + skill_overlap(0-30) + experience_fit(0-10)
Interest Score = LLM analysis of 2-turn conversation (0-100)
Combined Score = Match × 0.55 + Interest × 0.45
```

---

## Project Structure

```
talent-scout-backend/
├── main.py                  # FastAPI app, routes, caching
├── generate_candidates.py   # One-time script to create 100 synthetic profiles
├── agents/
│   ├── jd_parser.py         # Agent 1: LLM-based JD parsing
│   ├── candidate_matcher.py # Agent 2: ChromaDB semantic search + scoring
│   ├── outreach_agent.py    # Agent 3: 2-turn conversation simulation
│   └── interest_scorer.py   # Agent 4: Conversation analysis + interest score
├── data/
│   └── candidates.json      # 100 synthetic candidate profiles
├── requirements.txt
├── Procfile                 # Railway deployment config
└── runtime.txt              # Python version spec
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from console.groq.com |

---

## Deployment

Deployed on Railway. Push to `main` branch triggers automatic redeploy.

**Live URL:** `https://web-production-c301c.up.railway.app`

---

## APIs & Cost

All services used are within free tiers. Total cost: **$0**

| Service | Usage |
|---|---|
| Groq API | Free tier — 4 models with separate quotas |
| ChromaDB | Open source, runs locally |
| Railway | Free tier ($5 credit included) |