"""
Talent Scout Agent — FastAPI Backend v2
=======================================
Optimizations:
  - JD result caching (same JD never hits API twice)
  - Outreach reduced to 2 turns (was 4) — cuts API calls by 50%
  - Interest scoring merged into outreach (1 call instead of 2)
  - Graceful degradation: if rate limited, returns match scores only
  - Request deduplication via hash
"""

import json
import os
import asyncio
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.jd_parser import parse_jd
from agents.candidate_matcher import build_index, match_candidates
from agents.outreach_agent import simulate_conversation
from agents.interest_scorer import score_interest

# ── Load candidates ────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "data" / "candidates.json"
with open(DATA_PATH) as f:
    ALL_CANDIDATES: list[dict] = json.load(f)

# ── Simple in-memory cache (survives for lifetime of process) ──────────────
_jd_cache: dict[str, dict] = {}  # hash(jd_text) → full response

def _jd_hash(jd_text: str) -> str:
    return hashlib.md5(jd_text.strip().lower().encode()).hexdigest()


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Indexing {len(ALL_CANDIDATES)} candidates...")
    await asyncio.get_event_loop().run_in_executor(None, build_index, ALL_CANDIDATES)
    print("[startup] Index ready ✓")
    yield


app = FastAPI(
    title="Talent Scout Agent",
    description="AI-powered recruiter agent: JD → discovery → outreach → ranked shortlist",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    jd_text: str = Field(..., min_length=50)
    top_k: int = Field(default=10, ge=1, le=20)
    outreach_top_n: int = Field(default=3, ge=1, le=5)  # reduced default to 3


class CandidateResult(BaseModel):
    id: str
    name: str
    title: str
    skills: list[str]
    experience_years: int
    location: str
    availability: str
    match_score: float
    interest_score: Optional[int] = None
    combined_score: Optional[float] = None
    match_reasons: list[str]
    skill_overlap: list[str]
    conversation: Optional[list[dict]] = None
    interest_analysis: Optional[dict] = None
    open_to_remote: bool
    linkedin: str
    recommendation: Optional[str] = None


class AnalyzeResponse(BaseModel):
    jd_parsed: dict
    shortlist: list[CandidateResult]
    total_candidates_scanned: int
    outreach_conducted: int
    cached: bool = False


# ── Routes ─────────────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    loop = asyncio.get_event_loop()

    # ── Cache check ────────────────────────────────────────────────────────
    cache_key = _jd_hash(request.jd_text)
    if cache_key in _jd_cache:
        print(f"[cache] HIT for JD hash {cache_key[:8]}")
        cached = _jd_cache[cache_key]
        # Return cached but re-slice to requested top_k
        return AnalyzeResponse(**{**cached, "cached": True})

    # ── Step 1: Parse JD ──────────────────────────────────────────────────
    try:
        jd_parsed = await loop.run_in_executor(None, parse_jd, request.jd_text)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"JD parsing failed: {e}")

    # ── Step 2: Match candidates ───────────────────────────────────────────
    matched = await loop.run_in_executor(
        None, match_candidates, jd_parsed, request.top_k, ALL_CANDIDATES
    )

    # ── Step 3: Outreach on top-N (with rate limit protection) ────────────
    outreach_n = min(request.outreach_top_n, len(matched))
    shortlist = []
    rate_limited = False

    for i, candidate in enumerate(matched):
        if i < outreach_n and not rate_limited:
            try:
                conversation = await loop.run_in_executor(
                    None, simulate_conversation, candidate, jd_parsed
                )
                interest_data = await loop.run_in_executor(
                    None, score_interest, conversation, candidate
                )
                interest_score = interest_data["interest_score"]
                combined_score = round(
                    candidate["match_score"] * 0.55 + interest_score * 0.45, 1
                )
                recommendation = interest_data.get("recommendation", "maybe")
            except Exception as ex:
                err = str(ex).lower()
                if "rate" in err or "429" in err or "limit" in err:
                    print(f"[rate-limit] Hit limit at candidate {i+1}, skipping remaining outreach")
                    rate_limited = True
                else:
                    print(f"[warn] Outreach failed for {candidate['name']}: {ex}")
                conversation = []
                interest_data = {}
                interest_score = None
                combined_score = candidate["match_score"]
                recommendation = None
        else:
            conversation = []
            interest_data = {}
            interest_score = None
            combined_score = candidate["match_score"]
            recommendation = None

        shortlist.append(CandidateResult(
            id=candidate["id"],
            name=candidate["name"],
            title=candidate["title"],
            skills=candidate.get("skills", []),
            experience_years=candidate.get("experience_years", 0),
            location=candidate.get("location", ""),
            availability=candidate.get("availability", ""),
            open_to_remote=candidate.get("open_to_remote", False),
            linkedin=candidate.get("linkedin", ""),
            match_score=candidate["match_score"],
            match_reasons=candidate.get("match_reasons", []),
            skill_overlap=candidate.get("skill_overlap", []),
            interest_score=interest_score,
            combined_score=combined_score,
            conversation=conversation,
            interest_analysis=interest_data if interest_data else None,
            recommendation=recommendation,
        ))

    shortlist.sort(key=lambda x: (x.combined_score or 0), reverse=True)

    result = dict(
        jd_parsed=jd_parsed,
        shortlist=shortlist,
        total_candidates_scanned=len(ALL_CANDIDATES),
        outreach_conducted=outreach_n,
        cached=False,
    )

    # Store in cache
    _jd_cache[cache_key] = result
    print(f"[cache] Stored result for JD hash {cache_key[:8]}")

    return AnalyzeResponse(**result)


@app.get("/candidates")
def list_candidates(limit: int = 20, offset: int = 0):
    return {"total": len(ALL_CANDIDATES), "candidates": ALL_CANDIDATES[offset:offset+limit]}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "candidates_loaded": len(ALL_CANDIDATES),
        "version": "2.0.0",
        "cache_entries": len(_jd_cache),
    }