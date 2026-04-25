"""
Talent Scout Agent — FastAPI Backend
=====================================
Main entry point. Exposes:
  POST /analyze          — Full pipeline: JD → match → outreach → score → shortlist
  GET  /candidates       — List all indexed candidates (for debugging)
  GET  /health           — Health check
"""

import json
import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
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

# ── Lifespan: build vector index on startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Indexing {len(ALL_CANDIDATES)} candidates into ChromaDB...")
    await asyncio.get_event_loop().run_in_executor(
        None, build_index, ALL_CANDIDATES
    )
    print("[startup] Index ready ✓")
    yield
    print("[shutdown] Goodbye.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Talent Scout Agent",
    description="AI-powered recruiter agent: JD → candidate discovery → conversational outreach → ranked shortlist",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    jd_text: str = Field(..., min_length=50, description="Raw job description text")
    top_k: int = Field(default=10, ge=1, le=30, description="Total candidates to retrieve")
    outreach_top_n: int = Field(default=5, ge=1, le=10, description="How many top matches to run outreach on")


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


# ── Routes ─────────────────────────────────────────────────────────────────
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Full pipeline:
    1. Parse the JD with Claude
    2. Retrieve top-K matching candidates via semantic search + skill scoring
    3. For top-N, simulate conversational outreach and score interest
    4. Return combined ranked shortlist
    """
    # Step 1: Parse JD
    try:
        jd_parsed = await asyncio.get_event_loop().run_in_executor(
            None, parse_jd, request.jd_text
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"JD parsing failed: {e}")

    # Step 2: Match candidates
    matched = await asyncio.get_event_loop().run_in_executor(
        None, match_candidates, jd_parsed, request.top_k, ALL_CANDIDATES
    )

    # Step 3: Outreach on top-N
    outreach_n = min(request.outreach_top_n, len(matched))
    shortlist = []

    for i, candidate in enumerate(matched):
        if i < outreach_n:
            # Run outreach + interest scoring
            try:
                conversation = await asyncio.get_event_loop().run_in_executor(
                    None, simulate_conversation, candidate, jd_parsed
                )
                interest_data = await asyncio.get_event_loop().run_in_executor(
                    None, score_interest, conversation, candidate
                )
                interest_score = interest_data["interest_score"]
                # Weighted: 55% match, 45% interest
                combined_score = round(
                    candidate["match_score"] * 0.55 + interest_score * 0.45, 1
                )
                recommendation = interest_data.get("recommendation", "maybe")
            except Exception as ex:
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

        shortlist.append(
            CandidateResult(
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
            )
        )

    # Sort final shortlist by combined_score desc
    shortlist.sort(key=lambda x: (x.combined_score or 0), reverse=True)

    return AnalyzeResponse(
        jd_parsed=jd_parsed,
        shortlist=shortlist,
        total_candidates_scanned=len(ALL_CANDIDATES),
        outreach_conducted=outreach_n,
    )


@app.get("/candidates")
def list_candidates(limit: int = 20, offset: int = 0):
    """Return paginated raw candidate list (for debugging)."""
    return {
        "total": len(ALL_CANDIDATES),
        "candidates": ALL_CANDIDATES[offset : offset + limit],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "candidates_loaded": len(ALL_CANDIDATES),
        "version": "1.0.0",
    }