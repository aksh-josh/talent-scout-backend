"""
Candidate Matcher
-----------------
Uses ChromaDB's built-in default embedding function (onnxruntime-based,
no PyTorch / sentence-transformers required — fast build on Railway).
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

_VECTORSTORE_PATH = str(Path(__file__).parent.parent / "vectorstore")
_chroma_client: Optional[chromadb.PersistentClient] = None
_embed_fn = embedding_functions.DefaultEmbeddingFunction()


def _get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=_VECTORSTORE_PATH)
    return _chroma_client


def _candidate_to_text(c: dict) -> str:
    skills_str = " ".join(c.get("skills", []))
    return (
        f"{c['title']} {c.get('track', '')} "
        f"skills: {skills_str} "
        f"experience: {c.get('experience_years', 0)} years "
        f"location: {c.get('location', '')} "
        f"{c.get('summary', '')}"
    )


def build_index(candidates: list[dict], force_rebuild: bool = False) -> None:
    client = _get_chroma_client()

    if force_rebuild:
        try:
            client.delete_collection("candidates")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        "candidates",
        embedding_function=_embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids: set[str] = set()
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])
    except Exception:
        pass

    new_candidates = [c for c in candidates if c["id"] not in existing_ids]
    if not new_candidates:
        print(f"[index] All {len(candidates)} candidates already indexed.")
        return

    texts = [_candidate_to_text(c) for c in new_candidates]
    collection.add(
        ids=[c["id"] for c in new_candidates],
        documents=texts,
        metadatas=[
            {
                "name": c["name"],
                "title": c["title"],
                "track": c.get("track", ""),
                "skills_json": json.dumps(c.get("skills", [])),
                "experience_years": c.get("experience_years", 0),
                "location": c.get("location", ""),
                "availability": c.get("availability", ""),
                "summary": c.get("summary", ""),
                "open_to_remote": str(c.get("open_to_remote", False)),
                "open_to_relocation": str(c.get("open_to_relocation", False)),
                "linkedin": c.get("linkedin", ""),
                "expected_ctc_lpa": c.get("expected_ctc_lpa", 0),
            }
            for c in new_candidates
        ],
    )
    print(f"[index] Indexed {len(new_candidates)} candidates.")


def _skill_overlap_score(candidate_skills: list[str], jd: dict) -> float:
    required = set(s.lower() for s in jd.get("required_skills", []))
    nice = set(s.lower() for s in jd.get("nice_to_have_skills", []))
    candidate = set(s.lower() for s in candidate_skills)
    if not required:
        return 15.0
    req_hit = len(required & candidate) / len(required)
    nice_hit = len(nice & candidate) / max(len(nice), 1)
    return round(req_hit * 25 + nice_hit * 5, 1)


def _experience_score(candidate_exp: int, jd: dict) -> float:
    exp_min = jd.get("experience_years_min", 0)
    exp_max = jd.get("experience_years_max", 20)
    if exp_min <= candidate_exp <= exp_max:
        return 10.0
    elif candidate_exp < exp_min:
        return max(0.0, 10.0 - (exp_min - candidate_exp) * 2.5)
    return 8.0


def match_candidates(jd_parsed: dict, top_k: int = 10, all_candidates: list[dict] = None) -> list[dict]:
    client = _get_chroma_client()
    collection = client.get_collection("candidates", embedding_function=_embed_fn)

    query_text = (
        f"{jd_parsed.get('title', '')} "
        f"{' '.join(jd_parsed.get('required_skills', []))} "
        f"{jd_parsed.get('summary', '')}"
    )

    fetch_k = min(top_k * 3, 50)
    results = collection.query(
        query_texts=[query_text],
        n_results=fetch_k,
        include=["metadatas", "distances"],
    )

    candidate_lookup = {c["id"]: c for c in all_candidates} if all_candidates else {}
    enriched = []

    for i, (meta, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
        cid = results["ids"][0][i]
        candidate_skills = json.loads(meta.get("skills_json", "[]"))
        exp_years = meta.get("experience_years", 0)

        vector_score = round((1 - distance) * 60, 1)
        skill_score = _skill_overlap_score(candidate_skills, jd_parsed)
        exp_score = _experience_score(exp_years, jd_parsed)
        match_score = min(100.0, round(vector_score + skill_score + exp_score, 1))

        required_lower = set(s.lower() for s in jd_parsed.get("required_skills", []))
        skill_overlap = [s for s in candidate_skills if s.lower() in required_lower]

        match_reasons = []
        if skill_overlap:
            match_reasons.append(f"Matched required skills: {', '.join(skill_overlap)}")
        if exp_years >= jd_parsed.get("experience_years_min", 0):
            match_reasons.append(f"{exp_years} years experience meets requirement")
        if not match_reasons:
            match_reasons.append("Profile semantically similar to role requirements")

        full_candidate = candidate_lookup.get(cid, {
            "id": cid,
            "name": meta["name"],
            "title": meta["title"],
            "track": meta["track"],
            "skills": candidate_skills,
            "experience_years": exp_years,
            "location": meta["location"],
            "availability": meta["availability"],
            "summary": meta["summary"],
            "open_to_remote": meta.get("open_to_remote") == "True",
            "open_to_relocation": meta.get("open_to_relocation") == "True",
            "linkedin": meta.get("linkedin", ""),
            "expected_ctc_lpa": meta.get("expected_ctc_lpa", 0),
        })

        enriched.append({**full_candidate, "match_score": match_score,
                         "match_reasons": match_reasons, "skill_overlap": skill_overlap})

    enriched.sort(key=lambda x: x["match_score"], reverse=True)
    return enriched[:top_k]