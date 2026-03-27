from __future__ import annotations

import math
from typing import Any

from filters import apply_all_filters
from nl_query import parse_query
from recommender import get_recommendations_from_pool
from tmdb_api import (
    discover_movies_tmdb,
    get_movie_full_record_tmdb,
    get_movie_recommendations_tmdb,
    get_popular_movies,
    search_movies_tmdb,
)


def _filters_are_empty(filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    return (
        not filters.get("genres")
        and not filters.get("person")
        and filters.get("min_year") is None
        and filters.get("max_year") is None
        and filters.get("min_rating") is None
        and filters.get("min_votes") is None
        and filters.get("min_runtime") is None
        and filters.get("max_runtime") is None
    )


def _pick_seed_from_results(results: list[dict[str, Any]], max_seeds: int = 3) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = [
        r
        for r in results
        if isinstance(r, dict)
        and isinstance(r.get("id"), int)
        and str(r.get("title") or "").strip()
    ]
    if not usable:
        return []

    def _score_seed(x: dict[str, Any]) -> float:
        # Prefer widely-voted titles; fall back to rating.
        vc = x.get("vote_count")
        va = x.get("vote_average")
        vote_count = float(vc) if isinstance(vc, (int, float)) else 0.0
        vote_avg = float(va) if isinstance(va, (int, float)) else 0.0
        return (math.log10(1.0 + max(0.0, vote_count)) * 2.0) + (vote_avg / 10.0)

    usable.sort(key=_score_seed, reverse=True)
    deduped: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for m in usable:
        mid = m.get("id")
        if not isinstance(mid, int) or mid in seen_ids:
            continue
        seen_ids.add(mid)
        deduped.append(m)
        if len(deduped) >= max(1, int(max_seeds)):
            break
    return deduped


def _query_terms(query: str) -> list[str]:
    q = (query or "").strip().lower()
    if not q:
        return []
    raw = [t for t in q.replace("-", " ").split() if len(t) >= 3]
    # de-dupe, keep order
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:12]


def _collect_candidate_pool(
    *,
    query: str,
    filters: dict[str, Any],
    seeds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build a rich candidate pool from TMDB search/discover/recommendations/popular.
    """
    pool: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def _add(movie: dict[str, Any]) -> None:
        if not isinstance(movie, dict):
            return
        mid = movie.get("id")
        if not isinstance(mid, int) or mid in seen_ids:
            return
        full = get_movie_full_record_tmdb(mid)
        record = full if full else movie
        record = dict(record)
        record["year"] = record.get("year", record.get("release_year"))
        pool.append(record)
        seen_ids.add(mid)

    # 1) Free-text search pages
    for p in (1, 2):
        for m in search_movies_tmdb(query, page=p):
            _add(m)

    # 2) Discover pages using structured filters
    for p in (1, 2):
        for m in discover_movies_tmdb(
            page=p,
            genre_names=filters.get("genres") or [],
            min_year=filters.get("min_year"),
            max_year=filters.get("max_year"),
            min_rating=filters.get("min_rating"),
            min_votes=filters.get("min_votes"),
            min_runtime=filters.get("min_runtime"),
            max_runtime=filters.get("max_runtime"),
            person=filters.get("person"),
        ):
            _add(m)

    # 3) Seed-based TMDB recommendations
    for s in seeds:
        sid = s.get("id")
        if not isinstance(sid, int):
            continue
        for p in (1, 2):
            for m in get_movie_recommendations_tmdb(sid, page=p):
                _add(m)

    # 4) Popular fallback for diversity / cold-start
    for m in (get_popular_movies(page=1).get("results") or []):
        if isinstance(m, dict):
            _add(m)

    return pool


def _compute_reco_score(
    *,
    seed: dict[str, Any],
    candidate: dict[str, Any],
    query: str,
) -> float:
    """
    Lightweight score in [0, 1] used only for UI ranking.
    """
    # Rating component
    va = candidate.get("vote_average")
    rating = float(va) / 10.0 if isinstance(va, (int, float)) else 0.0

    # Genre overlap component
    seed_genres = {str(g).strip().lower() for g in (seed.get("genres") or []) if str(g).strip()}
    cand_genres = {str(g).strip().lower() for g in (candidate.get("genres") or []) if str(g).strip()}
    if seed_genres and cand_genres:
        genre_sim = len(seed_genres.intersection(cand_genres)) / float(len(seed_genres.union(cand_genres)))
    else:
        genre_sim = 0.0

    # Query-overview overlap component
    overview = str(candidate.get("overview") or "").lower()
    terms = _query_terms(query)
    if terms and overview:
        hits = sum(1 for t in terms if t in overview)
        text_sim = hits / float(len(terms))
    else:
        text_sim = 0.0

    # Vote-count confidence
    vc = candidate.get("vote_count")
    vote_count = float(vc) if isinstance(vc, (int, float)) else 0.0
    confidence = min(1.0, math.log10(1.0 + max(0.0, vote_count)) / 4.0)

    score = (0.40 * rating) + (0.30 * genre_sim) + (0.20 * text_sim) + (0.10 * confidence)
    if score != score:  # NaN
        return 0.0
    return float(max(0.0, min(1.0, score)))


def search_movies(query: str, top_n: int = 10) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Take a natural-language query and return filtered + recommended movies.

    Returns:
      - [] if no movies match the parsed filters
      - dict with keys: filters, seed_movie, recommendations
    """

    # Use the local rule-based parser for a stable submission (no external LLM dependency).
    # This still supports natural-language queries, but avoids runtime failures if an LLM
    # service/API key is unavailable.
    filters = parse_query(query)
    # Keep a consistent filter shape for downstream code.
    filters.setdefault("min_votes", None)
    filters.setdefault("min_runtime", None)
    filters.setdefault("max_runtime", None)
    if _filters_are_empty(filters):
        seed_candidates = search_movies_tmdb(query, page=1)
    else:
        seed_candidates = discover_movies_tmdb(
            page=1,
            genre_names=filters.get("genres") or [],
            min_year=filters.get("min_year"),
            max_year=filters.get("max_year"),
            min_rating=filters.get("min_rating"),
            min_votes=filters.get("min_votes"),
            min_runtime=filters.get("min_runtime"),
            max_runtime=filters.get("max_runtime"),
            person=filters.get("person"),
        )

    seeds = _pick_seed_from_results(seed_candidates, max_seeds=3)
    if not seeds:
        return []

    enriched_seeds: list[dict[str, Any]] = []
    for s in seeds:
        sid = s.get("id")
        if isinstance(sid, int):
            full = get_movie_full_record_tmdb(sid)
            enriched_seeds.append(full if full else s)
        else:
            enriched_seeds.append(s)
    seeds = enriched_seeds

    # Build one shared TMDB-backed pool, then apply your filter module.
    candidate_pool = _collect_candidate_pool(query=query, filters=filters, seeds=seeds)
    filtered_pool = apply_all_filters(candidate_pool, filters)

    # Fallback logic: real user queries often include constraints that are too strict
    # for the available candidate pool (especially year ranges). Instead of returning
    # "no results" and feeling broken, we relax constraints progressively to keep
    # the experience helpful.
    if not filtered_pool:
        relaxed_filters = dict(filters)
        # Year limits are common causes of empty matches; remove them first.
        relaxed_filters.pop("min_year", None)
        relaxed_filters.pop("max_year", None)
        filtered_pool = apply_all_filters(candidate_pool, relaxed_filters)

    # If still empty, fall back to the full candidate pool so we can return
    # recommendations rather than failing hard.
    if not filtered_pool:
        filtered_pool = candidate_pool

    # Build recommendations from each seed using your NLP/content recommender.
    merged_by_id: dict[int, dict[str, Any]] = {}
    fallback_without_id: list[dict[str, Any]] = []
    for s in seeds:
        seed_recs = get_recommendations_from_pool(s, filtered_pool, top_n=max(10, int(top_n) * 3))
        for r in seed_recs:
            rr = dict(r)
            rr["year"] = rr.get("year", rr.get("release_year"))
            rid = rr.get("id")
            if isinstance(rid, int):
                prev = merged_by_id.get(rid)
                if not prev or float(rr.get("score") or 0.0) > float(prev.get("score") or 0.0):
                    merged_by_id[rid] = rr
            else:
                fallback_without_id.append(rr)

    recommendations = list(merged_by_id.values()) + fallback_without_id

    # Optional blended score for UI readability (keeps recommender score as primary signal).
    primary_seed = seeds[0] if seeds else {}
    for r in recommendations:
        algo_score = float(r.get("score") or 0.0)
        heuristic = _compute_reco_score(seed=primary_seed, candidate=r, query=query)
        r["score"] = float(max(0.0, min(1.0, (0.75 * algo_score) + (0.25 * heuristic))))

    recommendations.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

    if top_n is not None:
        try:
            n = max(1, int(top_n))
            recommendations = recommendations[:n]
        except Exception:
            pass

    seed_titles = [str(s.get("title") or "").strip() for s in seeds if str(s.get("title") or "").strip()]

    return {
        "filters": filters,
        "seed_movie": seed_titles[0] if seed_titles else "",
        "seed_movies": seed_titles[:3],
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    queries = [
        "action movies after 2015",
        "science fiction with nolan",
        "drama before 2000",
    ]

    for q in queries:
        result = search_movies(q, top_n=10)
        print("QUERY:", q)

        if not result:
            print("Parsed filters:", parse_query(q))
            print("Seed movie:", None)
            print("Recommendations:", [])
        else:
            print("Parsed filters:", result["filters"])
            print("Seed movie:", result["seed_movie"])
            print("Recommendations:", result["recommendations"])

        print()