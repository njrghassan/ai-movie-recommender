"""TMDB API client utilities for movie recommendation pipelines.

This module provides reusable functions to fetch movie data from TMDB and
normalize key fields used by recommendation systems.
"""

from __future__ import annotations

import functools
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    load_dotenv = None

# Load environment variables from .env in the current working directory.
if load_dotenv is not None:
    load_dotenv()

BASE_URL = "https://api.themoviedb.org/3"
REQUEST_TIMEOUT = 10
DEFAULT_LANGUAGE = "en-US"

_DOTENV_CACHE: dict[str, str] | None = None
_TMDB_DEBUG = False


def _log_debug(message: str) -> None:
    if _TMDB_DEBUG:
        print(message)


def _load_dotenv_fallback() -> dict[str, str]:
    """
    Minimal `.env` loader (KEY=VALUE) for environments without python-dotenv.
    """
    global _DOTENV_CACHE
    if _DOTENV_CACHE is not None:
        return _DOTENV_CACHE

    env: dict[str, str] = {}
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and v:
                    env[k] = v
    except Exception:
        env = {}

    _DOTENV_CACHE = env
    return env


def _get_env(name: str) -> str | None:
    v = os.getenv(name)
    if isinstance(v, str) and v.strip():
        return v.strip()
    fallback = _load_dotenv_fallback().get(name)
    return fallback.strip() if isinstance(fallback, str) and fallback.strip() else None


_TMDB_DEBUG = (_get_env("TMDB_DEBUG") or "").strip().lower() in {"1", "true", "yes", "on"}


def _get_api_key() -> str | None:
    """Return TMDB API key from environment, or None if missing."""
    return _get_env("TMDB_API_KEY")


def _safe_get(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform a safe GET request to TMDB and return JSON or an empty dict.

    This helper centralizes request setup and error handling so endpoint
    functions stay concise and consistent.
    """
    api_key = _get_api_key()
    if not api_key:
        return {}

    query_params = dict(params or {})
    query_params["api_key"] = api_key
    query_params.setdefault("language", DEFAULT_LANGUAGE)

    qs = urllib.parse.urlencode({k: v for k, v in query_params.items() if v is not None})
    url = f"{BASE_URL}{endpoint}?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        _log_debug(f"TMDB request failed for {endpoint}: {exc}")
        return {}
    except ValueError as exc:
        _log_debug(f"Invalid JSON returned for {endpoint}: {exc}")
        return {}


def get_popular_movies(page: int = 1) -> dict[str, Any]:
    """Return JSON payload of popular movies for a page."""
    if page < 1:
        page = 1
    return _safe_get("/movie/popular", params={"page": page})


def get_trending_movies() -> list[dict[str, Any]]:
    """
    Return a clean list of trending movies from `/trending/movie/day`.

    Each item includes:
      - title
      - release_year
      - rating
      - overview

    Returns an empty list if TMDB request fails or payload is invalid.
    """
    payload = _safe_get("/trending/movie/day", params={})
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for movie in results:
        if not isinstance(movie, dict):
            continue
        title = movie.get("title") or movie.get("name") or ""
        if not isinstance(title, str) or not title.strip():
            continue

        cleaned.append(
            {
                "title": title.strip(),
                "release_year": _extract_release_year(movie.get("release_date")),
                "rating": movie.get("vote_average"),
                "overview": (movie.get("overview") or "").strip(),
            }
        )
    return cleaned


def get_movie_details(movie_id: int) -> dict[str, Any]:
    """Return detailed movie metadata (runtime, genres, overview, etc.)."""
    if not movie_id:
        return {}
    return _safe_get(f"/movie/{movie_id}")


def get_movie_credits(movie_id: int) -> dict[str, Any]:
    """Return cast and crew information for a movie."""
    if not movie_id:
        return {}
    return _safe_get(f"/movie/{movie_id}/credits")


def get_movie_keywords(movie_id: int) -> dict[str, Any]:
    """Return keyword metadata for NLP and content filtering."""
    if not movie_id:
        return {}
    return _safe_get(f"/movie/{movie_id}/keywords")


def fetch_full_movie_data(movie_id: int) -> dict[str, Any]:
    """Combine details, credits, and keywords into recommender-ready structure.

    Returns an empty dict if core details cannot be fetched.
    """
    details = get_movie_details(movie_id)
    if not details:
        return {}

    credits = get_movie_credits(movie_id)
    keywords_payload = get_movie_keywords(movie_id)

    genres = [genre.get("name") for genre in details.get("genres", []) if genre.get("name")]

    keywords = [
        kw.get("name")
        for kw in keywords_payload.get("keywords", [])
        if kw.get("name")
    ]

    cast = [
        person.get("name")
        for person in credits.get("cast", [])[:5]
        if person.get("name")
    ]

    director = next(
        (
            crew_member.get("name")
            for crew_member in credits.get("crew", [])
            if crew_member.get("job") == "Director" and crew_member.get("name")
        ),
        None,
    )

    return {
        "id": details.get("id", movie_id),
        "title": details.get("title", ""),
        "release_date": details.get("release_date", ""),
        "release_year": _extract_release_year(details.get("release_date")),
        "year": _extract_release_year(details.get("release_date")),
        "runtime": details.get("runtime"),
        "genres": genres,
        "overview": details.get("overview", ""),
        "keywords": keywords,
        "cast": cast,
        "director": director,
        "vote_average": details.get("vote_average"),
        "vote_count": details.get("vote_count"),
    }


@functools.lru_cache(maxsize=2048)
def get_movie_full_record_tmdb(movie_id: int) -> dict[str, Any]:
    """
    Cached normalized full record for a movie id.

    Shape matches what `filters.apply_all_filters()` expects.
    """
    if not isinstance(movie_id, int) or movie_id <= 0:
        return {}
    return fetch_full_movie_data(movie_id)


def _extract_release_year(release_date: Any) -> int | None:
    if not isinstance(release_date, str) or len(release_date) < 4:
        return None
    y = release_date[:4]
    return int(y) if y.isdigit() else None


def _normalize_list_of_dict_names(value: Any, key: str = "name") -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get(key)
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


@functools.lru_cache(maxsize=1)
def get_movie_genre_map() -> dict[int, str]:
    """
    Return TMDB genre_id -> name map (cached for process lifetime).
    """
    payload = _safe_get("/genre/movie/list", params={})
    genres = payload.get("genres", []) if isinstance(payload, dict) else []
    if not isinstance(genres, list):
        return {}
    out: dict[int, str] = {}
    for g in genres:
        if not isinstance(g, dict):
            continue
        gid = g.get("id")
        name = g.get("name")
        if isinstance(gid, int) and isinstance(name, str) and name.strip():
            out[gid] = name.strip()
    return out


def _movie_summary_from_tmdb(movie: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a TMDB movie result to the app's normalized movie shape.
    """
    genre_map = get_movie_genre_map()
    genre_ids = movie.get("genre_ids")
    if isinstance(genre_ids, list):
        genres = [genre_map.get(gid) for gid in genre_ids if isinstance(gid, int)]
        genres = [g for g in genres if g]
    else:
        genres = _normalize_list_of_dict_names(movie.get("genres"))

    release_date = movie.get("release_date")
    return {
        "id": movie.get("id"),
        "title": movie.get("title") or movie.get("name") or "",
        "release_year": _extract_release_year(release_date),
        "year": _extract_release_year(release_date),
        "runtime": movie.get("runtime"),
        "genres": genres,
        "overview": movie.get("overview") or "",
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
    }


def search_movies_tmdb(
    query: str,
    *,
    page: int = 1,
    include_adult: bool = False,
    year: int | None = None,
) -> list[dict[str, Any]]:
    """
    Free-text search for movies.
    """
    q = (query or "").strip()
    if not q:
        return []
    params: dict[str, Any] = {
        "query": q,
        "page": max(1, int(page)),
        "include_adult": include_adult,
    }
    if isinstance(year, int) and year > 1800:
        params["year"] = year
    payload = _safe_get("/search/movie", params=params)
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return []
    out: list[dict[str, Any]] = []
    for m in results:
        if not isinstance(m, dict):
            continue
        rec = _movie_summary_from_tmdb(m)
        if isinstance(rec.get("id"), int) and str(rec.get("title") or "").strip():
            out.append(rec)
    return out


def search_person_tmdb(query: str, *, page: int = 1) -> list[dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return []
    payload = _safe_get("/search/person", params={"query": q, "page": max(1, int(page))})
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return []
    out: list[dict[str, Any]] = []
    for p in results:
        if not isinstance(p, dict):
            continue
        pid = p.get("id")
        name = p.get("name")
        if isinstance(pid, int) and isinstance(name, str) and name.strip():
            out.append({"id": pid, "name": name.strip()})
    return out


def discover_movies_tmdb(
    *,
    page: int = 1,
    genre_names: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = None,
    min_votes: int | None = None,
    min_runtime: int | None = None,
    max_runtime: int | None = None,
    person: str | None = None,
    sort_by: str = "popularity.desc",
) -> list[dict[str, Any]]:
    """
    Structured discovery using TMDB's `/discover/movie`.
    """
    params: dict[str, Any] = {"page": max(1, int(page)), "sort_by": sort_by}

    if isinstance(min_rating, (int, float)):
        params["vote_average.gte"] = float(min_rating)
    if isinstance(min_votes, int) and min_votes > 0:
        params["vote_count.gte"] = int(min_votes)
    if isinstance(min_runtime, int) and min_runtime > 0:
        params["with_runtime.gte"] = int(min_runtime)
    if isinstance(max_runtime, int) and max_runtime > 0:
        params["with_runtime.lte"] = int(max_runtime)

    if isinstance(min_year, int) and min_year > 1800:
        params["primary_release_date.gte"] = f"{min_year}-01-01"
    if isinstance(max_year, int) and max_year > 1800:
        params["primary_release_date.lte"] = f"{max_year}-12-31"

    if genre_names:
        wanted = {str(g).strip().lower() for g in genre_names if str(g).strip()}
        if wanted:
            genre_map = get_movie_genre_map()
            name_to_id = {name.lower(): gid for gid, name in genre_map.items()}
            ids = [str(name_to_id[g]) for g in wanted if g in name_to_id]
            if ids:
                params["with_genres"] = ",".join(ids)

    if isinstance(person, str) and person.strip():
        people = search_person_tmdb(person.strip(), page=1)
        if people:
            params["with_people"] = people[0]["id"]

    payload = _safe_get("/discover/movie", params=params)
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return []
    out: list[dict[str, Any]] = []
    for m in results:
        if not isinstance(m, dict):
            continue
        rec = _movie_summary_from_tmdb(m)
        if isinstance(rec.get("id"), int) and str(rec.get("title") or "").strip():
            out.append(rec)
    return out


def get_movie_recommendations_tmdb(movie_id: int, *, page: int = 1) -> list[dict[str, Any]]:
    if not isinstance(movie_id, int) or movie_id <= 0:
        return []
    payload = _safe_get(
        f"/movie/{movie_id}/recommendations",
        params={"page": max(1, int(page))},
    )
    results = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(results, list):
        return []
    out: list[dict[str, Any]] = []
    for m in results:
        if not isinstance(m, dict):
            continue
        rec = _movie_summary_from_tmdb(m)
        if isinstance(rec.get("id"), int) and str(rec.get("title") or "").strip():
            out.append(rec)
    return out


if __name__ == "__main__":
    popular = get_popular_movies(page=1)
    results = popular.get("results", [])[:3] if popular else []

    if not results:
        print("No popular movies found or request failed.")
    else:
        print("Fetched 3 popular movies:")
        for movie in results:
            print(f"- {movie.get('title', 'Unknown Title')} (id={movie.get('id', 'N/A')})")

        first_movie_id = results[0].get("id")
        if first_movie_id:
            print("\nStructured data for first movie:")
            print(fetch_full_movie_data(first_movie_id))
        else:
            print("First movie missing ID; cannot fetch full movie data.")
