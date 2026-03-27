"""
Backend filtering utilities for a movie recommendation system.

Each movie is expected to be a dictionary with fields such as:
  - title (str)
  - release_year (int|None)
  - runtime (int|None)
  - genres (list[str])
  - vote_average (number)
  - vote_count (int)
  - cast (list[str])
  - director (str)

All filters are implemented defensively to safely handle missing/invalid values.
"""

from __future__ import annotations

from typing import Any, Iterable


def _to_int(value: Any) -> int | None:
    """Best-effort conversion to int; returns None on failure."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int; avoid surprising results
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    """Best-effort conversion to float; returns None on failure."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:  # NaN
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_text(value: Any) -> str:
    """Normalize to lowercase stripped text for matching."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _normalize_str_list(value: Any) -> list[str]:
    """Normalize a list-like field into lowercase strings."""
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        text = _normalize_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def filter_by_year(
    movies: Iterable[dict[str, Any]],
    min_year: int | None = None,
    max_year: int | None = None,
) -> list[dict[str, Any]]:
    """Return movies whose `release_year` is within [min_year, max_year]."""
    min_y = _to_int(min_year) if min_year is not None else None
    max_y = _to_int(max_year) if max_year is not None else None

    if min_y is None and max_y is None:
        return list(movies)

    filtered: list[dict[str, Any]] = []
    for movie in movies:
        year = _to_int(movie.get("release_year"))
        if year is None:
            continue
        if min_y is not None and year < min_y:
            continue
        if max_y is not None and year > max_y:
            continue
        filtered.append(movie)
    return filtered


def filter_by_rating(
    movies: Iterable[dict[str, Any]],
    min_rating: float | None = None,
    min_votes: int | None = None,
) -> list[dict[str, Any]]:
    """Filter by `vote_average` and/or `vote_count` thresholds."""
    rating = _to_float(min_rating) if min_rating is not None else None
    votes = _to_int(min_votes) if min_votes is not None else None

    if rating is None and votes is None:
        return list(movies)

    filtered: list[dict[str, Any]] = []
    for movie in movies:
        if rating is not None:
            avg = _to_float(movie.get("vote_average"))
            if avg is None or avg < rating:
                continue
        if votes is not None:
            count = _to_int(movie.get("vote_count"))
            if count is None or count < votes:
                continue
        filtered.append(movie)
    return filtered


def filter_by_runtime(
    movies: Iterable[dict[str, Any]],
    min_runtime: int | None = None,
    max_runtime: int | None = None,
) -> list[dict[str, Any]]:
    """Filter by `runtime` duration thresholds."""
    min_r = _to_int(min_runtime) if min_runtime is not None else None
    max_r = _to_int(max_runtime) if max_runtime is not None else None

    if min_r is None and max_r is None:
        return list(movies)

    filtered: list[dict[str, Any]] = []
    for movie in movies:
        runtime = _to_int(movie.get("runtime"))
        if runtime is None:
            continue
        if min_r is not None and runtime < min_r:
            continue
        if max_r is not None and runtime > max_r:
            continue
        filtered.append(movie)
    return filtered


def filter_by_genres(
    movies: Iterable[dict[str, Any]],
    genres: list[str] | None = None,
    mode: str = "OR",
) -> list[dict[str, Any]]:
    """
    Filter by genres.

    mode="OR": match if at least one genre overlaps.
    mode="AND": match only if all requested genres are present.
    """
    requested = _normalize_str_list(genres) if genres is not None else []
    if not requested:
        return list(movies)

    mode_norm = (mode or "OR").strip().upper()
    if mode_norm not in {"OR", "AND"}:
        raise ValueError("mode must be either 'OR' or 'AND'")

    filtered: list[dict[str, Any]] = []
    for movie in movies:
        movie_genres = set(_normalize_str_list(movie.get("genres")))
        if mode_norm == "OR":
            if movie_genres.intersection(requested):
                filtered.append(movie)
        else:  # AND
            if set(requested).issubset(movie_genres):
                filtered.append(movie)
    return filtered


def filter_by_person(
    movies: Iterable[dict[str, Any]],
    person_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Match `person_name` (case-insensitive, partial matching) against:
      - any name in `cast`
      - `director`
    """
    query = _normalize_text(person_name)
    if not query:
        return list(movies)

    filtered: list[dict[str, Any]] = []
    for movie in movies:
        # Director match
        director = _normalize_text(movie.get("director"))
        if director and query in director:
            filtered.append(movie)
            continue

        # Cast match (partial substring)
        cast_list = _normalize_str_list(movie.get("cast"))
        if any(query in actor for actor in cast_list):
            filtered.append(movie)
            continue

    return filtered


def apply_all_filters(
    movies: Iterable[dict[str, Any]],
    filters: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Apply all filters sequentially using a single configuration dict.

    Supported filter keys:
      - min_year, max_year
      - min_rating, min_votes
      - min_runtime, max_runtime
      - genres, genre_mode
      - person
    """
    current = list(movies)
    if not filters:
        return current

    current = filter_by_year(
        current,
        min_year=filters.get("min_year"),
        max_year=filters.get("max_year"),
    )
    current = filter_by_rating(
        current,
        min_rating=filters.get("min_rating"),
        min_votes=filters.get("min_votes"),
    )
    current = filter_by_runtime(
        current,
        min_runtime=filters.get("min_runtime"),
        max_runtime=filters.get("max_runtime"),
    )
    current = filter_by_genres(
        current,
        genres=filters.get("genres", []),
        mode=filters.get("genre_mode", "OR"),
    )
    current = filter_by_person(
        current,
        person_name=filters.get("person"),
    )
    return current


if __name__ == "__main__":
    from nlp_utils import load_movies

    movies = load_movies()
    print(f"Total movies in dataset: {len(movies)}\n")

    # --- Test 1 ---
    filters1 = {
        "min_year": 2000,
        "genres": ["action"],
        "genre_mode": "OR"
    }

    result1 = apply_all_filters(movies, filters1)
    print("Test 1 Results:", len(result1))
    for m in result1[:5]:
        print(m["title"])
    print("\n")

    # --- Test 2 ---
    filters2 = {
        "genres": ["action", "science fiction"],
        "genre_mode": "AND"
    }

    result2 = apply_all_filters(movies, filters2)
    print("Test 2 Results:", len(result2))
    for m in result2[:5]:
        print(m["title"])
    print("\n")

    # --- Test 3 ---
    filters3 = {
        "person": "nolan"
    }

    result3 = apply_all_filters(movies, filters3)
    print("Test 3 Results:", len(result3))
    for m in result3[:5]:
        print(m["title"])
    


