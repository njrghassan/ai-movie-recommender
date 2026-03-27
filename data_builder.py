"""Build a local movie dataset from TMDB popular movies."""

from __future__ import annotations

import json
import time
from typing import Any

from tmdb_api import fetch_full_movie_data, get_popular_movies

DATASET_PATH = "movies_dataset.json"
RATE_LIMIT_SECONDS = 0.2


def _to_lower_text(value: Any) -> str:
    """Return lowercase stripped text for string-like values."""
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def _to_lower_list(values: Any) -> list[str]:
    """Return cleaned lowercase list from iterable text values."""
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for item in values:
        text = _to_lower_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def _extract_release_year(release_date: Any) -> int | None:
    """Extract release year from YYYY-MM-DD string."""
    if not isinstance(release_date, str) or len(release_date) < 4:
        return None
    year_text = release_date[:4]
    return int(year_text) if year_text.isdigit() else None


def _build_movie_record(popular_movie: dict[str, Any]) -> dict[str, Any]:
    """Create a normalized movie record from popular + full TMDB data."""
    movie_id = popular_movie.get("id")
    if not isinstance(movie_id, int):
        return {}

    full_data = fetch_full_movie_data(movie_id)
    time.sleep(RATE_LIMIT_SECONDS)
    if not full_data:
        return {}

    overview = _to_lower_text(full_data.get("overview"))
    genres = _to_lower_list(full_data.get("genres"))

    # Skip low-quality rows that miss core content fields.
    if not overview or not genres:
        return {}

    release_date = full_data.get("release_date")
    record = {
        "id": movie_id,
        "title": _to_lower_text(full_data.get("title")),
        "release_year": _extract_release_year(release_date),
        "runtime": full_data.get("runtime"),
        "genres": genres,
        "overview": overview,
        "keywords": _to_lower_list(full_data.get("keywords")),
        "cast": _to_lower_list(full_data.get("cast"))[:5],
        "director": _to_lower_text(full_data.get("director")),
        "vote_average": popular_movie.get("vote_average", 0),
        "vote_count": popular_movie.get("vote_count", 0),
    }
    return record


def build_movie_dataset(num_pages) -> list[dict[str, Any]]:
    """Fetch, clean, and save a dataset of popular movies."""
    if num_pages < 1:
        num_pages = 1

    dataset: list[dict[str, Any]] = []

    for page in range(1, num_pages + 1):
        print(f"Fetching page {page}...")
        popular_payload = get_popular_movies(page=page)
        time.sleep(RATE_LIMIT_SECONDS)

        movies = popular_payload.get("results", []) if popular_payload else []
        if not isinstance(movies, list):
            continue

        for movie in movies:
            if not isinstance(movie, dict):
                continue

            movie_record = _build_movie_record(movie)
            if not movie_record:
                continue

            dataset.append(movie_record)
            print(f"Processed movie: {movie_record['title']}")

    with open(DATASET_PATH, "w", encoding="utf-8") as output_file:
        json.dump(dataset, output_file, ensure_ascii=False, indent=2)

    print(f"Saved {len(dataset)} movies to {DATASET_PATH}")
    return dataset


if __name__ == "__main__":
    build_movie_dataset(num_pages=10)
