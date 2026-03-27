from __future__ import annotations

from tmdb_api import get_trending_movies


def get_trending_display(limit: int = 10) -> list[dict]:
    """
    Return trending movies formatted for UI display.

    Falls back to an empty list if API is unavailable.
    """
    try:
        n = max(1, int(limit))
    except Exception:
        n = 10

    movies = get_trending_movies()
    if not movies:
        return []

    display_rows: list[dict] = []
    for m in movies[:n]:
        title = str(m.get("title") or "Untitled").strip()
        year = m.get("release_year")
        rating = m.get("rating")
        overview = str(m.get("overview") or "").strip()

        short_overview = overview if overview else "No overview available."
        if len(short_overview) > 220:
            short_overview = short_overview[:217].rstrip() + "..."

        display_rows.append(
            {
                "title": title,
                "year": year,
                "rating": rating,
                "overview": short_overview,
            }
        )

    return display_rows


if __name__ == "__main__":
    rows = get_trending_display(limit=5)
    if not rows:
        print("No trending movies available right now.")
    else:
        print("Top 5 trending movies:")
        for i, row in enumerate(rows, start=1):
            title = row.get("title")
            year = row.get("year")
            rating = row.get("rating")
            overview = row.get("overview")
            print(f"{i}. {title} ({year}) - Rating: {rating}\n{overview}\n")
