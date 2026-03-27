import re


GENRES = ["action", "comedy", "drama", "horror", "science fiction", "romance", "thriller"]


def _clean_person_name(text: str) -> str:
    # Strip trailing punctuation/whitespace from captured names.
    return re.sub(r"[,\.\!\?]+$", "", text.strip())


def _genre_regex(genre: str) -> str:
    # Build a regex that matches the genre phrase as words, allowing flexible whitespace.
    # Example: "science fiction" -> r"\bscience\s+fiction\b"
    words = genre.split()
    return r"\b" + r"\s+".join(map(re.escape, words)) + r"\b"


def parse_query(query: str) -> dict:
    """
    Convert a natural-language query into a structured filter dictionary.

    Example:
      "action movies after 2015 with nolan"
      -> {"min_year": 2015, "genres": ["action"], "person": "nolan", ...}
    """
    q = (query or "").strip()
    q_norm = q.lower().replace("-", " ")

    min_year = None
    max_year = None
    min_rating = None
    person = None
    genres: list[str] = []
    genre_mode = "OR"

    # --- Year parsing ---
    m_after = re.search(r"\bafter\s+(\d{4})\b", q_norm)
    if m_after:
        min_year = int(m_after.group(1))

    m_before = re.search(r"\bbefore\s+(\d{4})\b", q_norm)
    if m_before:
        max_year = int(m_before.group(1))

    # --- Rating parsing ---
    # Requirement: "above 7" -> min_rating = 7
    m_above = re.search(r"\babove\s+(\d+(?:\.\d+)?)\b", q_norm)
    if m_above:
        val = float(m_above.group(1))
        # Keep ints as ints when possible (purely cosmetic, but nicer for users).
        min_rating = int(val) if val.is_integer() else val

    # --- Person parsing ---
    # Detect patterns like:
    #   "with [name]" or "by [name]"
    # Stop capture at the next recognized keyword or end-of-string.
    m_person = re.search(
        r"\b(?:with|by)\s+(.+?)(?=\b(?:after|before|above|below|with|by)\b|$)",
        q_norm,
    )
    if m_person:
        person = _clean_person_name(m_person.group(1))

    # --- Genres parsing ---
    for g in GENRES:
        if re.search(_genre_regex(g), q_norm):
            genres.append(g)

    return {
        "min_year": min_year,
        "max_year": max_year,
        "genres": genres,
        "genre_mode": genre_mode,
        "person": person,
        "min_rating": min_rating,
    }


if __name__ == "__main__":
    test_queries = [
        "action movies after 2015",
        "drama before 2000",
        "science fiction with nolan",
    ]

    for t in test_queries:
        print("QUERY:", t)
        print("PARSED:", parse_query(t))
        print()

