movies = None
content_matrix = None
plot_matrix = None
content_sim = None
plot_sim = None
_title_to_index: dict[str, int] = {}
_normalized_title_to_index: dict[str, int] = {}
_normalized_title_to_indices: dict[str, list[int]] = {}


def _normalize_title_key(value):
    text = str(value or "").strip().lower()
    # Keep only alnum chars so "Lady Chatterley's Lover" and "lady chatterleys lover"
    # can still match.
    return "".join(ch for ch in text if ch.isalnum())


def _ensure_legacy_loaded():
    """
    Lazily initialize legacy dataset recommender artifacts.
    """
    global movies, content_matrix, plot_matrix, plot_sim, content_sim, _title_to_index, _normalized_title_to_index, _normalized_title_to_indices
    if movies is not None and content_sim is not None and plot_sim is not None and _title_to_index:
        return

    from nlp_utils import build_feature_matrices, compute_similarity, load_movies

    movies = load_movies()
    content_matrix, plot_matrix = build_feature_matrices(movies)
    content_sim = compute_similarity(content_matrix)
    plot_sim = compute_similarity(plot_matrix)

    _title_to_index = {}
    _normalized_title_to_index = {}
    _normalized_title_to_indices = {}
    for i, m in enumerate(movies):
        title = str(m.get("title", "")).strip().lower()
        if title and title not in _title_to_index:
            _title_to_index[title] = i
        nkey = _normalize_title_key(title)
        if nkey and nkey not in _normalized_title_to_index:
            _normalized_title_to_index[nkey] = i
        if nkey:
            _normalized_title_to_indices.setdefault(nkey, []).append(i)


def get_movie_index(title):
    """
    Return the index for `title` (case-insensitive) in the dataset,
    or None if not found.
    """
    if title is None:
        return None
    _ensure_legacy_loaded()
    query = str(title).strip().lower()
    idx = _title_to_index.get(query)
    if idx is not None:
        return idx

    # Fallback 1: punctuation-insensitive exact key
    nquery = _normalize_title_key(query)
    idx = _normalized_title_to_index.get(nquery)
    if idx is not None:
        return idx

    # Fallback 2: partial normalized containment for slight title variations
    if nquery:
        for ntitle, i in _normalized_title_to_index.items():
            if nquery in ntitle or ntitle in nquery:
                return i
    return None


def _pick_best_index_for_title(title):
    """
    Resolve ambiguous/duplicate titles to the richest metadata row.
    """
    _ensure_legacy_loaded()
    query = str(title or "").strip().lower()
    nquery = _normalize_title_key(query)
    if not nquery:
        return None

    candidates = list(_normalized_title_to_indices.get(nquery, []))
    if not candidates:
        # broaden to containment matches
        for ntitle, idx_list in _normalized_title_to_indices.items():
            if nquery in ntitle or ntitle in nquery:
                candidates.extend(idx_list)

    if not candidates:
        return None

    # Pick row with richest comparable metadata (genres/keywords/cast) and then votes.
    def _richness(idx):
        m = movies[idx]
        g = len(m.get("genres", []) or [])
        k = len(m.get("keywords", []) or [])
        c = len(m.get("cast", []) or [])
        v = m.get("vote_count")
        vote_count = int(v) if isinstance(v, (int, float)) else 0
        return (g + k + c, vote_count)

    return max(candidates, key=_richness)


def get_recommendations(title, top_n=10):
    """
    Recommend movies similar to `title` using a hybrid of:
      - content similarity (genres/keywords/cast/director)
      - NLP similarity of plot/overview
    """
    idx = get_movie_index(title)
    if idx is None:
        return []

    # Hybrid scoring (vector over all movies).
    final_scores = 0.6 * content_sim[idx] + 0.4 * plot_sim[idx]

    # Sort indices by score descending.
    sorted_indices = sorted(
        range(len(final_scores)),
        key=lambda i: float(final_scores[i]),
        reverse=True,
    )

    recommendations = []
    for i in sorted_indices:
        if i == idx:
            continue  # exclude the movie itself

        movie = movies[i]
        recommendations.append(
            {
                "title": movie.get("title"),
                "score": float(final_scores[i]),
                "genres": movie.get("genres", []),
                "year": movie.get("release_year"),
            }
        )

        if len(recommendations) >= top_n:
            break
            
    seen = set()
    unique_recommendations = []

    for r in recommendations:
        if r["title"] not in seen:
            seen.add(r["title"])
            unique_recommendations.append(r)

    return unique_recommendations


def _normalize_movie_for_matrix(movie):
    """
    Ensure missing text/list fields are present for NLP feature extraction.
    """
    if not isinstance(movie, dict):
        return {}
    normalized = dict(movie)
    normalized.setdefault("title", "")
    normalized.setdefault("genres", [])
    normalized.setdefault("keywords", [])
    normalized.setdefault("cast", [])
    normalized.setdefault("director", "")
    normalized.setdefault("overview", "")
    normalized.setdefault("release_year", normalized.get("year"))
    normalized.setdefault("year", normalized.get("release_year"))
    return normalized


def get_recommendations_from_pool(seed_movie, movies_pool, top_n=10):
    """
    Dynamic recommender over a runtime movie pool.

    This reuses the same hybrid approach:
      - content metadata similarity (genres/keywords/cast/director)
      - plot NLP similarity (overview text)

    Returns items with a `score` field in [0, 1] (approx) and
    keeps the richer movie payload so downstream filters/UI can use it.
    """
    if not isinstance(seed_movie, dict):
        return []
    if not isinstance(movies_pool, list) or not movies_pool:
        return []

    seed_norm = _normalize_movie_for_matrix(seed_movie)
    pool_norm = [_normalize_movie_for_matrix(m) for m in movies_pool if isinstance(m, dict)]
    if not pool_norm:
        return []

    # Remove exact duplicate ids to stabilize indexing.
    deduped = []
    seen_ids = set()
    for m in pool_norm:
        mid = m.get("id")
        if isinstance(mid, int):
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
        deduped.append(m)
    pool_norm = deduped

    # Ensure seed exists in comparison set.
    seed_id = seed_norm.get("id")
    has_seed = isinstance(seed_id, int) and any(m.get("id") == seed_id for m in pool_norm)
    combined = list(pool_norm)
    if not has_seed:
        combined = [seed_norm] + combined

    if len(combined) < 2:
        return []

    from nlp_utils import build_feature_matrices, compute_similarity

    content_matrix, plot_matrix = build_feature_matrices(combined)
    content_sim = compute_similarity(content_matrix)
    plot_sim = compute_similarity(plot_matrix)

    seed_idx = 0 if not has_seed else next(
        (i for i, m in enumerate(combined) if m.get("id") == seed_id),
        0,
    )

    final_scores = 0.6 * content_sim[seed_idx] + 0.4 * plot_sim[seed_idx]
    sorted_indices = sorted(
        range(len(final_scores)),
        key=lambda i: float(final_scores[i]),
        reverse=True,
    )

    recs = []
    for i in sorted_indices:
        if i == seed_idx:
            continue
        m = combined[i]
        # exclude seed again by id/title for safety
        if isinstance(seed_id, int) and m.get("id") == seed_id:
            continue
        if str(m.get("title", "")).strip().lower() == str(seed_norm.get("title", "")).strip().lower():
            continue

        out = dict(m)
        out["score"] = float(final_scores[i])
        out["year"] = out.get("year", out.get("release_year"))
        recs.append(out)
        if len(recs) >= max(1, int(top_n)):
            break

    return recs



def explain_recommendation(movie_a, movie_b):
    """
    Explain why two movies are similar by comparing overlap in:
      - genres
      - keywords
      - cast
    """

    _ensure_legacy_loaded()

    def _resolve_movie(x):
        if isinstance(x, int):
            return movies[x] if 0 <= x < len(movies) else {}
        if isinstance(x, dict):
            return x
        # Otherwise treat as a title.
        i = _pick_best_index_for_title(x)
        if i is None:
            i = get_movie_index(x)
        return movies[i] if i is not None else {}

    a = _resolve_movie(movie_a)
    b = _resolve_movie(movie_b)

    a_genres = set(g.lower().strip() for g in (a.get("genres", []) or []) if g)
    b_genres = set(g.lower().strip() for g in (b.get("genres", []) or []) if g)
    genres_overlap = sorted(a_genres.intersection(b_genres))

    a_keywords = set(k.lower().strip() for k in (a.get("keywords", []) or []) if k)
    b_keywords = set(k.lower().strip() for k in (b.get("keywords", []) or []) if k)
    keywords_overlap = sorted(a_keywords.intersection(b_keywords))

    a_cast = set(c.lower().strip() for c in (a.get("cast", []) or []) if c)
    b_cast = set(c.lower().strip() for c in (b.get("cast", []) or []) if c)
    cast_overlap = sorted(a_cast.intersection(b_cast))

    if genres_overlap:
        genres_part = f"Shares genres: {', '.join(genres_overlap)}"
    else:
        genres_part = "No shared genres"

    if keywords_overlap:
        keywords_part = f"Shares keywords: {', '.join(keywords_overlap)}"
    else:
        keywords_part = "No shared keywords"

    parts = [genres_part, keywords_part]
    if cast_overlap:
        parts.append(f"Common actors: {', '.join(cast_overlap)}")
    return ". ".join(parts) + "."


if __name__ == "__main__":
    test_title = "inception"

    print("TESTING\n")

    print("First 10 movies:")
    for m in movies[:10]:
        print(m["title"])
    print("\n")

    recs = get_recommendations(test_title, top_n=5)

    if not recs:
        print(f"Movie '{test_title}' not found in dataset.\n")
    else:
        print(f"Recommendations for '{test_title}':\n")
        for r in recs:
            print(r)

        # Debug deeper (safe local variables)
        idx = get_movie_index(test_title)
        final_scores = 0.6 * content_sim[idx] + 0.4 * plot_sim[idx]
        sorted_indices = sorted(
            range(len(final_scores)),
            key=lambda i: float(final_scores[i]),
            reverse=True,
        )

        print("\nTop similarity scores:")
        for i in sorted_indices[1:6]:
            print(movies[i]["title"], round(final_scores[i], 3))