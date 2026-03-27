import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_movies(filepath="movies_dataset.json"):
    """Load movies from a JSON file and return a list of movie dictionaries."""
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_value(value):
    """Normalize a single value to lowercase with spaces replaced by underscores."""
    return str(value).strip().lower().replace(" ", "_")


def _normalize_list_field(field_value):
    """Normalize list-like movie fields into a single whitespace-separated string."""
    if isinstance(field_value, list):
        normalized_items = [_normalize_value(item) for item in field_value if item]
        return " ".join(normalized_items)
    if field_value:
        return _normalize_value(field_value)
    return ""


def create_content_text(movie):
    genres = " ".join([str(x) for x in (movie.get("genres") or []) if x is not None])
    keywords = " ".join([str(x) for x in (movie.get("keywords") or []) if x is not None])
    cast = " ".join([str(x) for x in (movie.get("cast") or []) if x is not None])
    director = str(movie.get("director") or "")

    # Strong weighting
    text = (
        (genres + " ") * 4 +     # MOST important
        (keywords + " ") * 3 +
        (cast + " ") * 2 +
        (director + " ")
    )

    return re.sub(r"[^a-zA-Z0-9 ]", "", text).lower()


def build_feature_matrices(movies):
    """Create TF-IDF feature matrices for content metadata and plot overviews."""
    content_texts = []
    plot_texts = []

    for movie in movies:
        content_texts.append(create_content_text(movie))
        plot_texts.append(movie.get("overview", ""))

    content_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)
    plot_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

    content_matrix = content_vectorizer.fit_transform(content_texts)
    plot_matrix = plot_vectorizer.fit_transform(plot_texts)

    return content_matrix, plot_matrix


def compute_similarity(matrix):
    """Compute and return cosine similarity for the provided matrix."""
    return cosine_similarity(matrix)


if __name__ == "__main__":
    movies_data = load_movies()
    content_matrix_result, plot_matrix_result = build_feature_matrices(movies_data)
    content_similarity_result = compute_similarity(content_matrix_result)

    print("Content matrix shape:", content_matrix_result.shape)
    print("Plot matrix shape:", plot_matrix_result.shape)
    print("Content similarity matrix shape:", content_similarity_result.shape)
