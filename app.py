import streamlit as st

from search import search_movies
from api_features import get_trending_display


st.set_page_config(page_title="AI Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 AI Movie Recommender")

with st.sidebar:
    st.write("Discover movies using AI-powered recommendations and natural language search.")

st.write("")
st.header("🔍 Search for Movies")

query = st.text_input("Describe what you want to watch")

do_search = st.button("Search", type="primary")

if do_search:
    q = (query or "").strip()
    if not q:
        st.warning("Please enter a description to search.")
    else:
        result = search_movies(q)

        if not result:
            st.warning("No results found")
        else:
            seed_movie_title = str(result.get("seed_movie", "") or "").strip()
            st.subheader("Seed movie")
            st.subheader(seed_movie_title or "—")

            st.write("")
            st.subheader("Recommendations")

            recs = result.get("recommendations") or []
            if not recs:
                st.warning("No results found")
            else:
                for r in recs:
                    title = str(r.get("title") or "Untitled")
                    year = r.get("year")
                    genres = r.get("genres") or []
                    score = r.get("score")

                    year_text = str(year) if year is not None and str(year).strip() else "—"
                    genres_text = ", ".join([str(g) for g in genres if g]) if genres else "—"
                    score_text = f"{round(float(score), 2):.2f}" if score is not None else "—"

                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.write(f"Year: {year_text}")
                        st.write(f"Genres: {genres_text}")
                        st.write(f"Score: {score_text}")

                        # --- Simple, reliable explanation based on genres ---
                        explanation = ""

                        try:
                            # Extract genres from recommendation
                            rec_genres = set(
                                str(g).lower().strip()
                                for g in (genres or [])
                                if g
                            )

                            # Extract genres from seed via filters (if available)
                            seed_genres = set(
                                str(g).lower().strip()
                                for g in (result.get("filters", {}).get("genres") or [])
                                if g
                            )

                            shared_genres = rec_genres.intersection(seed_genres)

                            if shared_genres:
                                explanation = f"Shares genres: {', '.join(sorted(shared_genres))}"
                            else:
                                explanation = "Similar style and themes based on AI matching"

                        except Exception:
                            explanation = "Similar movies based on content similarity"

                        st.write("💡 " + explanation)

st.divider()

st.header("🔥 Trending Movies")
trending = get_trending_display(limit=10)
if not trending:
    st.caption("Trending movies are unavailable right now.")
else:
    for t in trending[:10]:
        title = str(t.get("title") or "Untitled")
        year = t.get("year")
        rating = t.get("rating")
        overview = str(t.get("overview") or "")

        year_text = str(year) if year is not None and str(year).strip() else "—"
        rating_text = f"{float(rating):.1f}" if isinstance(rating, (int, float)) else "—"

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(f"Year: {year_text}")
            st.write(f"Rating: {rating_text}")
            st.write(overview if overview else "No overview available.")
