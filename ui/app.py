"""
ui/app.py
=========
Production-grade Streamlit frontend for the SAR Movie Recommendation System.

Three modes:
  1. User Recommendations   — personalised top-K for a known user
  2. Similar Movies         — find movies similar to one you like
  3. Cold-Start / From List — recommendations from a hand-picked movie list
"""
from __future__ import annotations

import csv
import io
import time
from typing import List, Optional

import requests
import streamlit as st

# ── config ────────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:8000"
APP_TITLE   = "CineMatch · SAR Recommender"
TIMEOUT     = 20

st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide",
                   initial_sidebar_state="expanded")

# ── design system ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --ink:      #0d0d0f;
  --paper:    #f5f0e8;
  --cream:    #ede8dc;
  --gold:     #c9a84c;
  --gold2:    #f0d080;
  --red:      #b83232;
  --muted:    #7a7060;
  --card-bg:  #ffffff;
  --border:   #d8d0c0;
  --shadow:   0 2px 20px rgba(13,13,15,0.08);
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--paper) !important;
  color: var(--ink) !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }

/* ── masthead ── */
.masthead {
  display: flex; align-items: baseline; gap: 1rem;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 0.75rem; margin-bottom: 1.5rem;
}
.masthead-title {
  font-family: 'Playfair Display', serif;
  font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em;
  color: var(--ink); line-height: 1;
}
.masthead-sub {
  font-size: 0.8rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--muted); font-weight: 500;
}
.masthead-rule {
  flex: 1; border-top: 1px solid var(--border); margin: 0 0.5rem;
}

/* ── tab strip ── */
.tab-strip { display: flex; gap: 0; border-bottom: 2px solid var(--ink); margin-bottom: 1.5rem; }
.tab-btn {
  padding: 0.6rem 1.5rem;
  font-family: 'Playfair Display', serif;
  font-size: 0.95rem; font-style: italic;
  background: none; border: none; cursor: pointer;
  color: var(--muted); border-bottom: 3px solid transparent;
  transition: all 0.15s;
}
.tab-btn:hover { color: var(--ink); }
.tab-btn.active { color: var(--ink); border-bottom-color: var(--gold); font-weight: 700; }

/* ── movie card ── */
.movie-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1rem 1.2rem;
  margin-bottom: 0.55rem;
  display: flex; align-items: flex-start; gap: 1rem;
  box-shadow: var(--shadow);
  transition: transform 0.1s, box-shadow 0.1s;
}
.movie-card:hover {
  transform: translateX(3px);
  box-shadow: 0 4px 24px rgba(13,13,15,0.12);
}
.rank-num {
  font-family: 'Playfair Display', serif;
  font-size: 1.6rem; font-style: italic; color: var(--gold);
  min-width: 2.2rem; line-height: 1.2;
}
.movie-title {
  font-family: 'Playfair Display', serif;
  font-size: 1rem; font-weight: 700; color: var(--ink); margin: 0;
}
.movie-meta { font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; }
.score-pill {
  margin-left: auto; background: var(--cream);
  border: 1px solid var(--border); border-radius: 2px;
  padding: 0.15rem 0.5rem; font-size: 0.75rem; color: var(--muted);
  white-space: nowrap; font-weight: 500;
}

/* ── metric boxes ── */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1rem; }
.metric-box {
  background: var(--cream); border: 1px solid var(--border);
  border-radius: 4px; padding: 0.7rem 1rem; text-align: center; flex: 1;
}
.metric-val { font-family: 'Playfair Display', serif; font-size: 1.5rem; color: var(--gold); }
.metric-lbl { font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--ink) !important;
  color: var(--paper) !important;
}
section[data-testid="stSidebar"] * { color: var(--paper) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: var(--gold) !important; }

/* ── inputs ── */
.stTextInput > div > div { border-color: var(--border) !important; border-radius: 3px !important; }
.stSelectbox > div > div { border-color: var(--border) !important; border-radius: 3px !important; }
.stButton > button {
  background: var(--ink) !important; color: var(--paper) !important;
  border: none !important; border-radius: 3px !important;
  font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
  letter-spacing: 0.05em !important;
  transition: background 0.15s !important;
}
.stButton > button:hover { background: var(--red) !important; }
.stMultiSelect > div > div { border-color: var(--border) !important; border-radius: 3px !important; }

/* ── fallback badge ── */
.fallback-badge {
  display: inline-block; background: #fff3cd; border: 1px solid #ffc107;
  border-radius: 3px; padding: 0.25rem 0.75rem; font-size: 0.78rem; color: #856404;
  margin-bottom: 0.5rem;
}

/* ── section label ── */
.section-label {
  font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--muted); font-weight: 500; margin-bottom: 0.5rem;
}

/* ── divider ── */
.ornamental-divider { text-align: center; color: var(--gold); font-size: 1.2rem; margin: 0.5rem 0 1rem; }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def api_health(base: str = API_BASE):
    try:
        r = requests.get(f"{base}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def api_search_movies(q: str, limit: int = 50, base: str = API_BASE):
    try:
        r = requests.get(f"{base}/movies", params={"q": q, "limit": limit}, timeout=TIMEOUT)
        return r.json().get("movies", []) if r.status_code == 200 else []
    except Exception:
        return []


@st.cache_data(ttl=60)
def api_recommend(user_id: int, k: int, base: str = API_BASE):
    try:
        r = requests.get(f"{base}/recommend/{user_id}", params={"k": k}, timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def api_similar(movie_id: int, k: int, base: str = API_BASE):
    try:
        r = requests.get(f"{base}/similar/{movie_id}", params={"k": k}, timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def api_cold_start(item_ids: List[int], k: int, base: str = API_BASE):
    try:
        r = requests.post(f"{base}/recommend/items",
                          json={"item_ids": item_ids, "k": k}, timeout=TIMEOUT)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def api_users(limit: int = 943, base: str = API_BASE):
    try:
        r = requests.get(f"{base}/users", params={"limit": limit}, timeout=TIMEOUT)
        return r.json().get("users", []) if r.status_code == 200 else []
    except Exception:
        return []


# ── component helpers ─────────────────────────────────────────────────────────

def render_movie_card(rank: int, item: dict):
    score_html = f'<span class="score-pill">score {item["score"]:.4f}</span>' \
                 if item.get("score") is not None else ""
    st.markdown(f"""
    <div class="movie-card">
      <div class="rank-num">{rank}</div>
      <div style="flex:1">
        <div class="movie-title">{item['title']}</div>
        <div class="movie-meta">{item.get('genres','').replace('|',' · ')}</div>
      </div>
      {score_html}
    </div>""", unsafe_allow_html=True)


def recs_to_csv(recs: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["rank","item_id","title","genres","score"])
    w.writeheader()
    for i, r in enumerate(recs, 1):
        w.writerow({"rank": i, "item_id": r.get("item_id",""),
                    "title": r.get("title",""), "genres": r.get("genres",""),
                    "score": r.get("score","")})
    return buf.getvalue()


def movie_search_select(label: str, key: str, multi: bool = False):
    """
    Search-as-you-type movie selector.
    Returns a single dict (multi=False) or list of dicts (multi=True).
    """
    q = st.text_input(label, key=f"q_{key}", placeholder="Type a movie title…")
    if not q.strip():
        return [] if multi else None

    movies = api_search_movies(q.strip(), limit=50)
    if not movies:
        st.caption("No movies found.")
        return [] if multi else None

    options = {f"{m['title']} (ID:{m['item_id']})": m for m in movies}
    if multi:
        selected_keys = st.multiselect(
            f"Select movies ({len(options)} matches)",
            list(options.keys()), key=f"sel_{key}"
        )
        return [options[k] for k in selected_keys]
    else:
        chosen = st.selectbox(
            f"{len(options)} match(es)", ["— select —"] + list(options.keys()),
            key=f"sel_{key}"
        )
        return options.get(chosen) if chosen != "— select —" else None


# ── masthead ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="masthead">
  <div class="masthead-title">CineMatch</div>
  <div class="masthead-rule"></div>
  <div class="masthead-sub">Powered by Microsoft SAR · MovieLens 100K</div>
</div>
""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_base = st.text_input("API URL", value=API_BASE)
    k_value  = st.slider("Recommendations (K)", 5, 50, 10, 5)

    st.markdown("---")
    health = api_health(api_base)
    if health and health.get("model_loaded"):
        st.markdown("### 🟢 System Online")
        m = health.get("metrics", {})
        cols = st.columns(2)
        cols[0].metric("Users",  f"{health['n_users']:,}")
        cols[1].metric("Items",  f"{health['n_items']:,}")
        if m:
            st.markdown("**Eval Metrics (K=10)**")
            for key in ["precision@10","recall@10","ndcg@10","map@10","coverage"]:
                if key in m:
                    st.markdown(f"`{key}` → **{m[key]:.4f}**")
        st.caption(f"Similarity: {health.get('similarity','—')}")
    elif health:
        st.markdown("### 🟡 Model Not Loaded")
        st.caption("Run: `python scripts/train.py`")
    else:
        st.markdown("### 🔴 API Offline")
        st.caption(f"Expected at {api_base}")
        st.caption("Run: `python -m uvicorn api.main:app --port 8000`")

    st.markdown("---")
    st.markdown("**About SAR**")
    st.caption(
        "Smart Adaptive Recommendations computes item co-occurrence, "
        "builds a Jaccard/Lift similarity matrix, and scores users by "
        "their time-decayed affinity vector."
    )

# ── mode tabs ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🎯  User Recommendations",
    "🔗  Similar Movies",
    "🎲  Recommend From Movies",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — User Recommendations
# ════════════════════════════════════════════════════════════════════════════════

with tab1:
    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.markdown('<div class="section-label">Choose a user</div>', unsafe_allow_html=True)
        all_users = api_users(base=api_base)
        if all_users:
            user_id = st.selectbox("Select User ID", all_users, key="uid_select")
        else:
            user_id = st.number_input("Enter User ID (1–943)", min_value=1,
                                      max_value=943, value=1, step=1, key="uid_num")

        get_recs_btn = st.button("✦ Get Recommendations", key="btn_user",
                                  use_container_width=True)

    with col_res:
        if get_recs_btn:
            with st.spinner("Generating personalised recommendations …"):
                result = api_recommend(int(user_id), k_value, api_base)

            if "error" in result:
                st.error(f"API error: {result['error']}")
            else:
                recs = result.get("recommendations", [])
                if result.get("is_popular_fallback"):
                    st.markdown('<span class="fallback-badge">📈 Popularity Fallback — user not in training set</span>',
                                unsafe_allow_html=True)
                st.markdown(f'<div class="section-label">Top {len(recs)} recommendations for User {user_id}</div>',
                            unsafe_allow_html=True)
                for i, item in enumerate(recs, 1):
                    render_movie_card(i, item)

                st.download_button("📥 Export CSV", recs_to_csv(recs),
                                   file_name=f"recs_user{user_id}.csv",
                                   mime="text/csv")
        else:
            st.markdown('<div class="ornamental-divider">✦ ✦ ✦</div>', unsafe_allow_html=True)
            st.info("Select a user ID and click **Get Recommendations**.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Similar Movies
# ════════════════════════════════════════════════════════════════════════════════

with tab2:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown('<div class="section-label">Find a movie</div>', unsafe_allow_html=True)
        selected_movie = movie_search_select("Search movie title", key="sim_movie")

        if selected_movie:
            st.markdown(f"""
            <div style="background:var(--cream);border:1px solid var(--border);
                        border-radius:4px;padding:0.75rem 1rem;margin-top:0.5rem;">
              <div class="movie-title">{selected_movie['title']}</div>
              <div class="movie-meta">{selected_movie.get('genres','').replace('|',' · ')}</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:0.3rem;">ID: {selected_movie['item_id']}</div>
            </div>""", unsafe_allow_html=True)

        find_sim_btn = st.button("✦ Find Similar Movies", key="btn_sim",
                                  use_container_width=True,
                                  disabled=selected_movie is None)

    with col_r:
        if find_sim_btn and selected_movie:
            with st.spinner("Finding similar movies …"):
                result = api_similar(selected_movie["item_id"], k_value, api_base)

            if "error" in result:
                st.error(f"API error: {result['error']}")
            else:
                sims = result.get("similar", [])
                st.markdown(f'<div class="section-label">Movies similar to "{selected_movie["title"]}"</div>',
                            unsafe_allow_html=True)
                for i, item in enumerate(sims, 1):
                    render_movie_card(i, item)

                st.download_button("📥 Export CSV", recs_to_csv(sims),
                                   file_name=f"similar_{selected_movie['item_id']}.csv",
                                   mime="text/csv")
        else:
            st.markdown('<div class="ornamental-divider">✦ ✦ ✦</div>', unsafe_allow_html=True)
            st.info("Search for a movie above, select it, then click **Find Similar Movies**.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Cold Start / Recommend From Movies
# ════════════════════════════════════════════════════════════════════════════════

with tab3:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown('<div class="section-label">Build your movie list</div>',
                    unsafe_allow_html=True)
        st.caption("Search and select multiple movies you enjoy, then get recommendations.")

        picked = movie_search_select("Search and add movies", key="cold_movies", multi=True)

        # session state basket
        if "basket" not in st.session_state:
            st.session_state.basket = []

        if picked:
            for m in picked:
                if m["item_id"] not in {x["item_id"] for x in st.session_state.basket}:
                    st.session_state.basket.append(m)

        if st.session_state.basket:
            st.markdown(f'<div class="section-label">Your selection ({len(st.session_state.basket)} movies)</div>',
                        unsafe_allow_html=True)
            for m in st.session_state.basket:
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{m['title']}**")
                if c2.button("✕", key=f"rm_{m['item_id']}"):
                    st.session_state.basket = [x for x in st.session_state.basket
                                                if x["item_id"] != m["item_id"]]
                    st.rerun()

            if st.button("🗑 Clear All", key="clear_basket"):
                st.session_state.basket = []
                st.rerun()

        cold_btn = st.button("✦ Get Recommendations", key="btn_cold",
                              use_container_width=True,
                              disabled=len(st.session_state.basket) == 0)

    with col_r:
        if cold_btn and st.session_state.basket:
            seed_ids = [m["item_id"] for m in st.session_state.basket]
            with st.spinner(f"Computing recommendations from {len(seed_ids)} movies …"):
                result = api_cold_start(seed_ids, k_value, api_base)

            if "error" in result:
                st.error(f"API error: {result['error']}")
            else:
                recs = result.get("recommendations", [])
                st.markdown(f'<div class="section-label">Top {len(recs)} recommendations</div>',
                            unsafe_allow_html=True)
                for i, item in enumerate(recs, 1):
                    render_movie_card(i, item)

                st.download_button("📥 Export CSV", recs_to_csv(recs),
                                   file_name="cold_start_recs.csv", mime="text/csv")
        else:
            st.markdown('<div class="ornamental-divider">✦ ✦ ✦</div>', unsafe_allow_html=True)
            st.info("Add movies to your list on the left, then click **Get Recommendations**.")
