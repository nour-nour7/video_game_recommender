import streamlit as st
import os
import gdown
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import pickle
import google.generativeai as genai

genai.configure(api_key="AIzaSyBmDCcPFWahnd4Hu0Uc41SnsNDHn8fGRVQ")
model_llm = genai.GenerativeModel("gemini-2.0-flash-lite")

@st.cache_data
def explain_recommendation(searched_game, recommended_game, genres, tags):
    prompt = f"""
    A user liked "{searched_game}" and we recommended "{recommended_game}".
    Genres: {genres}
    Tags: {tags}
    In ONE sentence, explain why a fan of "{searched_game}" would enjoy "{recommended_game}". Max 30 words.
    """
    try:
        response = model_llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Explanation unavailable right now (API limit reached)"


GAMES_ID = "1xfIVgqLDi3qaB1Yn0_VqdixbaSu4E-MZ"
EMBEDDINGS_ID = "1uJjmI48C61TFsPT-lAvgO9r2Y0U5OL3y"

@st.cache_data
def load_data():
    os.makedirs("data", exist_ok=True)
    
    if not os.path.exists("data/games_final.pkl"):
        gdown.download(f"https://drive.google.com/uc?id={GAMES_ID}", "data/games_final.pkl", quiet=False)
    
    if not os.path.exists("data/embeddings.npy"):
        gdown.download(f"https://drive.google.com/uc?id={EMBEDDINGS_ID}", "data/embeddings.npy", quiet=False)
    
    with open("data/games_final.pkl", "rb") as f:
        games = pickle.load(f)
    embeddings = np.load("data/embeddings.npy")
    return games, embeddings

games_final, embeddings=load_data()

st.title("Steam Game Recommender")
#st.write("Loaded data: ",len(games_final), "games")


#search engine
#real time fuzzy matching

if 'selected_game' not in st.session_state:
    st.session_state.selected_game = None
if 'show_recs' not in st.session_state:
    st.session_state.show_recs = False

#Recherche
query = st.text_input("Start typing a game name...")

if query:
    normalized_query = query.lower().replace("-", " ")
    normalized_names = games_final['name'].str.lower().str.replace("-", " ")
    matches = process.extract(
        normalized_query,
        normalized_names,
        scorer=fuzz.token_sort_ratio,
        limit=5
    )
    suggestions = [games_final['name'].iloc[m[2]] for m in matches]
    
    # selection
    selected = st.selectbox("Select a game:", suggestions)
    preview_info = games_final[games_final['name'] == selected].iloc[0]
    prev_col1, prev_col2 = st.columns([1, 3])
    with prev_col1:
        st.image(preview_info['header_image'], use_container_width=True)
    with prev_col2:
        st.write(f"**{preview_info['name']}**")
        st.caption(f"{', '.join(preview_info['genres'])}")
        st.caption(f"{preview_info['positive_ratio']:.0%} positive")

    if st.button("Get Recommendations"):
        st.session_state.selected_game = selected
        st.session_state.show_recs = True

# recommendations
if st.session_state.show_recs and st.session_state.selected_game:
    selected_game = st.session_state.selected_game
    
    game_info = games_final[games_final['name'] == selected_game].iloc[0]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(game_info['header_image'], use_container_width=True)
    with col2:
        st.subheader(game_info['name'])
        st.write(f"**Price:** {game_info['price']}€")
        st.write(f"**Genres:** {', '.join(game_info['genres'])}")
        st.write(f"**Score:** {game_info['positive_ratio']:.0%}")
        st.write(f"**Total reviews:** {int(game_info['total_reviews'])}")
    
    st.divider()
    st.subheader(f"Games similar to {selected_game}")
    
    game_idx = games_final.index[games_final['name'] == selected_game][0]
    similarities = cosine_similarity([embeddings[game_idx]], embeddings)
    top_indices = np.argsort(similarities[0])[::-1][1:11]
    results = games_final.iloc[top_indices][['app_id','name','header_image','genres','price','positive_ratio','tags_text','short_description']]
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 3]:
            steam_url = f"https://store.steampowered.com/app/{row['app_id']}"
            st.markdown(
                f'<a href="{steam_url}" target="_blank">'
                f'<img src="{row["header_image"]}" style="width:100%"/>'
                f'</a>',
                unsafe_allow_html=True
            )
            st.write(f"**{row['name']}**")
            st.write(f"Price: {row['price']}€")
            st.write(f"Score: {row['positive_ratio']:.0%}")
            st.caption(f"{', '.join(row['genres'])}")
            with st.expander("Description"):
                st.write(row['short_description'])
            with st.expander("Why this game?"):
                explanation = explain_recommendation(
                    selected_game,
                    row['name'],
                    row['genres'],
                    row['tags_text']
                )
                st.write(explanation)