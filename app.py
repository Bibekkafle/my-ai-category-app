import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- PAGE CONFIG ---
st.set_page_config(page_title="Product AI Finder", page_icon="🧩", layout="wide")

# --- LOAD AI & DATA ---
@st.cache_resource
def load_ai():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    return pd.read_csv('Categorized_Product_List.csv')

model = load_ai()
df = load_data()
all_paths = df.iloc[:, 0].tolist() # Reads the first column

# --- INTERFACE ---
st.title("🤖 Intelligent Category Mapper")
st.write("Type a product name or a challenge (e.g., 'help with buttons').")

user_query = st.text_input("🔍 Search", placeholder="Describe what you need...")

if user_query:
    with st.spinner('AI is thinking...'):
        query_emb = model.encode(user_query, convert_to_tensor=True)
        choices_emb = model.encode(all_paths, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, choices_emb, top_k=3)[0]

    st.subheader("Recommended Matches")
    for hit in hits:
        score = int(hit['score'] * 100)
        if score > 30:
            match_text = all_paths[hit['corpus_id']]
            with st.expander(f"✅ {match_text} ({score}% Match)"):
                st.write(f"**Official Category Path:** {match_text}")
                st.progress(score / 100)