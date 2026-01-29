import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Category Mapper", page_icon="🏷️", layout="wide")

# Custom CSS for better look
st.markdown("""
    <style>
    .stExpander { border: 1px solid #e6e9ef; border-radius: 10px; margin-bottom: 10px; }
    .path-text { color: #555; font-family: monospace; font-size: 0.9rem; background: #f0f2f6; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD AI & DATA ---
@st.cache_resource
def load_ai():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    return pd.read_csv('Categorized_Product_List.csv')

model = load_ai()
df = load_data()
all_paths = df['Full Path'].tolist()

# --- 3. INTERFACE ---
st.title("🤖 Smart Product Categorizer")
st.write("Find any item's full path starting from its main Care Group.")

user_query = st.text_input("🔍 Search for a product or a need:", placeholder="e.g. wheelchair, kitchen help, or zipper")

if user_query:
    with st.spinner('AI is searching through 356 categories...'):
        query_emb = model.encode(user_query, convert_to_tensor=True)
        choices_emb = model.encode(all_paths, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, choices_emb, top_k=5)[0]

    st.subheader("Top Matches")
    for hit in hits:
        score = int(hit['score'] * 100)
        if score > 35:
            row = df.iloc[hit['corpus_id']]
            with st.expander(f"✅ {row['Last Level']} ({score}% match)"):
                st.write("**Full Category Hierarchy:**")
                st.markdown(f"<div class='path-text'>{row['Full Path']}</div>", unsafe_allow_html=True)
                st.progress(score / 100)