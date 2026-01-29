import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Product AI Mapper", page_icon="🔍", layout="wide")

# Custom CSS for a professional "Care-Group" layout
st.markdown("""
    <style>
    .group-header {
        color: #1f77b4;
        font-size: 0.85rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: -15px;
    }
    .item-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
    }
    .full-path-box {
        background-color: #f8f9fa;
        padding: 12px;
        border-left: 5px solid #1f77b4;
        border-radius: 4px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.95rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD DATA ---
@st.cache_resource
def load_ai():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    return pd.read_csv('Categorized_Product_List.csv')

model = load_ai()
df = load_data()
# We search against the full path which now includes the Care Group name at the start
all_searchable_text = df['Full Path'].tolist()

# --- 3. INTERFACE ---
st.title("🤖 Assistive Technology Categorizer")
st.write("Search by product name or need to find the correct Care Group and Hierarchy.")

query = st.text_input("What are you looking for?", placeholder="e.g. 'crutches', 'eating help', 'fixing a chair'")

if query:
    with st.spinner('Matching with Care Groups...'):
        query_emb = model.encode(query, convert_to_tensor=True)
        choice_embs = model.encode(all_searchable_text, convert_to_tensor=True)
        hits = util.semantic_search(query_emb, choice_embs, top_k=5)[0]

    st.subheader("Top Matches")
    for hit in hits:
        score = int(hit['score'] * 100)
        if score > 35:
            row = df.iloc[hit['corpus_id']]
            
            # This container makes the 'Care Group' (Mobility/Domestic) the hero
            with st.container():
                st.markdown(f"<p class='group-header'>{row['Category Group']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='item-title'>{row['Last Level']}</p>", unsafe_allow_html=True)
                st.markdown(f"<div class='full-path-box'>{row['Full Path']}</div>", unsafe_allow_html=True)
                st.write(f"Match Confidence: {score}%")
                st.divider()