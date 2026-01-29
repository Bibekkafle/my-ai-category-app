import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Product Category Mapper", 
    page_icon="🧩", 
    layout="wide"
)

# Custom Styling for the "Whole Path" look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stExpander { 
        background-color: white !important; 
        border: 1px solid #d1d5db !important; 
        border-radius: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .group-tag {
        color: #1d4ed8;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .full-path-display {
        background-color: #f3f4f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 6px solid #1d4ed8;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1f2937;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD AI MODEL & DATA ---
@st.cache_resource
def load_ai():
    # This model understands meanings and synonyms
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    # This reads your Master CSV file
    return pd.read_csv('Categorized_Product_List.csv')

try:
    model = load_ai()
    df = load_data()
    # We search against the Full Path because it contains the Group, Primary, and Item names
    search_corpus = df['Full Path'].tolist()

    # --- 3. SIDEBAR ---
    with st.sidebar:
        st.title("🧩 Category AI")
        st.write("This tool maps your needs to the official hierarchy using Semantic Search.")
        st.divider()
        st.info("The AI understands context. Try searching for a 'problem' instead of a 'product'.")

    # --- 4. MAIN INTERFACE ---
    st.title("🤖 Intelligent Product Mapper")
    st.write("Search for an item to see its complete path (e.g., *Products for Mobility > Walking > Rollators*).")

    user_query = st.text_input("🔍 Search", placeholder="e.g. 'help with stairs', 'bath equipment', or 'fixing my scooter'")

    if user_query:
        with st.spinner('AI is analyzing hierarchies...'):
            # Convert user text into AI coordinates
            query_embedding = model.encode(user_query, convert_to_tensor=True)
            corpus_embeddings = model.encode(search_corpus, convert_to_tensor=True)
            
            # Find the top 5 most relevant items
            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

        st.subheader("Recommended Hierarchies")
        
        for hit in hits:
            score = int(hit['score'] * 100)
            if score > 35: # Only show relevant results
                row = df.iloc[hit['corpus_id']]
                
                # Result Card
                with st.expander(f"✅ {row['Item']} ({score}% match)"):
                    st.markdown(f"<p class='group-tag'>CARE GROUP: {row['Category Group']}</p>", unsafe_allow_html=True)
                    st.write("**Complete Official Path:**")
                    st.markdown(f"<div class='full-path-display'>{row['Full Path']}</div>", unsafe_allow_html=True)
                    st.progress(score / 100)
            else:
                if hit == hits[0]:
                    st.warning("No highly confident matches. Try using different keywords.")
                break
    else:
        # Welcome Screen Visuals
        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1: st.success("**Correct Grouping**\nEnsures items are filed under Mobility, Self-care, etc.")
        with c2: st.info("**Full Path Display**\nShows the entire breadcrumb from your original lists.")
        with c3: st.warning("**Smart Search**\nHandles spelling errors and different product names.")

except FileNotFoundError:
    st.error("Error: 'Categorized_Product_List.csv' not found. Please upload the master CSV file to your GitHub.")
except Exception as e:
    st.error(f"An error occurred: {e}")