
import streamlit as st
import pandas as pd
import numpy as np
from preprocess_data import process_data
from content_based_filtering import content_based_recommendation
from collaborative_based_filtering import collaborative_filtering_recommendations
from rating_based_recommendation import get_top_rated_items
from hybrid_approach import hybrid_recommendation_filtering

# Page Configuration
st.set_page_config(
    page_title="AI Recommender - Premium Experience",
    page_icon="B", # Removed emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .product-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
        margin-bottom: 1.5rem;
        height: 420px; /* Fixed height for consistency */
        display: flex;
        flex-direction: column;
        border: 1px solid #eef2f6;
    }
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.1);
        border-color: #3498db;
    }
    .image-container {
        width: 100%;
        height: 200px; /* Fixed height for image area */
        overflow: hidden;
        border-radius: 8px;
        margin-bottom: 12px;
        background-color: #f8f9fa;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .product-image {
        width: 100%;
        height: 100%;
        object-fit: contain; /* Ensure image fits without stretching */
        transition: transform 0.3s;
    }
    .product_info {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .product-title {
        font-size: 1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        display: -webkit-box;
        -webkit-line-clamp: 3; /* Show up to 3 lines */
        -webkit-box-orient: vertical;
        overflow: hidden;
        height: 4.5rem; /* Fixed height for title area */
        line-height: 1.5rem;
    }
    .product-brand {
        color: #7f8c8d;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .product-rating {
        color: #f39c12;
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: auto;
        padding-top: 10px;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #3498db;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
    }
    .sidebar-section {
        background-color: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    /* Compact sidebar history items */
    .sidebar-item-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        background: white;
        padding: 5px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .sidebar-item-img {
        width: 50px;
        height: 50px;
        object-fit: contain;
        margin-right: 10px;
        border-radius: 4px;
    }
    .sidebar-item-text {
        font-size: 0.75rem;
        line-height: 1.1;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Helper to get first image URL
def get_first_image_url(image_url):
    if pd.isna(image_url) or not image_url:
        return "https://via.placeholder.com/200x200?text=No+Image"
    # Some URLs might be lists or strings with pipes
    url = str(image_url).split('|')[0].strip()
    return url

# Load data with caching
@st.cache_data
def load_and_preprocess():
    raw_data = pd.read_csv("clean_data.csv")
    return process_data(raw_data)

data = load_and_preprocess()

# Sidebar
with st.sidebar:
    st.title("AI Recommender")
    st.markdown("---")
    
    # User Selection
    st.subheader("User Selection")
    
    # Get existing user IDs
    real_user_ids = sorted(data['ID'].unique().tolist())
    
    # Add '0_guest' option for new user simulation
    # We maintain a list of options where 0_guest is first
    user_options = ['0_guest'] + [uid for uid in real_user_ids if uid != '0_guest']
    
    # Try to keep user ID in session state
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = user_options[0] # Default to guest
    
    # Ensure current session user is valid
    if st.session_state.current_user_id not in user_options:
        st.session_state.current_user_id = user_options[0]

    current_user_id = st.selectbox(
        "Select User ID", 
        user_options, 
        index=user_options.index(st.session_state.current_user_id)
    )
    st.session_state.current_user_id = current_user_id
    
    st.markdown("---")
    
    # Navigation
    st.subheader("Recommendation Type")
    
    if current_user_id == '0_guest':
        # New/Guest users only see Rating Based (Popularity)
        nav_options = ["Rating Based"]
        st.info("New users: Explore our top rated items!")
    else:
        # Existing users see personalized options first
        nav_options = ["Collaborative Based", "Content Based", "Hybrid Based", "Rating Based"]

    nav = st.radio("Select Type", nav_options)

    st.markdown("---")
    
    # NEW: Recent activity in Sidebar
    st.subheader("Recent Activity")
    user_history = data[data['ID'] == current_user_id][['Name', 'Brand', 'Rating', 'ImageURL', 'ProdID']].tail(5)
    
    if not user_history.empty:
        for idx, (_, row) in enumerate(user_history.iterrows()):
            img_url = get_first_image_url(row['ImageURL'])
            st.markdown(f"""
            <div class="sidebar-item-container">
                <img src="{img_url}" class="sidebar-item-img">
                <div class="sidebar-item-text">
                    <b>{row['Name'][:35]}...</b><br>
                    <span style="color: #f39c12;">Rate: {row['Rating']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent history found.")

    st.markdown("---")
    st.caption(f"Active User: {current_user_id}")

# Main Content Logic
def display_product_grid(products, cols=3):
    if products.empty:
        st.warning("No products found.")
        return
    
    # Pre-process image URLs for display
    products = products.copy()
    if 'ImageURL' in products.columns:
        products['ImageURL_Clean'] = products['ImageURL'].apply(get_first_image_url)
    else:
        products['ImageURL_Clean'] = "https://via.placeholder.com/200x200?text=No+Image"

    rows = len(products) // cols + (1 if len(products) % cols > 0 else 0)
    for i in range(rows):
        st_cols = st.columns(cols)
        for j in range(cols):
            idx = i * cols + j
            if idx < len(products):
                row = products.iloc[idx]
                with st_cols[j]:
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="image-container">
                            <img src="{row['ImageURL_Clean']}" class="product-image">
                        </div>
                        <div class="product_info">
                            <div class="product-title">{row['Name']}</div>
                            <div class="product-brand">{row['Brand']}</div>
                            <div class="product-rating">Rating: {row.get('Rating', 'N/A')} <span style="color:#7f8c8d; font-weight:normal; font-size:0.8rem;">({row.get('ReviewCount', 0)})</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# 2. Page Specific Content
if nav == "Rating Based":
    st.header("Rating Based Recommendations")
    top_n = st.slider("Number of products to show", 4, 20, 8)
    trending_recs = get_top_rated_items(data, top_n=top_n)
    display_product_grid(trending_recs, cols=4)

elif nav == "Collaborative Based":
    st.header(f"Collaborative Filtering Recommendations for User {current_user_id}")
    st.info("Based on your interests and similar users")
    top_n = st.slider("Number of products to show", 4, 12, 6)
    
    with st.spinner("Analyzing preferences..."):
        try:
            collab_recs = collaborative_filtering_recommendations(data, current_user_id, top_n=top_n)
            display_product_grid(collab_recs, cols=3)
        except Exception as e:
            st.error(f"Error generating collaborative recommendations: {e}")
            st.write("Showing trending items instead.")
            display_product_grid(get_top_rated_items(data, top_n=top_n), cols=3)

elif nav == "Content Based":
    st.header("Content Based Recommendations")
    product_names = sorted(data['Name'].unique().tolist())
    
    # Suggest a product from user's history if available
    default_product = user_history['Name'].iloc[0] if not user_history.empty else product_names[0]
    
    selected_product = st.selectbox("Select a product to find similarities:", product_names, index=product_names.index(default_product) if default_product in product_names else 0)
    
    top_n = st.slider("Matches to find", 3, 15, 6)
    
    if st.button("Find Similar"):
        with st.spinner("Finding matches..."):
            content_recs = content_based_recommendation(data, selected_product, top_n=top_n+1)
            # Remove the current product from recommendations if it's there
            if not content_recs.empty:
                content_recs = content_recs[content_recs['Name'] != selected_product].head(top_n)
            display_product_grid(content_recs, cols=3)

elif nav == "Hybrid Based":
    st.header("Hybrid Recommendations")
    st.write("Combining similarities and user behavior for precise recommendations.")
    
    product_names = sorted(data['Name'].unique().tolist())
    default_product = user_history['Name'].iloc[0] if not user_history.empty else product_names[0]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_product = st.selectbox("Product interest:", product_names, index=product_names.index(default_product) if default_product in product_names else 0)
    with col2:
        top_n = st.number_input("Count", 1, 15, 5)
        
    if st.button("Generate Hybrid Mix"):
        with st.spinner("Brewing recommendations..."):
            hybrid_recs = hybrid_recommendation_filtering(data, selected_product, current_user_id, top_n=top_n)
            display_product_grid(hybrid_recs, cols=3)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #7f8c8d;'>Powered by AI Recommendation Engines | 2026</div>", unsafe_allow_html=True)
