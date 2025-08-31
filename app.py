import pandas as pd
import streamlit as st
from src.cloud_io import MongoIO
from src.constants import SESSION_PRODUCT_KEY
from src.scrapper.scrape import ScrapeReviews
from src.scrapper.url_scrape import scrape_reviews_from_url
import re

st.set_page_config(
    page_title="Myntra Review Scrapper & ABSA",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Myntra Review Scrapper & ABSA Analysis")
st.markdown("---")

# Initialize session state
if "data" not in st.session_state:
    st.session_state["data"] = False

def validate_myntra_url(url):
    """Validate if the URL is a valid Myntra product URL."""
    myntra_pattern = r'https?://(www\.)?myntra\.com/.+'
    return bool(re.match(myntra_pattern, url))

def scrape_by_url():
    """Handle URL-based scraping."""
    st.header("🔗 Option 1: Scrape by Product URL")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        product_url = st.text_input(
            "Enter Myntra Product URL:",
            placeholder="https://www.myntra.com/shirts/roadster/roadster-men-blue-printed-cotton-casual-shirt/12345678/buy",
            help="Paste the direct URL of any Myntra product to analyze its reviews"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        scrape_url_button = st.button("🔍 Scrape from URL", type="primary")
    
    if scrape_url_button:
        if not product_url:
            st.error("Please enter a product URL")
            return None
            
        if not validate_myntra_url(product_url):
            st.error("Please enter a valid Myntra product URL")
            return None
        
        try:
            with st.spinner("🔄 Extracting reviews from URL... and wait for 20 sec at review page , manually scroll the whole review page"):
                scrapped_data = scrape_reviews_from_url(product_url)
                
            if scrapped_data is not None and len(scrapped_data) > 0:
                st.success(f"✅ Successfully extracted {len(scrapped_data)} reviews!")
                st.balloons()
                # Store in session state
                st.session_state["data"] = True
                st.session_state["scraped_reviews"] = scrapped_data
                
                # Extract product name from the data
                product_name = scrapped_data.iloc[0]['Product Name'] if len(scrapped_data) > 0 else "URL Product"
                st.session_state[SESSION_PRODUCT_KEY] = product_name
                
                # Store in MongoDB
                try:
                    mongoio = MongoIO()
                    mongoio.store_reviews(product_name=product_name, reviews=scrapped_data)
                    st.info("💾 Data saved to database")
                except Exception as e:
                    st.warning(f"⚠️ Could not save to database: {str(e)}")
                
                # Display preview
                st.subheader("📊 Reviews Preview")
                st.dataframe(scrapped_data.head(), use_container_width=True)
                
                # Show summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reviews", len(scrapped_data))
                with col2:
                    avg_rating = scrapped_data['Rating'].apply(lambda x: float(x) if str(x).replace('.', '').isdigit() else 0).mean()
                    st.metric("Avg Rating", f"{avg_rating:.1f}")
                with col3:
                    st.metric("Product", product_name.split(' - ')[0][:20] + "..." if len(product_name) > 20 else product_name)
                with col4:
                    st.metric("Price", scrapped_data.iloc[0]['Price'] if len(scrapped_data) > 0 else "N/A")
                
                return scrapped_data
                
            else:
                st.error("❌ No reviews found for this product")
                return None
                
        except Exception as e:
            st.error(f"❌ Error scraping reviews: {str(e)}")
            st.info("💡 Make sure the URL is correct and the product has reviews")
            return None
    
    return None

def scrape_by_search():
    """Handle search-based scraping."""
    st.header("🔍 Option 2: Scrape by Product Search")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        product = st.text_input(
            "Search Products:",
            placeholder="white shoes, denim jeans, cotton shirts...",
            help="Enter product keywords to search and scrape reviews"
        )
    
    with col2:
        no_of_products = st.number_input(
            "Number of products:",
            step=1,
            min_value=1,
            max_value=10,
            value=1,
            help="How many products to analyze"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        scrape_search_button = st.button("🛍️ Search & Scrape", type="secondary")
    
    if scrape_search_button:
        if not product:
            st.error("Please enter a product to search")
            return None
            
        try:
            with st.spinner(f"🔄 Searching for '{product}' and extracting reviews..."):
                scrapper = ScrapeReviews(
                    product_name=product,
                    no_of_products=int(no_of_products)
                )
                scrapped_data = scrapper.get_review_data()
                
            if scrapped_data is not None and len(scrapped_data) > 0:
                st.success(f"✅ Successfully extracted {len(scrapped_data)} reviews from {no_of_products} product(s)!")
                
                # Store in session state
                st.session_state["data"] = True
                st.session_state["scraped_reviews"] = scrapped_data
                st.session_state[SESSION_PRODUCT_KEY] = product
                
                # Store in MongoDB
                try:
                    mongoio = MongoIO()
                    mongoio.store_reviews(product_name=product, reviews=scrapped_data)
                    st.info("💾 Data saved to database")
                except Exception as e:
                    st.warning(f"⚠️ Could not save to database: {str(e)}")
                
                # Display preview
                st.subheader("📊 Reviews Preview")
                st.dataframe(scrapped_data.head(), use_container_width=True)
                
                # Show summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reviews", len(scrapped_data))
                with col2:
                    avg_rating = scrapped_data['Rating'].apply(lambda x: float(x) if str(x).replace('.', '').isdigit() else 0).mean()
                    st.metric("Avg Rating", f"{avg_rating:.1f}")
                with col3:
                    unique_products = scrapped_data['Product Name'].nunique()
                    st.metric("Unique Products", unique_products)
                with col4:
                    st.metric("Search Term", product)
                
                return scrapped_data
                
            else:
                st.error("❌ No reviews found for this search")
                return None
                
        except Exception as e:
            st.error(f"❌ Error scraping reviews: {str(e)}")
            st.info("💡 Try different keywords or check your internet connection")
            return None
    
    return None

def main():
    """Main function to coordinate the scraping options."""
    
    # Add instructions
    st.markdown("""
    ### 🎯 How to Use:
    
    **Option 1 - Direct URL**: Perfect when you already have a specific Myntra product in mind
    - Copy the URL from any Myntra product page
    - Paste it below and click "Scrape from URL"
    - Get instant analysis of all reviews for that product
    
    **Option 2 - Product Search**: Great for exploring products by keywords  
    - Enter search terms like "white shoes" or "cotton shirts"
    - Choose how many products to analyze
    - Get comprehensive analysis across multiple products
    """)
    
    st.markdown("---")
    
    # Create two main sections
    tab1, tab2 = st.tabs(["🔗 URL Scraping", "🔍 Search Scraping"])
    
    with tab1:
        data1 = scrape_by_url()
    
    with tab2:
        data2 = scrape_by_search()
    
    # Show next steps if data is available
    if st.session_state.get("data", False):
        st.markdown("---")
        st.success("🎉 **Reviews extracted successfully!**")
        st.info("📈 **Next Step**: Go to the **'Generate Analysis'** page to perform ABSA analysis on your scraped reviews!")
        
        # Quick action button to navigate to analysis
        if st.button("🚀 Go to ABSA Analysis", type="primary"):
            st.info("Please navigate to the 'Generate Analysis' page from the sidebar to perform ABSA analysis.")

if __name__ == "__main__":
    main()
