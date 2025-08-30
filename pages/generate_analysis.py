import pandas as pd
import streamlit as st
from src.cloud_io import MongoIO
from src.constants import SESSION_PRODUCT_KEY
from src.utils import fetch_product_names_from_cloud
from src.data_report.generate_data_report import DashboardGenerator

# Page configuration
st.set_page_config(
    page_title="Review Analysis",
    page_icon="📊",
    layout="wide"
)

mongo_con = MongoIO()

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f77b4, #17a2b8);
    color: white;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 2rem;
}

.analysis-card {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #dee2e6;
    margin: 1rem 0;
}

.ml-promo {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
}

.feature-list {
    background-color: rgba(255,255,255,0.1);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.btn-container {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

def create_analysis_page(review_data: pd.DataFrame):
    if review_data is not None:
        
        # Main header
        st.markdown("""
        <div class="main-header">
            <h1>📊 Myntra Review Analysis Dashboard</h1>
            <p>Comprehensive insights from your scraped product reviews</p>
        </div>
        """, unsafe_allow_html=True)

        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📦 Total Products", review_data['Product Name'].nunique())
        
        with col2:
            st.metric("📝 Total Reviews", len(review_data))
        
        with col3:
            avg_rating = review_data['Over_All_Rating'].astype(str).str.replace('₹', '').astype(float, errors='ignore').mean()
            st.metric("⭐ Avg Rating", f"{avg_rating:.1f}" if not pd.isna(avg_rating) else "N/A")
        
        with col4:
            unique_users = review_data['Name'].nunique()
            st.metric("👥 Unique Reviewers", unique_users)

        st.markdown("---")

        # Data preview
        with st.expander("🔍 **Preview Your Data**", expanded=False):
            st.dataframe(
                review_data.head(10), 
                use_container_width=True,
                height=400
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Full Dataset",
                    data=review_data.to_csv(index=False),
                    file_name=f"myntra_reviews_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            with col2:
                st.info(f"💡 Dataset contains {len(review_data.columns)} columns and {len(review_data)} rows")

        # Analysis options
        st.markdown("## 🎯 Choose Your Analysis Type")
        
        analysis_col1, analysis_col2 = st.columns([1, 1])
        
        with analysis_col1:
            st.markdown("""
            <div class="analysis-card">
                <h3>📈 Basic Analytics</h3>
                <p>Get fundamental insights including:</p>
                <ul>
                    <li>📊 Rating distributions and trends</li>
                    <li>💰 Price comparisons across products</li>
                    <li>🏷️ Product performance overview</li>
                    <li>📝 Review summaries and highlights</li>
                    <li>📉 Visual charts and graphs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 **Generate Basic Analysis**", type="primary", use_container_width=True):
                with st.spinner("🔄 Generating comprehensive analysis..."):
                    dashboard = DashboardGenerator(review_data)
                    
                    # Display general information
                    dashboard.display_general_info()
                    
                    st.markdown("---")
                    
                    # Display product-specific sections
                    dashboard.display_product_sections()
                    
                    st.success("✅ Basic analysis completed!")
        
        with analysis_col2:
            st.markdown("""
            <div class="ml-promo">
                <h3>🤖 Advanced ML Analysis</h3>
                <p>Unlock deeper insights with AI-powered analytics:</p>
                <div class="feature-list">
                    <strong>🧠 Sentiment Analysis</strong><br>
                    <small>Emotion detection and sentiment scoring</small><br><br>
                    
                    <strong>🔍 Review Quality Detection</strong><br>
                    <small>Identify fake reviews and quality assessment</small><br><br>
                    
                    <strong>💰 Price Prediction</strong><br>
                    <small>Fair pricing and value-for-money analysis</small><br><br>
                    
                    <strong>🎯 Smart Recommendations</strong><br>
                    <small>AI-powered product suggestions</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 **Launch ML Analysis**", type="secondary", use_container_width=True):
                st.switch_page("pages/ml_analysis.py")

        # Quick insights section
        st.markdown("---")
        st.markdown("## ⚡ Quick Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            # Top rated product
            try:
                avg_ratings = review_data.groupby('Product Name')['Over_All_Rating'].first()
                top_product = avg_ratings.idxmax()
                top_rating = avg_ratings.max()
                
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>🏆 Top Rated Product</h4>
                    <p><strong>{top_product}</strong></p>
                    <p>Rating: {top_rating}⭐</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.info("Unable to determine top rated product")
        
        with insight_col2:
            # Most reviewed product
            try:
                review_counts = review_data['Product Name'].value_counts()
                most_reviewed = review_counts.index[0]
                review_count = review_counts.iloc[0]
                
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>� Most Reviewed</h4>
                    <p><strong>{most_reviewed}</strong></p>
                    <p>{review_count} reviews</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.info("Unable to determine most reviewed product")
        
        with insight_col3:
            # Price range
            try:
                # Clean price data
                price_data = review_data['Price'].astype(str).str.replace('₹', '').str.replace(',', '')
                price_data = pd.to_numeric(price_data, errors='coerce')
                min_price = price_data.min()
                max_price = price_data.max()
                
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>💰 Price Range</h4>
                    <p><strong>₹{min_price:.0f} - ₹{max_price:.0f}</strong></p>
                    <p>Across all products</p>
                </div>
                """, unsafe_allow_html=True)
            except:
                st.info("Unable to determine price range")

def main():
    """Main function with error handling"""
    try:
        if st.session_state.get("data", False):
            data = mongo_con.get_reviews(product_name=st.session_state[SESSION_PRODUCT_KEY])
            if data is not None and not data.empty:
                create_analysis_page(data)
            else:
                st.error("❌ No data found for the selected product.")
                st.info("🔄 Try scraping some reviews first from the main page.")
        else:
            st.markdown("""
            <div class="main-header">
                <h1>📊 Review Analysis</h1>
                <p>No data available for analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 🚀 Get Started
            
            To analyze product reviews, you need to:
            
            1. **📱 Go to the main page** - Navigate to the homepage
            2. **🔍 Search for products** - Enter product keywords (e.g., "white shoes")
            3. **📊 Scrape reviews** - Click "Scrape Reviews" to collect data
            4. **📈 Analyze results** - Return here for comprehensive analysis
            
            """)
            
            if st.button("🏠 Go to Main Page", type="primary"):
                st.switch_page("app.py")
                
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("🔄 Please try refreshing the page or contact support.")

if __name__ == "__main__":
    main()

