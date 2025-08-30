"""
ML Analysis Page for Myntra Review Scrapper
Advanced analytics using machine learning models
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.cloud_io import MongoIO
from src.constants import SESSION_PRODUCT_KEY
from src.ml_models.ml_integration import MLAnalyzer

# Page configuration
st.set_page_config(page_title="ML Analysis", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Analysis")
st.markdown("---")

# Initialize components
mongo_con = MongoIO()
ml_analyzer = MLAnalyzer()

def display_sentiment_analysis(sentiment_data):
    """Display sentiment analysis results"""
    st.subheader("📊 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution pie chart
        sentiment_dist = sentiment_data['overall_sentiment_distribution']
        fig_pie = px.pie(
            values=list(sentiment_dist.values()),
            names=list(sentiment_dist.keys()),
            title="Overall Sentiment Distribution",
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545', 
                'neutral': '#6c757d'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Average sentiment score
        avg_sentiment = sentiment_data['average_sentiment_score']
        
        # Create gauge chart for sentiment score
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = avg_sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Average Sentiment Score"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightcoral"},
                    {'range': [-0.3, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

def display_quality_analysis(quality_data):
    """Display review quality analysis"""
    st.subheader("🔍 Review Quality Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", quality_data['total_reviews'])
    
    with col2:
        st.metric("High Quality Reviews", 
                 f"{quality_data['high_quality_count']} ({quality_data['high_quality_percentage']}%)")
    
    with col3:
        st.metric("Potential Fake Reviews",
                 f"{quality_data['potential_fake_count']} ({quality_data['fake_percentage']}%)")
    
    with col4:
        st.metric("Avg Quality Score", f"{quality_data['average_quality_score']:.3f}")
    
    # Common fake indicators
    if quality_data['common_fake_indicators']:
        st.markdown("**🚨 Common Fake Review Indicators:**")
        for indicator, count in quality_data['common_fake_indicators']:
            st.write(f"• {indicator}: {count} occurrences")

def display_price_analysis(price_data):
    """Display price analysis results"""
    st.subheader("💰 Price Analysis")
    
    # Price insights
    price_insights = price_data['price_insights']
    price_stats = price_insights['price_statistics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Price", f"₹{price_stats['average_price']}")
        st.metric("Median Price", f"₹{price_stats['median_price']}")
    
    with col2:
        st.metric("Price Range", 
                 f"₹{price_stats['price_range']['min']} - ₹{price_stats['price_range']['max']}")
        st.metric("Price Std Dev", f"₹{price_stats['price_std']}")
    
    with col3:
        correlation = price_insights['price_rating_correlation']
        st.metric("Price-Rating Correlation", f"{correlation:.3f}")
    
    # Value for money analysis
    st.markdown("**📈 Value for Money Analysis**")
    value_analysis = pd.DataFrame(price_data['value_analysis'])
    
    if not value_analysis.empty:
        # Top value products
        top_value = value_analysis.head(5)
        st.dataframe(top_value[['Product_Name', 'Price', 'Rating', 'Value_Score', 'Value_Category']])
        
        # Value distribution
        value_dist = value_analysis['Value_Category'].value_counts()
        fig_bar = px.bar(
            x=value_dist.index, 
            y=value_dist.values,
            title="Value Category Distribution",
            color=value_dist.values,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def display_recommendations(rec_data):
    """Display recommendation system results"""
    st.subheader("🎯 Product Recommendations")
    
    # Content-based recommendations
    st.markdown("**🔗 Content-Based Recommendations**")
    content_recs = rec_data['content_based']
    
    for product, recommendations in content_recs.items():
        with st.expander(f"Similar to: {product}"):
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                st.dataframe(rec_df)
            else:
                st.write("No recommendations available")
    
    # Price-based recommendations
    st.markdown("**💲 Price-Based Recommendations**")
    price_recs = rec_data['price_based']
    
    if price_recs:
        price_rec_df = pd.DataFrame(price_recs)
        st.dataframe(price_rec_df)
    else:
        st.write("No price-based recommendations available")

def display_insights(insights_data):
    """Display overall insights"""
    st.subheader("📈 Key Insights")
    
    # Product performance
    performance = insights_data['product_performance']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🏆 Best Rated Product**")
        best = performance['best_rated']
        st.write(f"**{best['name']}**")
        st.write(f"Rating: {best['rating']}/5")
        st.write(f"Price: ₹{best['price']}")
        st.write(f"Sentiment: {best['sentiment']:.2f}")
    
    with col2:
        st.markdown("**👎 Worst Rated Product**")
        worst = performance['worst_rated']
        st.write(f"**{worst['name']}**")
        st.write(f"Rating: {worst['rating']}/5")
        st.write(f"Price: ₹{worst['price']}")
        st.write(f"Sentiment: {worst['sentiment']:.2f}")
    
    with col3:
        st.markdown("**📝 Most Reviewed Product**")
        most_reviewed = performance['most_reviewed']
        st.write(f"**{most_reviewed['name']}**")
        st.write(f"Reviews: {most_reviewed['review_count']}")
        st.write(f"Rating: {most_reviewed['rating']}/5")
    
    # Overall statistics
    st.markdown("**📊 Overall Statistics**")
    overall = insights_data['overall_stats']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", overall['total_products'])
    
    with col2:
        st.metric("Total Reviews", overall['total_reviews'])
    
    with col3:
        st.metric("Average Rating", f"{overall['average_rating']}/5")
    
    with col4:
        st.metric("Average Price", f"₹{overall['average_price']}")

def main():
    """Main function for ML analysis page"""
    try:
        # Check if data exists
        if 'data' not in st.session_state or not st.session_state.data:
            st.warning("⚠️ No data available for analysis. Please scrape some reviews first!")
            st.markdown("Go to the main page to scrape product reviews.")
            return
        
        # Load data
        if SESSION_PRODUCT_KEY in st.session_state:
            product_name = st.session_state[SESSION_PRODUCT_KEY]
            
            with st.spinner("Loading review data..."):
                try:
                    review_data = mongo_con.get_reviews(product_name=product_name)
                    
                    if review_data is None or review_data.empty:
                        st.error("No review data found!")
                        return
                    
                    st.success(f"Loaded {len(review_data)} reviews for analysis")
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    return
            
            # Run ML analysis
            if st.button("🚀 Run ML Analysis", type="primary"):
                with st.spinner("Running machine learning analysis... This may take a few minutes."):
                    try:
                        # Run comprehensive ML analysis
                        analysis_results = ml_analyzer.analyze_reviews(review_data)
                        
                        # Store results in session state
                        st.session_state['ml_results'] = analysis_results
                        
                        st.success("✅ ML Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error in ML analysis: {str(e)}")
                        return
            
            # Display results if available
            if 'ml_results' in st.session_state:
                results = st.session_state['ml_results']
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Sentiment", "🔍 Quality", "💰 Price", "🎯 Recommendations", "📈 Insights"
                ])
                
                with tab1:
                    display_sentiment_analysis(results['sentiment_analysis'])
                
                with tab2:
                    display_quality_analysis(results['quality_analysis'])
                
                with tab3:
                    display_price_analysis(results['price_analysis'])
                
                with tab4:
                    display_recommendations(results['recommendations'])
                
                with tab5:
                    display_insights(results['insights'])
                
                # Option to download enhanced data
                st.markdown("---")
                st.subheader("📥 Download Enhanced Data")
                
                if st.button("Download Enhanced Dataset"):
                    enhanced_data = results['processed_data']
                    csv = enhanced_data.to_csv(index=False)
                    st.download_button(
                        label="📁 Download CSV",
                        data=csv,
                        file_name=f"{product_name}_enhanced_reviews.csv",
                        mime="text/csv"
                    )
        
        else:
            st.info("No product selected. Please scrape some reviews first.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
