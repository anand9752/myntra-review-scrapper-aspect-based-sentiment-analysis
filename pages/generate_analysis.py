import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.cloud_io import MongoIO
from src.constants import SESSION_PRODUCT_KEY
from src.utils import fetch_product_names_from_cloud
from src.data_report.generate_data_report import DashboardGenerator
from src.absa import ABSAAnalyzer, AdvancedABSAAnalyzer, SimpleAdvancedABSA

mongo_con = MongoIO()


def create_analysis_page(review_data: pd.DataFrame):
    if review_data is not None:
        
        # Show data source info
        st.info(f"📊 **Analyzing {len(review_data)} reviews** for product: **{st.session_state.get(SESSION_PRODUCT_KEY, 'Unknown Product')}**")
        
        # Show data preview with option to expand
        with st.expander("👀 View Raw Data", expanded=False):
            st.dataframe(review_data, use_container_width=True)
        
        # Create tabs for different types of analysis
        tab1, tab2, tab3 = st.tabs(["📊 General Analysis", "🎯 Aspect-Based Analysis", "📈 ABSA Insights"])
        
        with tab1:
            if st.button("Generate General Analysis"):
                dashboard = DashboardGenerator(review_data)

                # Display general information
                dashboard.display_general_info()

                # Display product-specific sections
                dashboard.display_product_sections()
        
        with tab2:
            st.write("### 🎯 Aspect-Based Sentiment Analysis Configuration")
            
            # Model selection
            col1, col2 = st.columns(2)
            with col1:
                analysis_method = st.selectbox(
                    "Choose Analysis Method:",
                    ["VADER (Recommended)", "Enhanced TextBlob+VADER (Experimental)", "Advanced Transformer (Experimental)"],
                    index=0,  # Default to VADER as it's most reliable
                    help="VADER is fastest and most reliable. Enhanced adds TextBlob analysis. Transformer is experimental."
                )
            
            with col2:
                st.info(
                    "⭐ **VADER**: Fast, reliable, and well-tested\n\n"
                    "🧪 **Enhanced**: Combines TextBlob + VADER\n\n"
                    "🤖 **Transformer**: AI-powered (may be slow)"
                )
            
            if st.button("Generate ABSA Analysis"):
                st.write("### Aspect-Based Sentiment Analysis Results")
                
                # Choose analyzer based on selection
                if "Enhanced" in analysis_method:
                    absa_analyzer = SimpleAdvancedABSA(use_textblob=True)
                    analysis_method_name = "Enhanced TextBlob+VADER"
                elif "Advanced" in analysis_method:
                    absa_analyzer = AdvancedABSAAnalyzer(use_transformers=True)
                    analysis_method_name = "Advanced Transformer"
                else:
                    absa_analyzer = ABSAAnalyzer()
                    analysis_method_name = "VADER"
                
                with st.spinner(f"Analyzing reviews using {analysis_method_name} model..."):
                    try:
                        # Perform ABSA analysis
                        if "Enhanced" in analysis_method:
                            analysis_results = absa_analyzer.analyze_product_aspects_enhanced(
                                review_data, 
                                product_name=st.session_state.get(SESSION_PRODUCT_KEY, "Unknown Product")
                            )
                        elif "Advanced" in analysis_method:
                            analysis_results = absa_analyzer.analyze_product_aspects_advanced(
                                review_data, 
                                product_name=st.session_state.get(SESSION_PRODUCT_KEY, "Unknown Product")
                            )
                        else:
                            analysis_results = absa_analyzer.analyze_product_aspects(
                                review_data, 
                                product_name=st.session_state.get(SESSION_PRODUCT_KEY, "Unknown Product")
                            )
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        st.info("Falling back to basic VADER analysis...")
                        absa_analyzer = ABSAAnalyzer()
                        analysis_results = absa_analyzer.analyze_product_aspects(
                            review_data, 
                            product_name=st.session_state.get(SESSION_PRODUCT_KEY, "Unknown Product")
                        )
                        analysis_method_name = "VADER (Fallback)"
                
                # Display results
                st.success(f"Analysis completed using {analysis_method_name}!")
                
                # Overall sentiment
                st.write(f"**Product:** {analysis_results['product_name']}")
                st.write(f"**Total Reviews:** {analysis_results['total_reviews']}")
                st.write(f"**Analysis Method:** {analysis_results.get('analysis_method', analysis_method_name)}")
                st.write(f"**Overall Sentiment Score:** {analysis_results['overall_avg_sentiment']:.3f}")
                
                # Show confidence if available
                if 'overall_avg_confidence' in analysis_results:
                    st.write(f"**Overall Confidence:** {analysis_results['overall_avg_confidence']:.3f}")
                
                # Sentiment interpretation
                if analysis_results['overall_avg_sentiment'] >= 0.05:
                    sentiment_emoji = "😊"
                    sentiment_text = "Positive"
                elif analysis_results['overall_avg_sentiment'] <= -0.05:
                    sentiment_emoji = "😞"
                    sentiment_text = "Negative"
                else:
                    sentiment_emoji = "😐"
                    sentiment_text = "Neutral"
                
                st.write(f"**Overall Sentiment:** {sentiment_emoji} {sentiment_text}")
                
                # Show model usage statistics if available
                if 'model_usage' in analysis_results:
                    st.write("**Model Usage Statistics:**")
                    for model, count in analysis_results['model_usage'].items():
                        st.write(f"  - {model}: {count} analyses")
                
                # Show enhanced confidence statistics if available
                if 'confidence_stats' in analysis_results:
                    st.write("**Enhanced Performance Metrics:**")
                    conf_stats = analysis_results['confidence_stats']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Positive Confidence", f"{conf_stats['avg_positive_confidence']:.3f}")
                    with col2:
                        st.metric("Negative Confidence", f"{conf_stats['avg_negative_confidence']:.3f}")
                    with col3:
                        st.metric("Neutral Confidence", f"{conf_stats['avg_neutral_confidence']:.3f}")
                    with col4:
                        if 'high_confidence_predictions' in conf_stats:
                            st.metric("High Confidence", f"{conf_stats['high_confidence_predictions']}/{analysis_results['total_reviews']}")
                
                # Show sentiment strength distribution if available
                if 'sentiment_strength' in analysis_results:
                    st.write("**Sentiment Strength Distribution:**")
                    strength = analysis_results['sentiment_strength']
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Very Positive", strength['very_positive'])
                    with col2:
                        st.metric("Positive", strength['positive'])
                    with col3:
                        st.metric("Neutral", strength['neutral'])
                    with col4:
                        st.metric("Negative", strength['negative'])
                    with col5:
                        st.metric("Very Negative", strength['very_negative'])
                
                # Store results in session state for the insights tab
                st.session_state['absa_results'] = analysis_results
        
        with tab3:
            if 'absa_results' in st.session_state:
                display_absa_insights(st.session_state['absa_results'])
            else:
                st.info("Please run ABSA Analysis first to see insights.")


def display_absa_insights(analysis_results):
    """Display detailed ABSA insights with visualizations."""
    st.write("### 🎯 Detailed Aspect-Based Sentiment Analysis")
    
    aspect_summary = analysis_results['aspect_summary']
    detailed_absa = analysis_results['detailed_absa']
    
    # Aspect Distribution Pie Chart
    st.write("#### Aspect Mentions Distribution")
    aspect_counts = aspect_summary.groupby('aspect')['count'].sum().reset_index()
    fig_pie = px.pie(aspect_counts, values='count', names='aspect', 
                     title='Distribution of Aspect Mentions')
    st.plotly_chart(fig_pie)
    
    # Sentiment by Aspect Bar Chart
    st.write("#### Sentiment Distribution by Aspect")
    fig_bar = px.bar(aspect_summary, x='aspect', y='count', color='sentiment_label',
                     title='Sentiment Distribution Across Aspects',
                     color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'})
    fig_bar.update_xaxes(tickangle=45)
    st.plotly_chart(fig_bar)
    
    # Sentiment Scores Heatmap
    st.write("#### Average Sentiment Scores by Aspect")
    pivot_data = aspect_summary.pivot(index='aspect', columns='sentiment_label', values='avg_sentiment_score')
    pivot_data = pivot_data.fillna(0)
    
    fig_heatmap = px.imshow(pivot_data, 
                           title='Average Sentiment Scores by Aspect and Sentiment Type',
                           color_continuous_scale='RdYlGn',
                           aspect='auto')
    st.plotly_chart(fig_heatmap)
    
    # Most Mentioned Aspects
    st.write("#### Most Mentioned Aspects")
    most_mentioned = analysis_results['most_mentioned_aspects']
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (aspect, count) in enumerate(list(most_mentioned.items())[:3], 1):
            st.metric(f"#{i} {aspect}", f"{count} mentions")
    
    with col2:
        # Show aspect-wise sentiment breakdown
        st.write("**Sentiment Breakdown:**")
        for aspect in most_mentioned.keys():
            aspect_data = aspect_summary[aspect_summary['aspect'] == aspect]
            if not aspect_data.empty:
                pos_count = aspect_data[aspect_data['sentiment_label'] == 'Positive']['count'].sum()
                neg_count = aspect_data[aspect_data['sentiment_label'] == 'Negative']['count'].sum()
                neu_count = aspect_data[aspect_data['sentiment_label'] == 'Neutral']['count'].sum()
                
                st.write(f"**{aspect}:** ✅{pos_count} ❌{neg_count} ⚪{neu_count}")
    
    # Detailed Review Analysis
    st.write("#### Sample Reviews by Aspect and Sentiment")
    
    # Allow user to select aspect and sentiment
    aspects = detailed_absa['aspect'].unique()
    sentiments = detailed_absa['sentiment_label'].unique()
    
    selected_aspect = st.selectbox("Select Aspect:", aspects)
    selected_sentiment = st.selectbox("Select Sentiment:", sentiments)
    
    # Filter and display sample reviews
    filtered_reviews = detailed_absa[
        (detailed_absa['aspect'] == selected_aspect) & 
        (detailed_absa['sentiment_label'] == selected_sentiment)
    ].head(5)
    
    if not filtered_reviews.empty:
        st.write(f"**Sample {selected_sentiment} reviews about {selected_aspect}:**")
        for idx, row in filtered_reviews.iterrows():
            with st.expander(f"Review {idx + 1} (Score: {row['compound_score']:.3f})"):
                st.write(row['original_review'])
    else:
        st.write(f"No {selected_sentiment.lower()} reviews found for {selected_aspect}")
    
    # Download enhanced data
    st.write("#### 📥 Download Enhanced Data")
    enhanced_reviews = analysis_results['enhanced_reviews']
    
    col1, col2 = st.columns(2)
    with col1:
        csv_enhanced = enhanced_reviews.to_csv(index=False)
        st.download_button(
            label="Download Enhanced Reviews (CSV)",
            data=csv_enhanced,
            file_name=f"enhanced_reviews_{analysis_results['product_name']}.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_detailed = detailed_absa.to_csv(index=False)
        st.download_button(
            label="Download Detailed ABSA (CSV)",
            data=csv_detailed,
            file_name=f"absa_analysis_{analysis_results['product_name']}.csv",
            mime="text/csv"
        )


def show_no_data_message():
    """Show message when no data is available."""
    st.markdown("""
    # 📊 ABSA Analysis Dashboard
    
    ### 🚫 No Data Available for Analysis
    
    To perform Aspect-Based Sentiment Analysis, you need to first scrape some reviews.
    
    #### 🔄 How to Get Started:
    
    1. **Go to the Home Page** (Main app)
    2. **Choose one of two options:**
       - 🔗 **URL Scraping**: Paste a direct Myntra product URL
       - 🔍 **Search Scraping**: Search for products by keywords
    3. **Extract Reviews** using either method
    4. **Return here** to perform ABSA analysis
    
    #### 💡 Example URLs:
    ```
    https://www.myntra.com/tshirts/roadster/roadster-men-navy-blue-printed-cotton-t-shirt/12345678/buy
    https://www.myntra.com/jeans/levis/levis-511-slim-fit-mid-rise-clean-look-stretchable-jeans/67890123/buy
    ```
    
    #### 🔍 Example Search Terms:
    - "white running shoes"
    - "cotton casual shirts" 
    - "denim jeans men"
    - "women ethnic wear"
    """)
    
    with st.sidebar:
        st.markdown("""
        ### 🎯 Quick Guide
        
        **No Data Available**
        
        Please go to the main page and:
        1. Scrape reviews using URL or search
        2. Return here for analysis
        
        **Need Help?**
        - Make sure you have a stable internet connection
        - Verify Myntra URLs are complete and valid
        - Try different search keywords if no results
        """)


try:
    # Check if we have scraped data in session state (from main app)
    if st.session_state.get("data", False) and "scraped_reviews" in st.session_state:
        review_data = st.session_state["scraped_reviews"]
        st.success("✅ Using recently scraped data from the main page")
        create_analysis_page(review_data)
    
    # Fallback: check if we have data from MongoDB (legacy method)
    elif st.session_state.get("data", False) and SESSION_PRODUCT_KEY in st.session_state:
        mongo_data = mongo_con.get_reviews(product_name=st.session_state[SESSION_PRODUCT_KEY])
        if mongo_data is not None and len(mongo_data) > 0:
            st.info("📂 Using data from database")
            create_analysis_page(mongo_data)
        else:
            st.warning("⚠️ No data found in database for the selected product")
            show_no_data_message()
    
    else:
        show_no_data_message()

except AttributeError:
    show_no_data_message()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    show_no_data_message()


def show_no_data_message():
    """Show message when no data is available."""
    st.markdown("""
    # 📊 ABSA Analysis Dashboard
    
    ### 🚫 No Data Available for Analysis
    
    To perform Aspect-Based Sentiment Analysis, you need to first scrape some reviews.
    
    #### 🔄 How to Get Started:
    
    1. **Go to the Home Page** (Main app)
    2. **Choose one of two options:**
       - 🔗 **URL Scraping**: Paste a direct Myntra product URL
       - 🔍 **Search Scraping**: Search for products by keywords
    3. **Extract Reviews** using either method
    4. **Return here** to perform ABSA analysis
    
    #### 💡 Example URLs:
    ```
    https://www.myntra.com/tshirts/roadster/roadster-men-navy-blue-printed-cotton-t-shirt/12345678/buy
    https://www.myntra.com/jeans/levis/levis-511-slim-fit-mid-rise-clean-look-stretchable-jeans/67890123/buy
    ```
    
    #### 🔍 Example Search Terms:
    - "white running shoes"
    - "cotton casual shirts" 
    - "denim jeans men"
    - "women ethnic wear"
    """)
    
    with st.sidebar:
        st.markdown("""
        ### 🎯 Quick Guide
        
        **No Data Available**
        
        Please go to the main page and:
        1. Scrape reviews using URL or search
        2. Return here for analysis
        
        **Need Help?**
        - Make sure you have a stable internet connection
        - Verify Myntra URLs are complete and valid
        - Try different search keywords if no results
        """)


# Call the main logic
try:
    # Check if we have scraped data in session state (from main app)
    if st.session_state.get("data", False) and "scraped_reviews" in st.session_state:
        review_data = st.session_state["scraped_reviews"]
        st.success("✅ Using recently scraped data from the main page")
        create_analysis_page(review_data)
    
    # Fallback: check if we have data from MongoDB (legacy method)
    elif st.session_state.get("data", False) and SESSION_PRODUCT_KEY in st.session_state:
        mongo_data = mongo_con.get_reviews(product_name=st.session_state[SESSION_PRODUCT_KEY])
        if mongo_data is not None and len(mongo_data) > 0:
            st.info("📂 Using data from database")
            create_analysis_page(mongo_data)
        else:
            st.warning("⚠️ No data found in database for the selected product")
            show_no_data_message()
    
    else:
        show_no_data_message()

except AttributeError:
    show_no_data_message()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    show_no_data_message()

