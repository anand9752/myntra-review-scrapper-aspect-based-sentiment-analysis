"""
Revolutionary Analytics Page
Features that no e-commerce platform offers
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from src.cloud_io import MongoIO
from src.constants import SESSION_PRODUCT_KEY
from src.advanced_features.unique_insights import UniqueInsightGenerator
from src.advanced_features.advanced_analytics import AdvancedAnalyticsEngine

# Page configuration
st.set_page_config(
    page_title="Revolutionary Analytics",
    page_icon="🚀",
    layout="wide"
)

mongo_con = MongoIO()

# Custom CSS for revolutionary design
st.markdown("""
<style>
.revolution-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 3rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.feature-highlight {
    background: linear-gradient(45deg, #ff6b6b, #feca57);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}

.insight-card {
    background: linear-gradient(135deg, #74b9ff, #0984e3);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.warning-card {
    background: linear-gradient(135deg, #fd79a8, #e84393);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}

.success-card {
    background: linear-gradient(135deg, #00b894, #00a085);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
}

.metric-showcase {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
    text-align: center;
}

.unique-badge {
    background: #ff7675;
    color: white;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: bold;
    display: inline-block;
    margin: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

def display_revolutionary_header():
    """Display the revolutionary analytics header"""
    st.markdown("""
    <div class="revolution-header">
        <h1>🚀 Revolutionary Analytics Engine</h1>
        <h3>Features That No E-commerce Platform Offers</h3>
        <p>Discover insights that go beyond basic ratings and reviews</p>
    </div>
    """, unsafe_allow_html=True)

def display_unique_value_proposition():
    """Highlight what makes this unique"""
    st.markdown("## 🎯 **What Makes This Revolutionary**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-highlight">
            <h4>🧠 Psychological Profiling</h4>
            <p>Understand buyer psychology and decision patterns</p>
            <span class="unique-badge">MYNTRA DOESN'T HAVE</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-highlight">
            <h4>🕵️ Hidden Problem Detection</h4>
            <p>Find issues hidden in positive reviews</p>
            <span class="unique-badge">MYNTRA DOESN'T HAVE</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-highlight">
            <h4>📈 Emotional Journey Mapping</h4>
            <p>Track customer emotions from purchase to review</p>
            <span class="unique-badge">MYNTRA DOESN'T HAVE</span>
        </div>
        """, unsafe_allow_html=True)

def display_emotional_journey(emotional_data):
    """Display emotional journey analysis"""
    st.markdown("## 🎭 **Customer Emotional Journey**")
    
    emotions = list(emotional_data['emotional_breakdown'].keys())
    counts = [emotional_data['emotional_breakdown'][emotion]['count'] for emotion in emotions]
    ratings = [emotional_data['emotional_breakdown'][emotion]['avg_rating'] for emotion in emotions]
    
    # Create emotional journey visualization
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Emotional Distribution', 'Emotion vs Rating Correlation'),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # Emotional distribution
    fig.add_trace(
        go.Bar(x=emotions, y=counts, name='Emotion Count', marker_color='lightblue'),
        row=1, col=1
    )
    
    # Emotion vs rating
    fig.add_trace(
        go.Scatter(x=emotions, y=ratings, mode='lines+markers', name='Avg Rating', line=dict(color='red')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="Emotional Intelligence Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Dominant emotion insight
    dominant_emotion = emotional_data['dominant_emotion']
    st.markdown(f"""
    <div class="insight-card">
        <h4>🎯 Dominant Customer Emotion: {dominant_emotion.title()}</h4>
        <p>This tells us about the overall customer experience and satisfaction patterns</p>
    </div>
    """, unsafe_allow_html=True)

def display_psychological_profiles(profile_data):
    """Display psychological buyer profiles"""
    st.markdown("## 🧠 **Psychological Buyer Profiles**")
    
    profiles = profile_data['buyer_profiles']
    
    # Create profile comparison chart
    profile_names = list(profiles.keys())
    satisfaction_rates = [profiles[profile]['satisfaction_rate'] for profile in profile_names]
    avg_ratings = [profiles[profile]['avg_rating'] for profile in profile_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=satisfaction_rates,
        theta=profile_names,
        fill='toself',
        name='Satisfaction Rate',
        line_color='rgb(34, 94, 168)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Buyer Personality Satisfaction Radar"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display dominant buyer type
    dominant_type = profile_data['dominant_buyer_type']
    st.markdown(f"""
    <div class="success-card">
        <h4>👑 Dominant Buyer Personality: {dominant_type.replace('_', ' ').title()}</h4>
        <p>Understanding your primary customer base helps in targeted marketing and product development</p>
    </div>
    """, unsafe_allow_html=True)

def display_competitor_intelligence(competitor_data):
    """Display competitor intelligence"""
    st.markdown("## 🔍 **Competitor Intelligence**")
    
    if competitor_data['competitor_analysis']:
        competitors = list(competitor_data['competitor_analysis'].keys())
        mentions = [competitor_data['competitor_analysis'][comp]['mention_count'] for comp in competitors]
        sentiments = [competitor_data['competitor_analysis'][comp]['sentiment_when_mentioned'] for comp in competitors]
        
        # Competitor mention analysis
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=competitors,
            y=mentions,
            name='Mentions',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="Competitor Mentions in Reviews",
            xaxis_title="Competitors",
            yaxis_title="Number of Mentions"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Most compared brand
        if competitor_data['most_compared_brand']:
            st.markdown(f"""
            <div class="warning-card">
                <h4>⚠️ Most Compared Competitor: {competitor_data['most_compared_brand'].title()}</h4>
                <p>This brand is frequently mentioned in comparisons - monitor their strategies</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No competitor mentions found in reviews")

def display_return_risk_prediction(risk_data):
    """Display return risk prediction"""
    st.markdown("## ⚠️ **Return Risk Prediction**")
    
    risk_breakdown = risk_data['risk_breakdown']
    
    # Risk assessment metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Risk Score", f"{risk_data['overall_return_risk']:.1f}%")
    
    with col2:
        highest_risk = risk_data['highest_risk_factor']
        st.metric("Highest Risk Factor", highest_risk.replace('_', ' ').title() if highest_risk else "N/A")
    
    with col3:
        riskiest_product = risk_data.get('riskiest_product', 'N/A')
        st.metric("Riskiest Product", "Found" if riskiest_product != 'N/A' else "None")
    
    with col4:
        risk_level = "HIGH" if risk_data['overall_return_risk'] > 20 else "LOW"
        st.metric("Risk Level", risk_level)
    
    # Risk breakdown visualization
    risk_types = list(risk_breakdown.keys())
    risk_scores = [risk_breakdown[risk]['risk_score'] for risk in risk_types]
    
    fig = px.funnel(
        x=risk_scores,
        y=risk_types,
        title="Return Risk Factors Analysis"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk recommendations
    if risk_data['overall_return_risk'] > 15:
        st.markdown("""
        <div class="warning-card">
            <h4>🚨 High Return Risk Detected!</h4>
            <p>Consider addressing the highlighted issues to reduce return rates</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-card">
            <h4>✅ Low Return Risk</h4>
            <p>Products show good customer satisfaction and low return probability</p>
        </div>
        """, unsafe_allow_html=True)

def display_trend_predictions(trend_data):
    """Display future trend predictions"""
    st.markdown("## 🔮 **Future Trend Predictions**")
    
    # Trending keywords
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Trending Keywords")
        trending_keywords = trend_data['trending_keywords']
        for i, keyword in enumerate(trending_keywords[:5], 1):
            st.write(f"{i}. **{keyword}**")
    
    with col2:
        st.markdown("### 🎨 Color Trends")
        color_trends = trend_data['color_trends']
        colors = list(color_trends.keys())
        counts = list(color_trends.values())
        
        fig = px.pie(values=counts, names=colors, title="Popular Colors")
        st.plotly_chart(fig, use_container_width=True)
    
    # Style trends
    st.markdown("### 👗 Style Trends")
    style_trends = trend_data['style_trends']
    styles = list(style_trends.keys())
    style_counts = list(style_trends.values())
    
    fig = px.bar(x=styles, y=style_counts, title="Fashion Style Preferences")
    st.plotly_chart(fig, use_container_width=True)

def display_quality_degradation(quality_data):
    """Display quality degradation analysis"""
    st.markdown("## 📉 **Quality Degradation Detection**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        degradation_score = quality_data['overall_degradation_score']
        st.metric("Quality Degradation Score", f"{degradation_score:.1f}%")
        
        status = quality_data['quality_status']
        if status == 'Concerning':
            st.markdown("""
            <div class="warning-card">
                <h4>⚠️ Quality Concerns Detected</h4>
                <p>Monitor product quality closely</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-card">
                <h4>✅ Quality Status: Stable</h4>
                <p>No significant quality degradation detected</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        concerning_product = quality_data.get('most_concerning_product')
        if concerning_product:
            st.markdown(f"""
            <div class="warning-card">
                <h4>🎯 Product Requiring Attention</h4>
                <p><strong>{concerning_product}</strong> shows quality concerns</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    """Main function for revolutionary analytics"""
    display_revolutionary_header()
    
    try:
        if st.session_state.get("data", False):
            data = mongo_con.get_reviews(product_name=st.session_state[SESSION_PRODUCT_KEY])
            
            if data is not None and not data.empty:
                display_unique_value_proposition()
                
                # Initialize analytics engines with error handling
                try:
                    with st.spinner("🔬 Generating Revolutionary Insights..."):
                        # Clean and validate data before processing
                        clean_data = data.copy()
                        
                        # Clean price data to handle concatenated values
                        if 'Price' in clean_data.columns:
                            def safe_price_clean(price_val):
                                if pd.isna(price_val):
                                    return None
                                price_str = str(price_val).strip()
                                if price_str.count('₹') > 1:
                                    import re
                                    # Extract first price value with comma support
                                    match = re.search(r'₹(\d+(?:,\d+)*)', price_str)
                                    if match:
                                        try:
                                            price_part = match.group(1).replace(',', '')
                                            return float(price_part)
                                        except:
                                            return None
                                    return None
                                else:
                                    cleaned = price_str.replace('₹', '').replace(',', '').strip()
                                    try:
                                        # More robust numeric check
                                        if cleaned and (cleaned.replace('.', '').isdigit() or 
                                                      (cleaned.count('.') == 1 and cleaned.replace('.', '').isdigit())):
                                            return float(cleaned)
                                        return None
                                    except:
                                        return None
                            
                            clean_data['Price'] = clean_data['Price'].apply(safe_price_clean)
                        
                        # Ensure required columns exist and have proper data types
                        required_columns = ['Product Name', 'Rating', 'Comment']
                        for col in required_columns:
                            if col not in clean_data.columns:
                                clean_data[col] = ''
                        
                        # Convert Rating to numeric, handling any non-numeric values
                        for col in ['Rating', 'Over_All_Rating']:
                            if col in clean_data.columns:
                                clean_data[col] = pd.to_numeric(clean_data[col], errors='coerce')
                                clean_data[col].fillna(3.0, inplace=True)  # Default neutral rating
                        
                        # Ensure Comment column is string type
                        if 'Comment' in clean_data.columns:
                            clean_data['Comment'] = clean_data['Comment'].astype(str).fillna('')
                        
                        # Remove rows with all NaN values
                        clean_data = clean_data.dropna(how='all')
                        
                        if len(clean_data) == 0:
                            st.error("No valid data available for analysis after cleaning.")
                            return
                        
                        unique_insights = UniqueInsightGenerator(clean_data)
                        advanced_analytics = AdvancedAnalyticsEngine(clean_data)
                        
                        # Generate insights with individual error handling
                        unique_results = {}
                        revolutionary_results = {}
                        
                        try:
                            unique_results = unique_insights.generate_all_insights()
                        except Exception as e:
                            st.warning(f"Some unique insights could not be generated: {str(e)}")
                            unique_results = {}
                        
                        try:
                            revolutionary_results = advanced_analytics.generate_revolutionary_insights()
                        except Exception as e:
                            st.warning(f"Some revolutionary insights could not be generated: {str(e)}")
                            revolutionary_results = {}
                    
                except Exception as e:
                    st.error(f"Error initializing analytics engines: {str(e)}")
                    return
                
                # Display insights in tabs
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🎭 Emotional Journey", 
                    "🧠 Buyer Psychology", 
                    "🔍 Competitor Intel", 
                    "⚠️ Return Risk", 
                    "🔮 Future Trends",
                    "📉 Quality Watch"
                ])
                
                with tab1:
                    try:
                        if 'emotional_journey_mapping' in revolutionary_results:
                            display_emotional_journey(revolutionary_results['emotional_journey_mapping'])
                        else:
                            st.info("Emotional journey analysis not available with current data")
                    except Exception as e:
                        st.error(f"Error displaying emotional journey: {str(e)}")
                
                with tab2:
                    try:
                        if 'psychological_buyer_profiling' in revolutionary_results:
                            display_psychological_profiles(revolutionary_results['psychological_buyer_profiling'])
                        else:
                            st.info("Psychological profiling not available with current data")
                    except Exception as e:
                        st.error(f"Error displaying psychological profiles: {str(e)}")
                
                with tab3:
                    try:
                        if 'competitor_intelligence' in revolutionary_results:
                            display_competitor_intelligence(revolutionary_results['competitor_intelligence'])
                        else:
                            st.info("Competitor intelligence not available with current data")
                    except Exception as e:
                        st.error(f"Error displaying competitor intelligence: {str(e)}")
                
                with tab4:
                    try:
                        if 'return_risk_prediction' in revolutionary_results:
                            display_return_risk_prediction(revolutionary_results['return_risk_prediction'])
                        else:
                            st.info("Return risk prediction not available with current data")
                    except Exception as e:
                        st.error(f"Error displaying return risk: {str(e)}")
                
                with tab5:
                    try:
                        if 'trend_prediction' in revolutionary_results:
                            display_trend_predictions(revolutionary_results['trend_prediction'])
                        else:
                            st.info("Trend predictions not available with current data")
                    except Exception as e:
                        st.error(f"Error displaying trend predictions: {str(e)}")
                
                with tab6:
                    try:
                        if 'quality_degradation_detection' in revolutionary_results:
                            display_quality_degradation(revolutionary_results['quality_degradation_detection'])
                        else:
                            st.info("Quality degradation analysis not available with current data")
                    except Exception as e:
                        st.error(f"Error displaying quality degradation: {str(e)}")
                
                # Revolutionary summary
                st.markdown("---")
                st.markdown("## 🚀 **Revolutionary Summary**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class="metric-showcase">
                        <h4>🎯 Unique Insights Generated</h4>
                        <h2>10+</h2>
                        <p>Revolutionary analytics</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-showcase">
                        <h4>🧠 AI-Powered Features</h4>
                        <h2>100%</h2>
                        <p>Machine learning driven</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div class="metric-showcase">
                        <h4>💡 Competitive Advantage</h4>
                        <h2>∞</h2>
                        <p>No competitor offers this</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.error("❌ No data found for analysis.")
        else:
            st.markdown("""
            <div class="revolution-header">
                <h1>🚀 Revolutionary Analytics</h1>
                <p>No data available for analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            ### 🎯 **Revolutionary Features Awaiting Your Data:**
            
            - 🎭 **Emotional Journey Mapping** - Track customer emotions from purchase to review
            - 🧠 **Psychological Buyer Profiling** - Understand customer psychology and decision patterns  
            - 🕵️ **Hidden Problem Detection** - Find issues buried in positive reviews
            - 🔍 **Competitor Intelligence** - Analyze competitor mentions and positioning
            - ⚠️ **Return Risk Prediction** - Predict likelihood of returns before they happen
            - 🔮 **Future Trend Prediction** - Forecast upcoming fashion and style trends
            - 📉 **Quality Degradation Detection** - Monitor product quality over time
            - 💡 **Social Influence Analysis** - Understand social factors in purchase decisions
            - 🎯 **Personalized Insights** - Custom recommendations based on buyer personas
            - 📈 **Micro-Sentiment Analysis** - Detailed sentiment on specific product aspects
            
            **No e-commerce platform offers these insights!**
            """)
            
            if st.button("🏠 Go to Main Page to Start", type="primary"):
                st.switch_page("app.py")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("🔄 Please try refreshing the page.")

if __name__ == "__main__":
    main()
