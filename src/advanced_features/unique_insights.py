"""
Advanced Unique Features for Myntra Review Analysis
Features that go beyond basic ratings available on Myntra
"""

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from collections import Counter
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


class UniqueInsightGenerator:
    """
    Generate unique insights that Myntra doesn't provide
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.insights = {}
        
    def generate_all_insights(self):
        """Generate all unique insights"""
        self.insights.update({
            'size_fit_analysis': self.analyze_size_and_fit(),
            'review_authenticity': self.detect_review_patterns(),
            'seasonal_trends': self.analyze_seasonal_patterns(),
            'reviewer_behavior': self.analyze_reviewer_behavior(),
            'quality_durability': self.extract_quality_concerns(),
            'competitor_comparison': self.cross_brand_analysis(),
            'hidden_issues': self.discover_hidden_problems(),
            'price_value_insights': self.advanced_price_analysis(),
            'styling_suggestions': self.extract_styling_tips(),
            'purchase_decision_factors': self.identify_decision_factors()
        })
        return self.insights
    
    def analyze_size_and_fit(self):
        """Analyze size and fit complaints - unique insight Myntra doesn't show"""
        size_keywords = {
            'too_small': ['small', 'tight', 'fitted', 'narrow', 'size up', 'smaller'],
            'too_large': ['large', 'loose', 'big', 'oversized', 'size down', 'bigger'],
            'perfect_fit': ['perfect fit', 'true to size', 'fits well', 'good fit'],
            'size_inconsistent': ['inconsistent', 'varies', 'different sizes']
        }
        
        size_analysis = {}
        comments = self.df['Comment'].fillna('').str.lower()
        
        for category, keywords in size_keywords.items():
            pattern = '|'.join(keywords)
            matches = comments.str.contains(pattern, regex=True, na=False)
            size_analysis[category] = {
                'count': matches.sum(),
                'percentage': (matches.sum() / len(self.df)) * 100,
                'sample_reviews': comments[matches].head(3).tolist()
            }
        
        # Size recommendation
        if size_analysis['too_small']['count'] > size_analysis['too_large']['count']:
            recommendation = "Consider ordering one size larger"
        elif size_analysis['too_large']['count'] > size_analysis['too_small']['count']:
            recommendation = "Consider ordering one size smaller"
        else:
            recommendation = "Sizes appear to be true to size"
            
        return {
            'analysis': size_analysis,
            'recommendation': recommendation,
            'confidence_score': max(size_analysis['too_small']['percentage'], 
                                  size_analysis['too_large']['percentage'])
        }
    
    def detect_review_patterns(self):
        """Detect suspicious review patterns and authenticity issues"""
        
        # Group by reviewer name
        reviewer_analysis = self.df.groupby('Name').agg({
            'Comment': ['count', 'nunique'],
            'Rating': ['mean', 'std'],
            'Product Name': 'nunique'
        }).round(2)
        
        reviewer_analysis.columns = ['review_count', 'unique_comments', 'avg_rating', 'rating_std', 'products_reviewed']
        
        # Detect suspicious patterns
        suspicious_reviewers = reviewer_analysis[
            (reviewer_analysis['review_count'] > 5) |  # Too many reviews
            (reviewer_analysis['rating_std'] == 0) |   # Always same rating
            (reviewer_analysis['unique_comments'] < reviewer_analysis['review_count'] * 0.5)  # Repetitive comments
        ]
        
        # Detect copy-paste reviews
        comment_duplicates = self.df['Comment'].value_counts()
        duplicate_reviews = comment_duplicates[comment_duplicates > 1]
        
        return {
            'suspicious_reviewers': len(suspicious_reviewers),
            'duplicate_reviews': len(duplicate_reviews),
            'authenticity_score': max(0, 100 - (len(suspicious_reviewers) * 10) - (len(duplicate_reviews) * 5)),
            'red_flags': {
                'mass_reviewers': suspicious_reviewers.head().to_dict(),
                'copied_reviews': duplicate_reviews.head().to_dict()
            }
        }
    
    def analyze_seasonal_patterns(self):
        """Analyze if reviews show seasonal buying patterns"""
        # This would require timestamp data, simulating with random patterns
        
        seasonal_keywords = {
            'summer': ['summer', 'hot', 'cooling', 'breathable', 'light'],
            'winter': ['winter', 'warm', 'cozy', 'thick', 'insulated'],
            'festival': ['festival', 'diwali', 'christmas', 'wedding', 'party'],
            'casual': ['daily wear', 'casual', 'office', 'work', 'comfortable']
        }
        
        seasonal_data = {}
        comments = self.df['Comment'].fillna('').str.lower()
        
        for season, keywords in seasonal_keywords.items():
            pattern = '|'.join(keywords)
            matches = comments.str.contains(pattern, regex=True, na=False)
            seasonal_data[season] = {
                'mentions': matches.sum(),
                'percentage': (matches.sum() / len(self.df)) * 100
            }
        
        dominant_season = max(seasonal_data.keys(), key=lambda x: seasonal_data[x]['mentions'])
        
        return {
            'seasonal_breakdown': seasonal_data,
            'dominant_use_case': dominant_season,
            'recommendation': f"This product seems most popular for {dominant_season} use"
        }
    
    def analyze_reviewer_behavior(self):
        """Analyze reviewer behavior patterns"""
        
        # Review length analysis
        self.df['comment_length'] = self.df['Comment'].fillna('').str.len()
        self.df['word_count'] = self.df['Comment'].fillna('').str.split().str.len()
        
        # Correlation between review length and rating
        length_rating_corr = self.df['comment_length'].corr(
            pd.to_numeric(self.df['Rating'], errors='coerce')
        )
        
        # Sentiment vs Rating mismatch
        sentiments = []
        for comment in self.df['Comment'].fillna(''):
            try:
                sentiment = TextBlob(str(comment)).sentiment.polarity
                sentiments.append(sentiment)
            except:
                sentiments.append(0)
        
        self.df['sentiment_score'] = sentiments
        
        # Find reviews where sentiment doesn't match rating
        numeric_ratings = pd.to_numeric(self.df['Rating'], errors='coerce')
        high_rating_negative_sentiment = (
            (numeric_ratings >= 4) & (self.df['sentiment_score'] < -0.1)
        ).sum()
        
        return {
            'avg_review_length': self.df['comment_length'].mean(),
            'length_rating_correlation': length_rating_corr,
            'sentiment_rating_mismatch': high_rating_negative_sentiment,
            'detailed_reviewers_rating': self.df[self.df['word_count'] > 20]['Rating'].mean(),
            'brief_reviewers_rating': self.df[self.df['word_count'] <= 5]['Rating'].mean()
        }
    
    def extract_quality_concerns(self):
        """Extract specific quality and durability concerns"""
        
        quality_issues = {
            'fabric_quality': ['fabric', 'material', 'cloth', 'texture', 'feel'],
            'durability': ['durability', 'lasting', 'wear', 'tear', 'damage'],
            'stitching': ['stitching', 'seams', 'threads', 'loose'],
            'color_fade': ['color', 'fade', 'bleeding', 'wash'],
            'sizing_issues': ['size', 'fit', 'length', 'width'],
            'delivery_packaging': ['packaging', 'delivery', 'condition', 'damaged']
        }
        
        issue_analysis = {}
        comments = self.df['Comment'].fillna('').str.lower()
        
        for issue, keywords in quality_issues.items():
            # Look for negative context around these keywords
            negative_words = ['bad', 'poor', 'terrible', 'awful', 'worst', 'horrible', 'cheap']
            
            issue_mentions = 0
            problematic_reviews = []
            
            for keyword in keywords:
                for neg_word in negative_words:
                    pattern = f"{neg_word}.*{keyword}|{keyword}.*{neg_word}"
                    matches = comments.str.contains(pattern, regex=True, na=False)
                    issue_mentions += matches.sum()
                    if matches.any():
                        problematic_reviews.extend(comments[matches].head(2).tolist())
            
            issue_analysis[issue] = {
                'mentions': issue_mentions,
                'severity': 'High' if issue_mentions > len(self.df) * 0.1 else 'Low',
                'sample_complaints': problematic_reviews[:3]
            }
        
        return issue_analysis
    
    def cross_brand_analysis(self):
        """Compare across different brands in the dataset"""
        
        if 'Brand' not in self.df.columns:
            # Extract brand from product name
            self.df['Brand'] = self.df['Product Name'].str.extract(r'^([A-Za-z]+)')
        
        # Clean price data before aggregation
        def clean_price_for_analysis(price_value):
            """Clean price data for brand analysis"""
            if pd.isna(price_value):
                return None
            
            price_str = str(price_value).strip()
            
            # Handle concatenated prices
            if price_str.count('₹') > 1:
                import re
                price_match = re.search(r'₹(\d+)', price_str)
                if price_match:
                    try:
                        return float(price_match.group(1))
                    except (ValueError, TypeError):
                        return None
                else:
                    numbers_only = re.sub(r'[^\d]', '', price_str)
                    if numbers_only:
                        estimated_length = min(4, len(numbers_only) // max(1, price_str.count('₹')))
                        try:
                            return float(numbers_only[:estimated_length]) if estimated_length > 0 else None
                        except (ValueError, TypeError):
                            return None
                    return None
            else:
                cleaned = price_str.replace('₹', '').replace(',', '').strip()
                try:
                    return float(cleaned) if cleaned and cleaned.replace('.', '').isdigit() else None
                except (ValueError, TypeError):
                    return None
        
        # Create a copy with cleaned price data
        df_clean = self.df.copy()
        df_clean['Price_Clean'] = df_clean['Price'].apply(clean_price_for_analysis)
        
        brand_analysis = df_clean.groupby('Brand').agg({
            'Rating': ['mean', 'count'],
            'Price_Clean': 'mean',
            'Comment': lambda x: ' '.join(x.fillna('')).count('recommend') / len(x)
        }).round(2)
        
        brand_analysis.columns = ['avg_rating', 'review_count', 'avg_price', 'recommendation_rate']
        
        # Value for money analysis
        brand_analysis['value_score'] = (
            brand_analysis['avg_rating'] / (brand_analysis['avg_price'] / 1000)
        ).round(2)
        
        return {
            'brand_comparison': brand_analysis.to_dict(),
            'best_value_brand': brand_analysis['value_score'].idxmax(),
            'highest_rated_brand': brand_analysis['avg_rating'].idxmax(),
            'most_recommended_brand': brand_analysis['recommendation_rate'].idxmax()
        }
    
    def discover_hidden_problems(self):
        """Use NLP to discover problems not explicitly mentioned in ratings"""
        
        # Extract complaints from positive ratings (hidden issues)
        positive_ratings = self.df[pd.to_numeric(self.df['Rating'], errors='coerce') >= 4]
        
        complaint_keywords = [
            'but', 'however', 'although', 'except', 'issue', 'problem', 'concern',
            'wish', 'could be better', 'improvement', 'disappointed'
        ]
        
        hidden_issues = []
        for comment in positive_ratings['Comment'].fillna(''):
            comment_lower = str(comment).lower()
            for keyword in complaint_keywords:
                if keyword in comment_lower:
                    # Extract sentence containing the complaint
                    sentences = comment.split('.')
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            hidden_issues.append(sentence.strip())
                            break
        
        # Cluster common hidden issues
        if hidden_issues:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            try:
                X = vectorizer.fit_transform(hidden_issues)
                kmeans = KMeans(n_clusters=min(3, len(hidden_issues)), random_state=42)
                clusters = kmeans.fit_predict(X)
                
                clustered_issues = {}
                for i, cluster in enumerate(clusters):
                    if cluster not in clustered_issues:
                        clustered_issues[cluster] = []
                    clustered_issues[cluster].append(hidden_issues[i])
                
                return {
                    'hidden_issue_count': len(hidden_issues),
                    'issue_clusters': clustered_issues,
                    'sample_hidden_issues': hidden_issues[:5]
                }
            except:
                return {
                    'hidden_issue_count': len(hidden_issues),
                    'sample_hidden_issues': hidden_issues[:5]
                }
        
        return {'hidden_issue_count': 0, 'sample_hidden_issues': []}
    
    def advanced_price_analysis(self):
        """Advanced price-value analysis"""
        
        # Clean price data with improved handling for concatenated prices
        if 'Price' in self.df.columns:
            def clean_price_data(price_value):
                """Clean concatenated price data"""
                if pd.isna(price_value):
                    return None  # Use None instead of np.nan for better compatibility
                
                price_str = str(price_value).strip()
                
                # Handle concatenated prices like ₹919₹919₹919...
                if price_str.count('₹') > 1:
                    import re
                    # Extract first price value
                    price_match = re.search(r'₹(\d+)', price_str)
                    if price_match:
                        try:
                            return float(price_match.group(1))
                        except (ValueError, TypeError):
                            return None
                    else:
                        # Fallback: extract first numeric sequence
                        numbers_only = re.sub(r'[^\d]', '', price_str)
                        if numbers_only:
                            # Estimate original price length (typically 3-5 digits for clothing)
                            estimated_length = min(4, len(numbers_only) // max(1, price_str.count('₹')))
                            try:
                                return float(numbers_only[:estimated_length]) if estimated_length > 0 else None
                            except (ValueError, TypeError):
                                return None
                        return None
                else:
                    # Regular price cleaning
                    cleaned = price_str.replace('₹', '').replace(',', '').strip()
                    try:
                        return float(cleaned) if cleaned and cleaned.replace('.', '').isdigit() else None
                    except (ValueError, TypeError):
                        return None
            
            # Apply cleaning and convert to Series with proper dtype
            prices_cleaned = self.df['Price'].apply(clean_price_data)
            prices = pd.Series(prices_cleaned).astype('float64', errors='ignore')
            # Remove any remaining non-numeric values
            prices = prices.dropna()
            
            ratings_cleaned = pd.to_numeric(self.df['Rating'], errors='coerce')
            ratings = ratings_cleaned.dropna()
            
            # Ensure we have valid data to work with
            if len(prices) == 0 or len(ratings) == 0:
                return {'error': 'No valid price or rating data available for analysis'}
            
            # Price vs satisfaction correlation
            price_satisfaction_corr = prices.corr(ratings)
            
            # Price segments analysis
            price_quartiles = prices.quantile([0.25, 0.5, 0.75])
            
            price_segments = {
                'budget': (prices <= price_quartiles[0.25]),
                'mid_range': ((prices > price_quartiles[0.25]) & (prices <= price_quartiles[0.75])),
                'premium': (prices > price_quartiles[0.75])
            }
            
            segment_analysis = {}
            for segment, mask in price_segments.items():
                segment_analysis[segment] = {
                    'avg_rating': ratings[mask].mean(),
                    'avg_price': prices[mask].mean(),
                    'satisfaction_rate': (ratings[mask] >= 4).mean() * 100
                }
            
            return {
                'price_satisfaction_correlation': price_satisfaction_corr,
                'segment_analysis': segment_analysis,
                'sweet_spot': max(segment_analysis.keys(), 
                                key=lambda x: segment_analysis[x]['satisfaction_rate'])
            }
        
        return {'error': 'Price data not available'}
    
    def extract_styling_tips(self):
        """Extract styling and usage tips from reviews"""
        
        styling_keywords = {
            'occasions': ['office', 'party', 'casual', 'formal', 'wedding', 'date'],
            'combinations': ['with', 'pair', 'match', 'goes well', 'combine'],
            'styling': ['style', 'look', 'outfit', 'fashion', 'trendy'],
            'seasons': ['summer', 'winter', 'monsoon', 'spring']
        }
        
        styling_tips = {}
        comments = self.df['Comment'].fillna('').str.lower()
        
        for category, keywords in styling_keywords.items():
            tips = []
            for comment in comments:
                for keyword in keywords:
                    if keyword in str(comment):
                        # Extract sentence containing styling tip
                        sentences = str(comment).split('.')
                        for sentence in sentences:
                            if keyword in sentence and len(sentence.strip()) > 10:
                                tips.append(sentence.strip())
                                break
            
            styling_tips[category] = {
                'count': len(tips),
                'tips': tips[:5]  # Top 5 tips
            }
        
        return styling_tips
    
    def identify_decision_factors(self):
        """Identify what factors influence purchase decisions"""
        
        decision_factors = {
            'price_conscious': ['cheap', 'affordable', 'budget', 'value', 'price'],
            'quality_focused': ['quality', 'durable', 'lasting', 'premium'],
            'style_focused': ['style', 'fashion', 'trendy', 'design', 'look'],
            'comfort_focused': ['comfortable', 'soft', 'fit', 'ease'],
            'brand_loyal': ['brand', 'trust', 'reliable', 'known']
        }
        
        factor_analysis = {}
        comments = self.df['Comment'].fillna('').str.lower()
        
        for factor, keywords in decision_factors.items():
            pattern = '|'.join(keywords)
            matches = comments.str.contains(pattern, regex=True, na=False)
            
            factor_analysis[factor] = {
                'mentions': matches.sum(),
                'percentage': (matches.sum() / len(self.df)) * 100,
                'avg_rating_for_factor': pd.to_numeric(
                    self.df[matches]['Rating'], errors='coerce'
                ).mean()
            }
        
        dominant_factor = max(factor_analysis.keys(), 
                            key=lambda x: factor_analysis[x]['mentions'])
        
        return {
            'factor_breakdown': factor_analysis,
            'dominant_factor': dominant_factor,
            'buyer_persona': f"Most buyers are {dominant_factor.replace('_', ' ')}"
        }


def display_unique_insights(insights_data):
    """Display unique insights in Streamlit"""
    
    st.markdown("## 🎯 **Unique Insights** - What Myntra Doesn't Tell You")
    
    # Size & Fit Analysis
    st.markdown("### 👕 Size & Fit Intelligence")
    size_data = insights_data.get('size_fit_analysis', {})
    if size_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Size Recommendation", size_data.get('recommendation', 'N/A'))
        with col2:
            st.metric("Confidence Score", f"{size_data.get('confidence_score', 0):.1f}%")
        with col3:
            fit_issues = size_data.get('analysis', {})
            total_issues = sum([fit_issues.get(key, {}).get('count', 0) for key in ['too_small', 'too_large']])
            st.metric("Fit Issues Reported", total_issues)
    
    # Review Authenticity
    st.markdown("### 🔍 Review Authenticity Score")
    auth_data = insights_data.get('review_authenticity', {})
    if auth_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Authenticity Score", f"{auth_data.get('authenticity_score', 0):.0f}/100")
        with col2:
            st.metric("Suspicious Reviewers", auth_data.get('suspicious_reviewers', 0))
        with col3:
            st.metric("Duplicate Reviews", auth_data.get('duplicate_reviews', 0))
    
    # Hidden Problems
    st.markdown("### 🕵️ Hidden Issues in Positive Reviews")
    hidden_data = insights_data.get('hidden_issues', {})
    if hidden_data and hidden_data.get('hidden_issue_count', 0) > 0:
        st.warning(f"Found {hidden_data['hidden_issue_count']} hidden complaints in positive reviews!")
        
        with st.expander("View Hidden Issues"):
            for issue in hidden_data.get('sample_hidden_issues', [])[:3]:
                st.write(f"• {issue}")
    
    # Quality Concerns Heatmap
    st.markdown("### 🎨 Quality Concerns Breakdown")
    quality_data = insights_data.get('quality_durability', {})
    if quality_data:
        concern_names = list(quality_data.keys())
        concern_counts = [quality_data[key].get('mentions', 0) for key in concern_names]
        
        if any(concern_counts):
            fig = px.bar(
                x=concern_names, 
                y=concern_counts,
                title="Most Common Quality Concerns",
                color=concern_counts,
                color_continuous_scale="Reds"
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Purchase Decision Factors
    st.markdown("### 🧠 What Drives Purchase Decisions")
    decision_data = insights_data.get('purchase_decision_factors', {})
    if decision_data:
        st.info(f"**Buyer Persona**: {decision_data.get('buyer_persona', 'Analysis not available')}")
        
        factors = decision_data.get('factor_breakdown', {})
        if factors:
            factor_names = [name.replace('_', ' ').title() for name in factors.keys()]
            factor_percentages = [factors[key].get('percentage', 0) for key in factors.keys()]
            
            fig = px.pie(
                values=factor_percentages, 
                names=factor_names,
                title="Primary Purchase Motivations"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Styling Tips
    st.markdown("### 💡 Styling Tips from Real Users")
    styling_data = insights_data.get('styling_suggestions', {})
    if styling_data:
        for category, data in styling_data.items():
            if data.get('tips'):
                with st.expander(f"{category.title()} Suggestions ({data['count']} mentions)"):
                    for tip in data['tips'][:3]:
                        st.write(f"💡 {tip}")
