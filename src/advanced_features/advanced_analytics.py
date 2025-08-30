"""
Advanced Analytics Engine
Unique features that make this project stand out from basic Myntra ratings
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from collections import Counter
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class AdvancedAnalyticsEngine:
    """
    Revolutionary analytics that go beyond what any e-commerce platform offers
    """
    
    def safe_numpy_operation(self, operation, data, default=0):
        """Safely perform numpy operations with proper error handling"""
        try:
            # Ensure data is numeric and remove any NaN values
            if isinstance(data, list):
                clean_data = [float(x) for x in data if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit()]
            else:
                clean_data = data.dropna() if hasattr(data, 'dropna') else data
            
            if len(clean_data) == 0:
                return default
                
            # Convert to numpy array with proper dtype
            np_data = np.array(clean_data, dtype=np.float64)
            
            if operation == 'mean':
                return np.mean(np_data)
            elif operation == 'std':
                return np.std(np_data)
            elif operation == 'max':
                return np.max(np_data)
            elif operation == 'min':
                return np.min(np_data)
            else:
                return default
        except Exception as e:
            print(f"Warning: Numpy operation {operation} failed: {e}")
            return default
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analytics_results = {}
        
    def generate_revolutionary_insights(self):
        """Generate insights that are truly unique to this platform"""
        
        return {
            'emotional_journey_mapping': self.map_customer_emotional_journey(),
            'review_evolution_analysis': self.analyze_review_evolution_patterns(),
            'micro_sentiment_analysis': self.perform_micro_sentiment_analysis(),
            'psychological_buyer_profiling': self.create_psychological_buyer_profiles(),
            'competitor_intelligence': self.extract_competitor_intelligence(),
            'trend_prediction': self.predict_future_trends(),
            'quality_degradation_detection': self.detect_quality_degradation(),
            'social_influence_analysis': self.analyze_social_influence_patterns(),
            'return_risk_prediction': self.predict_return_risk(),
            'personalized_recommendations': self.generate_personalized_insights()
        }
    
    def map_customer_emotional_journey(self):
        """Map the emotional journey of customers from purchase to review"""
        
        # Extract emotional indicators from reviews
        emotion_patterns = {
            'excitement': ['excited', 'amazing', 'love', 'fantastic', 'awesome', 'perfect'],
            'satisfaction': ['satisfied', 'good', 'nice', 'decent', 'okay', 'fine'],
            'disappointment': ['disappointed', 'expected better', 'not as shown', 'misleading'],
            'frustration': ['frustrated', 'angry', 'terrible', 'worst', 'horrible', 'awful'],
            'surprise': ['surprised', 'unexpected', 'wow', 'didn\'t expect', 'shocking'],
            'regret': ['regret', 'waste', 'should have', 'mistake', 'wrong choice']
        }
        
        emotional_data = {}
        comments = self.df['Comment'].fillna('').str.lower()
        ratings = pd.to_numeric(self.df['Rating'], errors='coerce')
        
        for emotion, keywords in emotion_patterns.items():
            pattern = '|'.join(keywords)
            matches = comments.str.contains(pattern, regex=True, na=False)
            
            emotional_data[emotion] = {
                'count': matches.sum(),
                'avg_rating': ratings[matches].mean() if matches.any() else 0,
                'percentage': (matches.sum() / len(self.df)) * 100,
                'sample_quotes': comments[matches].head(3).tolist()
            }
        
        # Create emotional journey map
        journey_stages = {
            'pre_purchase_excitement': emotional_data['excitement']['count'],
            'post_purchase_satisfaction': emotional_data['satisfaction']['count'],
            'experience_disappointment': emotional_data['disappointment']['count'],
            'final_regret': emotional_data['regret']['count']
        }
        
        return {
            'emotional_breakdown': emotional_data,
            'journey_map': journey_stages,
            'dominant_emotion': max(emotional_data.keys(), key=lambda x: emotional_data[x]['count']),
            'emotional_volatility': self.safe_numpy_operation('std', [data['count'] for data in emotional_data.values()])
        }
    
    def analyze_review_evolution_patterns(self):
        """Analyze how reviews change over time for same products"""
        
        # Simulate review timeline analysis (in real scenario, you'd use timestamps)
        product_evolution = {}
        
        for product in self.df['Product Name'].unique():
            product_reviews = self.df[self.df['Product Name'] == product]
            
            if len(product_reviews) > 3:  # Only analyze products with multiple reviews
                ratings = pd.to_numeric(product_reviews['Rating'], errors='coerce')
                
                # Analyze rating trend (assuming reviews are chronologically ordered)
                trend_direction = 'stable'
                if len(ratings) > 1:
                    first_half_avg = ratings[:len(ratings)//2].mean()
                    second_half_avg = ratings[len(ratings)//2:].mean()
                    
                    if second_half_avg > first_half_avg + 0.5:
                        trend_direction = 'improving'
                    elif second_half_avg < first_half_avg - 0.5:
                        trend_direction = 'declining'
                
                # Analyze comment sentiment evolution
                sentiments = []
                for comment in product_reviews['Comment'].fillna(''):
                    try:
                        sentiment = TextBlob(str(comment)).sentiment.polarity
                        sentiments.append(sentiment)
                    except:
                        sentiments.append(0)
                
                product_evolution[product] = {
                    'rating_trend': trend_direction,
                    'rating_volatility': float(ratings.std()) if len(ratings) > 0 else 0,
                    'sentiment_trend': self.safe_numpy_operation('mean', sentiments),
                    'review_count': len(product_reviews),
                    'quality_consistency': 'high' if (len(ratings) > 0 and float(ratings.std()) < 0.5) else 'low'
                }
        
        return {
            'product_evolution': product_evolution,
            'improving_products': [p for p, data in product_evolution.items() 
                                 if data['rating_trend'] == 'improving'],
            'declining_products': [p for p, data in product_evolution.items() 
                                 if data['rating_trend'] == 'declining'],
            'most_consistent': min(product_evolution.keys(), 
                                 key=lambda x: product_evolution[x]['rating_volatility'])
            if product_evolution else None
        }
    
    def perform_micro_sentiment_analysis(self):
        """Detailed sentiment analysis on specific product aspects"""
        
        aspects = {
            'fabric_quality': ['fabric', 'material', 'cloth', 'texture'],
            'fit_comfort': ['fit', 'comfort', 'comfortable', 'size'],
            'design_style': ['design', 'style', 'look', 'appearance'],
            'value_money': ['value', 'price', 'worth', 'money'],
            'delivery_service': ['delivery', 'shipping', 'packaging', 'service'],
            'durability': ['durable', 'lasting', 'quality', 'wear']
        }
        
        aspect_sentiments = {}
        comments = self.df['Comment'].fillna('')
        
        for aspect, keywords in aspects.items():
            aspect_comments = []
            
            for comment in comments:
                comment_lower = str(comment).lower()
                if any(keyword in comment_lower for keyword in keywords):
                    # Extract sentences containing aspect keywords
                    sentences = str(comment).split('.')
                    for sentence in sentences:
                        if any(keyword in sentence.lower() for keyword in keywords):
                            aspect_comments.append(sentence.strip())
                            break
            
            if aspect_comments:
                # Calculate sentiment for aspect-specific comments
                sentiments = []
                for comment in aspect_comments:
                    try:
                        sentiment = TextBlob(comment).sentiment.polarity
                        sentiments.append(sentiment)
                    except:
                        sentiments.append(0)
                
                aspect_sentiments[aspect] = {
                    'avg_sentiment': self.safe_numpy_operation('mean', sentiments),
                    'sentiment_range': self.safe_numpy_operation('max', sentiments) - self.safe_numpy_operation('min', sentiments),
                    'mention_count': len(aspect_comments),
                    'sentiment_category': self._categorize_sentiment(self.safe_numpy_operation('mean', sentiments)),
                    'sample_comments': aspect_comments[:3]
                }
        
        return {
            'aspect_sentiments': aspect_sentiments,
            'best_aspect': max(aspect_sentiments.keys(), 
                             key=lambda x: aspect_sentiments[x]['avg_sentiment'])
            if aspect_sentiments else None,
            'worst_aspect': min(aspect_sentiments.keys(), 
                              key=lambda x: aspect_sentiments[x]['avg_sentiment'])
            if aspect_sentiments else None
        }
    
    def create_psychological_buyer_profiles(self):
        """Create psychological profiles of different buyer types"""
        
        # Define psychological indicators in reviews
        psychological_traits = {
            'perfectionist': ['perfect', 'exactly', 'precise', 'flawless', 'ideal'],
            'price_sensitive': ['cheap', 'expensive', 'affordable', 'budget', 'deal'],
            'trend_follower': ['trendy', 'fashionable', 'latest', 'trending', 'modern'],
            'quality_seeker': ['quality', 'premium', 'durable', 'superior', 'excellent'],
            'comfort_prioritizer': ['comfortable', 'cozy', 'soft', 'easy', 'relaxed'],
            'brand_conscious': ['brand', 'branded', 'reputation', 'trust', 'known']
        }
        
        buyer_profiles = {}
        comments = self.df['Comment'].fillna('').str.lower()
        ratings = pd.to_numeric(self.df['Rating'], errors='coerce')
        
        for trait, keywords in psychological_traits.items():
            pattern = '|'.join(keywords)
            matches = comments.str.contains(pattern, regex=True, na=False)
            
            if matches.any():
                buyer_profiles[trait] = {
                    'count': matches.sum(),
                    'avg_rating': ratings[matches].mean(),
                    'satisfaction_rate': (ratings[matches] >= 4).mean() * 100,
                    'typical_concerns': self._extract_concerns(comments[matches]),
                    'buying_motivation': self._analyze_motivation(comments[matches])
                }
        
        # Identify dominant buyer type
        dominant_type = max(buyer_profiles.keys(), 
                          key=lambda x: buyer_profiles[x]['count']) if buyer_profiles else None
        
        return {
            'buyer_profiles': buyer_profiles,
            'dominant_buyer_type': dominant_type,
            'buyer_diversity_score': len(buyer_profiles),
            'recommendations_by_type': self._generate_type_recommendations(buyer_profiles)
        }
    
    def extract_competitor_intelligence(self):
        """Extract mentions of competitors and comparative insights"""
        
        # Common fashion brand names that might be mentioned
        competitor_brands = [
            'zara', 'h&m', 'uniqlo', 'forever 21', 'nike', 'adidas', 'puma',
            'vero moda', 'only', 'mango', 'bershka', 'pull and bear'
        ]
        
        competitor_mentions = {}
        comments = self.df['Comment'].fillna('').str.lower()
        
        for brand in competitor_brands:
            mentions = comments.str.contains(brand, regex=False, na=False)
            
            if mentions.any():
                # Extract context around competitor mentions
                comparative_comments = []
                for comment in comments[mentions]:
                    if any(comp_word in str(comment) for comp_word in ['better', 'worse', 'similar', 'like', 'compared']):
                        comparative_comments.append(comment)
                
                competitor_mentions[brand] = {
                    'mention_count': mentions.sum(),
                    'comparative_context': comparative_comments[:3],
                    'sentiment_when_mentioned': self._analyze_competitor_sentiment(comments[mentions])
                }
        
        return {
            'competitor_analysis': competitor_mentions,
            'most_compared_brand': max(competitor_mentions.keys(), 
                                     key=lambda x: competitor_mentions[x]['mention_count'])
            if competitor_mentions else None,
            'competitive_advantage': self._identify_competitive_advantages(competitor_mentions)
        }
    
    def predict_future_trends(self):
        """Predict future trends based on review patterns"""
        
        # Identify emerging keywords and themes
        all_comments = ' '.join(self.df['Comment'].fillna('').astype(str))
        
        # Extract trending keywords
        words = re.findall(r'\b\w+\b', all_comments.lower())
        word_freq = Counter(words)
        
        # Filter for fashion/product related terms
        fashion_terms = [word for word, count in word_freq.most_common(100) 
                        if len(word) > 3 and word not in ['good', 'nice', 'very', 'really']]
        
        # Analyze color trends
        colors = ['black', 'white', 'blue', 'red', 'green', 'yellow', 'pink', 
                 'purple', 'orange', 'brown', 'grey', 'navy', 'maroon']
        color_mentions = {color: all_comments.lower().count(color) for color in colors}
        
        # Analyze style trends
        styles = ['casual', 'formal', 'party', 'office', 'summer', 'winter', 
                 'trendy', 'classic', 'modern', 'vintage']
        style_mentions = {style: all_comments.lower().count(style) for style in styles}
        
        return {
            'trending_keywords': fashion_terms[:10],
            'color_trends': dict(sorted(color_mentions.items(), key=lambda x: x[1], reverse=True)[:5]),
            'style_trends': dict(sorted(style_mentions.items(), key=lambda x: x[1], reverse=True)[:5]),
            'emerging_themes': self._identify_emerging_themes(all_comments),
            'seasonal_predictions': self._predict_seasonal_trends(all_comments)
        }
    
    def detect_quality_degradation(self):
        """Detect if product quality is degrading over time"""
        
        # Simulate temporal analysis (in real scenario, use actual timestamps)
        quality_indicators = ['quality', 'durable', 'lasting', 'premium', 'excellent']
        degradation_indicators = ['cheap', 'poor', 'bad', 'terrible', 'worst', 'decline']
        
        comments = self.df['Comment'].fillna('').str.lower()
        
        quality_mentions = sum(comments.str.count(indicator) for indicator in quality_indicators)
        degradation_mentions = sum(comments.str.count(indicator) for indicator in degradation_indicators)
        
        # Calculate quality degradation score
        total_mentions = quality_mentions + degradation_mentions
        degradation_score = (degradation_mentions / total_mentions * 100) if total_mentions > 0 else 0
        
        # Analyze quality consistency across products
        product_quality = {}
        for product in self.df['Product Name'].unique():
            product_comments = self.df[self.df['Product Name'] == product]['Comment'].fillna('').str.lower()
            
            product_quality_mentions = sum(product_comments.str.count(indicator) for indicator in quality_indicators)
            product_degradation_mentions = sum(product_comments.str.count(indicator) for indicator in degradation_indicators)
            
            product_total = product_quality_mentions + product_degradation_mentions
            product_degradation_score = (product_degradation_mentions / product_total * 100) if product_total > 0 else 0
            
            product_quality[product] = {
                'degradation_score': product_degradation_score,
                'quality_mentions': product_quality_mentions,
                'concern_level': 'High' if product_degradation_score > 30 else 'Low'
            }
        
        return {
            'overall_degradation_score': degradation_score,
            'quality_status': 'Concerning' if degradation_score > 25 else 'Stable',
            'product_quality_map': product_quality,
            'most_concerning_product': max(product_quality.keys(), 
                                         key=lambda x: product_quality[x]['degradation_score'])
            if product_quality else None
        }
    
    def analyze_social_influence_patterns(self):
        """Analyze how social factors influence purchasing decisions"""
        
        social_indicators = {
            'influencer_impact': ['influencer', 'blogger', 'youtube', 'instagram', 'social media'],
            'friend_recommendation': ['friend', 'recommended', 'suggested', 'told me'],
            'celebrity_association': ['celebrity', 'star', 'actor', 'actress', 'famous'],
            'trend_following': ['trending', 'viral', 'everyone wearing', 'popular'],
            'peer_pressure': ['everyone has', 'all my friends', 'must have', 'seen others']
        }
        
        social_analysis = {}
        comments = self.df['Comment'].fillna('').str.lower()
        ratings = pd.to_numeric(self.df['Rating'], errors='coerce')
        
        for influence_type, keywords in social_indicators.items():
            pattern = '|'.join(keywords)
            matches = comments.str.contains(pattern, regex=True, na=False)
            
            if matches.any():
                social_analysis[influence_type] = {
                    'mentions': matches.sum(),
                    'avg_satisfaction': ratings[matches].mean(),
                    'influence_strength': (matches.sum() / len(self.df)) * 100,
                    'typical_comments': comments[matches].head(3).tolist()
                }
        
        return {
            'social_influence_breakdown': social_analysis,
            'strongest_influence': max(social_analysis.keys(), 
                                     key=lambda x: social_analysis[x]['mentions'])
            if social_analysis else None,
            'social_dependency_score': sum(data['influence_strength'] for data in social_analysis.values())
        }
    
    def predict_return_risk(self):
        """Predict likelihood of product returns based on review patterns"""
        
        return_indicators = {
            'size_issues': ['wrong size', 'doesn\'t fit', 'too small', 'too large', 'size issue'],
            'quality_issues': ['poor quality', 'cheap material', 'broke', 'damaged', 'defective'],
            'expectation_mismatch': ['not as expected', 'different from photo', 'misleading', 'not what I ordered'],
            'color_issues': ['color different', 'not the same color', 'faded', 'color not matching'],
            'delivery_issues': ['damaged packaging', 'wrong item', 'delayed delivery', 'poor packaging']
        }
        
        risk_analysis = {}
        comments = self.df['Comment'].fillna('').str.lower()
        
        total_risk_score = 0
        for risk_type, indicators in return_indicators.items():
            pattern = '|'.join(indicators)
            matches = comments.str.contains(pattern, regex=True, na=False)
            
            risk_score = (matches.sum() / len(self.df)) * 100
            total_risk_score += risk_score
            
            risk_analysis[risk_type] = {
                'risk_score': risk_score,
                'affected_reviews': matches.sum(),
                'risk_level': 'High' if risk_score > 10 else 'Low'
            }
        
        # Product-wise risk assessment
        product_risk = {}
        for product in self.df['Product Name'].unique():
            product_comments = self.df[self.df['Product Name'] == product]['Comment'].fillna('').str.lower()
            
            product_risk_score = 0
            for indicators in return_indicators.values():
                pattern = '|'.join(indicators)
                matches = product_comments.str.contains(pattern, regex=True, na=False)
                product_risk_score += (matches.sum() / len(product_comments)) * 100 if len(product_comments) > 0 else 0
            
            product_risk[product] = {
                'total_risk_score': product_risk_score,
                'risk_category': 'High Risk' if product_risk_score > 20 else 'Low Risk'
            }
        
        return {
            'overall_return_risk': total_risk_score,
            'risk_breakdown': risk_analysis,
            'highest_risk_factor': max(risk_analysis.keys(), 
                                     key=lambda x: risk_analysis[x]['risk_score'])
            if risk_analysis else None,
            'product_risk_assessment': product_risk,
            'riskiest_product': max(product_risk.keys(), 
                                  key=lambda x: product_risk[x]['total_risk_score'])
            if product_risk else None
        }
    
    def generate_personalized_insights(self):
        """Generate personalized recommendations based on review analysis"""
        
        # Analyze review patterns to create buyer personas
        personas = {
            'budget_conscious': self._analyze_budget_conscious_reviews(),
            'quality_seeker': self._analyze_quality_focused_reviews(),
            'fashion_forward': self._analyze_trend_focused_reviews(),
            'practical_buyer': self._analyze_practical_reviews()
        }
        
        return {
            'persona_insights': personas,
            'personalized_recommendations': self._generate_persona_recommendations(personas),
            'shopping_tips': self._generate_shopping_tips(),
            'warning_signs': self._identify_warning_signs()
        }
    
    # Helper methods
    def _categorize_sentiment(self, sentiment_score):
        if sentiment_score > 0.1:
            return 'Positive'
        elif sentiment_score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'
    
    def _extract_concerns(self, comments):
        concern_words = ['concern', 'issue', 'problem', 'worry', 'disappointed']
        concerns = []
        for comment in comments:
            for word in concern_words:
                if word in str(comment):
                    concerns.append(str(comment)[:100] + '...')
                    break
        return concerns[:3]
    
    def _analyze_motivation(self, comments):
        motivations = []
        for comment in comments:
            if any(word in str(comment) for word in ['because', 'since', 'as', 'for']):
                motivations.append(str(comment)[:100] + '...')
        return motivations[:3]
    
    def _generate_type_recommendations(self, buyer_profiles):
        recommendations = {}
        for buyer_type, data in buyer_profiles.items():
            if data['satisfaction_rate'] > 80:
                recommendations[buyer_type] = f"Products highly satisfy {buyer_type} buyers"
            else:
                recommendations[buyer_type] = f"Consider {buyer_type} concerns before purchase"
        return recommendations
    
    def _analyze_competitor_sentiment(self, comments):
        sentiments = []
        for comment in comments:
            try:
                sentiment = TextBlob(str(comment)).sentiment.polarity
                sentiments.append(sentiment)
            except:
                sentiments.append(0)
        return self.safe_numpy_operation('mean', sentiments)
    
    def _identify_competitive_advantages(self, competitor_mentions):
        if not competitor_mentions:
            return []
        
        advantages = []
        for brand, data in competitor_mentions.items():
            if data['sentiment_when_mentioned'] > 0:
                advantages.append(f"Positively compared to {brand}")
            else:
                advantages.append(f"Room for improvement vs {brand}")
        
        return advantages
    
    def _identify_emerging_themes(self, text):
        # Simple theme identification based on keyword clustering
        themes = []
        if 'sustainable' in text or 'eco' in text:
            themes.append('Sustainability Focus')
        if 'comfort' in text and 'work' in text:
            themes.append('Work-from-Home Comfort')
        if 'versatile' in text or 'multi' in text:
            themes.append('Versatility Trend')
        
        return themes
    
    def _predict_seasonal_trends(self, text):
        seasonal_predictions = {}
        if text.count('summer') > text.count('winter'):
            seasonal_predictions['dominant_season'] = 'Summer wear trending'
        else:
            seasonal_predictions['dominant_season'] = 'Winter wear trending'
        
        return seasonal_predictions
    
    def _analyze_budget_conscious_reviews(self):
        budget_keywords = ['cheap', 'affordable', 'budget', 'value', 'deal']
        comments = self.df['Comment'].fillna('').str.lower()
        
        budget_pattern = '|'.join(budget_keywords)
        budget_reviews = comments.str.contains(budget_pattern, regex=True, na=False)
        
        return {
            'count': budget_reviews.sum(),
            'avg_satisfaction': pd.to_numeric(self.df[budget_reviews]['Rating'], errors='coerce').mean(),
            'key_concerns': ['price vs quality', 'value for money', 'durability']
        }
    
    def _analyze_quality_focused_reviews(self):
        quality_keywords = ['quality', 'premium', 'durable', 'excellent', 'superior']
        comments = self.df['Comment'].fillna('').str.lower()
        
        quality_pattern = '|'.join(quality_keywords)
        quality_reviews = comments.str.contains(quality_pattern, regex=True, na=False)
        
        return {
            'count': quality_reviews.sum(),
            'avg_satisfaction': pd.to_numeric(self.df[quality_reviews]['Rating'], errors='coerce').mean(),
            'key_concerns': ['material quality', 'craftsmanship', 'longevity']
        }
    
    def _analyze_trend_focused_reviews(self):
        trend_keywords = ['trendy', 'fashionable', 'style', 'modern', 'latest']
        comments = self.df['Comment'].fillna('').str.lower()
        
        trend_pattern = '|'.join(trend_keywords)
        trend_reviews = comments.str.contains(trend_pattern, regex=True, na=False)
        
        return {
            'count': trend_reviews.sum(),
            'avg_satisfaction': pd.to_numeric(self.df[trend_reviews]['Rating'], errors='coerce').mean(),
            'key_concerns': ['design relevance', 'style longevity', 'trend accuracy']
        }
    
    def _analyze_practical_reviews(self):
        practical_keywords = ['comfortable', 'practical', 'daily', 'office', 'work']
        comments = self.df['Comment'].fillna('').str.lower()
        
        practical_pattern = '|'.join(practical_keywords)
        practical_reviews = comments.str.contains(practical_pattern, regex=True, na=False)
        
        return {
            'count': practical_reviews.sum(),
            'avg_satisfaction': pd.to_numeric(self.df[practical_reviews]['Rating'], errors='coerce').mean(),
            'key_concerns': ['comfort', 'functionality', 'versatility']
        }
    
    def _generate_persona_recommendations(self, personas):
        recommendations = {}
        for persona, data in personas.items():
            if data['avg_satisfaction'] > 4:
                recommendations[persona] = f"Highly recommended for {persona} buyers"
            else:
                recommendations[persona] = f"Exercise caution - {persona} buyers show mixed satisfaction"
        
        return recommendations
    
    def _generate_shopping_tips(self):
        return [
            "Read reviews focusing on your primary concern (price/quality/style)",
            "Check for mentions of sizing issues before ordering",
            "Look for recent reviews to gauge current product quality",
            "Pay attention to reviewer profiles similar to your preferences"
        ]
    
    def _identify_warning_signs(self):
        return [
            "Multiple mentions of quality degradation",
            "Consistent sizing issues across reviews",
            "High number of comparison mentions with competitors",
            "Sudden shift in review sentiment patterns"
        ]
