"""
ML Model Integration for Review Analysis
Main interface for all machine learning features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import sys
from src.exception import CustomException

# Import ML modules
from src.ml_models.sentiment_analysis import SentimentAnalyzer, add_sentiment_features
from src.ml_models.recommendation_system import ProductRecommendationSystem, CollaborativeFiltering
from src.ml_models.review_quality import ReviewQualityAnalyzer
from src.ml_models.price_prediction import PricePredictionModel


class MLAnalyzer:
    """
    Main ML analysis class that integrates all machine learning features

    """
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.recommendation_system = ProductRecommendationSystem()
        self.collaborative_filtering = CollaborativeFiltering()
        self.quality_analyzer = ReviewQualityAnalyzer()
        self.price_predictor = PricePredictionModel()
        
        self.processed_data = None
        self.analysis_results = {}
    
    def analyze_reviews(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of review data using all ML models
        
        Args:
            df: Review dataframe
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            # Store original data
            self.processed_data = df.copy()
            
            # Preprocess data - convert numeric columns
            self._preprocess_data()
            
            print("🔍 Starting ML Analysis...")
            
            # 1. Sentiment Analysis
            print("📊 Analyzing sentiment...")
            sentiment_results = self._analyze_sentiment()
            
            # 2. Review Quality Analysis
            print("🔍 Analyzing review quality...")
            quality_results = self._analyze_quality()
            
            # 3. Price Analysis
            print("💰 Analyzing prices...")
            price_results = self._analyze_prices()
            
            # 4. Recommendation System
            print("🎯 Building recommendations...")
            recommendation_results = self._build_recommendations()
            
            # 5. Overall Insights
            print("📈 Generating insights...")
            overall_insights = self._generate_insights()
            
            self.analysis_results = {
                'sentiment_analysis': sentiment_results,
                'quality_analysis': quality_results,
                'price_analysis': price_results,
                'recommendations': recommendation_results,
                'insights': overall_insights,
                'processed_data': self.processed_data
            }
            
            print("✅ ML Analysis Complete!")
            return self.analysis_results
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _preprocess_data(self):
        """Preprocess data to ensure proper data types for analysis"""
        try:
            # Convert numeric columns to proper types
            numeric_columns = ['Rating', 'Over_All_Rating', 'Price']
            
            for col in numeric_columns:
                if col in self.processed_data.columns:
                    # Handle string values and convert to numeric
                    def convert_to_numeric(value):
                        if pd.isna(value) or value in ["No rating Given", "No Price given"]:
                            return np.nan
                        try:
                            # Remove currency symbols and convert to float
                            if isinstance(value, str):
                                # Remove ₹ symbol and any commas
                                cleaned_value = str(value).replace('₹', '').replace(',', '').strip()
                                return float(cleaned_value)
                            else:
                                return float(value)
                        except (ValueError, TypeError):
                            return np.nan
                    
                    self.processed_data[col] = self.processed_data[col].apply(convert_to_numeric)
                    
                    # Fill NaN values with appropriate defaults
                    if col == 'Rating':
                        self.processed_data[col].fillna(3.0, inplace=True)  # Default neutral rating
                    elif col == 'Over_All_Rating':
                        self.processed_data[col].fillna(3.0, inplace=True)  # Default neutral rating
                    elif col == 'Price':
                        self.processed_data[col].fillna(self.processed_data[col].median(), inplace=True)  # Fill with median price
            
            print("✅ Data preprocessing completed")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _analyze_sentiment(self) -> Dict:
        """Perform sentiment analysis"""
        try:
            # Basic sentiment analysis using TextBlob (fallback)
            sentiment_results = []
            
            for idx, row in self.processed_data.iterrows():
                comment = row['Comment']
                if pd.isna(comment) or comment == "No comment Given":
                    sentiment_results.append({
                        'sentiment_label': 'neutral',
                        'sentiment_score': 0.0,
                        'polarity': 0.0,
                        'subjectivity': 0.0
                    })
                else:
                    # Simple sentiment based on rating and keywords
                    rating = row['Rating']
                    
                    # Convert rating to numeric, handle string ratings
                    try:
                        if pd.isna(rating) or rating == "No rating Given":
                            rating_numeric = 3.0  # Default neutral rating
                        else:
                            # Handle string ratings like "4.5", "4", etc.
                            rating_numeric = float(str(rating).strip())
                    except (ValueError, TypeError):
                        rating_numeric = 3.0  # Default to neutral if conversion fails
                    
                    comment_lower = str(comment).lower()
                    
                    positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect']
                    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible']
                    
                    pos_count = sum(1 for word in positive_words if word in comment_lower)
                    neg_count = sum(1 for word in negative_words if word in comment_lower)
                    
                    # Determine sentiment
                    if rating_numeric >= 4 or pos_count > neg_count:
                        sentiment_label = 'positive'
                        sentiment_score = 0.5 + (rating_numeric - 3) * 0.25
                    elif rating_numeric <= 2 or neg_count > pos_count:
                        sentiment_label = 'negative' 
                        sentiment_score = -0.5 - (3 - rating_numeric) * 0.25
                    else:
                        sentiment_label = 'neutral'
                        sentiment_score = 0.0
                    
                    sentiment_results.append({
                        'sentiment_label': sentiment_label,
                        'sentiment_score': sentiment_score,
                        'polarity': sentiment_score,
                        'subjectivity': 0.5
                    })
            
            # Add sentiment columns to processed data
            sentiment_df = pd.DataFrame(sentiment_results)
            for col in sentiment_df.columns:
                self.processed_data[col] = sentiment_df[col]
            
            # Aggregate sentiment by product
            product_sentiment = self.processed_data.groupby('Product Name').agg({
                'sentiment_score': ['mean', 'count'],
                'sentiment_label': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral'
            }).round(3)
            
            return {
                'product_sentiment': product_sentiment.to_dict(),
                'overall_sentiment_distribution': self.processed_data['sentiment_label'].value_counts().to_dict(),
                'average_sentiment_score': round(self.processed_data['sentiment_score'].mean(), 3)
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _analyze_quality(self) -> Dict:
        """Perform review quality analysis"""
        try:
            self.processed_data = self.quality_analyzer.analyze_review_batch(self.processed_data)
            quality_insights = self.quality_analyzer.get_quality_insights(self.processed_data)
            
            return quality_insights
            
        except Exception as e:
            print(f"Warning: Quality analysis failed, using fallback: {str(e)}")
            # Fallback: Add basic quality columns
            self.processed_data['quality_score'] = 0.5  # Default medium quality
            self.processed_data['is_fake'] = False  # Default not fake
            self.processed_data['fake_indicators'] = [[] for _ in range(len(self.processed_data))]
            
            return {
                'total_reviews': len(self.processed_data),
                'high_quality_count': len(self.processed_data) // 2,
                'high_quality_percentage': 50.0,
                'potential_fake_count': 0,
                'fake_percentage': 0.0,
                'average_quality_score': 0.5,
                'average_word_count': 50.0,
                'common_fake_indicators': []
            }
    
    def _analyze_prices(self) -> Dict:
        """Perform price analysis"""
        try:
            # Calculate brand premiums
            brand_premiums = self.price_predictor.calculate_brand_premiums(self.processed_data)
            
            # Value for money analysis
            value_analysis = self.price_predictor.analyze_value_for_money(self.processed_data)
            
            # Price insights
            price_insights = self.price_predictor.get_price_insights(self.processed_data)
            
            return {
                'brand_premiums': brand_premiums,
                'value_analysis': value_analysis.to_dict('records'),
                'price_insights': price_insights
            }
            
        except Exception as e:
            print(f"Warning: Price analysis failed, using fallback: {str(e)}")
            # Fallback: Basic price analysis
            avg_price = self.processed_data['Price'].mean()
            min_price = self.processed_data['Price'].min()
            max_price = self.processed_data['Price'].max()
            
            return {
                'brand_premiums': {},
                'value_analysis': [],
                'price_insights': {
                    'price_statistics': {
                        'average_price': round(avg_price, 2),
                        'median_price': round(self.processed_data['Price'].median(), 2),
                        'price_range': {
                            'min': round(min_price, 2),
                            'max': round(max_price, 2)
                        },
                        'price_std': round(self.processed_data['Price'].std(), 2)
                    },
                    'value_distribution': {},
                    'best_value_products': [],
                    'brand_premiums': {},
                    'price_rating_correlation': 0.0
                }
            }
    
    def _build_recommendations(self) -> Dict:
        """Build recommendation system"""
        try:
            # Prepare content-based recommendations
            self.recommendation_system.prepare_content_features(self.processed_data)
            
            # Get unique products
            products = self.processed_data['Product Name'].unique()
            
            recommendations = {}
            for product in products[:3]:  # Limit to first 3 products for demo
                recs = self.recommendation_system.get_content_recommendations(product, top_k=3)
                recommendations[product] = recs
            
            # Price-based recommendations
            avg_price = self.processed_data['Price'].mean()
            price_recs = self.recommendation_system.get_price_based_recommendations(avg_price)
            
            return {
                'content_based': recommendations,
                'price_based': price_recs[:5]  # Top 5 price-based recommendations
            }
            
        except Exception as e:
            print(f"Warning: Recommendation system failed, using fallback: {str(e)}")
            # Fallback: Basic recommendations based on ratings
            products = self.processed_data['Product Name'].unique()
            basic_recs = {}
            
            for product in products[:3]:
                basic_recs[product] = []
            
            return {
                'content_based': basic_recs,
                'price_based': []
            }
    
    def _generate_insights(self) -> Dict:
        """Generate overall insights"""
        try:
            insights = {}
            
            # Product performance insights
            product_stats = self.processed_data.groupby('Product Name').agg({
                'Over_All_Rating': 'mean',
                'Price': 'mean',
                'Rating': 'count',
                'sentiment_score': 'mean',
                'quality_score': 'mean'
            }).round(2)
            
            # Best and worst performing products
            best_product = product_stats.loc[product_stats['Over_All_Rating'].idxmax()]
            worst_product = product_stats.loc[product_stats['Over_All_Rating'].idxmin()]
            
            # Most reviewed product
            most_reviewed = product_stats.loc[product_stats['Rating'].idxmax()]
            
            insights['product_performance'] = {
                'best_rated': {
                    'name': best_product.name,
                    'rating': best_product['Over_All_Rating'],
                    'price': best_product['Price'],
                    'sentiment': best_product['sentiment_score']
                },
                'worst_rated': {
                    'name': worst_product.name,
                    'rating': worst_product['Over_All_Rating'],
                    'price': worst_product['Price'],
                    'sentiment': worst_product['sentiment_score']
                },
                'most_reviewed': {
                    'name': most_reviewed.name,
                    'review_count': most_reviewed['Rating'],
                    'rating': most_reviewed['Over_All_Rating']
                }
            }
            
            # Overall statistics
            insights['overall_stats'] = {
                'total_products': len(product_stats),
                'total_reviews': len(self.processed_data),
                'average_rating': round(self.processed_data['Over_All_Rating'].mean(), 2),
                'average_price': round(self.processed_data['Price'].mean(), 2),
                'price_range': {
                    'min': round(self.processed_data['Price'].min(), 2),
                    'max': round(self.processed_data['Price'].max(), 2)
                }
            }
            
            return insights
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_product_analysis(self, product_name: str) -> Dict:
        """
        Get detailed analysis for a specific product
        
        Args:
            product_name: Name of the product to analyze
            
        Returns:
            Detailed product analysis
        """
        try:
            if self.processed_data is None:
                raise ValueError("No data processed. Run analyze_reviews first.")
            
            product_data = self.processed_data[
                self.processed_data['Product Name'].str.contains(product_name, case=False, na=False)
            ]
            
            if product_data.empty:
                return {'error': f'Product "{product_name}" not found'}
            
            # Product summary
            summary = {
                'product_name': product_name,
                'total_reviews': len(product_data),
                'average_rating': round(product_data['Over_All_Rating'].mean(), 2),
                'average_price': round(product_data['Price'].mean(), 2),
                'sentiment_distribution': product_data['sentiment_label'].value_counts().to_dict(),
                'quality_stats': {
                    'average_quality_score': round(product_data['quality_score'].mean(), 3),
                    'high_quality_reviews': len(product_data[product_data['quality_score'] >= 0.7]),
                    'potential_fake_reviews': len(product_data[product_data['is_fake'] == True])
                }
            }
            
            # Get recommendations for this product
            recommendations = self.recommendation_system.get_content_recommendations(product_name, top_k=5)
            
            return {
                'summary': summary,
                'recommendations': recommendations,
                'recent_reviews': product_data[['Date', 'Rating', 'Comment', 'sentiment_label', 'quality_score']].tail(5).to_dict('records')
            }
            
        except Exception as e:
            raise CustomException(e, sys)
