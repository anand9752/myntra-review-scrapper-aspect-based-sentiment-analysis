"""
Advanced ABSA Analyzer using HuggingFace Transformers
This module provides enhanced sentiment analysis using pre-trained BERT models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re
import os
import sys
from src.exception import CustomException
from src.absa.absa_analyzer import ABSAAnalyzer

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available. Using VADER sentiment analysis only.")


class AdvancedABSAAnalyzer(ABSAAnalyzer):
    """
    Advanced ABSA analyzer using HuggingFace transformers for better sentiment analysis.
    Falls back to VADER if transformers are not available.
    """
    
    def __init__(self, use_transformers: bool = True):
        """
        Initialize the advanced ABSA analyzer.
        
        Args:
            use_transformers (bool): Whether to use transformer models for sentiment analysis
        """
        super().__init__()
        
        self.use_transformers = use_transformers and TRANSFORMERS_AVAILABLE
        self.sentiment_pipeline = None
        
        if self.use_transformers:
            try:
                # Use a simpler, more reliable sentiment model
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name
                )
                print("✅ Loaded advanced transformer model for sentiment analysis")
            except Exception as e:
                try:
                    # Fallback to a more basic but reliable model
                    self.sentiment_pipeline = pipeline("sentiment-analysis")
                    print("✅ Loaded basic transformer model for sentiment analysis")
                except Exception as e2:
                    print(f"⚠️ Failed to load any transformer model: {e2}")
                    print("Falling back to VADER sentiment analysis")
                    self.use_transformers = False
    
    def classify_sentiment_advanced(self, text: str) -> Dict[str, float]:
        """
        Advanced sentiment classification using transformers or VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Enhanced sentiment scores and classification
        """
        try:
            if self.use_transformers and self.sentiment_pipeline:
                return self._classify_with_transformers(text)
            else:
                return self.classify_sentiment(text)
                
        except Exception as e:
            # Fallback to VADER if transformer fails
            return self.classify_sentiment(text)
    
    def _classify_with_transformers(self, text: str) -> Dict[str, float]:
        """
        Classify sentiment using transformer model.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores and classification
        """
        try:
            # Truncate text if too long
            max_length = 500
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get predictions
            result = self.sentiment_pipeline(text)
            
            # Handle different output formats
            if isinstance(result, list):
                result = result[0]
            
            label = result['label'].upper()
            score = result['score']
            
            # Initialize scores
            positive_score = 0.0
            negative_score = 0.0
            neutral_score = 0.0
            
            # Map labels to standard format
            if 'POSITIVE' in label or label in ['LABEL_2']:
                positive_score = score
                negative_score = 1 - score
            elif 'NEGATIVE' in label or label in ['LABEL_0']:
                negative_score = score
                positive_score = 1 - score
            else:  # NEUTRAL or LABEL_1
                neutral_score = score
                positive_score = (1 - score) / 2
                negative_score = (1 - score) / 2
            
            # Calculate compound score
            compound = positive_score - negative_score
            
            # Determine sentiment label with better thresholds
            if compound >= 0.2:
                sentiment_label = 'Positive'
            elif compound <= -0.2:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
            
            return {
                'positive': positive_score,
                'negative': negative_score,
                'neutral': neutral_score,
                'compound': compound,
                'sentiment_label': sentiment_label,
                'model_used': 'transformer',
                'confidence': score
            }
            
        except Exception as e:
            print(f"Error in transformer classification: {e}")
            # Fallback to VADER
            result = self.classify_sentiment(text)
            result['model_used'] = 'vader_fallback'
            return result
    
    def analyze_reviews_advanced(self, reviews_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform advanced ABSA on a DataFrame of reviews using transformers.
        
        Args:
            reviews_data (pd.DataFrame): DataFrame containing review data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Enhanced reviews and detailed ABSA results
        """
        try:
            if 'Comment' not in reviews_data.columns:
                raise ValueError("DataFrame must contain 'Comment' column")
            
            # Preprocess reviews
            reviews = reviews_data['Comment'].fillna('').tolist()
            preprocessed_reviews = self.preprocess_reviews(reviews)
            
            # Store results
            absa_results = []
            
            print(f"🔍 Analyzing {len(reviews)} reviews with {'transformer' if self.use_transformers else 'VADER'} model...")
            
            for idx, (original_review, preprocessed_review) in enumerate(zip(reviews, preprocessed_reviews)):
                if idx % 10 == 0:
                    print(f"Progress: {idx}/{len(reviews)} reviews processed")
                
                # Extract aspects
                detected_aspects = self.extract_aspects(preprocessed_review)
                
                if not detected_aspects:
                    # If no specific aspects detected, classify overall sentiment
                    overall_sentiment = self.classify_sentiment_advanced(preprocessed_review)
                    absa_results.append({
                        'review_index': idx,
                        'original_review': original_review,
                        'aspect': 'Overall',
                        'sentiment_label': overall_sentiment['sentiment_label'],
                        'positive_score': overall_sentiment['positive'],
                        'negative_score': overall_sentiment['negative'],
                        'neutral_score': overall_sentiment['neutral'],
                        'compound_score': overall_sentiment['compound'],
                        'model_used': overall_sentiment.get('model_used', 'vader')
                    })
                else:
                    # Analyze sentiment for each detected aspect
                    for aspect in detected_aspects:
                        # Extract context around aspect mentions
                        aspect_context = self.extract_aspect_context(preprocessed_review, aspect)
                        
                        # Classify sentiment for this aspect using advanced method
                        aspect_sentiment = self.classify_sentiment_advanced(aspect_context)
                        
                        absa_results.append({
                            'review_index': idx,
                            'original_review': original_review,
                            'aspect': aspect,
                            'sentiment_label': aspect_sentiment['sentiment_label'],
                            'positive_score': aspect_sentiment['positive'],
                            'negative_score': aspect_sentiment['negative'],
                            'neutral_score': aspect_sentiment['neutral'],
                            'compound_score': aspect_sentiment['compound'],
                            'model_used': aspect_sentiment.get('model_used', 'vader')
                        })
            
            print("✅ Analysis completed!")
            
            # Convert results to DataFrame
            absa_df = pd.DataFrame(absa_results)
            
            # Merge with original data
            result_df = reviews_data.copy()
            result_df['review_index'] = range(len(result_df))
            
            # Create a summary of aspects and sentiments per review
            aspect_summary = absa_df.groupby('review_index').agg({
                'aspect': lambda x: ', '.join(x),
                'sentiment_label': lambda x: ', '.join(x),
                'compound_score': 'mean'
            }).reset_index()
            
            result_df = result_df.merge(aspect_summary, on='review_index', how='left')
            result_df = result_df.rename(columns={
                'aspect': 'detected_aspects',
                'sentiment_label': 'aspect_sentiments',
                'compound_score': 'avg_sentiment_score'
            })
            
            return result_df, absa_df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_product_aspects_advanced(self, reviews_data: pd.DataFrame, product_name: Optional[str] = None) -> Dict:
        """
        Comprehensive advanced aspect analysis for a product.
        
        Args:
            reviews_data (pd.DataFrame): DataFrame containing review data
            product_name (str, optional): Name of the product
            
        Returns:
            Dict: Comprehensive analysis results with advanced insights
        """
        try:
            # Perform advanced ABSA
            enhanced_reviews, absa_detailed = self.analyze_reviews_advanced(reviews_data)
            
            # Get aspect summary
            aspect_summary = self.get_aspect_summary(absa_detailed)
            
            # Calculate overall statistics
            total_reviews = len(reviews_data)
            avg_sentiment = absa_detailed['compound_score'].mean()
            
            most_mentioned_aspects = aspect_summary.groupby('aspect')['count'].sum().sort_values(ascending=False)
            
            # Calculate model usage statistics
            model_usage = absa_detailed['model_used'].value_counts().to_dict()
            
            # Calculate confidence metrics
            confidence_stats = {
                'avg_positive_confidence': absa_detailed['positive_score'].mean(),
                'avg_negative_confidence': absa_detailed['negative_score'].mean(),
                'avg_neutral_confidence': absa_detailed['neutral_score'].mean(),
                'sentiment_distribution': absa_detailed['sentiment_label'].value_counts().to_dict()
            }
            
            return {
                'product_name': product_name or 'Unknown Product',
                'total_reviews': total_reviews,
                'overall_avg_sentiment': avg_sentiment,
                'enhanced_reviews': enhanced_reviews,
                'detailed_absa': absa_detailed,
                'aspect_summary': aspect_summary,
                'most_mentioned_aspects': most_mentioned_aspects.to_dict(),
                'model_usage': model_usage,
                'confidence_stats': confidence_stats,
                'analysis_method': 'advanced_transformer' if self.use_transformers else 'vader'
            }
            
        except Exception as e:
            raise CustomException(e, sys)
