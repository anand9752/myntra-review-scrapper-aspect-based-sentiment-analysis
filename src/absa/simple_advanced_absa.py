"""
Simple and reliable transformer-based sentiment analyzer for ABSA
This uses a lightweight model that should work consistently.
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
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("Warning: TextBlob not available. Using VADER only.")


class SimpleAdvancedABSA(ABSAAnalyzer):
    """
    Simple advanced ABSA analyzer using TextBlob for better sentiment analysis.
    More reliable than complex transformer models.
    """
    
    def __init__(self, use_textblob: bool = True):
        """
        Initialize the simple advanced ABSA analyzer.
        
        Args:
            use_textblob (bool): Whether to use TextBlob for sentiment analysis
        """
        super().__init__()
        
        self.use_textblob = use_textblob and TEXTBLOB_AVAILABLE
        
        if self.use_textblob:
            print("✅ Using TextBlob for enhanced sentiment analysis")
        else:
            print("⚪ Using VADER sentiment analysis only")
    
    def classify_sentiment_enhanced(self, text: str) -> Dict[str, float]:
        """
        Enhanced sentiment classification using TextBlob + VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Enhanced sentiment scores and classification
        """
        try:
            # Get VADER scores
            vader_result = self.classify_sentiment(text)
            
            if self.use_textblob:
                # Get TextBlob scores
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # -1 to 1
                subjectivity = blob.sentiment.subjectivity  # 0 to 1
                
                # Convert TextBlob polarity to positive/negative scores
                if polarity > 0:
                    textblob_pos = polarity
                    textblob_neg = 0
                elif polarity < 0:
                    textblob_pos = 0
                    textblob_neg = abs(polarity)
                else:
                    textblob_pos = 0
                    textblob_neg = 0
                
                textblob_neu = 1 - textblob_pos - textblob_neg
                
                # Use VADER as primary, TextBlob as secondary validation
                # If both agree, boost confidence; if they disagree, be more conservative
                vader_sentiment = vader_result['sentiment_label']
                
                # Determine TextBlob sentiment
                if polarity >= 0.1:
                    textblob_sentiment = 'Positive'
                elif polarity <= -0.1:
                    textblob_sentiment = 'Negative'
                else:
                    textblob_sentiment = 'Neutral'
                
                # If both models agree, use stronger weighting
                if vader_sentiment == textblob_sentiment:
                    # Both agree - boost the signal
                    vader_weight = 0.7
                    textblob_weight = 0.3
                    confidence_boost = 1.2
                else:
                    # Disagreement - be more conservative, trust VADER more
                    vader_weight = 0.8
                    textblob_weight = 0.2
                    confidence_boost = 0.8
                
                # Combine scores
                combined_pos = (vader_result['positive'] * vader_weight + 
                               textblob_pos * textblob_weight) * confidence_boost
                combined_neg = (vader_result['negative'] * vader_weight + 
                               textblob_neg * textblob_weight) * confidence_boost
                combined_neu = (vader_result['neutral'] * vader_weight + 
                               textblob_neu * textblob_weight)
                
                # Normalize scores
                total = combined_pos + combined_neg + combined_neu
                if total > 0:
                    combined_pos /= total
                    combined_neg /= total
                    combined_neu /= total
                
                # Calculate enhanced compound score
                compound = combined_pos - combined_neg
                
                # Use more sensitive thresholds when models agree
                if vader_sentiment == textblob_sentiment:
                    if compound >= 0.05:
                        sentiment_label = 'Positive'
                    elif compound <= -0.05:
                        sentiment_label = 'Negative'
                    else:
                        sentiment_label = 'Neutral'
                else:
                    # Use stricter thresholds when models disagree
                    if compound >= 0.15:
                        sentiment_label = 'Positive'
                    elif compound <= -0.15:
                        sentiment_label = 'Negative'
                    else:
                        sentiment_label = 'Neutral'
                
                return {
                    'positive': combined_pos,
                    'negative': combined_neg,
                    'neutral': combined_neu,
                    'compound': compound,
                    'sentiment_label': sentiment_label,
                    'model_used': 'textblob_vader_enhanced',
                    'subjectivity': subjectivity,
                    'textblob_polarity': polarity,
                    'vader_compound': vader_result['compound'],
                    'models_agree': vader_sentiment == textblob_sentiment,
                    'confidence_boost': confidence_boost
                }
            else:
                # Use VADER only
                vader_result['model_used'] = 'vader_only'
                return vader_result
                
        except Exception as e:
            print(f"Error in enhanced sentiment classification: {e}")
            # Fallback to pure VADER
            result = self.classify_sentiment(text)
            result['model_used'] = 'vader_fallback'
            return result
    
    def analyze_reviews_enhanced(self, reviews_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform enhanced ABSA on a DataFrame of reviews.
        
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
            
            model_name = "Enhanced TextBlob+VADER" if self.use_textblob else "VADER"
            print(f"🔍 Analyzing {len(reviews)} reviews with {model_name} model...")
            
            for idx, (original_review, preprocessed_review) in enumerate(zip(reviews, preprocessed_reviews)):
                if idx % 5 == 0:
                    print(f"Progress: {idx}/{len(reviews)} reviews processed")
                
                # Extract aspects
                detected_aspects = self.extract_aspects(preprocessed_review)
                
                if not detected_aspects:
                    # If no specific aspects detected, classify overall sentiment
                    overall_sentiment = self.classify_sentiment_enhanced(preprocessed_review)
                    absa_results.append({
                        'review_index': idx,
                        'original_review': original_review,
                        'aspect': 'Overall',
                        'sentiment_label': overall_sentiment['sentiment_label'],
                        'positive_score': overall_sentiment['positive'],
                        'negative_score': overall_sentiment['negative'],
                        'neutral_score': overall_sentiment['neutral'],
                        'compound_score': overall_sentiment['compound'],
                        'model_used': overall_sentiment.get('model_used', 'vader'),
                        'confidence': overall_sentiment.get('subjectivity', 0.5)
                    })
                else:
                    # Analyze sentiment for each detected aspect
                    for aspect in detected_aspects:
                        # Extract context around aspect mentions
                        aspect_context = self.extract_aspect_context(preprocessed_review, aspect)
                        
                        # Classify sentiment for this aspect using enhanced method
                        aspect_sentiment = self.classify_sentiment_enhanced(aspect_context)
                        
                        absa_results.append({
                            'review_index': idx,
                            'original_review': original_review,
                            'aspect': aspect,
                            'sentiment_label': aspect_sentiment['sentiment_label'],
                            'positive_score': aspect_sentiment['positive'],
                            'negative_score': aspect_sentiment['negative'],
                            'neutral_score': aspect_sentiment['neutral'],
                            'compound_score': aspect_sentiment['compound'],
                            'model_used': aspect_sentiment.get('model_used', 'vader'),
                            'confidence': aspect_sentiment.get('subjectivity', 0.5)
                        })
            
            print("✅ Enhanced analysis completed!")
            
            # Convert results to DataFrame
            absa_df = pd.DataFrame(absa_results)
            
            # Merge with original data
            result_df = reviews_data.copy()
            result_df['review_index'] = range(len(result_df))
            
            # Create a summary of aspects and sentiments per review
            aspect_summary = absa_df.groupby('review_index').agg({
                'aspect': lambda x: ', '.join(x),
                'sentiment_label': lambda x: ', '.join(x),
                'compound_score': 'mean',
                'confidence': 'mean'
            }).reset_index()
            
            result_df = result_df.merge(aspect_summary, on='review_index', how='left')
            result_df = result_df.rename(columns={
                'aspect': 'detected_aspects',
                'sentiment_label': 'aspect_sentiments',
                'compound_score': 'avg_sentiment_score',
                'confidence': 'avg_confidence'
            })
            
            return result_df, absa_df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_product_aspects_enhanced(self, reviews_data: pd.DataFrame, product_name: Optional[str] = None) -> Dict:
        """
        Comprehensive enhanced aspect analysis for a product.
        
        Args:
            reviews_data (pd.DataFrame): DataFrame containing review data
            product_name (str, optional): Name of the product
            
        Returns:
            Dict: Comprehensive analysis results with enhanced insights
        """
        try:
            # Perform enhanced ABSA
            enhanced_reviews, absa_detailed = self.analyze_reviews_enhanced(reviews_data)
            
            # Get aspect summary
            aspect_summary = self.get_aspect_summary(absa_detailed)
            
            # Calculate overall statistics
            total_reviews = len(reviews_data)
            avg_sentiment = absa_detailed['compound_score'].mean()
            avg_confidence = absa_detailed['confidence'].mean()
            
            most_mentioned_aspects = aspect_summary.groupby('aspect')['count'].sum().sort_values(ascending=False)
            
            # Calculate model usage statistics
            model_usage = absa_detailed['model_used'].value_counts().to_dict()
            
            # Calculate enhanced confidence metrics
            confidence_stats = {
                'avg_positive_confidence': absa_detailed['positive_score'].mean(),
                'avg_negative_confidence': absa_detailed['negative_score'].mean(),
                'avg_neutral_confidence': absa_detailed['neutral_score'].mean(),
                'avg_overall_confidence': avg_confidence,
                'sentiment_distribution': absa_detailed['sentiment_label'].value_counts().to_dict(),
                'high_confidence_predictions': len(absa_detailed[absa_detailed['confidence'] > 0.7])
            }
            
            # Calculate sentiment strength distribution
            sentiment_strength = {
                'very_positive': len(absa_detailed[absa_detailed['compound_score'] > 0.5]),
                'positive': len(absa_detailed[(absa_detailed['compound_score'] > 0.1) & (absa_detailed['compound_score'] <= 0.5)]),
                'neutral': len(absa_detailed[(absa_detailed['compound_score'] >= -0.1) & (absa_detailed['compound_score'] <= 0.1)]),
                'negative': len(absa_detailed[(absa_detailed['compound_score'] >= -0.5) & (absa_detailed['compound_score'] < -0.1)]),
                'very_negative': len(absa_detailed[absa_detailed['compound_score'] < -0.5])
            }
            
            return {
                'product_name': product_name or 'Unknown Product',
                'total_reviews': total_reviews,
                'overall_avg_sentiment': avg_sentiment,
                'overall_avg_confidence': avg_confidence,
                'enhanced_reviews': enhanced_reviews,
                'detailed_absa': absa_detailed,
                'aspect_summary': aspect_summary,
                'most_mentioned_aspects': most_mentioned_aspects.to_dict(),
                'model_usage': model_usage,
                'confidence_stats': confidence_stats,
                'sentiment_strength': sentiment_strength,
                'analysis_method': 'enhanced_textblob_vader' if self.use_textblob else 'vader_only'
            }
            
        except Exception as e:
            raise CustomException(e, sys)
