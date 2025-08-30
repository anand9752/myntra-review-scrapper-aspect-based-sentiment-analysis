"""
Review Quality and Fake Review Detection
Identifies helpful reviews and detects potential fake/spam reviews
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List
import sys
from src.exception import CustomException


class ReviewQualityAnalyzer:
    def __init__(self):
        self.quality_features = [
            'review_length', 'word_count', 'sentence_count', 
            'capital_ratio', 'punctuation_ratio', 'rating_text_mismatch',
            'specific_words_count', 'repetitive_patterns'
        ]
    
    def extract_quality_features(self, review_text: str, rating: float) -> Dict:
        """
        Extract quality features from a review
        
        Args:
            review_text: The review comment
            rating: Numerical rating given
            
        Returns:
            Dictionary of quality features
        """
        try:
            if pd.isna(review_text) or review_text == "No comment Given":
                return {feature: 0 for feature in self.quality_features}
            
            text = str(review_text)
            
            # Basic text statistics
            review_length = len(text)
            word_count = len(text.split())
            sentence_count = len([s for s in text.split('.') if s.strip()])
            
            # Character analysis
            capital_count = sum(1 for c in text if c.isupper())
            capital_ratio = capital_count / len(text) if len(text) > 0 else 0
            
            punctuation_count = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', text))
            punctuation_ratio = punctuation_count / len(text) if len(text) > 0 else 0
            
            # Rating-text sentiment mismatch (basic heuristic)
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'awesome']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor']
            
            positive_count = sum(1 for word in positive_words if word in text.lower())
            negative_count = sum(1 for word in negative_words if word in text.lower())
            
            # Mismatch: high rating with negative words or low rating with positive words
            rating_text_mismatch = 0
            if rating >= 4 and negative_count > positive_count:
                rating_text_mismatch = 1
            elif rating <= 2 and positive_count > negative_count:
                rating_text_mismatch = 1
            
            # Specific vs generic words
            specific_words = ['quality', 'material', 'size', 'fit', 'comfort', 'delivery', 'packaging']
            specific_words_count = sum(1 for word in specific_words if word in text.lower())
            
            # Repetitive patterns (potential spam indicator)
            words = text.lower().split()
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            repetitive_patterns = max(word_freq.values()) if word_freq else 0
            
            return {
                'review_length': review_length,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'capital_ratio': capital_ratio,
                'punctuation_ratio': punctuation_ratio,
                'rating_text_mismatch': rating_text_mismatch,
                'specific_words_count': specific_words_count,
                'repetitive_patterns': repetitive_patterns
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def calculate_quality_score(self, features: Dict) -> float:
        """
        Calculate overall quality score based on features
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            score = 0.0
            
            # Length score (optimal range: 50-300 characters)
            if 50 <= features['review_length'] <= 300:
                score += 0.2
            elif features['review_length'] > 300:
                score += 0.15
            
            # Word count score (optimal: 10-60 words)
            if 10 <= features['word_count'] <= 60:
                score += 0.2
            elif features['word_count'] > 60:
                score += 0.15
            
            # Sentence structure score
            if features['sentence_count'] >= 2:
                score += 0.1
            
            # Capital ratio (not too many caps - spam indicator)
            if features['capital_ratio'] <= 0.1:
                score += 0.1
            
            # Specific words bonus
            if features['specific_words_count'] >= 2:
                score += 0.2
            elif features['specific_words_count'] >= 1:
                score += 0.1
            
            # Penalty for rating-text mismatch
            if features['rating_text_mismatch']:
                score -= 0.3
            
            # Penalty for repetitive patterns
            if features['repetitive_patterns'] > 3:
                score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def detect_fake_review(self, features: Dict, user_review_count: int = 1) -> Dict:
        """
        Detect potential fake reviews based on patterns
        
        Args:
            features: Review quality features
            user_review_count: Number of reviews by this user
            
        Returns:
            Dictionary with fake detection results
        """
        try:
            fake_indicators = []
            fake_score = 0.0
            
            # Very short reviews with extreme ratings
            if features['word_count'] < 5:
                fake_indicators.append("Very short review")
                fake_score += 0.3
            
            # Excessive capital letters
            if features['capital_ratio'] > 0.3:
                fake_indicators.append("Excessive capital letters")
                fake_score += 0.2
            
            # Rating-text mismatch
            if features['rating_text_mismatch']:
                fake_indicators.append("Rating doesn't match review content")
                fake_score += 0.4
            
            # Repetitive patterns
            if features['repetitive_patterns'] > 5:
                fake_indicators.append("Repetitive word patterns")
                fake_score += 0.3
            
            # Single review user (potential bot)
            if user_review_count == 1 and features['word_count'] < 10:
                fake_indicators.append("Single review from user with minimal content")
                fake_score += 0.2
            
            # Generic review (no specific words)
            if features['specific_words_count'] == 0 and features['word_count'] > 10:
                fake_indicators.append("Generic review with no specific details")
                fake_score += 0.2
            
            is_fake = fake_score > 0.5
            
            return {
                'is_fake': is_fake,
                'fake_score': min(1.0, fake_score),
                'fake_indicators': fake_indicators,
                'confidence': min(1.0, fake_score) if is_fake else 1.0 - min(1.0, fake_score)
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_review_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze quality and authenticity for a batch of reviews
        
        Args:
            df: DataFrame with review data
            
        Returns:
            Enhanced DataFrame with quality metrics
        """
        try:
            # Count reviews per user
            user_counts = df['Name'].value_counts().to_dict()
            
            quality_results = []
            fake_results = []
            
            for idx, row in df.iterrows():
                # Extract features
                features = self.extract_quality_features(row['Comment'], row['Rating'])
                
                # Calculate quality score
                quality_score = self.calculate_quality_score(features)
                
                # Detect fake reviews
                user_count = user_counts.get(row['Name'], 1)
                fake_detection = self.detect_fake_review(features, user_count)
                
                quality_results.append({
                    'quality_score': quality_score,
                    **features
                })
                
                fake_results.append(fake_detection)
            
            # Add results to dataframe
            quality_df = pd.DataFrame(quality_results)
            fake_df = pd.DataFrame(fake_results)
            
            # Combine with original dataframe
            result_df = pd.concat([df.reset_index(drop=True), quality_df, fake_df], axis=1)
            
            return result_df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_quality_insights(self, df: pd.DataFrame) -> Dict:
        """
        Get overall quality insights for the dataset
        
        Args:
            df: DataFrame with quality analysis results
            
        Returns:
            Dictionary with quality insights
        """
        try:
            total_reviews = len(df)
            high_quality_reviews = len(df[df['quality_score'] >= 0.7])
            potential_fake_reviews = len(df[df['is_fake'] == True])
            
            avg_quality_score = df['quality_score'].mean()
            avg_word_count = df['word_count'].mean()
            
            # Most common fake indicators
            all_indicators = []
            for indicators in df['fake_indicators']:
                all_indicators.extend(indicators)
            
            from collections import Counter
            common_fake_indicators = Counter(all_indicators).most_common(5)
            
            return {
                'total_reviews': total_reviews,
                'high_quality_count': high_quality_reviews,
                'high_quality_percentage': round((high_quality_reviews / total_reviews) * 100, 2),
                'potential_fake_count': potential_fake_reviews,
                'fake_percentage': round((potential_fake_reviews / total_reviews) * 100, 2),
                'average_quality_score': round(avg_quality_score, 3),
                'average_word_count': round(avg_word_count, 1),
                'common_fake_indicators': common_fake_indicators
            }
            
        except Exception as e:
            raise CustomException(e, sys)
