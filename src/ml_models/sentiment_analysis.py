"""
Sentiment Analysis Module for Review Comments
Provides sentiment classification and scoring for product reviews
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline
import re
from typing import Dict, List, Tuple
import sys
from src.exception import CustomException


class SentimentAnalyzer:
    def __init__(self, model_type: str = "textblob"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_type: "textblob" for basic analysis or "transformer" for advanced
        """
        self.model_type = model_type
        
        if model_type == "transformer":
            try:
                # Using pre-trained sentiment analysis pipeline
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                print("Falling back to TextBlob...")
                self.model_type = "textblob"
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess review text"""
        if pd.isna(text) or text == "No comment Given":
            return ""
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
    
    def analyze_sentiment_textblob(self, text: str) -> Dict:
        """Analyze sentiment using TextBlob"""
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return {
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.0,
                    'polarity': 0.0,
                    'subjectivity': 0.0
                }
            
            blob = TextBlob(cleaned_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convert polarity to label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'sentiment_label': label,
                'sentiment_score': polarity,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_sentiment_transformer(self, text: str) -> Dict:
        """Analyze sentiment using transformer model"""
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return {
                    'sentiment_label': 'neutral',
                    'sentiment_score': 0.0,
                    'confidence': 0.0
                }
            
            result = self.sentiment_pipeline(cleaned_text)[0]
            
            # Extract highest confidence prediction
            best_pred = max(result, key=lambda x: x['score'])
            
            # Map labels
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            sentiment_label = label_mapping.get(best_pred['label'], best_pred['label'].lower())
            
            # Convert to sentiment score (-1 to 1)
            if sentiment_label == 'positive':
                sentiment_score = best_pred['score']
            elif sentiment_label == 'negative':
                sentiment_score = -best_pred['score']
            else:
                sentiment_score = 0.0
                
            return {
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score,
                'confidence': best_pred['score']
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_batch(self, comments: List[str]) -> pd.DataFrame:
        """Analyze sentiment for a batch of comments"""
        try:
            results = []
            
            for comment in comments:
                if self.model_type == "textblob":
                    result = self.analyze_sentiment_textblob(comment)
                else:
                    result = self.analyze_sentiment_transformer(comment)
                
                results.append(result)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_aspect_sentiment(self, text: str, aspects: List[str]) -> Dict:
        """
        Extract sentiment for specific aspects mentioned in reviews
        
        Args:
            text: Review comment
            aspects: List of aspects to check (e.g., ['quality', 'price', 'fit', 'delivery'])
        """
        try:
            cleaned_text = self.clean_text(text)
            aspect_sentiments = {}
            
            for aspect in aspects:
                # Check if aspect is mentioned in the text
                if aspect.lower() in cleaned_text:
                    # Extract sentences containing the aspect
                    sentences = cleaned_text.split('.')
                    aspect_sentences = [s for s in sentences if aspect.lower() in s]
                    
                    if aspect_sentences:
                        # Analyze sentiment of aspect-specific sentences
                        aspect_text = '. '.join(aspect_sentences)
                        
                        if self.model_type == "textblob":
                            sentiment = self.analyze_sentiment_textblob(aspect_text)
                        else:
                            sentiment = self.analyze_sentiment_transformer(aspect_text)
                        
                        aspect_sentiments[aspect] = sentiment
                    else:
                        aspect_sentiments[aspect] = None
                else:
                    aspect_sentiments[aspect] = None
            
            return aspect_sentiments
            
        except Exception as e:
            raise CustomException(e, sys)


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sentiment analysis features to the review dataframe
    
    Args:
        df: Review dataframe with 'Comment' column
        
    Returns:
        Enhanced dataframe with sentiment features
    """
    try:
        analyzer = SentimentAnalyzer(model_type="textblob")  # Start with TextBlob for speed
        
        # Analyze sentiment for all comments
        sentiment_results = analyzer.analyze_batch(df['Comment'].tolist())
        
        # Add sentiment columns to original dataframe
        for col in sentiment_results.columns:
            df[f'sentiment_{col}'] = sentiment_results[col]
        
        # Add aspect-based sentiment for common e-commerce aspects
        aspects = ['quality', 'price', 'fit', 'size', 'delivery', 'comfort', 'material']
        
        for aspect in aspects:
            df[f'{aspect}_sentiment'] = df['Comment'].apply(
                lambda x: analyzer.get_aspect_sentiment(x, [aspect]).get(aspect, {}).get('sentiment_score', 0) 
                if analyzer.get_aspect_sentiment(x, [aspect]).get(aspect) else 0
            )
        
        return df
        
    except Exception as e:
        raise CustomException(e, sys)
