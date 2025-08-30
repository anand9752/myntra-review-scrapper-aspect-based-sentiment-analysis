import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import sys
from src.exception import CustomException

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class ABSAAnalyzer:
    """
    Aspect-Based Sentiment Analysis (ABSA) analyzer for product reviews.
    """
    
    def __init__(self):
        """Initialize the ABSA analyzer with predefined aspects and keywords."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Define aspect categories and their associated keywords
        self.aspect_keywords = {
            'Style/Design': [
                'style', 'design', 'look', 'looks', 'appearance', 'color', 'colour', 
                'beautiful', 'pretty', 'ugly', 'attractive', 'stylish', 'fashionable',
                'trendy', 'modern', 'classic', 'elegant', 'cute', 'nice looking',
                'aesthetic', 'pattern', 'print', 'design'
            ],
            'Quality/Material': [
                'quality', 'material', 'fabric', 'leather', 'cotton', 'polyester',
                'durable', 'durability', 'cheap', 'flimsy', 'sturdy', 'strong',
                'weak', 'tear', 'torn', 'rip', 'ripped', 'stitching', 'construction',
                'built', 'made', 'craftsmanship', 'workmanship', 'texture'
            ],
            'Size/Fit': [
                'size', 'fit', 'fitting', 'fitted', 'tight', 'loose', 'big', 'small',
                'large', 'medium', 'xl', 'xxl', 'length', 'width', 'comfortable',
                'comfort', 'uncomfortable', 'perfect fit', 'sizing', 'measurements'
            ],
            'Price/Value': [
                'price', 'cost', 'expensive', 'cheap', 'affordable', 'value',
                'money', 'worth', 'budget', 'overpriced', 'reasonable', 'deal',
                'bargain', 'costly', 'investment', 'price point'
            ],
            'Delivery/Service': [
                'delivery', 'shipping', 'service', 'customer service', 'support',
                'fast', 'slow', 'delayed', 'on time', 'quick', 'package',
                'packaging', 'arrived', 'received', 'order', 'dispatch'
            ]
        }
        
        # Compile regex patterns for each aspect
        self.aspect_patterns = {}
        for aspect, keywords in self.aspect_keywords.items():
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            self.aspect_patterns[aspect] = re.compile(pattern, re.IGNORECASE)
    
    def preprocess_reviews(self, reviews: List[str]) -> List[str]:
        """
        Preprocess a list of reviews by cleaning and normalizing text.
        
        Args:
            reviews (List[str]): List of review texts
            
        Returns:
            List[str]: Preprocessed review texts
        """
        try:
            preprocessed = []
            
            for review in reviews:
                if not isinstance(review, str):
                    review = str(review) if review is not None else ""
                
                # Convert to lowercase
                review = review.lower()
                
                # Remove extra whitespaces
                review = re.sub(r'\s+', ' ', review)
                
                # Remove special characters but keep punctuation for sentiment
                review = re.sub(r'[^\w\s.,!?-]', '', review)
                
                # Strip leading/trailing whitespace
                review = review.strip()
                
                preprocessed.append(review)
            
            return preprocessed
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def extract_aspects(self, review: str) -> List[str]:
        """
        Extract aspects mentioned in a review using keyword matching.
        
        Args:
            review (str): Review text
            
        Returns:
            List[str]: List of detected aspects
        """
        try:
            detected_aspects = []
            
            for aspect, pattern in self.aspect_patterns.items():
                if pattern.search(review):
                    detected_aspects.append(aspect)
            
            return detected_aspects
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def classify_sentiment(self, text: str) -> Dict[str, float]:
        """
        Classify sentiment of text using VADER sentiment analyzer.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores and classification
        """
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Determine overall sentiment based on compound score
            if scores['compound'] >= 0.05:
                sentiment_label = 'Positive'
            elif scores['compound'] <= -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
            
            return {
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'compound': scores['compound'],
                'sentiment_label': sentiment_label
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def extract_aspect_context(self, review: str, aspect: str, window_size: int = 50) -> str:
        """
        Extract context around aspect mentions for more accurate sentiment analysis.
        
        Args:
            review (str): Review text
            aspect (str): Aspect to find context for
            window_size (int): Number of characters around the aspect mention
            
        Returns:
            str: Context around aspect mentions
        """
        try:
            pattern = self.aspect_patterns[aspect]
            matches = list(pattern.finditer(review))
            
            if not matches:
                return review  # Return full review if no specific mention found
            
            contexts = []
            for match in matches:
                start = max(0, match.start() - window_size)
                end = min(len(review), match.end() + window_size)
                context = review[start:end]
                contexts.append(context)
            
            return ' '.join(contexts)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_reviews(self, reviews_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform ABSA on a DataFrame of reviews.
        
        Args:
            reviews_data (pd.DataFrame): DataFrame containing review data
            
        Returns:
            pd.DataFrame: DataFrame with ABSA results
        """
        try:
            if 'Comment' not in reviews_data.columns:
                raise ValueError("DataFrame must contain 'Comment' column")
            
            # Preprocess reviews
            reviews = reviews_data['Comment'].fillna('').tolist()
            preprocessed_reviews = self.preprocess_reviews(reviews)
            
            # Store results
            absa_results = []
            
            for idx, (original_review, preprocessed_review) in enumerate(zip(reviews, preprocessed_reviews)):
                # Extract aspects
                detected_aspects = self.extract_aspects(preprocessed_review)
                
                if not detected_aspects:
                    # If no specific aspects detected, classify overall sentiment
                    overall_sentiment = self.classify_sentiment(preprocessed_review)
                    absa_results.append({
                        'review_index': idx,
                        'original_review': original_review,
                        'aspect': 'Overall',
                        'sentiment_label': overall_sentiment['sentiment_label'],
                        'positive_score': overall_sentiment['positive'],
                        'negative_score': overall_sentiment['negative'],
                        'neutral_score': overall_sentiment['neutral'],
                        'compound_score': overall_sentiment['compound']
                    })
                else:
                    # Analyze sentiment for each detected aspect
                    for aspect in detected_aspects:
                        # Extract context around aspect mentions
                        aspect_context = self.extract_aspect_context(preprocessed_review, aspect)
                        
                        # Classify sentiment for this aspect
                        aspect_sentiment = self.classify_sentiment(aspect_context)
                        
                        absa_results.append({
                            'review_index': idx,
                            'original_review': original_review,
                            'aspect': aspect,
                            'sentiment_label': aspect_sentiment['sentiment_label'],
                            'positive_score': aspect_sentiment['positive'],
                            'negative_score': aspect_sentiment['negative'],
                            'neutral_score': aspect_sentiment['neutral'],
                            'compound_score': aspect_sentiment['compound']
                        })
            
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
    
    def get_aspect_summary(self, absa_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for aspects and sentiments.
        
        Args:
            absa_df (pd.DataFrame): ABSA results DataFrame
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        try:
            summary = absa_df.groupby(['aspect', 'sentiment_label']).agg({
                'review_index': 'count',
                'compound_score': 'mean'
            }).reset_index()
            
            summary = summary.rename(columns={
                'review_index': 'count',
                'compound_score': 'avg_sentiment_score'
            })
            
            # Calculate percentages
            total_by_aspect = summary.groupby('aspect')['count'].sum().reset_index()
            total_by_aspect = total_by_aspect.rename(columns={'count': 'total_count'})
            
            summary = summary.merge(total_by_aspect, on='aspect')
            summary['percentage'] = (summary['count'] / summary['total_count'] * 100).round(2)
            
            return summary
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_product_aspects(self, reviews_data: pd.DataFrame, product_name: Optional[str] = None) -> Dict:
        """
        Comprehensive aspect analysis for a product.
        
        Args:
            reviews_data (pd.DataFrame): DataFrame containing review data
            product_name (str, optional): Name of the product
            
        Returns:
            Dict: Comprehensive analysis results
        """
        try:
            # Perform ABSA
            enhanced_reviews, absa_detailed = self.analyze_reviews(reviews_data)
            
            # Get aspect summary
            aspect_summary = self.get_aspect_summary(absa_detailed)
            
            # Calculate overall statistics
            total_reviews = len(reviews_data)
            avg_sentiment = absa_detailed['compound_score'].mean()
            
            most_mentioned_aspects = aspect_summary.groupby('aspect')['count'].sum().sort_values(ascending=False)
            
            return {
                'product_name': product_name or 'Unknown Product',
                'total_reviews': total_reviews,
                'overall_avg_sentiment': avg_sentiment,
                'enhanced_reviews': enhanced_reviews,
                'detailed_absa': absa_detailed,
                'aspect_summary': aspect_summary,
                'most_mentioned_aspects': most_mentioned_aspects.to_dict()
            }
            
        except Exception as e:
            raise CustomException(e, sys)
