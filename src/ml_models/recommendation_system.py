"""
Product Recommendation System
Provides content-based and collaborative filtering recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import sys
from typing import List, Dict, Tuple
from src.exception import CustomException


class ProductRecommendationSystem:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.product_features = None
        self.similarity_matrix = None
        self.products_df = None
        
    def prepare_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare content-based features from review data
        
        Args:
            df: Review dataframe
            
        Returns:
            Product features dataframe
        """
        try:
            # Aggregate data by product
            product_agg = df.groupby('Product Name').agg({
                'Over_All_Rating': 'mean',
                'Price': 'mean',
                'Rating': ['mean', 'count'],
                'Comment': lambda x: ' '.join(x.dropna())
            }).reset_index()
            
            # Flatten column names
            product_agg.columns = [
                'Product_Name', 'Avg_Overall_Rating', 'Avg_Price', 
                'Avg_User_Rating', 'Review_Count', 'All_Comments'
            ]
            
            # Create TF-IDF features from comments
            tfidf_features = self.tfidf_vectorizer.fit_transform(product_agg['All_Comments'])
            
            # Create feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Convert to dataframe
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=feature_names)
            
            # Combine with numerical features
            numerical_features = product_agg[['Avg_Overall_Rating', 'Avg_Price', 'Avg_User_Rating', 'Review_Count']]
            
            # Normalize numerical features
            numerical_features_norm = (numerical_features - numerical_features.min()) / (numerical_features.max() - numerical_features.min())
            
            # Combine all features
            self.product_features = pd.concat([
                product_agg[['Product_Name']], 
                numerical_features_norm, 
                tfidf_df
            ], axis=1)
            
            self.products_df = product_agg
            
            return self.product_features
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def compute_similarity_matrix(self):
        """Compute cosine similarity matrix between products"""
        try:
            if self.product_features is None:
                raise ValueError("Product features not prepared. Call prepare_content_features first.")
            
            # Get feature columns (exclude product name)
            feature_cols = self.product_features.columns[1:]
            features = self.product_features[feature_cols].values
            
            # Compute cosine similarity
            self.similarity_matrix = cosine_similarity(features)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_content_recommendations(self, product_name: str, top_k: int = 5) -> List[Dict]:
        """
        Get content-based recommendations for a product
        
        Args:
            product_name: Name of the product to get recommendations for
            top_k: Number of recommendations to return
            
        Returns:
            List of recommended products with similarity scores
        """
        try:
            if self.similarity_matrix is None:
                self.compute_similarity_matrix()
            
            # Find product index
            product_idx = None
            for idx, name in enumerate(self.product_features['Product_Name']):
                if product_name.lower() in name.lower():
                    product_idx = idx
                    break
            
            if product_idx is None:
                return []
            
            # Get similarity scores for this product
            sim_scores = list(enumerate(self.similarity_matrix[product_idx]))
            
            # Sort by similarity (excluding the product itself)
            sim_scores = [(i, score) for i, score in sim_scores if i != product_idx]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get top recommendations
            recommendations = []
            for i, (idx, score) in enumerate(sim_scores[:top_k]):
                rec_product = self.product_features.iloc[idx]
                product_info = self.products_df.iloc[idx]
                
                recommendations.append({
                    'rank': i + 1,
                    'product_name': rec_product['Product_Name'],
                    'similarity_score': round(score, 3),
                    'avg_rating': round(product_info['Avg_Overall_Rating'], 2),
                    'avg_price': round(product_info['Avg_Price'], 2),
                    'review_count': int(product_info['Review_Count'])
                })
            
            return recommendations
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_price_based_recommendations(self, target_price: float, price_range: float = 0.2) -> List[Dict]:
        """
        Get recommendations based on price range
        
        Args:
            target_price: Target price point
            price_range: Price range as percentage (0.2 = ±20%)
            
        Returns:
            List of products within price range sorted by rating
        """
        try:
            if self.products_df is None:
                raise ValueError("Products data not available.")
            
            min_price = target_price * (1 - price_range)
            max_price = target_price * (1 + price_range)
            
            # Filter products by price range
            price_filtered = self.products_df[
                (self.products_df['Avg_Price'] >= min_price) & 
                (self.products_df['Avg_Price'] <= max_price)
            ].copy()
            
            # Sort by rating
            price_filtered = price_filtered.sort_values('Avg_Overall_Rating', ascending=False)
            
            recommendations = []
            for idx, row in price_filtered.iterrows():
                recommendations.append({
                    'product_name': row['Product_Name'],
                    'price': round(row['Avg_Price'], 2),
                    'rating': round(row['Avg_Overall_Rating'], 2),
                    'review_count': int(row['Review_Count']),
                    'value_score': round(row['Avg_Overall_Rating'] / (row['Avg_Price'] / 1000), 3)  # Rating per 1000 rupees
                })
            
            return recommendations
            
        except Exception as e:
            raise CustomException(e, sys)


class CollaborativeFiltering:
    def __init__(self):
        self.user_item_matrix = None
        self.svd_model = None
        
    def create_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item rating matrix"""
        try:
            # Use Name as user identifier and Product Name as item
            user_item = df.pivot_table(
                index='Name', 
                columns='Product Name', 
                values='Rating', 
                fill_value=0
            )
            
            self.user_item_matrix = user_item
            return user_item
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def fit_svd_model(self, n_components: int = 50):
        """Fit SVD model for collaborative filtering"""
        try:
            if self.user_item_matrix is None:
                raise ValueError("User-item matrix not created.")
            
            self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd_model.fit(self.user_item_matrix)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_user_recommendations(self, user_name: str, top_k: int = 5) -> List[Dict]:
        """Get recommendations for a specific user"""
        try:
            if self.svd_model is None:
                raise ValueError("SVD model not fitted.")
            
            if user_name not in self.user_item_matrix.index:
                return []
            
            # Get user vector
            user_idx = self.user_item_matrix.index.get_loc(user_name)
            user_vector = self.user_item_matrix.iloc[user_idx].values.reshape(1, -1)
            
            # Transform using SVD
            user_svd = self.svd_model.transform(user_vector)
            
            # Reconstruct ratings
            reconstructed = self.svd_model.inverse_transform(user_svd)[0]
            
            # Get products user hasn't rated
            unrated_products = self.user_item_matrix.columns[self.user_item_matrix.iloc[user_idx] == 0]
            
            # Get recommendations
            recommendations = []
            for product in unrated_products:
                product_idx = self.user_item_matrix.columns.get_loc(product)
                predicted_rating = reconstructed[product_idx]
                
                recommendations.append({
                    'product_name': product,
                    'predicted_rating': round(predicted_rating, 2)
                })
            
            # Sort by predicted rating
            recommendations = sorted(recommendations, key=lambda x: x['predicted_rating'], reverse=True)
            
            return recommendations[:top_k]
            
        except Exception as e:
            raise CustomException(e, sys)
