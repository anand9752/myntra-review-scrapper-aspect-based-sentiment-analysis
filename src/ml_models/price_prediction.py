"""
Price Prediction and Value Analysis
Predicts fair prices and analyzes value for money
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
from src.exception import CustomException


class PricePredictionModel:
    def __init__(self):
        self.price_factors = {}
        self.category_avg_prices = {}
        self.brand_premiums = {}
        
    def extract_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features that influence price
        
        Args:
            df: Review dataframe
            
        Returns:
            DataFrame with price prediction features
        """
        try:
            # Aggregate by product
            product_features = df.groupby('Product Name').agg({
                'Over_All_Rating': 'mean',
                'Price': 'mean',
                'Rating': ['mean', 'count', 'std'],
                'Comment': 'count'
            }).reset_index()
            
            # Flatten columns
            product_features.columns = [
                'Product_Name', 'Avg_Overall_Rating', 'Avg_Price',
                'Avg_User_Rating', 'Review_Count', 'Rating_Std', 'Comment_Count'
            ]
            
            # Extract brand and category information from product names
            product_features['Brand'] = product_features['Product_Name'].apply(self._extract_brand)
            product_features['Category'] = product_features['Product_Name'].apply(self._extract_category)
            
            # Calculate price per rating point
            product_features['Price_Per_Rating'] = product_features['Avg_Price'] / (product_features['Avg_Overall_Rating'] + 0.1)
            
            # Rating consistency (lower std = more consistent ratings)
            product_features['Rating_Consistency'] = 1 / (product_features['Rating_Std'] + 0.1)
            
            # Popularity score
            product_features['Popularity_Score'] = np.log1p(product_features['Review_Count'])
            
            return product_features
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _extract_brand(self, product_name: str) -> str:
        """Extract brand from product name (basic heuristic)"""
        try:
            # Common brand patterns
            common_brands = ['nike', 'adidas', 'puma', 'reebok', 'vans', 'converse', 'fila']
            
            name_lower = product_name.lower()
            for brand in common_brands:
                if brand in name_lower:
                    return brand.title()
            
            # Return first word as potential brand
            return product_name.split()[0] if product_name.split() else 'Unknown'
            
        except Exception as e:
            return 'Unknown'
    
    def _extract_category(self, product_name: str) -> str:
        """Extract category from product name (basic heuristic)"""
        try:
            name_lower = product_name.lower()
            
            if any(word in name_lower for word in ['shoe', 'sneaker', 'boot', 'sandal']):
                return 'Footwear'
            elif any(word in name_lower for word in ['shirt', 't-shirt', 'top', 'blouse']):
                return 'Tops'
            elif any(word in name_lower for word in ['jean', 'pant', 'trouser', 'short']):
                return 'Bottoms'
            elif any(word in name_lower for word in ['dress', 'skirt', 'kurta']):
                return 'Dresses'
            else:
                return 'Other'
                
        except Exception as e:
            return 'Other'
    
    def calculate_brand_premiums(self, df: pd.DataFrame) -> Dict:
        """Calculate price premiums for different brands"""
        try:
            product_features = self.extract_price_features(df)
            
            # Calculate average price by brand
            brand_prices = product_features.groupby('Brand')['Avg_Price'].mean()
            overall_avg_price = product_features['Avg_Price'].mean()
            
            # Calculate premium as ratio to overall average
            brand_premiums = {}
            for brand, price in brand_prices.items():
                premium = price / overall_avg_price
                brand_premiums[brand] = {
                    'average_price': round(price, 2),
                    'premium_ratio': round(premium, 2),
                    'premium_percentage': round((premium - 1) * 100, 1)
                }
            
            self.brand_premiums = brand_premiums
            return brand_premiums
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_fair_price(self, product_data: Dict) -> Dict:
        """
        Predict fair price based on product characteristics
        
        Args:
            product_data: Dictionary with product info (rating, brand, category, etc.)
            
        Returns:
            Price prediction with confidence intervals
        """
        try:
            base_price = 1000  # Base price assumption
            
            # Rating factor (higher rating = higher price)
            rating = product_data.get('rating', 3.5)
            rating_factor = 0.8 + (rating - 1) * 0.1  # Range: 0.8 to 1.2
            
            # Brand premium
            brand = product_data.get('brand', 'Unknown')
            brand_premium = self.brand_premiums.get(brand, {}).get('premium_ratio', 1.0)
            
            # Category factor
            category = product_data.get('category', 'Other')
            category_factors = {
                'Footwear': 1.2,
                'Tops': 0.8,
                'Bottoms': 1.0,
                'Dresses': 1.1,
                'Other': 0.9
            }
            category_factor = category_factors.get(category, 1.0)
            
            # Quality factor (based on review quality if available)
            quality_score = product_data.get('quality_score', 0.5)
            quality_factor = 0.9 + quality_score * 0.2  # Range: 0.9 to 1.1
            
            # Calculate predicted price
            predicted_price = base_price * rating_factor * brand_premium * category_factor * quality_factor
            
            # Confidence intervals (±20%)
            lower_bound = predicted_price * 0.8
            upper_bound = predicted_price * 1.2
            
            return {
                'predicted_price': round(predicted_price, 2),
                'price_range': {
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                },
                'factors': {
                    'base_price': base_price,
                    'rating_factor': round(rating_factor, 2),
                    'brand_premium': round(brand_premium, 2),
                    'category_factor': round(category_factor, 2),
                    'quality_factor': round(quality_factor, 2)
                }
            }
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def analyze_value_for_money(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze value for money for all products
        
        Args:
            df: Review dataframe
            
        Returns:
            DataFrame with value analysis
        """
        try:
            product_features = self.extract_price_features(df)
            
            value_analysis = []
            
            for idx, product in product_features.iterrows():
                # Calculate value score
                rating_score = product['Avg_Overall_Rating'] / 5.0  # Normalize to 0-1
                price_score = 1 / (1 + product['Avg_Price'] / 1000)  # Lower price = higher score
                popularity_score = min(1.0, product['Review_Count'] / 100)  # Cap at 100 reviews
                
                # Weighted value score
                value_score = (rating_score * 0.4 + price_score * 0.4 + popularity_score * 0.2)
                
                # Price efficiency (rating per 1000 rupees)
                price_efficiency = product['Avg_Overall_Rating'] / (product['Avg_Price'] / 1000)
                
                value_analysis.append({
                    'Product_Name': product['Product_Name'],
                    'Price': product['Avg_Price'],
                    'Rating': product['Avg_Overall_Rating'],
                    'Review_Count': product['Review_Count'],
                    'Value_Score': round(value_score, 3),
                    'Price_Efficiency': round(price_efficiency, 2),
                    'Value_Category': self._categorize_value(value_score)
                })
            
            value_df = pd.DataFrame(value_analysis)
            return value_df.sort_values('Value_Score', ascending=False)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _categorize_value(self, value_score: float) -> str:
        """Categorize value based on score"""
        if value_score >= 0.8:
            return 'Excellent Value'
        elif value_score >= 0.6:
            return 'Good Value'
        elif value_score >= 0.4:
            return 'Average Value'
        else:
            return 'Poor Value'
    
    def get_price_insights(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive price insights
        
        Args:
            df: Review dataframe
            
        Returns:
            Dictionary with price insights
        """
        try:
            product_features = self.extract_price_features(df)
            value_analysis = self.analyze_value_for_money(df)
            
            insights = {
                'price_statistics': {
                    'average_price': round(product_features['Avg_Price'].mean(), 2),
                    'median_price': round(product_features['Avg_Price'].median(), 2),
                    'price_range': {
                        'min': round(product_features['Avg_Price'].min(), 2),
                        'max': round(product_features['Avg_Price'].max(), 2)
                    },
                    'price_std': round(product_features['Avg_Price'].std(), 2)
                },
                'value_distribution': value_analysis['Value_Category'].value_counts().to_dict(),
                'best_value_products': value_analysis.head(3)[['Product_Name', 'Value_Score', 'Price_Efficiency']].to_dict('records'),
                'brand_premiums': self.brand_premiums,
                'price_rating_correlation': round(np.corrcoef(product_features['Avg_Price'], product_features['Avg_Overall_Rating'])[0,1], 3)
            }
            
            return insights
            
        except Exception as e:
            raise CustomException(e, sys)
