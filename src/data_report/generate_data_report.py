import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os, sys
from src.exception import CustomException


class DashboardGenerator:
    def __init__(self, data):
        self.data = data
        # Set up custom CSS for better styling
        self._apply_custom_css()

    def _apply_custom_css(self):
        """Apply custom CSS for better UI"""
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 0.5rem 0;
        }
        
        .product-card {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border: 1px solid #e1e5e9;
        }
        
        .review-item {
            background-color: #f8f9fa;
            padding: 0.8rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 3px solid #28a745;
        }
        
        .negative-review {
            border-left-color: #dc3545 !important;
        }
        
        .section-header {
            color: #1f77b4;
            font-size: 1.5rem;
            font-weight: 600;
            margin: 1rem 0;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 0.5rem;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: -5px;
        }
        
        .rating-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: bold;
            margin: 0.2rem;
        }
        
        .rating-excellent {
            background-color: #28a745;
            color: white;
        }
        
        .rating-good {
            background-color: #17a2b8;
            color: white;
        }
        
        .rating-average {
            background-color: #ffc107;
            color: black;
        }
        
        .rating-poor {
            background-color: #dc3545;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    def display_general_info(self):
        """Display improved general information with better layout"""
        st.markdown('<h1 class="section-header">📊 Dashboard Overview</h1>', unsafe_allow_html=True)

        # Data preprocessing
        self.data['Over_All_Rating'] = pd.to_numeric(self.data['Over_All_Rating'], errors='coerce')
        # Clean and convert price data with improved handling for concatenated prices
        def clean_price_data(price_value):
            """Clean concatenated price data"""
            if pd.isna(price_value):
                return np.nan
            
            price_str = str(price_value).strip()
            
            # Handle concatenated prices like ₹919₹919₹919...
            if price_str.count('₹') > 1:
                import re
                # Extract first price value
                price_match = re.search(r'₹(\d+)', price_str)
                if price_match:
                    return float(price_match.group(1))
                else:
                    # Fallback: extract first numeric sequence
                    numbers_only = re.sub(r'[^\d]', '', price_str)
                    if numbers_only:
                        # Estimate original price length (typically 3-5 digits for clothing)
                        estimated_length = min(4, len(numbers_only) // price_str.count('₹'))
                        return float(numbers_only[:estimated_length]) if estimated_length > 0 else np.nan
                    return np.nan
            else:
                # Regular price cleaning
                cleaned = price_str.replace('₹', '').replace(',', '').strip()
                try:
                    return float(cleaned) if cleaned else np.nan
                except ValueError:
                    return np.nan
        
        self.data['Price'] = self.data['Price'].apply(clean_price_data)
        self.data["Rating"] = pd.to_numeric(self.data['Rating'], errors='coerce')

        # Key metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products = self.data['Product Name'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{total_products}</div>
                <div class="stat-label">Total Products</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_reviews = len(self.data)
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{total_reviews}</div>
                <div class="stat-label">Total Reviews</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_rating = self.data['Over_All_Rating'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{avg_rating:.1f}⭐</div>
                <div class="stat-label">Average Rating</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_price = self.data['Price'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">₹{avg_price:.0f}</div>
                <div class="stat-label">Average Price</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Charts in responsive layout
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Enhanced pie chart
            product_ratings = self.data.groupby('Product Name', as_index=False)['Over_All_Rating'].mean().dropna()
            
            fig_pie = px.pie(
                product_ratings, 
                values='Over_All_Rating', 
                names='Product Name',
                title='🎯 Average Ratings by Product',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_layout(
                height=400,
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5),
                title_font_size=16,
                title_x=0.5
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            # Enhanced bar chart
            avg_prices = self.data.groupby('Product Name', as_index=False)['Price'].mean().dropna()
            
            fig_bar = px.bar(
                avg_prices, 
                x='Product Name', 
                y='Price', 
                color='Price',
                title='💰 Price Comparison Between Products',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(
                height=400,
                xaxis_tickangle=-45,
                title_font_size=16,
                title_x=0.5,
                xaxis_title="Product Name",
                yaxis_title="Average Price (₹)"
            )
            fig_bar.update_traces(texttemplate='₹%{y:.0f}', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        # Additional analytics
        st.markdown('<h2 class="section-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
        
        # Rating distribution
        rating_dist_col1, rating_dist_col2 = st.columns(2)
        
        with rating_dist_col1:
            # Rating distribution histogram
            fig_hist = px.histogram(
                self.data, 
                x='Rating', 
                nbins=10, 
                title='📊 Rating Distribution',
                color_discrete_sequence=['#1f77b4']
            )
            fig_hist.update_layout(
                height=350,
                title_font_size=16,
                title_x=0.5,
                xaxis_title="Rating",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with rating_dist_col2:
            # Price vs Rating scatter plot
            fig_scatter = px.scatter(
                self.data, 
                x='Price', 
                y='Over_All_Rating',
                color='Product Name',
                title='💎 Price vs Rating Analysis',
                size_max=10
            )
            fig_scatter.update_layout(
                height=350,
                title_font_size=16,
                title_x=0.5,
                xaxis_title="Price (₹)",
                yaxis_title="Overall Rating"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

    def display_product_sections(self):
        """Display product sections with improved responsive layout"""
        st.markdown('<h1 class="section-header">🛍️ Product Analysis</h1>', unsafe_allow_html=True)

        product_names = self.data['Product Name'].unique()
        
        # Display products in a responsive grid (max 2 per row)
        for i in range(0, len(product_names), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(product_names):
                    product_name = product_names[i + j]
                    product_data = self.data[self.data['Product Name'] == product_name]
                    
                    with col:
                        self._display_product_card(product_name, product_data)

    def _display_product_card(self, product_name, product_data):
        """Display individual product card with improved styling"""
        
        # Product header
        st.markdown(f"""
        <div class="product-card">
            <h3 style="color: #1f77b4; margin-bottom: 1rem;">🏷️ {product_name}</h3>
        """, unsafe_allow_html=True)
        
        # Product metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_price = product_data['Price'].mean()
            st.metric("💰 Avg Price", f"₹{avg_price:.0f}")
        
        with col2:
            avg_rating = product_data['Over_All_Rating'].mean()
            st.metric("⭐ Avg Rating", f"{avg_rating:.1f}/5")
        
        with col3:
            review_count = len(product_data)
            st.metric("📝 Reviews", review_count)

        # Rating distribution badges
        st.markdown("**📊 Rating Distribution:**")
        rating_counts = product_data['Rating'].value_counts().sort_index(ascending=False)
        
        rating_html = ""
        for rating, count in rating_counts.items():
            if rating >= 4:
                badge_class = "rating-excellent"
            elif rating >= 3:
                badge_class = "rating-good"
            elif rating >= 2:
                badge_class = "rating-average"
            else:
                badge_class = "rating-poor"
            
            rating_html += f'<span class="rating-badge {badge_class}">{rating}⭐: {count}</span>'
        
        st.markdown(rating_html, unsafe_allow_html=True)

        # Reviews sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✨ Top Positive Reviews**")
            positive_reviews = product_data[product_data['Rating'] >= 4].nlargest(3, 'Rating')
            
            if not positive_reviews.empty:
                for _, row in positive_reviews.iterrows():
                    comment_preview = str(row['Comment'])[:100] + "..." if len(str(row['Comment'])) > 100 else str(row['Comment'])
                    st.markdown(f"""
                    <div class="review-item">
                        <strong>{row['Rating']}⭐</strong><br>
                        <small>{comment_preview}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No positive reviews found")
        
        with col2:
            st.markdown("**💢 Critical Reviews**")
            negative_reviews = product_data[product_data['Rating'] <= 2].nsmallest(3, 'Rating')
            
            if not negative_reviews.empty:
                for _, row in negative_reviews.iterrows():
                    comment_preview = str(row['Comment'])[:100] + "..." if len(str(row['Comment'])) > 100 else str(row['Comment'])
                    st.markdown(f"""
                    <div class="review-item negative-review">
                        <strong>{row['Rating']}⭐</strong><br>
                        <small>{comment_preview}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No critical reviews found")

        # Product-specific chart
        if len(product_data) > 1:
            # Time series of ratings (if date available)
            if 'Date' in product_data.columns:
                try:
                    product_data['Date'] = pd.to_datetime(product_data['Date'], errors='coerce')
                    daily_ratings = product_data.groupby('Date')['Rating'].mean().reset_index()
                    
                    if len(daily_ratings) > 1:
                        fig_line = px.line(
                            daily_ratings, 
                            x='Date', 
                            y='Rating',
                            title=f'📈 Rating Trend for {product_name}',
                            markers=True
                        )
                        fig_line.update_layout(height=250, title_font_size=14)
                        st.plotly_chart(fig_line, use_container_width=True)
                except:
                    pass
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
