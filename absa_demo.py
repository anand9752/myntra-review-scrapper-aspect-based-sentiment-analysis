"""
Demo script for Aspect-Based Sentiment Analysis (ABSA)
This script demonstrates how to use the ABSA functionality with sample data.
"""

import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.absa import ABSAAnalyzer

def create_sample_data():
    """Create sample review data for testing ABSA functionality."""
    sample_reviews = [
        {
            'Product Name': 'Men\'s White Running Shoes',
            'Over_All_Rating': '4.2',
            'Price': '₹2999',
            'Date': '2024-01-15',
            'Rating': '5',
            'Name': 'John Doe',
            'Comment': 'Great quality shoes! The design is very stylish and modern. Fits perfectly and very comfortable for running. Worth the price!'
        },
        {
            'Product Name': 'Men\'s White Running Shoes',
            'Over_All_Rating': '4.2',
            'Price': '₹2999',
            'Date': '2024-01-16',
            'Rating': '2',
            'Name': 'Jane Smith',
            'Comment': 'Poor quality material. The shoes started tearing after just one week. Very disappointed with the durability. Overpriced for such cheap quality.'
        },
        {
            'Product Name': 'Men\'s White Running Shoes',
            'Over_All_Rating': '4.2',
            'Price': '₹2999',
            'Date': '2024-01-17',
            'Rating': '4',
            'Name': 'Mike Johnson',
            'Comment': 'Good looking shoes with nice design. Size fits well. Delivery was fast and packaging was good. Price is reasonable for the style.'
        },
        {
            'Product Name': 'Men\'s White Running Shoes',
            'Over_All_Rating': '4.2',
            'Price': '₹2999',
            'Date': '2024-01-18',
            'Rating': '3',
            'Name': 'Sarah Wilson',
            'Comment': 'Average quality. The color is nice but the material feels cheap. Size runs a bit small. Customer service was helpful though.'
        },
        {
            'Product Name': 'Men\'s White Running Shoes',
            'Over_All_Rating': '4.2',
            'Price': '₹2999',
            'Date': '2024-01-19',
            'Rating': '5',
            'Name': 'Alex Brown',
            'Comment': 'Excellent shoes! Beautiful design, perfect fit, great quality construction. Fast delivery and well packaged. Highly recommend!'
        }
    ]
    
    return pd.DataFrame(sample_reviews)

def run_absa_demo():
    """Run the ABSA demo with sample data."""
    print("🎯 Aspect-Based Sentiment Analysis (ABSA) Demo")
    print("=" * 50)
    
    # Create sample data
    print("📝 Creating sample review data...")
    review_data = create_sample_data()
    print(f"✅ Created {len(review_data)} sample reviews")
    print()
    
    # Initialize ABSA analyzer
    print("🔧 Initializing ABSA Analyzer...")
    absa_analyzer = ABSAAnalyzer()
    print("✅ ABSA Analyzer initialized successfully")
    print()
    
    # Perform analysis
    print("🔍 Performing Aspect-Based Sentiment Analysis...")
    analysis_results = absa_analyzer.analyze_product_aspects(
        review_data, 
        product_name="Men's White Running Shoes"
    )
    print("✅ Analysis completed!")
    print()
    
    # Display results
    print("📊 ANALYSIS RESULTS")
    print("-" * 30)
    print(f"Product: {analysis_results['product_name']}")
    print(f"Total Reviews: {analysis_results['total_reviews']}")
    print(f"Overall Sentiment Score: {analysis_results['overall_avg_sentiment']:.3f}")
    
    # Interpret overall sentiment
    if analysis_results['overall_avg_sentiment'] >= 0.05:
        print("Overall Sentiment: 😊 Positive")
    elif analysis_results['overall_avg_sentiment'] <= -0.05:
        print("Overall Sentiment: 😞 Negative")
    else:
        print("Overall Sentiment: 😐 Neutral")
    print()
    
    # Most mentioned aspects
    print("🎯 MOST MENTIONED ASPECTS")
    print("-" * 30)
    for i, (aspect, count) in enumerate(analysis_results['most_mentioned_aspects'].items(), 1):
        print(f"{i}. {aspect}: {count} mentions")
    print()
    
    # Aspect summary
    print("📈 ASPECT-WISE SENTIMENT SUMMARY")
    print("-" * 40)
    aspect_summary = analysis_results['aspect_summary']
    
    for aspect in aspect_summary['aspect'].unique():
        print(f"\n{aspect}:")
        aspect_data = aspect_summary[aspect_summary['aspect'] == aspect]
        
        for _, row in aspect_data.iterrows():
            emoji = "✅" if row['sentiment_label'] == 'Positive' else "❌" if row['sentiment_label'] == 'Negative' else "⚪"
            print(f"  {emoji} {row['sentiment_label']}: {row['count']} ({row['percentage']:.1f}%) - Avg Score: {row['avg_sentiment_score']:.3f}")
    
    # Sample reviews for each sentiment
    print("\n📝 SAMPLE REVIEWS BY ASPECT")
    print("-" * 35)
    detailed_absa = analysis_results['detailed_absa']
    
    for aspect in ['Style/Design', 'Quality/Material', 'Price/Value']:
        if aspect in detailed_absa['aspect'].values:
            print(f"\n{aspect}:")
            aspect_reviews = detailed_absa[detailed_absa['aspect'] == aspect]
            
            for sentiment in ['Positive', 'Negative']:
                sentiment_reviews = aspect_reviews[aspect_reviews['sentiment_label'] == sentiment]
                if not sentiment_reviews.empty:
                    sample_review = sentiment_reviews.iloc[0]
                    emoji = "😊" if sentiment == 'Positive' else "😞"
                    print(f"  {emoji} {sentiment} (Score: {sample_review['compound_score']:.3f}):")
                    print(f"    \"{sample_review['original_review'][:100]}...\"")
    
    print("\n" + "=" * 50)
    print("✨ ABSA Demo completed successfully!")
    
    return analysis_results

if __name__ == "__main__":
    try:
        results = run_absa_demo()
        print("\n💾 You can now use these results in your Streamlit app!")
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")
        import traceback
        traceback.print_exc()
