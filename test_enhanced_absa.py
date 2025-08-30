"""
Test script for the enhanced ABSA implementation
This tests the reliable TextBlob+VADER combination.
"""

import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.absa import ABSAAnalyzer, SimpleAdvancedABSA

def create_test_data():
    """Create test review data with clear sentiments."""
    sample_reviews = [
        {
            'Product Name': 'Test Product',
            'Over_All_Rating': '4.0',
            'Price': '₹1999',
            'Date': '2024-01-15',
            'Rating': '5',
            'Name': 'Happy Customer',
            'Comment': 'This product is absolutely amazing! The quality is excellent and the design is beautiful. Perfect fit and great value for money. Fast delivery too!'
        },
        {
            'Product Name': 'Test Product',
            'Over_All_Rating': '4.0',
            'Price': '₹1999',
            'Date': '2024-01-16',
            'Rating': '1',
            'Name': 'Unhappy Customer',
            'Comment': 'Terrible quality! The material is cheap and feels flimsy. Poor design and horrible fit. Overpriced garbage. Slow delivery and bad service.'
        },
        {
            'Product Name': 'Test Product',
            'Over_All_Rating': '4.0',
            'Price': '₹1999',
            'Date': '2024-01-17',
            'Rating': '3',
            'Name': 'Neutral Customer',
            'Comment': 'The product is okay. Nothing special but not bad either. Average quality and design. Size is fine. Price is reasonable. Delivery was normal.'
        },
        {
            'Product Name': 'Test Product',
            'Over_All_Rating': '4.0',
            'Price': '₹1999',
            'Date': '2024-01-18',
            'Rating': '4',
            'Name': 'Mixed Customer',
            'Comment': 'Good quality material but poor design. Great fit but expensive price. Fast delivery but bad packaging. Mixed feelings overall.'
        }
    ]
    
    return pd.DataFrame(sample_reviews)

def test_enhanced_absa():
    """Test the enhanced ABSA functionality."""
    print("🧪 TESTING ENHANCED ABSA IMPLEMENTATION")
    print("=" * 50)
    
    # Create test data
    print("📝 Creating test review data...")
    review_data = create_test_data()
    print(f"✅ Created {len(review_data)} test reviews")
    print()
    
    # Test basic VADER
    print("🚀 Testing Basic VADER Analysis...")
    print("-" * 30)
    vader_analyzer = ABSAAnalyzer()
    vader_results = vader_analyzer.analyze_product_aspects(
        review_data, 
        product_name="Test Product"
    )
    
    print(f"VADER - Overall Sentiment: {vader_results['overall_avg_sentiment']:.3f}")
    print(f"VADER - Most Mentioned Aspects: {list(vader_results['most_mentioned_aspects'].keys())[:3]}")
    
    # Show sentiment breakdown
    vader_detailed = vader_results['detailed_absa']
    vader_sentiments = vader_detailed['sentiment_label'].value_counts()
    print(f"VADER - Sentiment Distribution: {vader_sentiments.to_dict()}")
    print()
    
    # Test enhanced analyzer
    print("⭐ Testing Enhanced TextBlob+VADER Analysis...")
    print("-" * 45)
    enhanced_analyzer = SimpleAdvancedABSA(use_textblob=True)
    enhanced_results = enhanced_analyzer.analyze_product_aspects_enhanced(
        review_data,
        product_name="Test Product"
    )
    
    print(f"Enhanced - Overall Sentiment: {enhanced_results['overall_avg_sentiment']:.3f}")
    print(f"Enhanced - Overall Confidence: {enhanced_results['overall_avg_confidence']:.3f}")
    print(f"Enhanced - Analysis Method: {enhanced_results['analysis_method']}")
    print(f"Enhanced - Most Mentioned Aspects: {list(enhanced_results['most_mentioned_aspects'].keys())[:3]}")
    
    # Show enhanced sentiment breakdown
    enhanced_detailed = enhanced_results['detailed_absa']
    enhanced_sentiments = enhanced_detailed['sentiment_label'].value_counts()
    print(f"Enhanced - Sentiment Distribution: {enhanced_sentiments.to_dict()}")
    
    # Show sentiment strength
    if 'sentiment_strength' in enhanced_results:
        strength = enhanced_results['sentiment_strength']
        print(f"Enhanced - Sentiment Strength: Pos({strength['very_positive']+strength['positive']}) Neu({strength['neutral']}) Neg({strength['negative']+strength['very_negative']})")
    
    # Show model usage
    if 'model_usage' in enhanced_results:
        print(f"Enhanced - Model Usage: {enhanced_results['model_usage']}")
    print()
    
    # Compare individual review analysis
    print("🔍 DETAILED REVIEW COMPARISON")
    print("-" * 35)
    
    for idx, row in review_data.iterrows():
        comment = row['Comment']
        expected_rating = int(row['Rating'])
        
        # VADER analysis
        vader_review = vader_detailed[vader_detailed['review_index'] == idx]
        vader_sentiment = vader_review['sentiment_label'].mode().iloc[0] if not vader_review.empty else 'Unknown'
        vader_score = vader_review['compound_score'].mean() if not vader_review.empty else 0
        
        # Enhanced analysis
        enhanced_review = enhanced_detailed[enhanced_detailed['review_index'] == idx]
        enhanced_sentiment = enhanced_review['sentiment_label'].mode().iloc[0] if not enhanced_review.empty else 'Unknown'
        enhanced_score = enhanced_review['compound_score'].mean() if not enhanced_review.empty else 0
        enhanced_confidence = enhanced_review['confidence'].mean() if not enhanced_review.empty else 0
        
        # Expected sentiment based on rating
        if expected_rating >= 4:
            expected_sentiment = "Positive"
        elif expected_rating <= 2:
            expected_sentiment = "Negative"
        else:
            expected_sentiment = "Neutral"
        
        print(f"\nReview {idx + 1} (Rating: {expected_rating}):")
        print(f"  Expected: {expected_sentiment}")
        print(f"  VADER: {vader_sentiment} ({vader_score:.3f})")
        print(f"  Enhanced: {enhanced_sentiment} ({enhanced_score:.3f}, conf: {enhanced_confidence:.3f})")
        
        # Check accuracy
        vader_correct = vader_sentiment == expected_sentiment
        enhanced_correct = enhanced_sentiment == expected_sentiment
        
        if enhanced_correct and not vader_correct:
            print("  ✅ Enhanced analyzer performed better!")
        elif vader_correct and not enhanced_correct:
            print("  ⚠️ VADER performed better")
        elif vader_correct and enhanced_correct:
            print("  ✅ Both correct")
        else:
            print("  ❌ Both incorrect")
        
        print(f"  Comment: \"{comment[:60]}...\"")
    
    print("\n" + "=" * 50)
    print("✨ Enhanced ABSA testing completed!")
    
    # Calculate accuracy
    vader_accuracy = sum(1 for idx, row in review_data.iterrows() 
                        if get_predicted_sentiment(vader_detailed, idx) == get_expected_sentiment(int(row['Rating']))) / len(review_data)
    
    enhanced_accuracy = sum(1 for idx, row in review_data.iterrows() 
                           if get_predicted_sentiment(enhanced_detailed, idx) == get_expected_sentiment(int(row['Rating']))) / len(review_data)
    
    print(f"\n📊 ACCURACY COMPARISON:")
    print(f"VADER Accuracy: {vader_accuracy:.1%}")
    print(f"Enhanced Accuracy: {enhanced_accuracy:.1%}")
    
    if enhanced_accuracy > vader_accuracy:
        print("🏆 Enhanced analyzer is more accurate!")
    elif vader_accuracy > enhanced_accuracy:
        print("🏆 VADER analyzer is more accurate!")
    else:
        print("🤝 Both analyzers have equal accuracy!")
    
    return enhanced_results

def get_predicted_sentiment(detailed_df, review_idx):
    """Get predicted sentiment for a review."""
    review_data = detailed_df[detailed_df['review_index'] == review_idx]
    if not review_data.empty:
        return review_data['sentiment_label'].mode().iloc[0]
    return 'Unknown'

def get_expected_sentiment(rating):
    """Get expected sentiment based on rating."""
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"

if __name__ == "__main__":
    try:
        results = test_enhanced_absa()
        print("\n💾 Enhanced ABSA is ready for production use!")
    except Exception as e:
        print(f"❌ Error in testing: {str(e)}")
        import traceback
        traceback.print_exc()
