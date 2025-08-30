"""
Advanced ABSA Demo using HuggingFace Transformers
This script demonstrates the enhanced ABSA functionality with transformer models.
"""

import pandas as pd
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.absa import ABSAAnalyzer, AdvancedABSAAnalyzer

def create_complex_sample_data():
    """Create more complex sample review data for testing advanced ABSA functionality."""
    sample_reviews = [
        {
            'Product Name': 'Premium Leather Jacket',
            'Over_All_Rating': '4.1',
            'Price': '₹8999',
            'Date': '2024-01-15',
            'Rating': '5',
            'Name': 'John Doe',
            'Comment': 'Absolutely love this jacket! The leather quality is outstanding and feels premium. The design is modern and stylish. Perfect fit and very comfortable. Worth every penny despite the high price. Fast delivery too!'
        },
        {
            'Product Name': 'Premium Leather Jacket',
            'Over_All_Rating': '4.1',
            'Price': '₹8999',
            'Date': '2024-01-16',
            'Rating': '2',
            'Name': 'Jane Smith',
            'Comment': 'Very disappointed with this purchase. The leather looks cheap and feels synthetic. Poor stitching quality. Size runs small - ordered Large but fits like Medium. Overpriced for such poor quality. Took forever to arrive and packaging was damaged.'
        },
        {
            'Product Name': 'Premium Leather Jacket',
            'Over_All_Rating': '4.1',
            'Price': '₹8999',
            'Date': '2024-01-17',
            'Rating': '4',
            'Name': 'Mike Johnson',
            'Comment': 'Great looking jacket with excellent style. The color is exactly as shown. Material feels good but not as premium as expected for the price. Fits well and comfortable to wear. Delivery was quick and well packaged.'
        },
        {
            'Product Name': 'Premium Leather Jacket',
            'Over_All_Rating': '4.1',
            'Price': '₹8999',
            'Date': '2024-01-18',
            'Rating': '3',
            'Name': 'Sarah Wilson',
            'Comment': 'Average quality jacket. The design is nice and trendy but the material could be better. Size is true to fit. Price seems a bit high for what you get. Customer service was responsive when I had questions.'
        },
        {
            'Product Name': 'Premium Leather Jacket',
            'Over_All_Rating': '4.1',
            'Price': '₹8999',
            'Date': '2024-01-19',
            'Rating': '5',
            'Name': 'Alex Brown',
            'Comment': 'Fantastic jacket! Premium leather material, beautiful craftsmanship, perfect fit. The style is timeless and looks expensive. Great value for money. Super fast delivery and excellent packaging. Highly recommended!'
        },
        {
            'Product Name': 'Premium Leather Jacket',
            'Over_All_Rating': '4.1',
            'Price': '₹8999',
            'Date': '2024-01-20',
            'Rating': '1',
            'Name': 'Tom Davis',
            'Comment': 'Terrible quality! The jacket started peeling after just one week. Cheap material that looks nothing like real leather. Poor construction and loose threads everywhere. Size is completely wrong. Worst purchase ever. Slow delivery and poor customer service.'
        }
    ]
    
    return pd.DataFrame(sample_reviews)

def compare_analysis_methods():
    """Compare VADER vs Transformer-based analysis."""
    print("🔬 COMPARATIVE ABSA ANALYSIS")
    print("=" * 60)
    
    # Create sample data
    print("📝 Creating complex sample review data...")
    review_data = create_complex_sample_data()
    print(f"✅ Created {len(review_data)} sample reviews")
    print()
    
    # Standard VADER Analysis
    print("🚀 Running VADER Analysis...")
    print("-" * 30)
    vader_analyzer = ABSAAnalyzer()
    vader_results = vader_analyzer.analyze_product_aspects(
        review_data, 
        product_name="Premium Leather Jacket"
    )
    
    print(f"VADER - Overall Sentiment Score: {vader_results['overall_avg_sentiment']:.3f}")
    print(f"VADER - Most Mentioned Aspects: {list(vader_results['most_mentioned_aspects'].keys())[:3]}")
    print()
    
    # Advanced Transformer Analysis  
    print("🤖 Running Advanced Transformer Analysis...")
    print("-" * 40)
    advanced_analyzer = AdvancedABSAAnalyzer(use_transformers=True)
    
    try:
        advanced_results = advanced_analyzer.analyze_product_aspects_advanced(
            review_data,
            product_name="Premium Leather Jacket"
        )
        
        print(f"Transformer - Overall Sentiment Score: {advanced_results['overall_avg_sentiment']:.3f}")
        print(f"Transformer - Analysis Method: {advanced_results.get('analysis_method', 'Unknown')}")
        print(f"Transformer - Most Mentioned Aspects: {list(advanced_results['most_mentioned_aspects'].keys())[:3]}")
        
        if 'model_usage' in advanced_results:
            print(f"Model Usage: {advanced_results['model_usage']}")
        
        if 'confidence_stats' in advanced_results:
            conf = advanced_results['confidence_stats']
            print(f"Confidence - Pos: {conf['avg_positive_confidence']:.3f}, Neg: {conf['avg_negative_confidence']:.3f}, Neu: {conf['avg_neutral_confidence']:.3f}")
        
    except Exception as e:
        print(f"⚠️ Transformer analysis failed: {e}")
        print("Falling back to VADER results for comparison")
        advanced_results = vader_results
    
    print()
    
    # Detailed Comparison
    print("📊 DETAILED COMPARISON")
    print("-" * 25)
    
    # Compare aspect sentiment distributions
    vader_detailed = vader_results['detailed_absa']
    advanced_detailed = advanced_results['detailed_absa']
    
    print("\n🎯 Aspect-wise Sentiment Comparison:")
    for aspect in ['Quality/Material', 'Style/Design', 'Price/Value', 'Size/Fit']:
        if aspect in vader_detailed['aspect'].values:
            # VADER results
            vader_aspect = vader_detailed[vader_detailed['aspect'] == aspect]
            vader_avg = vader_aspect['compound_score'].mean()
            vader_sentiment = vader_aspect['sentiment_label'].mode().iloc[0] if not vader_aspect.empty else 'N/A'
            
            # Advanced results
            advanced_aspect = advanced_detailed[advanced_detailed['aspect'] == aspect]
            advanced_avg = advanced_aspect['compound_score'].mean()
            advanced_sentiment = advanced_aspect['sentiment_label'].mode().iloc[0] if not advanced_aspect.empty else 'N/A'
            
            print(f"\n{aspect}:")
            print(f"  VADER      : {vader_sentiment} ({vader_avg:.3f})")
            print(f"  Transformer: {advanced_sentiment} ({advanced_avg:.3f})")
            
            # Show difference
            diff = advanced_avg - vader_avg
            if abs(diff) > 0.1:
                print(f"  📈 Significant difference: {diff:+.3f}")
    
    print("\n" + "=" * 60)
    print("✨ Comparative analysis completed!")
    
    return vader_results, advanced_results

def run_detailed_absa_demo():
    """Run a detailed ABSA demo showing all features."""
    print("🎯 DETAILED ASPECT-BASED SENTIMENT ANALYSIS DEMO")
    print("=" * 70)
    
    try:
        # Run comparative analysis
        vader_results, advanced_results = compare_analysis_methods()
        
        # Show detailed insights for advanced results
        print("\n🔍 DETAILED ADVANCED ANALYSIS INSIGHTS")
        print("-" * 45)
        
        detailed_absa = advanced_results['detailed_absa']
        
        # Show some interesting patterns
        print("\n📝 Sample Reviews with Different Sentiment Scores:")
        
        # Get most positive and most negative reviews
        most_positive = detailed_absa.loc[detailed_absa['compound_score'].idxmax()]
        most_negative = detailed_absa.loc[detailed_absa['compound_score'].idxmin()]
        
        print(f"\n😊 Most Positive ({most_positive['aspect']}):")
        print(f"Score: {most_positive['compound_score']:.3f}")
        print(f"Review: \"{most_positive['original_review'][:100]}...\"")
        
        print(f"\n😞 Most Negative ({most_negative['aspect']}):")
        print(f"Score: {most_negative['compound_score']:.3f}")
        print(f"Review: \"{most_negative['original_review'][:100]}...\"")
        
        # Show aspect extraction accuracy
        print("\n🎯 Aspect Detection Results:")
        aspect_counts = detailed_absa['aspect'].value_counts()
        for aspect, count in aspect_counts.items():
            print(f"  {aspect}: {count} detections")
        
        print("\n💾 Analysis complete! Ready for integration with Streamlit app.")
        
        return advanced_results
        
    except Exception as e:
        print(f"❌ Error in detailed demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        results = run_detailed_absa_demo()
        if results:
            print("\n🚀 Advanced ABSA implementation is ready!")
            print("You can now use both VADER and Transformer models in your Streamlit app.")
    except Exception as e:
        print(f"❌ Error running advanced demo: {str(e)}")
        import traceback
        traceback.print_exc()
