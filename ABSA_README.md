# Aspect-Based Sentiment Analysis (ABSA) Implementation

## Overview

This implementation provides comprehensive Aspect-Based Sentiment Analysis (ABSA) for product reviews. It can analyze reviews across multiple aspects like Style/Design, Quality/Material, Size/Fit, Price/Value, and Delivery/Service, providing detailed sentiment analysis for each aspect.

## Features

### 🎯 Core ABSA Functionality
- **Aspect Extraction**: Automatically detects product aspects mentioned in reviews
- **Sentiment Classification**: Classifies sentiment for each detected aspect
- **Multiple Models**: Supports both VADER (fast) and Transformer models (accurate)
- **Comprehensive Analysis**: Provides detailed statistics and insights

### 📊 Supported Aspects
1. **Style/Design**: Appearance, color, aesthetics, fashion
2. **Quality/Material**: Durability, construction, materials used
3. **Size/Fit**: Sizing accuracy, comfort, fitting
4. **Price/Value**: Cost effectiveness, value for money
5. **Delivery/Service**: Shipping, packaging, customer service

## Implementation Structure

```
src/absa/
├── __init__.py                 # Module initialization
├── absa_analyzer.py           # Core ABSA analyzer (VADER-based)
└── advanced_absa.py           # Advanced analyzer (Transformer-based)
```

## Usage Examples

### Basic ABSA Analysis

```python
from src.absa import ABSAAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = ABSAAnalyzer()

# Analyze reviews (DataFrame with 'Comment' column required)
results = analyzer.analyze_product_aspects(reviews_df, product_name="Your Product")

# Access results
print(f"Overall sentiment: {results['overall_avg_sentiment']}")
print(f"Most mentioned aspects: {results['most_mentioned_aspects']}")
```

### Advanced ABSA Analysis

```python
from src.absa import AdvancedABSAAnalyzer

# Initialize advanced analyzer
advanced_analyzer = AdvancedABSAAnalyzer(use_transformers=True)

# Perform advanced analysis
results = advanced_analyzer.analyze_product_aspects_advanced(
    reviews_df, 
    product_name="Your Product"
)

# Access enhanced results
print(f"Analysis method: {results['analysis_method']}")
print(f"Model usage: {results['model_usage']}")
print(f"Confidence stats: {results['confidence_stats']}")
```

### Streamlit Integration

The ABSA functionality is integrated into the Streamlit app via the `pages/generate_analysis.py` file:

1. **General Analysis Tab**: Traditional dashboard with charts
2. **Aspect-Based Analysis Tab**: ABSA configuration and execution
3. **ABSA Insights Tab**: Detailed visualizations and downloadable results

## Analysis Output Structure

### Enhanced Reviews DataFrame
- Original review data plus:
  - `detected_aspects`: Comma-separated list of detected aspects
  - `aspect_sentiments`: Corresponding sentiment labels
  - `avg_sentiment_score`: Average compound sentiment score

### Detailed ABSA DataFrame
Each row represents one aspect-sentiment pair:
- `review_index`: Index of the original review
- `original_review`: Original review text
- `aspect`: Detected aspect category
- `sentiment_label`: Positive/Negative/Neutral
- `positive_score`, `negative_score`, `neutral_score`: Individual sentiment scores
- `compound_score`: Overall sentiment score (-1 to +1)
- `model_used`: Which model was used for analysis

### Aspect Summary DataFrame
Aggregated statistics:
- `aspect`: Aspect category
- `sentiment_label`: Sentiment type
- `count`: Number of mentions
- `avg_sentiment_score`: Average sentiment score
- `percentage`: Percentage within aspect category

## Visualization Features

### 📈 Charts and Graphs
- **Aspect Distribution Pie Chart**: Shows which aspects are mentioned most
- **Sentiment Bar Chart**: Sentiment distribution across aspects
- **Sentiment Heatmap**: Average sentiment scores by aspect and sentiment type
- **Metrics Dashboard**: Key statistics and confidence scores

### 📥 Export Options
- Enhanced reviews as CSV
- Detailed ABSA results as CSV
- Aspect summaries for further analysis

## Model Comparison

### VADER Sentiment Analysis
- **Pros**: Fast, lightweight, no external dependencies
- **Cons**: Rule-based, less context-aware
- **Best for**: Quick analysis, real-time processing

### Transformer Models
- **Pros**: Context-aware, higher accuracy, AI-powered
- **Cons**: Slower, requires more memory, internet for first download
- **Best for**: Detailed analysis, research purposes

## Configuration and Customization

### Adding New Aspects
Edit the `aspect_keywords` dictionary in `absa_analyzer.py`:

```python
self.aspect_keywords = {
    'Your_New_Aspect': [
        'keyword1', 'keyword2', 'keyword3'
    ]
}
```

### Adjusting Sentiment Thresholds
Modify the sentiment classification logic in `classify_sentiment()`:

```python
if scores['compound'] >= 0.05:  # Adjust threshold
    sentiment_label = 'Positive'
elif scores['compound'] <= -0.05:  # Adjust threshold
    sentiment_label = 'Negative'
```

## Performance Considerations

### Memory Usage
- VADER: Minimal memory footprint
- Transformers: ~1-2GB for model loading

### Processing Speed
- VADER: ~100-500 reviews/second
- Transformers: ~10-50 reviews/second

### Recommendations
- Use VADER for real-time analysis or large datasets
- Use Transformers for detailed analysis of smaller datasets
- Consider batching for large datasets with Transformers

## Error Handling

The implementation includes comprehensive error handling:
- Graceful fallback from Transformers to VADER
- Missing data handling
- Invalid input validation
- Custom exception reporting via `CustomException`

## Dependencies

### Required Packages
```bash
nltk>=3.8.1
pandas>=1.5.0
numpy>=1.20.0
```

### Optional Packages (for Transformers)
```bash
transformers>=4.30.0
torch>=1.13.0
```

### Installation
```bash
pip install nltk pandas numpy
# For advanced features:
pip install transformers torch
```

## Demo Scripts

### Basic Demo
```bash
python absa_demo.py
```
Demonstrates basic ABSA functionality with sample data.

### Advanced Demo
```bash
python advanced_absa_demo.py
```
Compares VADER vs Transformer analysis methods.

## Integration with Existing App

### Streamlit App
The ABSA functionality is integrated into the existing Streamlit app:
- Navigate to the "Generate Analysis" page
- Choose between General Analysis and ABSA
- Select analysis method (VADER/Transformer)
- View comprehensive results and insights

### Flask App
Can be integrated into the Flask app by modifying `application.py`:

```python
from src.absa import ABSAAnalyzer

@app.route("/absa", methods=['POST'])
def absa_analysis():
    analyzer = ABSAAnalyzer()
    # ... integration code
```

## Future Enhancements

### Potential Improvements
1. **Custom Aspect Training**: Train models on domain-specific aspects
2. **Multi-language Support**: Extend to other languages
3. **Real-time Analysis**: Stream processing capabilities
4. **Aspect Relationships**: Analyze how aspects relate to each other
5. **Trend Analysis**: Track sentiment changes over time

### API Development
Consider creating REST API endpoints for:
- Single review analysis
- Batch review processing
- Model comparison
- Aspect customization

## Troubleshooting

### Common Issues

1. **NLTK Data Not Found**
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

2. **Transformer Model Loading Failed**
   - Check internet connection
   - Verify disk space (>2GB required)
   - Falls back to VADER automatically

3. **Memory Issues with Large Datasets**
   - Use VADER for datasets >1000 reviews
   - Process in batches
   - Consider streaming analysis

### Performance Optimization
- Cache model loading for repeated analysis
- Use multiprocessing for large datasets
- Implement lazy loading for transformers

## Conclusion

This ABSA implementation provides a robust, scalable solution for analyzing product reviews across multiple aspects and sentiment dimensions. It offers flexibility in model choice, comprehensive analysis capabilities, and seamless integration with existing applications.

The modular design allows for easy customization and extension, making it suitable for various e-commerce and review analysis use cases.
