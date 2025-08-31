# 📊 ABSA Data Flow Analysis - Myntra Review Scrapper Project

## 🔄 Complete Data Flow Diagram

```
📱 USER INPUT
     ↓
🌐 STREAMLIT APP (app.py)
     ↓
📋 TWO INPUT METHODS
     ↓
┌─────────────────────────────────────────────────────────────────┐
│  METHOD 1: URL-BASED SCRAPING    │  METHOD 2: SEARCH-BASED      │
│                                   │                               │
│  1. User enters Myntra URL        │  1. User enters product name │
│     ↓                            │  2. User selects # products   │
│  2. validate_myntra_url()         │     ↓                        │
│     ↓                            │  3. ScrapeReviews class       │
│  3. scrape_reviews_from_url()     │     (src/scrapper/scrape.py)  │
│     (calls URLScrapeReviews)      │     ↓                        │
│     ↓                            │  4. Chrome WebDriver          │
│  4. URLScrapeReviews class        │  5. Web scraping process      │
│     (src/scrapper/url_scrape.py)  │  6. BeautifulSoup parsing     │
│     ↓                            │     ↓                        │
│  5. Chrome WebDriver              │  7. Returns DataFrame         │
│  6. Direct URL scraping           │                               │
│  7. BeautifulSoup parsing         │                               │
│     ↓                            │                               │
│  8. Returns DataFrame             │                               │
└─────────────────────────────────────────────────────────────────┘
     ↓
💾 DATA STORAGE & SESSION MANAGEMENT
     ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. Store in st.session_state["scraped_reviews"]                 │
│  2. Store product name in st.session_state[SESSION_PRODUCT_KEY]  │
│  3. Set st.session_state["data"] = True                         │
│  4. Save to MongoDB via MongoIO class                           │
│     (src/cloud_io/__init__.py)                                  │
└─────────────────────────────────────────────────────────────────┘
     ↓
📄 ANALYSIS PAGE (pages/generate_analysis.py)
     ↓
🎯 ABSA ANALYSIS FLOW
     ↓
┌─────────────────────────────────────────────────────────────────┐
│  USER SELECTS ANALYSIS METHOD:                                   │
│                                                                 │
│  🚀 VADER (Recommended)    🧪 Enhanced TextBlob+VADER    🤖 Transformer │
│       ↓                         ↓                           ↓    │
│  ABSAAnalyzer            SimpleAdvancedABSA         AdvancedABSAAnalyzer │
│  (absa_analyzer.py)      (simple_advanced_absa.py)  (advanced_absa.py)  │
└─────────────────────────────────────────────────────────────────┘
     ↓
🔍 ABSA PROCESSING PIPELINE
     ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. PREPROCESSING (preprocess_reviews)                           │
│     • Text cleaning and normalization                           │
│     • Remove special characters                                 │
│     • Convert to lowercase                                      │
│                                                                 │
│  2. ASPECT EXTRACTION (extract_aspects)                         │
│     • Pattern matching using keyword dictionaries               │
│     • Detect: Style/Design, Quality/Material, Size/Fit,        │
│       Price/Value, Delivery/Service                            │
│                                                                 │
│  3. SENTIMENT ANALYSIS (classify_sentiment)                     │
│     VADER: Uses rule-based sentiment analysis                   │
│     Enhanced: Combines VADER + TextBlob polarity               │
│     Transformer: Uses HuggingFace models (if available)        │
│                                                                 │
│  4. CONTEXT EXTRACTION (extract_aspect_context)                │
│     • Extract text around aspect mentions                       │
│     • Window-based context for better accuracy                  │
│                                                                 │
│  5. RESULT COMPILATION                                          │
│     • Create detailed ABSA DataFrame                           │
│     • Generate aspect summary statistics                        │
│     • Calculate confidence metrics                              │
└─────────────────────────────────────────────────────────────────┘
     ↓
📊 RESULTS & VISUALIZATION
     ↓
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT DATA STRUCTURES:                                         │
│                                                                 │
│  1. enhanced_reviews: Original data + ABSA columns             │
│     • detected_aspects                                          │
│     • aspect_sentiments                                         │
│     • avg_sentiment_score                                       │
│                                                                 │
│  2. detailed_absa: Row per aspect-sentiment pair               │
│     • review_index, aspect, sentiment_label                    │
│     • positive_score, negative_score, neutral_score            │
│     • compound_score, confidence                               │
│                                                                 │
│  3. aspect_summary: Aggregated statistics                      │
│     • count, percentage by aspect and sentiment                │
│     • avg_sentiment_score                                      │
│                                                                 │
│  4. analysis_results: Complete result dictionary               │
│     • most_mentioned_aspects                                   │
│     • model_usage, confidence_stats                            │
│     • sentiment_strength distribution                          │
└─────────────────────────────────────────────────────────────────┘
     ↓
🎨 STREAMLIT VISUALIZATION (Tab 3: ABSA Insights)
     ↓
┌─────────────────────────────────────────────────────────────────┐
│  INTERACTIVE CHARTS & METRICS:                                   │
│                                                                 │
│  • Aspect Distribution Pie Chart                               │
│  • Sentiment Bar Charts by Aspect                              │
│  • Sentiment Heatmaps                                          │
│  • Performance Metrics Dashboard                               │
│  • Sample Reviews by Aspect/Sentiment                          │
│  • Downloadable CSV Reports                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🗂️ Detailed File-by-File Flow

### 1. **Entry Point: `app.py`**
```python
# Main Streamlit application
- User chooses between URL input or product search
- Calls either scrape_by_url() or scrape_by_search()
- Stores results in session state
- Saves to MongoDB via MongoIO
```

### 2. **Data Scraping Layer**

#### **Option A: URL-based** (`src/scrapper/url_scrape.py`)
```python
URLScrapeReviews class:
├── __init__() - Setup Chrome driver
├── validate_myntra_url() - URL validation
├── extract_product_reviews() - Main scraping logic
├── scroll_to_load_reviews() - Dynamic content loading
├── extract_review_data() - Parse review elements
└── get_review_dataframe() - Return formatted DataFrame
```

#### **Option B: Search-based** (`src/scrapper/scrape.py`)
```python
ScrapeReviews class:
├── __init__() - Setup Chrome driver
├── scrape_product_urls() - Search and get product URLs
├── extract_reviews() - Get product details
├── extract_products() - Parse review pages
└── get_review_data() - Return formatted DataFrame
```

### 3. **Data Storage Layer** (`src/cloud_io/__init__.py`)
```python
MongoIO class:
├── store_reviews() - Save to MongoDB
├── get_reviews() - Retrieve from MongoDB
└── Database connection management
```

### 4. **Analysis Page** (`pages/generate_analysis.py`)
```python
Analysis Flow:
├── Data source detection (URL vs Search vs MongoDB)
├── Method selection (VADER/Enhanced/Transformer)
├── ABSA analysis execution
├── Results visualization
└── Export functionality
```

### 5. **ABSA Processing Layer** (`src/absa/`)

#### **Core ABSA** (`absa_analyzer.py`)
```python
ABSAAnalyzer class:
├── aspect_keywords - Predefined keyword dictionaries
├── preprocess_reviews() - Text cleaning
├── extract_aspects() - Pattern-based aspect detection
├── classify_sentiment() - VADER sentiment analysis
├── extract_aspect_context() - Context window extraction
├── analyze_reviews() - Main analysis pipeline
├── get_aspect_summary() - Statistical summaries
└── analyze_product_aspects() - Complete analysis
```

#### **Enhanced ABSA** (`simple_advanced_absa.py`)
```python
SimpleAdvancedABSA class (extends ABSAAnalyzer):
├── classify_sentiment_enhanced() - VADER + TextBlob
├── analyze_reviews_enhanced() - Enhanced pipeline
└── analyze_product_aspects_enhanced() - Complete enhanced analysis
```

#### **Advanced ABSA** (`advanced_absa.py`)
```python
AdvancedABSAAnalyzer class (extends ABSAAnalyzer):
├── HuggingFace transformer model loading
├── _classify_with_transformers() - AI-powered analysis
├── analyze_reviews_advanced() - Transformer pipeline
└── analyze_product_aspects_advanced() - Complete AI analysis
```

## 🎯 Key Data Structures

### **Input Data (DataFrame from scraping)**
```python
Columns: [Product Name, Over_All_Rating, Price, Date, Rating, Name, Comment]
```

### **ABSA Output (enhanced_reviews)**
```python
Original columns + [detected_aspects, aspect_sentiments, avg_sentiment_score]
```

### **Detailed ABSA (detailed_absa)**
```python
Columns: [review_index, original_review, aspect, sentiment_label, 
          positive_score, negative_score, neutral_score, compound_score, 
          model_used, confidence]
```

### **Analysis Results Dictionary**
```python
{
    'product_name': str,
    'total_reviews': int,
    'overall_avg_sentiment': float,
    'enhanced_reviews': DataFrame,
    'detailed_absa': DataFrame,
    'aspect_summary': DataFrame,
    'most_mentioned_aspects': dict,
    'model_usage': dict,
    'confidence_stats': dict,
    'sentiment_strength': dict
}
```

## 🔧 Aspect Detection Logic

### **Keyword Mapping**
```python
aspect_keywords = {
    'Style/Design': ['style', 'design', 'look', 'color', 'beautiful', ...],
    'Quality/Material': ['quality', 'material', 'durable', 'cheap', ...],
    'Size/Fit': ['size', 'fit', 'tight', 'loose', 'comfortable', ...],
    'Price/Value': ['price', 'expensive', 'affordable', 'worth', ...],
    'Delivery/Service': ['delivery', 'shipping', 'fast', 'slow', ...]
}
```

### **Processing Pipeline**
1. **Text preprocessing** → Clean review text
2. **Regex matching** → Find aspect keywords in text
3. **Context extraction** → Get surrounding text for sentiment
4. **Sentiment analysis** → Apply chosen model
5. **Result aggregation** → Combine all aspects per review

## 📈 Performance & Model Comparison

### **VADER (Recommended)**
- **Speed**: ⚡ Fastest (~500 reviews/sec)
- **Accuracy**: ✅ 75% on test data
- **Reliability**: 🏆 Most stable

### **Enhanced TextBlob+VADER**
- **Speed**: 🚀 Fast (~200 reviews/sec)
- **Accuracy**: ⚠️ 50% on test data
- **Features**: 🔬 Additional confidence metrics

### **Advanced Transformer**
- **Speed**: 🐌 Slowest (~50 reviews/sec)
- **Accuracy**: 🎯 Variable (depends on model)
- **Features**: 🤖 AI-powered context understanding

## 🎨 Visualization Flow

The analysis results flow into interactive Streamlit visualizations:
- **Plotly charts** for aspect distribution and sentiment analysis
- **Metrics dashboard** for key performance indicators
- **Expandable sections** for detailed review exploration
- **Download buttons** for CSV export

This complete data flow ensures that from user input to final visualization, every step is tracked, processed, and optimized for both performance and accuracy.
