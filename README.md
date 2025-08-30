# Myntra Review Scraper Project with Machine Learning

## Project Detail/Summary

This project is an advanced Myntra review scraper that not only extracts customer reviews but also provides comprehensive machine learning analysis. The application collects product ratings, reviews, and user feedback, then applies sophisticated ML algorithms to provide actionable insights including sentiment analysis, quality assessment, price prediction, and intelligent recommendations.

## ✨ Key Features

### 📊 **Basic Features**
- Web scraping of Myntra product reviews
- Real-time data extraction with Selenium
- MongoDB integration for data storage
- Interactive Streamlit dashboard
- Data visualization with Plotly

### 🤖 **Advanced ML Features**
- **Sentiment Analysis**: Emotion detection and aspect-based sentiment analysis
- **Review Quality Assessment**: Fake review detection and quality scoring
- **Product Recommendation System**: Content-based and collaborative filtering
- **Price Prediction**: Fair price estimation and value analysis
- **Advanced Analytics**: Comprehensive insights and business intelligence

## How to Setup Locally

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/PWskills-DataScienceTeam/myntra-review-scrapper.git
   cd myntra-review-scraper
   ```

2. Create a new conda environment and activate it
```bash
conda create -p ./env python=3.10 -y
#to activate the environment
conda activate ./env 
#or 
source activate ./env
```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Replace the environment variable in `.env` file
    Add the MongoDB environment variable in the `.env` file

5. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

6. Access the application in your web browser at [http://localhost:8501](http://localhost:8501).

## 🧠 Machine Learning Integration

### Available ML Models:

1. **Sentiment Analysis**
   - TextBlob for basic sentiment analysis
   - Transformer models (RoBERTa) for advanced analysis
   - Aspect-based sentiment extraction

2. **Review Quality Analysis**
   - Quality scoring based on linguistic features
   - Fake review detection algorithms
   - Content authenticity assessment

3. **Recommendation System**
   - Content-based filtering using TF-IDF
   - Collaborative filtering with SVD
   - Price-based recommendations

4. **Price Prediction**
   - Fair price estimation models
   - Value-for-money analysis
   - Brand premium calculations

### Usage:
Navigate to the "ML Analysis" page in the Streamlit app to access all machine learning features. The system provides:
- Interactive dashboards for each ML feature
- Downloadable enhanced datasets
- Comprehensive insights and recommendations

## Dependencies

The project relies on the following dependencies:

### Core Dependencies:
- **Streamlit**: Interactive web application framework
- **MongoDB**: NoSQL database for data storage
- **Selenium**: Web scraping automation
- **BeautifulSoup**: HTML parsing
- **Pandas**: Data manipulation and analysis

### Machine Learning Dependencies:
- **scikit-learn**: Machine learning algorithms
- **textblob**: Natural language processing
- **transformers**: Advanced NLP models
- **torch**: Deep learning framework
- **plotly**: Interactive visualizations

## 📁 Project Structure

```
myntra-review-scrapper/
├── src/
│   ├── scrapper/          # Web scraping modules
│   ├── cloud_io/          # MongoDB integration
│   ├── ml_models/         # Machine learning models
│   │   ├── sentiment_analysis.py
│   │   ├── review_quality.py
│   │   ├── recommendation_system.py
│   │   ├── price_prediction.py
│   │   └── ml_integration.py
│   ├── data_report/       # Basic analytics
│   └── constants/         # Configuration
├── pages/
│   ├── generate_analysis.py  # Basic analysis page
│   └── ml_analysis.py        # ML analysis page
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
└── ML_FEATURES_GUIDE.md     # Detailed ML documentation
```

## 🚀 Quick Start Guide

1. **Scrape Reviews**: Use the main page to search and scrape product reviews
2. **Basic Analysis**: View traditional analytics on the "Generate Analysis" page
3. **ML Analysis**: Access advanced ML insights on the "ML Analysis" page
4. **Download Data**: Export enhanced datasets with ML features

## Replacing chromedriver.exe with ChromeDriver Binary

The decision to replace `chromedriver.exe` with the `ChromeDriver binary pypi package` was made to provide better compatibility and flexibility across different operating systems. By using the binary, users can easily switch between operating systems without the need to manage different driver versions.

## MongoDB Connection

The project utilizes MongoDB as the backend database for storing scraped data. The `database-connect` package is employed to streamline the connection process, making it easier for developers to interact with MongoDB in their applications.

## 📖 Documentation

For detailed information about machine learning features, see [ML_FEATURES_GUIDE.md](ML_FEATURES_GUIDE.md).

## 🤝 Contributing

Feel free to explore the codebase and customize the scraper to suit your specific requirements. If you encounter any issues or have suggestions for improvement, please open an issue on the GitHub repository.

## 📄 License

This project is open source and available under the MIT License.

---

Happy scraping with AI-powered insights! 🕵️‍♂️🤖🚀