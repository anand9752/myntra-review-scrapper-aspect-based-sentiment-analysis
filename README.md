# Myntra Review Scraper Project

## Project Detail/Summary

This project is a Myntra review scraper that allows users to extract and analyze customer reviews from the Myntra website. The scraper collects valuable information, such as product ratings, reviews, and user feedback, providing insights into customer sentiments and preferences.

## How to Setup Locally

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/PWskills-DataScienceTeam/myntra-review-scrapper.git

   # 🛍️ Myntra Review Scrapper & ABSA Analysis

   ## 🚀 Overview

   **Myntra Review Scrapper & ABSA Analysis** is a full-stack, production-grade web application that scrapes product reviews from Myntra, stores them in MongoDB, and provides advanced Aspect-Based Sentiment Analysis (ABSA) using VADER, TextBlob, and Transformer models. The app features a beautiful Streamlit dashboard for interactive exploration and insights.

   ---

   ## 🏗️ Architecture

   ```
   User Input (Streamlit UI)
      │
      ├──► Scraping Engine (Selenium + BeautifulSoup)
      │       ├─ URL-based scraping
      │       └─ Search-based scraping
      │
      ├──► Data Storage (MongoDB via src/cloud_io)
      │
      ├──► ABSA Analysis (src/absa)
      │       ├─ VADER (fast, reliable)
      │       ├─ TextBlob+VADER (hybrid)
      │       └─ Transformers (context-aware)
      │
      └──► Interactive Dashboard (Streamlit)
   ```

   ---

   ## 🔄 Workflow

   1. **User Input**: Enter Myntra product URL or search keywords in the Streamlit app.
   2. **Scraping**: Selenium automates Chrome to fetch reviews, BeautifulSoup parses review data.
   3. **Storage**: Reviews are stored in MongoDB for persistence and fast access.
   4. **Analysis**: ABSA modules extract aspects (Quality, Price, Size/Fit, Delivery, Design) and classify sentiment using multiple models.
   5. **Visualization**: Streamlit dashboard displays aspect-wise sentiment, trends, and allows CSV export.

   ---

   ## ✨ Features

   - **Dual Scraping Modes**: URL-based and search-based review extraction
   - **Robust Data Storage**: MongoDB integration for scalable review management
   - **Advanced ABSA**: Three-tier sentiment analysis (VADER, TextBlob+VADER, Transformers)
   - **Aspect Extraction**: Automatic detection of product aspects in reviews
   - **Interactive Dashboard**: Real-time charts, tables, and insights
   - **Export Capability**: Download analysis results as CSV
   - **Session State**: Seamless navigation and data persistence
   - **Error Handling**: Graceful fallbacks and user-friendly messages

   ---

   ## 🛠️ Setup Instructions

   1. **Clone the repository**
      ```bash
      git clone https://github.com/anand9752/review-scrapper-main.git
      cd review-scrapper-main
      ```

   2. **Create and activate environment**
      ```bash
      conda create -p ./env python=3.10 -y
      conda activate ./env
      # or
      source activate ./env
      ```

   3. **Install dependencies**
      ```bash
      pip install -r requirements.txt
      ```

   4. **Configure MongoDB**
      - Add your MongoDB connection string to the `.env` file as `MONGODB_URL_KEY`

   5. **Run the app**
      ```bash
      streamlit run app.py
      ```

   6. **Access the dashboard**
      - Open [http://localhost:8501](http://localhost:8501) in your browser

   ---

   ## 📦 Dependencies

   - streamlit
   - selenium
   - beautifulsoup4
   - pandas
   - numpy
   - plotly
   - nltk
   - textblob
   - transformers
   - torch
   - python-dotenv
   - database-connect
   - chromedriver-binary
   - flask-cors
   - gunicorn

   ---

   ## 🌐 Free Deployment Options

   - **Streamlit Cloud**: Easiest, free, supports most features (except Selenium scraping)
   - **Heroku**: Free tier, supports MongoDB, but may need custom buildpacks for Chrome/Selenium
   - **Render**: Free tier, similar to Heroku

   **Note:** Selenium-based scraping may not work on all free cloud platforms due to browser limitations. For demo purposes, use sample data or switch to requests-based scraping.

   ---

   ## 📊 Example Workflow

   1. **Paste Myntra URL**: `https://www.myntra.com/shirts/roadster/roadster-men-blue-solid-casual-shirt/1376577/buy`
   2. **Scrape Reviews**: App fetches and stores reviews in MongoDB
   3. **Analyze Sentiment**: ABSA module extracts aspects and classifies sentiment
   4. **View Insights**: Dashboard shows aspect-wise sentiment, trends, and allows export

   ---

   ## 🧠 Sentiment Models Explained

   - **VADER**: Fast, rule-based, 75% accuracy
   - **TextBlob+VADER**: Hybrid, 50% accuracy
   - **Transformers**: Context-aware, experimental

   ---

   ## 📁 Project Structure

   ```
   src/
     ├── absa/                # ABSA analyzers (VADER, TextBlob, Transformers)
     ├── cloud_io/            # MongoDB integration
     ├── scrapper/            # Scraping logic (Selenium, BeautifulSoup)
     ├── constants/           # Project constants
     ├── utils/               # Utility functions
     ├── exception.py         # Custom exceptions
   pages/
     ├── generate_analysis.py # ABSA dashboard page
   app.py                     # Main Streamlit app
   requirements.txt           # Python dependencies
   .env                       # Environment variables
   ```

   ---

   ## 🏆 Credits

   - Built by Anand (anand.gp.97@gmail.com)
   - Open source, MIT License

   ---

   ## 🎉 Happy Scraping & Analyzing! 🕵️‍♂️🚀
