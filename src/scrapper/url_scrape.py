"""
URL-based review scraper for direct Myntra product analysis
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from src.exception import CustomException
from bs4 import BeautifulSoup as bs
import pandas as pd
import os, sys
import time
import re
from urllib.parse import urlparse


class URLScrapeReviews:
    def __init__(self, product_url: str):
        """
        Initialize URL-based scraper for a specific Myntra product URL.
        
        Args:
            product_url (str): Direct Myntra product URL
        """
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--window-size=1920,1080")
        # Uncomment for headless mode
        # options.add_argument('--headless')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            print(f"Failed to initialize Chrome driver: {e}")
            raise
        
        self.product_url = product_url
        self.product_title = None
        self.product_rating_value = None
        self.product_price = None
        
    def validate_myntra_url(self, url: str) -> bool:
        """
        Validate if the URL is a valid Myntra product URL.
        
        Args:
            url (str): URL to validate
            
        Returns:
            bool: True if valid Myntra URL
        """
        try:
            parsed_url = urlparse(url)
            return (parsed_url.netloc == 'www.myntra.com' or 
                   parsed_url.netloc == 'myntra.com') and len(parsed_url.path) > 1
        except:
            return False
    
    def extract_product_details(self):
        """
        Extract product details from the Myntra product page.
        
        Returns:
            dict: Product details including title, rating, price
        """
        try:
            if not self.validate_myntra_url(self.product_url):
                raise ValueError("Invalid Myntra product URL")
            
            print(f"Navigating to product URL: {self.product_url}")
            self.driver.get(self.product_url)
            time.sleep(3)  # Wait for page to load
            
            page_source = self.driver.page_source
            soup = bs(page_source, "html.parser")
            
            # Extract product title
            title_elements = soup.findAll("title")
            if title_elements:
                self.product_title = title_elements[0].text.strip()
            else:
                self.product_title = "Unknown Product"
            
            # Extract overall rating
            rating_elements = soup.findAll("div", {"class": "index-overallRating"})
            if rating_elements:
                try:
                    self.product_rating_value = rating_elements[0].find("div").text.strip()
                except:
                    self.product_rating_value = "No rating"
            else:
                self.product_rating_value = "No rating"
            
            # Extract price
            price_elements = soup.findAll("span", {"class": "pdp-price"})
            if price_elements:
                try:
                    self.product_price = price_elements[0].text.strip()
                except:
                    self.product_price = "Price not found"
            else:
                self.product_price = "Price not found"
            
            return {
                'title': self.product_title,
                'rating': self.product_rating_value,
                'price': self.product_price,
                'url': self.product_url
            }
            
        except Exception as e:
            print(f"Error extracting product details: {str(e)}")
            raise CustomException(e, sys)
    
    def get_reviews_link(self):
        """
        Find and return the reviews page link.
        
        Returns:
            str: Reviews page URL
        """
        try:
            page_source = self.driver.page_source
            soup = bs(page_source, "html.parser")
            
            # Find reviews link
            reviews_element = soup.find("a", {"class": "detailed-reviews-allReviews"})
            if reviews_element and 'href' in reviews_element.attrs:
                reviews_path = reviews_element['href']
                reviews_url = "https://www.myntra.com" + reviews_path
                return reviews_url
            else:
                raise ValueError("No reviews found for this product")
                
        except Exception as e:
            print(f"Error finding reviews link: {str(e)}")
            raise CustomException(e, sys)
    
    def wait_for_user_scrolling(self, streamlit_callback=None):
        """
        Instead of automatic scrolling, let the user manually scroll through reviews.
        This is much faster and more reliable.
        """
        try:
            print("📜 Manual scrolling approach activated...")
            
            # Maximize window for better user experience
            self.driver.maximize_window()
            time.sleep(2)
            
            # Get initial review count
            initial_reviews = self.count_reviews_on_page()
            print(f"Initial reviews found: {initial_reviews}")
            
            # Display instructions to user
            print("\n" + "="*60)
            print("🖱️  MANUAL SCROLLING INSTRUCTIONS")
            print("="*60)
            print("1. A browser window has opened with the reviews page")
            print("2. Please MANUALLY SCROLL through ALL the reviews")
            print("3. Scroll slowly to let all reviews load")
            print("4. When you've scrolled through everything, come back here")
            print("5. Press ENTER in this terminal to continue scraping")
            print("="*60)
            
            # Wait for user to indicate they're done scrolling
            if streamlit_callback:
                # Use Streamlit UI for user interaction
                import os
                import time
                
                # Create a signal file for communication
                signal_file = "scroll_complete.txt"
                if os.path.exists(signal_file):
                    os.remove(signal_file)
                
                print("\n" + "="*60)
                print("🖱️  MANUAL SCROLLING - UI MODE")
                print("="*60)
                print("1. A browser window has opened with the reviews page")
                print("2. Please MANUALLY SCROLL through ALL the reviews")
                print("3. Go back to the Streamlit app and click 'Continue' button")
                print("="*60)
                
                # Wait for the signal file to be created by Streamlit UI
                while not os.path.exists(signal_file):
                    time.sleep(1)
                
                # Clean up the signal file
                os.remove(signal_file)
                
                # Count final reviews after user scrolling
                final_reviews = self.count_reviews_on_page()
                print(f"✅ User scrolling completed!")
                print(f"📊 Total reviews found: {final_reviews}")
                
                if final_reviews > initial_reviews:
                    print(f"📈 Successfully loaded {final_reviews - initial_reviews} additional reviews")
                else:
                    print("📋 Using initially loaded reviews")
            else:
                # Fallback to terminal interaction
                input("\n⏳ Press ENTER after you've finished scrolling through all reviews...")
            
            # Count final reviews after user scrolling
            final_reviews = self.count_reviews_on_page()
            print(f"✅ User scrolling completed!")
            print(f"� Total reviews found: {final_reviews}")
            
            if final_reviews > initial_reviews:
                print(f"📈 Successfully loaded {final_reviews - initial_reviews} additional reviews")
            else:
                print("� Using initially loaded reviews")
            
        except Exception as e:
            print(f"⚠️ Error during manual scrolling setup: {str(e)}")
            # Continue anyway - we'll work with whatever reviews we have
    
    def count_reviews_on_page(self):
        """Count the number of review elements currently loaded on the page"""
        try:
            # Try multiple selectors for review elements
            selectors = [
                "div[class*='user-review']",
                "div[class*='review']",
                "div.detailed-reviews-userReviewsContainer div.user-review-main",
                "[class*='userReview']"
            ]
            
            max_count = 0
            for selector in selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    max_count = max(max_count, len(elements))
                except:
                    continue
            
            return max_count
        except:
            return 0
    
    def click_load_more_buttons(self):
        """Try to click any 'Load More' or 'Show More' buttons"""
        try:
            # Common button texts for loading more content
            button_texts = ["Load More", "Show More", "View More", "More Reviews", "Load More Reviews"]
            button_selectors = [
                "button[class*='load']",
                "button[class*='more']",
                "a[class*='load']",
                "a[class*='more']",
                "[role='button']"
            ]
            
            # Try clicking buttons by text
            for text in button_texts:
                try:
                    button = self.driver.find_element(By.XPATH, f"//button[contains(text(), '{text}')]")
                    if button.is_displayed() and button.is_enabled():
                        print(f"🔘 Clicking '{text}' button")
                        self.driver.execute_script("arguments[0].click();", button)
                        time.sleep(2)
                        return True
                except:
                    continue
            
            # Try clicking buttons by selector
            for selector in button_selectors:
                try:
                    buttons = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for button in buttons:
                        if button.is_displayed() and button.is_enabled():
                            button_text = button.text.lower()
                            if any(word in button_text for word in ['load', 'more', 'show']):
                                print(f"🔘 Clicking button: {button.text}")
                                self.driver.execute_script("arguments[0].click();", button)
                                time.sleep(2)
                                return True
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️ Error clicking load more buttons: {str(e)}")
        
        return False
    
    def extract_reviews_from_url(self, streamlit_callback=None) -> pd.DataFrame:
        """
        Extract all reviews from the Myntra product URL.
        
        Returns:
            pd.DataFrame: DataFrame containing reviews
        """
        try:
            # Extract product details
            product_details = self.extract_product_details()
            print(f"Product found: {product_details['title']}")
            
            # Get reviews page
            reviews_url = self.get_reviews_link()
            print(f"Navigating to reviews: {reviews_url}")
            
            self.driver.get(reviews_url)
            time.sleep(20)
            
            # Wait for user to manually scroll through reviews
            self.wait_for_user_scrolling(streamlit_callback)
            
            # Extract reviews
            review_page = self.driver.page_source
            soup = bs(review_page, "html.parser")
            
            review_containers = soup.findAll("div", {"class": "detailed-reviews-userReviewsContainer"})
            
            if not review_containers:
                raise ValueError("No reviews found on this page")
            
            reviews = []
            
            for container in review_containers:
                user_ratings = container.findAll("div", {"class": "user-review-main user-review-showRating"})
                user_comments = container.findAll("div", {"class": "user-review-reviewTextWrapper"})
                user_names = container.findAll("div", {"class": "user-review-left"})
                
                # Process each review
                for i in range(min(len(user_ratings), len(user_comments), len(user_names))):
                    try:
                        # Extract rating
                        rating_elem = user_ratings[i].find("span", class_="user-review-starRating")
                        rating = rating_elem.get_text().strip() if rating_elem else "No rating"
                    except:
                        rating = "No rating"
                    
                    try:
                        # Extract comment
                        comment = user_comments[i].text.strip()
                    except:
                        comment = "No comment"
                    
                    try:
                        # Extract name and date
                        name_spans = user_names[i].find_all("span")
                        name = name_spans[0].text.strip() if len(name_spans) > 0 else "Anonymous"
                        date = name_spans[1].text.strip() if len(name_spans) > 1 else "No date"
                    except:
                        name = "Anonymous"
                        date = "No date"
                    
                    # Create review entry
                    review_entry = {
                        "Product Name": product_details['title'],
                        "Over_All_Rating": product_details['rating'],
                        "Price": product_details['price'],
                        "Date": date,
                        "Rating": rating,
                        "Name": name,
                        "Comment": comment,
                        "Source_URL": self.product_url
                    }
                    
                    reviews.append(review_entry)
            
            if not reviews:
                raise ValueError("No valid reviews could be extracted")
            
            # Create DataFrame
            df = pd.DataFrame(reviews, columns=[
                "Product Name", "Over_All_Rating", "Price", "Date", 
                "Rating", "Name", "Comment", "Source_URL"
            ])
            
            print(f"Successfully extracted {len(df)} reviews")
            return df
            
        except Exception as e:
            print(f"Error extracting reviews: {str(e)}")
            raise CustomException(e, sys)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up the WebDriver instance."""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                print("WebDriver cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


def scrape_reviews_from_url(product_url: str, streamlit_callback=None) -> pd.DataFrame:
    """
    Convenience function to scrape reviews from a Myntra product URL.
    
    Args:
        product_url (str): Myntra product URL
        
    Returns:
        pd.DataFrame: Reviews data
    """
    scraper = URLScrapeReviews(product_url)
    return scraper.extract_reviews_from_url(streamlit_callback)
