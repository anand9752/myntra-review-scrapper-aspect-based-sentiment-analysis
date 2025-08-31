from flask import request
from selenium import webdriver
from selenium.webdriver.common.by import By
from src.exception import CustomException
from bs4 import BeautifulSoup as bs
import pandas as pd
import os, sys
import time
from selenium.webdriver.chrome.options import Options
from urllib.parse import quote


class ScrapeReviews:
    def __init__(self,
                 product_name:str,
                 no_of_products:int):
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--remote-debugging-port=9222")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-plugins")
        options.add_argument("--window-size=1920,1080")
        # Uncomment the line below for headless mode (no browser window)
        # options.add_argument('--headless')
        
        # Start a new Chrome browser session with better error handling
        try:
            self.driver = webdriver.Chrome(options=options)
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)
        except Exception as e:
            print(f"Failed to initialize Chrome driver: {e}")
            raise

        self.product_name = product_name
        self.no_of_products = no_of_products
        self.product_rating_value = None
        self.product_title = None
        self.product_price = None

    def scrape_product_urls(self, product_name):
        try:
            # Check if driver is still available
            if not self.driver or not self.driver.service.is_connectable():
                raise Exception("WebDriver connection lost")
                
            search_string = product_name.replace(" ","-")
            encoded_query = quote(search_string)
            
            # Navigate to the URL with retry mechanism
            url = f"https://www.myntra.com/{search_string}?rawQuery={encoded_query}"
            print(f"Navigating to: {url}")
            
            self.driver.get(url)
            time.sleep(3)  # Wait for page to load
            
            myntra_text = self.driver.page_source
            myntra_html = bs(myntra_text, "html.parser")
            pclass = myntra_html.findAll("ul", {"class": "results-base"})

            product_urls = []
            for i in pclass:
                href = i.find_all("a", href=True)

                for product_no in range(len(href)):
                    t = href[product_no]["href"]
                    product_urls.append(t)

            return product_urls

        except Exception as e:
            print(f"Error in scrape_product_urls: {str(e)}")
            raise CustomException(e, sys)

    def extract_reviews(self, product_link):
        try:
            productLink = "https://www.myntra.com/" + product_link
            self.driver.get(productLink)
            prodRes = self.driver.page_source
            prodRes_html = bs(prodRes, "html.parser")
            title_h = prodRes_html.findAll("title")

            self.product_title = title_h[0].text

            overallRating = prodRes_html.findAll(
                "div", {"class": "index-overallRating"}
            )
            for i in overallRating:
                self.product_rating_value = i.find("div").text
            price = prodRes_html.findAll("span", {"class": "pdp-price"})
            for i in price:
                self.product_price = i.text
            product_reviews = prodRes_html.find(
                "a", {"class": "detailed-reviews-allReviews"}
            )

            if not product_reviews:
                return None
            return product_reviews
        except Exception as e:
            raise CustomException(e, sys)
        
    def wait_for_user_scrolling(self, streamlit_callback=None):
        """
        Instead of automatic scrolling, let the user manually scroll through reviews.
        This is much faster and more reliable. Can use UI callback instead of terminal input.
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



    def extract_products(self, product_reviews: list, streamlit_callback=None):
        try:
            t2 = product_reviews["href"]
            Review_link = "https://www.myntra.com" + t2
            self.driver.get(Review_link)
            
            self.wait_for_user_scrolling(streamlit_callback)
            
            # Enhanced review extraction after user scrolling
            review_page = self.driver.page_source

            review_html = bs(review_page, "html.parser")
            
            # Try multiple selectors to find all review containers
            review_containers = []
            selectors = [
                "div.detailed-reviews-userReviewsContainer",
                "div[class*='user-review-main']",
                "div[class*='review-container']",
                "[class*='userReview']"
            ]
            
            for selector in selectors:
                containers = review_html.select(selector)
                if containers:
                    review_containers = containers
                    print(f"✅ Found {len(containers)} review containers using selector: {selector}")
                    break
            
            if not review_containers:
                print("⚠️ No review containers found, trying fallback method")
                review_containers = review_html.findAll("div", {"class": "detailed-reviews-userReviewsContainer"})

            all_user_ratings = []
            all_user_comments = []
            all_user_names = []
            
            # Extract all review elements from all containers
            for container in review_containers:
                # Extract ratings
                ratings = container.findAll("div", {"class": "user-review-main user-review-showRating"})
                all_user_ratings.extend(ratings)
                
                # Extract comments
                comments = container.findAll("div", {"class": "user-review-reviewTextWrapper"})
                all_user_comments.extend(comments)
                
                # Extract user info
                names = container.findAll("div", {"class": "user-review-left"})
                all_user_names.extend(names)
            
            # Also try direct extraction from the entire page
            if not all_user_ratings:
                print("⚠️ No ratings found in containers, trying direct page extraction")
                all_user_ratings = review_html.findAll("div", {"class": "user-review-main user-review-showRating"})
                all_user_comments = review_html.findAll("div", {"class": "user-review-reviewTextWrapper"}) 
                all_user_names = review_html.findAll("div", {"class": "user-review-left"})

            print(f"📊 Extracted: {len(all_user_ratings)} ratings, {len(all_user_comments)} comments, {len(all_user_names)} user info")
            
            # Use the maximum count to handle any mismatched arrays
            max_reviews = max(len(all_user_ratings), len(all_user_comments), len(all_user_names))
            
            reviews = []
            for i in range(max_reviews):
                try:
                    # Extract rating
                    if i < len(all_user_ratings):
                        rating_element = all_user_ratings[i].find("span", class_="user-review-starRating")
                        rating = rating_element.get_text().strip() if rating_element else "No rating Given"
                    else:
                        rating = "No rating Given"
                except:
                    rating = "No rating Given"
                
                try:
                    # Extract comment
                    if i < len(all_user_comments):
                        comment = all_user_comments[i].text.strip()
                    else:
                        comment = "No comment Given"
                except:
                    comment = "No comment Given"
                
                try:
                    # Extract name
                    if i < len(all_user_names):
                        name_element = all_user_names[i].find("span")
                        name = name_element.text.strip() if name_element else "No Name given"
                    else:
                        name = "No Name given"
                except:
                    name = "No Name given"
                
                try:
                    # Extract date
                    if i < len(all_user_names):
                        date_elements = all_user_names[i].find_all("span")
                        date = date_elements[1].text.strip() if len(date_elements) > 1 else "No Date given"
                    else:
                        date = "No Date given"
                except:
                    date = "No Date given"

                # Only add review if we have at least a rating or comment
                if rating != "No rating Given" or comment != "No comment Given":
                    mydict = {
                        "Product Name": self.product_title,
                        "Over_All_Rating": self.product_rating_value,
                        "Price": self.product_price,
                        "Date": date,
                        "Rating": rating,
                        "Name": name,
                        "Comment": comment,
                    }
                    reviews.append(mydict)

            print(f"✅ Successfully extracted {len(reviews)} complete reviews")
            
            if len(reviews) == 0:
                print("⚠️ No reviews were extracted. This might be due to:")
                print("   - Product has no reviews")
                print("   - Page structure has changed")
                print("   - Network issues during loading")
                return None

            review_data = pd.DataFrame(
                reviews,
                columns=[
                    "Product Name",
                    "Over_All_Rating",
                    "Price",
                    "Date",
                    "Rating",
                    "Name",
                    "Comment",
                ],
            )

            return review_data

        except Exception as e:
            raise CustomException(e, sys)
        
    
    def skip_products(self, search_string, no_of_products, skip_index):
        product_urls: list = self.scrape_product_urls(search_string, no_of_products + 1)

        product_urls.pop(skip_index)

    def get_review_data(self, streamlit_callback=None) -> pd.DataFrame:
        try:
            # search_string = self.request.form["content"].replace(" ", "-")
            # no_of_products = int(self.request.form["prod_no"])

            product_urls = self.scrape_product_urls(product_name=self.product_name)

            product_details = []

            review_len = 0


            while review_len < self.no_of_products:
                product_url = product_urls[review_len]
                review = self.extract_reviews(product_url)

                if review:
                    product_detail = self.extract_products(review, streamlit_callback)
                    product_details.append(product_detail)

                    review_len += 1
                else:
                    product_urls.pop(review_len)

            self.driver.quit()

            data = pd.concat(product_details, axis=0)
            
            data.to_csv("data.csv", index=False)
            
            return data   # For running Streamlit app, you can return the data as dataframe directly
                
            # For running Flask app, you can return the columns and values separately. Uncomment the following lines:

            # columns = data.columns

            # values = [[data.loc[i, col] for col in data.columns ] for i in range(len(data)) ]
            
            # return columns, values
        
        except Exception as e:
            print(f"Error in get_review_data: {str(e)}")
            self.cleanup()
            raise CustomException(e, sys)
        finally:
            # Always cleanup driver
            self.cleanup()
    
    def cleanup(self):
        """Clean up the WebDriver instance."""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.quit()
                print("WebDriver cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
