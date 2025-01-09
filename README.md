# Denebunu Review Sentiment Analysis

This repository is developed for performing sentiment analysis on product reviews scraped from Denebunu.com. The project aims to analyze the emotional tone of reviews using both supervised learning algorithms and zero-shot classification methods.

## Project Purpose

The main purpose of this project is to analyze customer feedback collected from e-commerce platforms and understand the emotional content of these feedbacks. The project can be a useful resource for businesses that want to improve customer experience and contribute to product development processes.

## Contents

The repo contains two main Python files:

1.  `web_scraper.py`:
    *   Scrapes product reviews from Denebunu.com by crawling the product pages.
    *   Saves the collected reviews into a JSON file.
    *   Uses `concurrent.futures.ThreadPoolExecutor` for parallel data collection.

2.  `sentiment_analysis.py`:
    *   Loads data from the `review_balanced.json` file.
    *   Applies data preprocessing steps (lowercasing, special character removal, tokenization, stop word removal, stemming).
    *   Transforms text data into numerical format using TF-IDF vectorization.
    *   Performs sentiment analysis using supervised learning algorithms (Logistic Regression, SVC, Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost).
    *   Performs sentiment analysis using zero-shot learning.
    *   Evaluates model performance (Accuracy, Precision, Recall, F1-Score).

## Setup

1.  **Clone the repo:**

    ```bash
    git clone https://github.com/tamerkanak/Sentiment-Analysis.git
    ```
2.  **Install required libraries:**

    ```bash
    pip install requests beautifulsoup4 scikit-learn transformers pandas nltk TurkishStemmer xgboost lightgbm catboost
    ```
3.  **Download NLTK datasets:**
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

## Usage

1.  **Web scraping:**

    Run the `web_scraper.py` file to scrape reviews from Denebunu.com:

    ```bash
    python web_scraper.py
    ```

    This process will create the `review_balanced.json` file.

2.  **Sentiment analysis:**

    Run the `sentiment_analysis.py` file to perform sentiment analysis:

    ```bash
    python sentiment_analysis.py
    ```

    This process will train supervised learning models and evaluate results with zero-shot learning.

## Contributing

Contributions are welcome! Feel free to create pull requests for bug fixes, improvements, and new features.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please open a GitHub issue or contact us directly.
