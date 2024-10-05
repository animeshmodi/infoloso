import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import os
import requests
import json
from abc import ABC, abstractmethod
from transformers import pipeline
from textblob import TextBlob
import logging
import re
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ModelInterface(ABC):
    @abstractmethod
    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        pass

class LocalModel(ModelInterface):
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            logger.error(f"Error in local summarization: {e}")
            return text[:max_length]

class HuggingFaceAPI(ModelInterface):
    def __init__(self):
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HuggingFace API key not found in environment variables")
        
        self.api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        try:
            payload = {
                "inputs": text,
                "parameters": {"max_length": max_length, "min_length": min_length}
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)

            # Check if the response is valid
            if response.status_code == 200:
                return response.json()[0]['summary_text']
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return text[:max_length]

        except Exception as e:
            logger.error(f"Error in HuggingFace API summarization: {e}")
            return text[:max_length]

class GroqAPI(ModelInterface):
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found in environment variables")
        
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        try:
            payload = {
                "model": "mixtral-8x7b-32768",
                "messages": [{
                    "role": "system",
                    "content": f"Summarize the following text in {min_length} to {max_length} words:"
                }, {
                    "role": "user",
                    "content": text
                }]
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)

            # Check if the response is valid
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return text[:max_length]

        except Exception as e:
            logger.error(f"Error in Groq API summarization: {e}")
            return text[:max_length]

class ReviewAnalyzer:
    def __init__(self, model_type: str = "local"):
        self.model_type = model_type
        if model_type == "local":
            self.model = LocalModel()
        elif model_type == "huggingface":
            self.model = HuggingFaceAPI()
        elif model_type == "groq":
            self.model = GroqAPI()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
    def preprocess_text(self, text: str) -> str:
        text = str(text).lower().strip()
        return text

    def get_sentiment_score(self, text: str) -> float:
        return TextBlob(text).sentiment.polarity

    def calculate_review_score(self, review: str, avg_length: float) -> float:
        length_score = len(review) / avg_length
        sentiment_score = abs(self.get_sentiment_score(review))
        relevance_score = len(set(review.split())) / 100  # Unique words ratio
        
        total_score = (length_score * 0.3) + (sentiment_score * 0.4) + (relevance_score * 0.3)
        return total_score

    def process_review_batch(self, batch: List[str]) -> List[Dict]:
        results = []
        for text in batch:
            summary = self.model.summarize(text)
            sentiment = self.get_sentiment_score(text)
            results.append({
                'original': text,
                'summary': summary,
                'sentiment': sentiment
            })
        return results

    def analyze_reviews(self, df: pd.DataFrame, text_column: str, batch_size: int = 5) -> Dict:
        logger.info("Starting review analysis...")
        
        processed_reviews = [self.preprocess_text(text) for text in df[text_column]]
        avg_length = np.mean([len(review) for review in processed_reviews])
        
        review_scores = [self.calculate_review_score(review, avg_length) for review in processed_reviews]
        top_indices = sorted(range(len(review_scores)), key=lambda i: review_scores[i], reverse=True)[:5]
        top_reviews = df.iloc[top_indices][text_column].tolist()
        
        summarized_reviews = []
        for i in range(0, len(top_reviews), batch_size):
            batch = top_reviews[i:i + batch_size]
            batch_results = self.process_review_batch(batch)
            summarized_reviews.extend(batch_results)
        
        all_text = " ".join(df[text_column].astype(str))
        overall_summary = self.model.summarize(all_text, max_length=200)
        
        return {
            'overall_summary': overall_summary,
            'top_reviews': summarized_reviews
        }

def load_and_preprocess_reviews(file_path: str) -> pd.DataFrame:
    """
    Load reviews from a text file and preprocess them by splitting 
    based on specific delimiters.
    
    Args:
    - file_path (str): Path to the text file containing reviews.
    
    Returns:
    - pd.DataFrame: DataFrame containing preprocessed reviews.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Split reviews based on patterns:
    # - A period followed by space and a capital letter (indicates the start of a new sentence)
    # - Double newlines or newline followed by indentation
    reviews = re.split(r'\.\s+(?=[A-Z])|\n\s*\n', content)
    
    # Remove empty reviews and strip whitespace
    reviews = [review.strip() for review in reviews if review.strip()]
    
    # Create a DataFrame
    df = pd.DataFrame(reviews, columns=['review'])
    
    return df

def main():
    # Use .env for API keys
    MODEL_TYPE = os.getenv("MODEL_TYPE", "local")  # local, huggingface, or groq
    
    # Path to the reviews file
    file_path = r"C:\Users\SRIVATSAL NARAYAN\Desktop\llm\summary\reviews.txt"
    
    # Load and preprocess reviews from file
    preprocessed_reviews_df = load_and_preprocess_reviews(file_path)
    
    try:
        analyzer = ReviewAnalyzer(model_type=MODEL_TYPE)
        results = analyzer.analyze_reviews(preprocessed_reviews_df, 'review')
        
        print("\nOverall Summary:")
        print(results['overall_summary'])
        
        print("\nTop 5 Reviews with Summaries:")
        for i, review in enumerate(results['top_reviews'], 1):
            print(f"\n{i}. Original: {review['original'][:100]}...")
            print(f"   Summary: {review['summary']}")
            print(f"   Sentiment: {review['sentiment']:.2f}")
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
