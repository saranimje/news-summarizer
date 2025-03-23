# Live Company News Analyzer 
**A project by Sara Nimje - [Visit Portfolio Website](https://saranimje.github.io/)**
This application fetches live news articles for a company, analyzes sentiment, summarizes content, and converts it into Hindi audio. 
## Objective:
I have developed a web-based application that extracts key details from multiple news articles related to a given company. The application performs sentiment analysis, conducts a comparative analysis, and generates a text-to-speech (TTS) output in Hindi. Users can input a company name and receive a structured sentiment report along with an audio summary, making the information more accessible and insightful.
# Project Setup
## Installation:

 - Clone this repository -
`git clone https://github.com/saranimje/news-summarizer.git `
- Navigate to directory -
`cd news-summarizer`

 - Install Dependencies - 
 `pip install -r requirements.txt`
    
 - Run Gradio App - 
 `python app.py`
    
 - Run API (Optional) - 
 `uvicorn api:app --reload`

# Model Details
## Summarization Model
 - Uses transformers from Hugging Face.
 - Model: `google/long-t5-tglobal-base`

## Sentiment Analysis
Uses default sentiment-analysis pipeline from Hugging Face.

## Topic Modelling
-   Uses TF-IDF vectorization with NMF (Non-Negative Matrix Factorization) to extract key topics from news articles.
-   Utilizes cosine similarity to measure relationships between articles.

## Text-to-Speech 
Uses `gTTS (Google Text-to-Speech)`
## Translation
Uses `GoogleTranslator` (source: English, target: Hindi).


# API Development
This project includes a **FastAPI-based API** to fetch news articles and analyze them.
## **Endpoints:**
**1. Home**
-   `GET /`
-   Returns: `{"message": "News Summarization API is running!"}`   
**2. Fetch News**
-   `GET /news/?company_name=Tesla&article_number=5`
-   Returns JSON output containing news articles and analysis.
# API Development
## Using Postman or Curl:
1.  Open **Postman** or any API testing tool.
2.  Send a `GET` request to:
   ` http://127.0.0.1:8000/news/?company_name=Tesla&article_number=5`
3.  View JSON response with news articles and summaries.

## Third-Party API Usage
-   **News Sources**: Google Search (`googlesearch` Python module).
-   **Libraries Used**:
    - `requests` for API calls
    - `gensim`, `deep_translator`, `nltk` for text processing.
    - `googlesearch` to fetch news links.
    - `feedparser` for RSS feeds.