from fastapi import FastAPI
from utils import fetch_news_data

app = FastAPI()

@app.get("/")
def home():
    return {"message": "News Summarization API is running!"}

@app.get("/news/")
def get_news(company_name: str, article_number: int):
    results = fetch_news_data(company_name, article_number)
    return {"news": results}

# run locally with: uvicorn api:app --reload