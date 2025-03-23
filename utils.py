# ==========================
# Data Handling & Storage
# ==========================
import json
import ast
import pandas as pd
import numpy as np

# ==========================
# Web Scraping & Data Retrieval
# ==========================
import requests
import httpx
import feedparser
import concurrent.futures
from bs4 import BeautifulSoup
from googlesearch import search
from urllib.parse import urlparse

# ==========================
# Natural Language Processing (NLP)
# ==========================
import nltk
import spacy
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS  # Text-to-speech

# ==========================
# Machine Learning & Text Analysis
# ==========================
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import RandomizedSearchCV

# ==========================
# Data Visualization
# ==========================
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Utility & Performance Optimization
# ==========================
import re
import os
import io
from collections import Counter
from tqdm import tqdm  # progress bar


def fetch_news_data(company_name: str, article_number: int):
    excluded_domains = ["youtube.com", "en.wikipedia.org", "m.economictimes.com", "www.prnewswire.com", "economictimes.indiatimes.com", "www.moneycontrol.com"]

    def is_valid_news_article(url, company_name):
        try:
            domain = urlparse(url).netloc  # extracts the domain
            if company_name.lower() in domain.lower() or any(excluded_domain in domain for excluded_domain in excluded_domains):
                return False
            return True
        except Exception:
            return False  # handle unexpected errors

    def get_top_articles(company_name, article_number):
        query = f"{company_name} latest news article"
        valid_urls = []

        for url in search(query, num_results = article_number*2):
            if is_valid_news_article(url, company_name):
                valid_urls.append(url)
            if len(valid_urls) > article_number+1:
                break

        return valid_urls

    def extract_article_data(url):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # handle HTTP errors

            soup = BeautifulSoup(response.content, 'html.parser')

            # extract title
            title = soup.title.string.strip() if soup.title else None
            source = url.split('/')[2]  # Extract domain

            # validate data
            if not title:
                return None

            return {"title": title, "link": url, "source": source}

        except (requests.exceptions.RequestException, AttributeError):
            return None  # skip articles with invalid data

    def main(company_name, article_number):
        urls = get_top_articles(company_name, article_number)
        # extract and validate article data
        articles_data = [extract_article_data(url) for url in urls]
        articles_data = [article for article in articles_data if article]  # remove None values

        # create DataFrame only if valid articles exist
        if articles_data:
            df = pd.DataFrame(articles_data)
        else:
            df = pd.DataFrame(columns=["title", "link"])  # empty DataFrame if nothing was found

        return df

    df = main(company_name, article_number+1)
    news_df_output = df[["title", "source"]].rename(columns={"title": "Headline", "source": "Source"})
    news_df_output["Source"] = news_df_output["Source"].str.replace(r"^www\.", "", regex=True).str.split('.').str[0]
    
    yield {"news_df_output": news_df_output}

    def get_article_text(url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            # remove unwanted elements
            for unwanted in soup.select("nav, aside, footer, header, .ad, .advertisement, .promo, .sidebar, .related-articles"):
                unwanted.extract()

            # try extracting from known article containers
            article_body = soup.find(['article', 'div', 'section'], class_=['article-body', 'post-body', 'entry-content', 'main-content'])

            if article_body:
                paragraphs = article_body.find_all('p')
                article_text = " ".join([p.get_text() for p in paragraphs]).strip()
                return article_text if article_text else None  # return None if empty

            # fallback to all <p> tags
            paragraphs = soup.find_all('p')
            article_text = " ".join([p.get_text() for p in paragraphs]).strip()

            return article_text if article_text else None  # return None if empty

        except Exception:
            return None  # return None in case of an error
    df['article_text'] = df['link'].apply(get_article_text)

    df = df.reset_index(drop=True)

    block_patterns = [
        # Error messages (with variations)
        r'Oops[!,\.]? something went wrong',
        r'An error has occurred',
        r'This content is not available',
        r'Please enable JavaScript to continue',
        r'Error loading content',
        r'Follow Us',

        # JavaScript patterns
        r'var .*?;',
        r'alert\(.*?\)',
        r'console\.log\(.*?\)',
        r'<script.*?</script>',
        r'<noscript>.*?</noscript>',
        r'<style.*?</style>',

        # Loading or restricted content messages
        r'Loading[\.]*',
        r'You must be logged in to view this content',
        r'This content is restricted',
        r'Access denied',
        r'Please disable your ad blocker',

        # GDPR and cookie consent banners
        r'This site uses cookies',
        r'We use cookies to improve your experience',
        r'By using this site, you agree to our use of cookies',
        r'Accept Cookies',

        # Stories or content teasers with any number
        r'\d+\s*Stories',

        # Miscellaneous
        r'<iframe.*?</iframe>',
        r'<meta.*?>',
        r'<link.*?>',
        r'Refresh the page and try again',
        r'Click here if the page does not load',
        r'© [0-9]{4}.*? All rights reserved',
        r'Unauthorized access',
        r'Terms of Service',
        r'Privacy Policy',
        r'<.*?>',
        ]

    pattern = '|'.join(block_patterns)
    df['article_text'] = df['article_text'].str.replace(pattern, '', regex=True).str.strip()
    df['article_text'] = df['article_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    custom_stop_words = set(ENGLISH_STOP_WORDS.union({company_name.lower(), 'company', 'ttm', 'rs'}))

    # add numeric values (integer, decimal, comma-separated, monetary)
    numeric_patterns = re.compile(r'\b\d+(?:[\.,]\d+)?(?:,\d+)*\b|\$\d+(?:[\.,]\d+)?')
    numeric_matches = set(re.findall(numeric_patterns, ' '.join(df['article_text'])))
    custom_stop_words.update(numeric_matches)

    # remove unwanted unicode characters (like \u2018, \u2019, etc.)
    unicode_patterns = re.compile(r'[\u2018\u2019\u2020\u2021\u2014]')  # Add more if needed
    df['article_text'] = df['article_text'].apply(lambda x: unicode_patterns.sub('', x))

    custom_stop_words = list(custom_stop_words)

    summarizer = pipeline("summarization", model="google/long-t5-tglobal-base")

    def generate_summary(text):
        try:
            if len(text.split()) > 50:  # skip very short texts
                summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                return summary
            else:
                return text
        except Exception as e:
            print(f"Error processing text: {e}")
            return None

    # apply summarization to the 'article_text' column
    df['summary'] = df['article_text'].apply(generate_summary)

    # load a pre-trained BERT-based sentiment model from Hugging Faces
    sentiment_pipeline = pipeline("sentiment-analysis")

    def analyze_sentiment(text):
        """Analyze sentiment with a confidence-based neutral zone."""
        if not text.strip():
            return "Neutral"

        try:
            result = sentiment_pipeline(text)[0]
            sentiment_label = result["label"]
            confidence = round(result["score"], 2)

            if confidence < 0.7:
                return "Neutral"
            return f"{sentiment_label.capitalize()} ({confidence})"
        except Exception:
            return "Error in sentiment analysis."

    # apply sentiment analysis on the summary column
    df['sentiment'] = df['summary'].apply(analyze_sentiment)

    df['sentiment_label'] = df['sentiment'].str.extract(r'(Positive|Negative|Neutral)')

    sentiment_bars = plt.figure(figsize=(7, 7))
    sns.countplot(x=df['sentiment_label'], palette={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'})
    plt.title("Sentiment Analysis of Articles")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    # save the figure as an image file to use in gradio interface
    sentiment_bars_file = "sentiment_bars.png"
    sentiment_bars.savefig(sentiment_bars_file)
    plt.close(sentiment_bars)

    sentiment_counts = df['sentiment_label'].value_counts()

    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}

    sentiment_pie = plt.figure(figsize=(7, 7))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=[colors[label] for label in sentiment_counts.index])
    plt.title("Sentiment Distribution of Articles")

    sentiment_pie_file = "sentiment_pie.png"
    sentiment_pie.savefig(sentiment_pie_file)
    plt.close(sentiment_pie)

    df['combined_text'] = df['title'] + ' ' + df['summary'] # combine text for analysis

    vectorizer = TfidfVectorizer(max_features=1000, stop_words=custom_stop_words)
    tfidf = vectorizer.fit_transform(df['combined_text'])

    n_topics = 5  # number of topics
    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(H):
        top_words = [feature_names[i] for i in topic.argsort()[-5:]][::-1]  # 5 words per topic
        topics.append(", ".join(top_words))


    def get_top_topics(row):
        topic_indices = W[row].argsort()[-3:][::-1]  # get top 3 topics
        return [topics[i] for i in topic_indices]

    df['top_topics'] = [get_top_topics(i) for i in range(len(df))]
    df['dominant_topic'] = W.argmax(axis=1)
    df['topic_distribution'] = W.tolist()
    similarity_matrix = cosine_similarity(W)

    df['similarity_scores'] = similarity_matrix.mean(axis=1)
    df['most_similar_article'] = similarity_matrix.argsort(axis=1)[:, -2]  # second highest value
    df['least_similar_article'] = similarity_matrix.argsort(axis=1)[:, 0]  # lowest value

    similarity_heatmap = plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=False, yticklabels=False)
    plt.title("Comparative Analysis of News Coverage Across Articles")

    comparisons = []
    for i in range(len(df)):
        # find most similar and least similar articles
        similar_idx = similarity_matrix[i].argsort()[-2]  # most similar (excluding itself)
        least_similar_idx = similarity_matrix[i].argsort()[0]  # least similar

        # build comparison text
        comparison = {
            "Most Similar": f"Article {i + 1} focuses on '{topics[df['dominant_topic'][i]]}', similar to Article {similar_idx + 1} which also discusses '{topics[df['dominant_topic'][similar_idx]]}'.",
            "Least Similar": f"Article {i + 1} focuses on '{topics[df['dominant_topic'][i]]}', contrasting with Article {least_similar_idx + 1} which discusses '{topics[df['dominant_topic'][least_similar_idx]]}'."
        }
        comparisons.append(comparison)

    df['coverage_comparison'] = comparisons
    # find common and unique topics
    all_topics = df['dominant_topic'].tolist()
    topic_counter = Counter(all_topics)
    common_topics = [topics[i] for i, count in topic_counter.items() if count > 1]
    unique_topics = [topics[i] for i, count in topic_counter.items() if count == 1]

    topic_overlap = {
        "Common Topics": common_topics,
        "Unique Topics": unique_topics
    }
    sentiment_counts = df['sentiment_label'].value_counts()
    if sentiment_counts.get('Positive', 0) > sentiment_counts.get('Negative', 0):
        sentiment = "Overall sentiment is positive."
    elif sentiment_counts.get('Negative', 0) > sentiment_counts.get('Positive', 0):
        sentiment = "Overall sentiment is negative."
    else:
        sentiment = "Overall sentiment is mixed."

    def extract_relevant_topics(topics):
        if isinstance(topics, str):
            topics = ast.literal_eval(topics)  # convert string to list if needed

        if len(topics) <= 2:
            return topics

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(topics)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # sum similarity scores for each topic
        topic_scores = similarity_matrix.sum(axis=1)

        # get top 2 highest scoring topics
        top_indices = topic_scores.argsort()[-2:][::-1]
        top_topics = [topics[i] for i in top_indices]

        return top_topics


    # ensure 'top_topics' is a list
    df['top_topics'] = df['top_topics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # convert lists to sets for easy comparison
    df['top_topics_set'] = df['top_topics'].apply(lambda x: set(x) if isinstance(x, list) else set())

    # find common topics across all articles
    if len(df) > 1:
        common_topics = set.intersection(*df['top_topics_set'])
    else:
        common_topics = set()  # no common topics if only one article

    # extract unique topics by removing common ones
    df['unique_topics'] = df['top_topics_set'].apply(lambda x: list(x - common_topics) if x else [])

    # drop the temporary 'top_topics_set' column
    df.drop(columns=['top_topics_set'], inplace=True)


    coverage_differences = []
    for _, row in df.iterrows():
        if row['most_similar_article'] in df.index and row['least_similar_article'] in df.index:
            most_similar = df.loc[row['most_similar_article']]
            least_similar = df.loc[row['least_similar_article']]

            # extract most relevant topics
            most_relevant_topics = extract_relevant_topics(row['top_topics'])
            least_relevant_topics = extract_relevant_topics(least_similar['top_topics'])

            if most_relevant_topics and least_relevant_topics:
                comparison = {
                    "Comparison": f"{row['title']} highlights {', '.join(row['top_topics'])}, while {most_similar['title']} discusses {', '.join(most_similar['top_topics'])}.",
                    "Impact": f"The article emphasizes {most_relevant_topics[0]} and {most_relevant_topics[1]}, contrasting with {least_relevant_topics[0]} and {least_relevant_topics[1]} in the least similar article."
                }
                coverage_differences.append(comparison)
    structured_summary = {
        "Company": company_name,
        "Articles": [
            {
                "Title": row['title'],
                "Summary": row['summary'],
                "Sentiment": row['sentiment'],
                "Topics": row['top_topics'],
                "Unique Topics": row['unique_topics']
            }
            for _, row in df.iterrows()
        ],
        "Comparative Sentiment Score": {
            "Sentiment Distribution": df['sentiment'].value_counts().to_dict(),
        },
        "Topic Overlap": {
            "Common Topics": list(common_topics) if common_topics else ["No common topics found"],
            "Unique Topics": [
                {"Title": row['title'], "Unique Topics": row['unique_topics']}
                for _, row in df.iterrows()
            ]
        },
        "Final Sentiment Analysis": f"{company_name}’s latest news coverage is mostly {df['sentiment'].mode()[0].lower()}. Potential market impact expected."
    }

    yield {"json_summary": structured_summary}
    english_news = [f"Name of Company: {company_name}"]

    for i, row in df.iterrows():
        article_entry = f"Article {i + 1}: "
        article_entry += f"{row['title']}; "
        article_entry += f"Summary: {row['summary']} This article has a {row['sentiment_label'].lower()} sentiment."
        english_news.append(article_entry)
    yield {"english_news_list": english_news}
    translator = GoogleTranslator(source='en', target='hi')  # 'hi' = Hindi

    translated_news = []
    for text in tqdm(english_news, desc="Translating"):
        translated_news.append(translator.translate(text))
    yield {"hindi_news_list": translated_news}
    hindi_news = '; '.join(translated_news)
    # yield {"hindi_news_text": hindi_news}
    def text_to_speech(text, language='hi'):
      tts = gTTS(text=text, lang=language, slow=False)
      filename = "hindi_news.mp3"  # save file to path
      tts.save(filename)
      return filename
    print(df)
    news_audio = text_to_speech(hindi_news)
    yield {"hindi_news_audio": news_audio}

    yield {"bar_chart": sentiment_bars_file}

    yield {"pie_chart": sentiment_pie_file}