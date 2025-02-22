import os
import sys
import sqlite3
import time
import tempfile
import shutil
import asyncio
import logging
import json
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
from pydantic import BaseModel
import requests
import openai
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
ZYTE_API_KEY = os.getenv("ZYTE_API_KEY")
OPENAI_SECRET = os.getenv("OPENAI_SECRET")
if not ZYTE_API_KEY:
    raise Exception("Missing ZYTE_API_KEY in environment variables. Please check your .env file.")
if not OPENAI_SECRET:
    raise Exception("Missing OPENAI_SECRET in environment variables. Please check your .env file.")

# Set OpenAI API key and create an OpenAI client for gpt-4o-mini-2024-07-18.
openai.api_key = OPENAI_SECRET
client_gpt4o = OpenAI(api_key=OPENAI_SECRET)

app = FastAPI()

# Pydantic model for structured URL classification output.
class URLClassification(BaseModel):
    url: str
    useful: bool

# Pydantic model for the final quiz item.
class QuizItem(BaseModel):
    url: str
    description: str
    content: str
    question: str
    answer: str

def find_chrome_history_path() -> str:
    """Returns the Chrome history database path for Windows, macOS, or Linux."""
    home = os.path.expanduser("~")
    if sys.platform.startswith("win"):
        return os.path.join(home, "AppData", "Local", "Google", "Chrome", "User Data", "Default", "History")
    elif sys.platform.startswith("darwin"):
        return os.path.join(home, "Library", "Application Support", "Google", "Chrome", "Default", "History")
    elif sys.platform.startswith("linux"):
        return os.path.join(home, ".config", "google-chrome", "Default", "History")
    else:
        return None

def get_history_urls(days: int = 7) -> list:
    """
    Connects to a temporary copy of the Chrome history database,
    retrieves full URL paths visited within the past `days` days,
    and returns a list of unique URLs.
    """
    history_db = find_chrome_history_path()
    if not history_db or not os.path.exists(history_db):
        raise HTTPException(
            status_code=404,
            detail="Chrome history database not found at expected location: " + str(history_db)
        )
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_history_db = tmp.name
        shutil.copy2(history_db, temp_history_db)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error copying history database: " + str(e))
    
    cutoff_unix = time.time() - (days * 86400)
    threshold = int((cutoff_unix + 11644473600) * 1000000)
    
    try:
        conn = sqlite3.connect(temp_history_db)
        cursor = conn.cursor()
        select_statement = (
            "SELECT urls.url FROM urls, visits "
            "WHERE urls.id = visits.url AND visits.visit_time > ?;"
        )
        cursor.execute(select_statement, (threshold,))
        rows = cursor.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error reading database: " + str(e))
    finally:
        conn.close()
        os.remove(temp_history_db)
    
    urls = {row[0] for row in rows if row[0]}
    logger.debug(f"Extracted {len(urls)} unique URLs from Chrome history.")
    return list(urls)

async def classify_url(url: str) -> bool:
    """
    Uses the gpt-4o-mini-2024-07-18 model to classify whether a URL is useful.
    Returns True if classified as useful, else False.
    """
    prompt = (
        "You are a URL classifier. Based on the following URL, determine if it contains informative content "
        "(such as articles, research, or educational material) that a user would want to be quized on as part "
        "of a knowledge retention system. Return a JSON with the following keys:\n"
        '{\n  "url": "<the URL>",\n  "useful": <true or false>\n}\n'
        "Only output valid JSON according to this schema. Do not include any additional text.\n\n"
        f"URL: {url}"
    )
    logger.debug(f"Classifying URL: {url}")
    logger.debug(f"Classification prompt: {prompt}")
    try:
        loop = asyncio.get_running_loop()
        classification = await loop.run_in_executor(
            None,
            lambda: client_gpt4o.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": prompt}],
                response_format=URLClassification
            )
        )
        parsed = classification.choices[0].message.parsed
        logger.debug(f"Parsed classification result for {url}: {parsed}")
        return parsed.useful
    except Exception as e:
        logger.error(f"Error classifying {url}: {e}")
        return False

async def filter_useful_urls(urls: list) -> list:
    """Filters the provided list of URLs using GPT classification and returns the useful URLs."""
    tasks = [classify_url(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    useful_urls = [url for url, is_useful in zip(urls, results) if is_useful is True]
    logger.debug(f"Out of {len(urls)} URLs, {len(useful_urls)} were classified as useful.")
    return useful_urls

def fetch_with_zyte_sync(url: str) -> dict:
    """
    Uses the provided code snippet via requests to fetch page content with article extraction.
    Retries up to 2 times before giving up.
    Returns a dictionary containing the keys: headline, description, url, and content.
    The 'url' is taken from the extracted article if available; otherwise, the original URL is used.
    """
    max_retries = 2
    attempt = 0
    while attempt <= max_retries:
        try:
            api_response = requests.post(
                "https://api.zyte.com/v1/extract",
                auth=(ZYTE_API_KEY, ""),
                json={
                    "url": url,
                    "article": True,
                    "articleOptions": {"extractFrom": "httpResponseBody"},
                },
                timeout=15
            )
            response_json = api_response.json()
            if "article" in response_json:
                article = response_json["article"]
                result = {
                    "headline": article.get("headline", ""),
                    "description": article.get("description", ""),
                    "url": article.get("url", url),
                    "content": article.get("articleBody", "")
                }
            else:
                result = {"error": "No article extraction found", "url": url}
            return result
        except Exception as e:
            attempt += 1
            logger.error(f"Error fetching {url} via Zyte on attempt {attempt}: {e}")
            if attempt > max_retries:
                return {"error": str(e), "url": url}

async def fetch_with_zyte(url: str) -> dict:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_with_zyte_sync, url)

async def scrape_all(urls: list) -> list:
    """Scrapes all given URLs concurrently using Zyte."""
    tasks = [fetch_with_zyte(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results

async def generate_quiz_item_for_source(source: dict) -> QuizItem:
    """
    Uses the gpt-4o-mini-2024-07-18 model to generate a quiz question and answer.
    The entire scraped source dictionary is provided as context.
    The LLM is instructed to return the same JSON with two new keys: 'question' and 'answer'.
    The final JSON should include the keys: url, description, content, question, and answer.
    """
    prompt = (
        "You are a quiz question generation assistant. Given the following article information as JSON, "
        "add two keys to the JSON: 'question' and 'answer'. 'question' should be one concise quiz question "
        "that tests the reader's understanding of the key concepts in the article. 'answer' should be a single "
        "correct answer based solely on the information provided. Do not remove or modify any existing keys. "
        "Return a valid JSON object with the same keys as the input plus the new keys 'question' and 'answer'.\n\n"
        f"{json.dumps(source)}"
    )
    logger.debug(f"Generating quiz item for source: {source.get('url', '')}")
    logger.debug(f"Quiz item generation prompt: {prompt}")
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client_gpt4o.beta.chat.completions.parse(
                model="gpt-4o-mini-2024-07-18",
                messages=[{"role": "user", "content": prompt}],
                response_format=QuizItem
            )
        )
        parsed = response.choices[0].message.parsed
        logger.debug(f"Generated quiz item for {source.get('url', '')}: {parsed.question}")
        return parsed
    except Exception as e:
        logger.error(f"Error generating quiz item for {source.get('url', '')}: {e}")
        return QuizItem(
            url=source.get("url", ""),
            description=source.get("description", ""),
            content=source.get("content", ""),
            question="Error generating question.",
            answer="Error generating answer."
        )

@app.get("/generate_quiz")
async def generate_quiz(
    days: int = Query(7, ge=1, le=7, description="Number of days back to retrieve history (max 7)")
):
    """
    Endpoint that:
    1. Retrieves Chrome history URLs from the last `days` days.
    2. Uses the gpt-4o-mini-2024-07-18 model to classify each URL as useful.
    3. Scrapes the useful URLs using the Zyte API (with up to 2 retries).
    4. For each scraped source, generates a quiz item by feeding the entire scraped dictionary,
       which adds a 'question' and an 'answer' field.
    5. Returns a JSON object with a key 'quiz_items' mapping to a list of dictionaries, each containing:
       url, description, content, question, and answer.
    """
    urls = get_history_urls(days)
    if not urls:
        raise HTTPException(status_code=404, detail="No history URLs found.")
    
    useful_urls = await filter_useful_urls(urls)
    if not useful_urls:
        raise HTTPException(status_code=404, detail="No useful URLs found after classification.")
    
    scraped_data = await scrape_all(useful_urls)
    # Generate one quiz item per scraped source that has a valid URL.
    quiz_tasks = [generate_quiz_item_for_source(source) for source in scraped_data if source.get("url")]
    quiz_items = await asyncio.gather(*quiz_tasks, return_exceptions=False)
    final_quiz_items = [
        {
            "url": item.url,
            "description": item.description,
            "content": item.content,
            "question": item.question,
            "answer": item.answer
        }
        for item in quiz_items
    ]
    return {"quiz_items": final_quiz_items}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
