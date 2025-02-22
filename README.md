# Quiz Generator from Chrome History

This FastAPI application extracts URLs from your Chrome history, classifies them using GPT-4 (via the `gpt-4o-mini-2024-07-18` model), scrapes useful articles via the Zyte API, and generates quiz items (a question and an answer) based on the scraped content.

## Features

- **Chrome History Extraction:** Reads Chrome history and extracts unique URLs.
- **URL Classification:** Uses GPT-4 to determine if a URL contains informative content.
- **Article Scraping:** Uses the Zyte API (with retry logic) to extract article details.
- **Quiz Generation:** Generates one quiz question and answer per source based on its headline, description, and content.
- **REST API Endpoint:** Exposes a `/generate_quiz` endpoint that returns the quiz items.

## Requirements

- Python 3.10+
- [ngrok](https://ngrok.com/) (optional, for public tunneling)
- Dependencies (install via `pip install -r requirements.txt`):
  - fastapi
  - uvicorn[standard]
  - python-dotenv
  - zyte_api
  - openai
  - pydantic
  - requests

## Setup

1. **Clone the Repository** (if applicable) or create your project directory.

2. **Create a `.env` file** in the project root with your API keys:

   ```dotenv
   ZYTE_API_KEY=your_actual_zyte_api_key_here
   OPENAI_SECRET=your_openai_secret_here

## Run the Application

1. **Start the FastAPI Server:**

   ```bash
   uvicorn main:app --reload

## 1. Start the FastAPI Server

Open a terminal in your project directory and run:

```bash
uvicorn main:app --reload
