"""
STEP 2 (optional expansion): Scrape public bank FAQ pages to grow your dataset
beyond the seed data/bank_faqs.csv file.

Usage: python scripts/scrape_faqs.py

NOTE: Always check a site's robots.txt and terms of use before scraping.
This is a generic template — you'll need to inspect each bank's FAQ page HTML
structure (right-click > Inspect) and adjust the CSS selectors below, since
every bank's page layout is different.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

HEADERS = {"User-Agent": "Mozilla/5.0 (educational project data collection)"}


def scrape_faq_page(url: str, question_selector: str, answer_selector: str, category: str):
    """Generic FAQ scraper — pass in CSS selectors for question/answer elements."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    questions = soup.select(question_selector)
    answers = soup.select(answer_selector)

    rows = []
    for q, a in zip(questions, answers):
        q_text = q.get_text(strip=True)
        a_text = a.get_text(strip=True)
        if q_text and a_text:
            rows.append({"question": q_text, "answer": a_text, "category": category})
    return rows


def main():
    all_rows = []

    # Example placeholder — replace with actual bank FAQ page URLs + selectors
    # after inspecting each page's HTML structure.
    sources = [
        # {
        #     "url": "https://www.examplebank.com/faqs/loans",
        #     "question_selector": ".faq-question",
        #     "answer_selector": ".faq-answer",
        #     "category": "Loans",
        # },
    ]

    for src in sources:
        print(f"Scraping: {src['url']}")
        try:
            rows = scrape_faq_page(
                src["url"], src["question_selector"], src["answer_selector"], src["category"]
            )
            all_rows.extend(rows)
            print(f"  -> collected {len(rows)} Q&A pairs")
        except Exception as e:
            print(f"  -> failed: {e}")
        time.sleep(2)  # be polite, avoid hammering the server

    if all_rows:
        new_df = pd.DataFrame(all_rows)
        existing_df = pd.read_csv("data/bank_faqs.csv")
        combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(
            subset=["question"]
        )
        combined.to_csv("data/bank_faqs.csv", index=False)
        print(f"Dataset now has {len(combined)} total Q&A pairs.")
    else:
        print("No sources configured yet — add bank FAQ URLs + selectors in `sources` above.")
        print("Alternative: download the Kaggle 'Bank FAQ Dataset' and merge it into data/bank_faqs.csv")


if __name__ == "__main__":
    main()
