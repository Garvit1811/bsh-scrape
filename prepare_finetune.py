#!/usr/bin/env python3
"""
Convert scraped BSH articles into fine-tuning datasets.

Reads the JSONL output from scrape_bsh.py and produces:
  1. A ChatML / OpenAI-style JSONL for fine-tuning (messages format)
  2. A simple prompt-completion JSONL (legacy format)

Usage:
    python prepare_finetune.py
    python prepare_finetune.py --min-words 200 --format chat
    python prepare_finetune.py --format completion
"""

import argparse
import json
import logging
import os
import textwrap

INPUT_FILE = os.path.join("output", "bsh_articles.jsonl")
OUTPUT_DIR = "output"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a writer for the Balanced Supply of Housing (BSH), a UBC research "
    "cluster focused on housing policy, equity, and community-based research. "
    "Write in BSH's style: evidence-based, accessible, policy-oriented, and "
    "grounded in community perspectives on housing justice."
)


def load_articles(path: str, min_words: int) -> list[dict]:
    articles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            word_count = len(article["content"].split())
            if word_count >= min_words:
                articles.append(article)
            else:
                log.debug("Skipping '%s' (%d words)", article["title"], word_count)
    log.info("Loaded %d articles (min %d words)", len(articles), min_words)
    return articles


def make_chat_prompt(article: dict) -> str:
    """Build a natural prompt from article metadata."""
    parts = []
    if article.get("categories"):
        cats = ", ".join(article["categories"])
        parts.append(f"Topic: {cats}")
    parts.append(f"Write an article titled \"{article['title']}\".")
    if article.get("excerpt"):
        parts.append(f"Summary: {article['excerpt']}")
    return " ".join(parts)


def to_chat_format(articles: list[dict]) -> list[dict]:
    """OpenAI chat fine-tuning format (messages array)."""
    rows = []
    for a in articles:
        prompt = make_chat_prompt(a)
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": a["content"]},
                ]
            }
        )
    return rows


def to_completion_format(articles: list[dict]) -> list[dict]:
    """Simple prompt/completion format."""
    rows = []
    for a in articles:
        prompt = make_chat_prompt(a)
        rows.append({"prompt": prompt + "\n\n###\n\n", "completion": a["content"] + " END"})
    return rows


def save_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %d rows to %s", len(rows), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BSH articles for LLM fine-tuning")
    parser.add_argument("--min-words", type=int, default=150, help="Minimum word count per article")
    parser.add_argument(
        "--format",
        choices=["chat", "completion", "both"],
        default="both",
        help="Output format (default: both)",
    )
    args = parser.parse_args()

    articles = load_articles(INPUT_FILE, args.min_words)
    if not articles:
        log.error("No articles found in %s", INPUT_FILE)
        return

    if args.format in ("chat", "both"):
        rows = to_chat_format(articles)
        save_jsonl(rows, os.path.join(OUTPUT_DIR, "bsh_finetune_chat.jsonl"))

    if args.format in ("completion", "both"):
        rows = to_completion_format(articles)
        save_jsonl(rows, os.path.join(OUTPUT_DIR, "bsh_finetune_completion.jsonl"))

    log.info("Done.")


if __name__ == "__main__":
    main()
