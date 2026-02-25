#!/usr/bin/env python3
"""
Scrape all articles from bsh.ubc.ca (Balanced Supply of Housing — UBC Research Cluster).

Outputs JSONL files ready for LLM fine-tuning.

Strategies (tried in order):
  1. WordPress REST API  — structured JSON, most reliable
  2. WordPress sitemap   — discover all URLs, then scrape HTML
  3. Recursive crawl     — follow internal links from the homepage

Usage:
    python scrape_bsh.py                    # run all strategies
    python scrape_bsh.py --strategy api     # WordPress REST API only
    python scrape_bsh.py --strategy sitemap # sitemap + HTML scrape
    python scrape_bsh.py --strategy crawl   # recursive link crawl
"""

import argparse
import html
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://bsh.ubc.ca"
WP_API_URL = f"{BASE_URL}/wp-json/wp/v2"
OUTPUT_DIR = "output"
JSONL_FILE = os.path.join(OUTPUT_DIR, "bsh_articles.jsonl")
JSON_FILE = os.path.join(OUTPUT_DIR, "bsh_articles.json")

REQUEST_DELAY = 1.0  # seconds between requests (be polite)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Pages that are not articles (navigation / structural pages)
SKIP_SLUGS = {
    "about",
    "contact",
    "our-people",
    "in-the-news",
    "research",
    "events",
    "resources",
    "partners",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Article:
    url: str
    title: str
    content: str  # plain-text body
    date: Optional[str] = None
    author: Optional[str] = None
    categories: list[str] = field(default_factory=list)
    excerpt: Optional[str] = None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch(session: requests.Session, url: str) -> Optional[requests.Response]:
    """GET *url* with retries and polite delay."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp
        except requests.RequestException as exc:
            wait = 2**attempt
            log.warning("Attempt %d for %s failed: %s — retrying in %ds", attempt, url, exc, wait)
            time.sleep(wait)
    log.error("Failed to fetch %s after %d attempts", url, MAX_RETRIES)
    return None


# ---------------------------------------------------------------------------
# HTML → plain-text helpers
# ---------------------------------------------------------------------------


def html_to_text(html_str: str) -> str:
    """Convert an HTML fragment to clean plain text."""
    soup = BeautifulSoup(html_str, "lxml")

    # Remove scripts, styles, nav, footer
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text.strip()


def extract_article_from_html(soup: BeautifulSoup, url: str) -> Optional[Article]:
    """Extract article fields from a full-page BeautifulSoup object."""

    # --- title ---
    title = None
    for sel in ["h1.entry-title", "h1.page-title", "h1"]:
        tag = soup.select_one(sel)
        if tag:
            title = tag.get_text(strip=True)
            break
    if not title:
        og = soup.find("meta", property="og:title")
        if og:
            title = og.get("content", "").strip()
    if not title:
        return None

    # --- main content ---
    content_html = None
    for sel in [
        "div.entry-content",
        "article .entry-content",
        "div.post-content",
        "div.page-content",
        "div.content-area",
        "article",
        "main",
        "div#content",
        "div.site-content",
    ]:
        tag = soup.select_one(sel)
        if tag:
            content_html = str(tag)
            break

    if not content_html:
        return None

    content = html_to_text(content_html)
    if len(content) < 100:
        return None  # too short to be a real article

    # --- date ---
    date = None
    time_tag = soup.find("time")
    if time_tag:
        date = time_tag.get("datetime") or time_tag.get_text(strip=True)
    if not date:
        meta = soup.find("meta", property="article:published_time")
        if meta:
            date = meta.get("content")

    # --- author ---
    author = None
    for sel in ["span.author", "a[rel='author']", ".byline", ".post-author"]:
        tag = soup.select_one(sel)
        if tag:
            author = tag.get_text(strip=True)
            break
    if not author:
        meta = soup.find("meta", attrs={"name": "author"})
        if meta:
            author = meta.get("content")

    # --- categories ---
    categories = []
    for sel in ["a[rel='category tag']", ".cat-links a", ".post-categories a"]:
        for tag in soup.select(sel):
            categories.append(tag.get_text(strip=True))

    # --- excerpt ---
    excerpt = None
    meta = soup.find("meta", property="og:description")
    if meta:
        excerpt = meta.get("content", "").strip()

    return Article(
        url=url,
        title=title,
        content=content,
        date=date,
        author=author,
        categories=categories,
        excerpt=excerpt,
    )


# ---------------------------------------------------------------------------
# Strategy 1: WordPress REST API
# ---------------------------------------------------------------------------


def fetch_wp_authors(session: requests.Session) -> dict[int, str]:
    """Fetch author id → name mapping from the WP REST API."""
    authors: dict[int, str] = {}
    page = 1
    while True:
        url = f"{WP_API_URL}/users?per_page=100&page={page}"
        resp = fetch(session, url)
        if resp is None or resp.status_code != 200:
            break
        data = resp.json()
        if not data:
            break
        for u in data:
            authors[u["id"]] = u.get("name", "Unknown")
        page += 1
    return authors


def fetch_wp_categories(session: requests.Session) -> dict[int, str]:
    """Fetch category id → name mapping."""
    cats: dict[int, str] = {}
    page = 1
    while True:
        url = f"{WP_API_URL}/categories?per_page=100&page={page}"
        resp = fetch(session, url)
        if resp is None or resp.status_code != 200:
            break
        data = resp.json()
        if not data:
            break
        for c in data:
            cats[c["id"]] = c.get("name", "")
        page += 1
    return cats


def scrape_via_api(session: requests.Session) -> list[Article]:
    """Use the WordPress REST API to pull all posts and pages."""
    articles: list[Article] = []

    # Pre-fetch author and category maps
    log.info("Fetching authors and categories from WP API …")
    authors = fetch_wp_authors(session)
    categories = fetch_wp_categories(session)
    log.info("Found %d authors, %d categories", len(authors), len(categories))

    for post_type in ("posts", "pages"):
        page = 1
        while True:
            url = f"{WP_API_URL}/{post_type}?per_page=100&page={page}&_embed"
            log.info("WP API: fetching %s page %d", post_type, page)
            resp = fetch(session, url)

            if resp is None:
                break
            if resp.status_code == 400:
                # WP returns 400 when page > total
                break
            if resp.status_code != 200:
                log.warning("WP API returned %d for %s page %d", resp.status_code, post_type, page)
                break

            items = resp.json()
            if not items:
                break

            for item in items:
                slug = item.get("slug", "")
                if slug in SKIP_SLUGS:
                    continue

                title = html.unescape(item["title"]["rendered"]).strip()
                content = html_to_text(item["content"]["rendered"])

                if len(content) < 100:
                    continue

                date = item.get("date")
                author_id = item.get("author")
                author_name = authors.get(author_id)
                cat_ids = item.get("categories", [])
                cat_names = [categories[cid] for cid in cat_ids if cid in categories]
                excerpt_html = item.get("excerpt", {}).get("rendered", "")
                excerpt = html_to_text(excerpt_html) if excerpt_html else None
                link = item.get("link", "")

                articles.append(
                    Article(
                        url=link,
                        title=title,
                        content=content,
                        date=date,
                        author=author_name,
                        categories=cat_names,
                        excerpt=excerpt,
                    )
                )

            total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
            if page >= total_pages:
                break
            page += 1

    log.info("WP API: collected %d articles", len(articles))
    return articles


# ---------------------------------------------------------------------------
# Strategy 2: Sitemap-based discovery + HTML scraping
# ---------------------------------------------------------------------------

SITEMAP_URLS = [
    f"{BASE_URL}/wp-sitemap.xml",
    f"{BASE_URL}/sitemap.xml",
    f"{BASE_URL}/sitemap_index.xml",
    f"{BASE_URL}/post-sitemap.xml",
    f"{BASE_URL}/page-sitemap.xml",
]


def discover_urls_from_sitemap(session: requests.Session) -> set[str]:
    """Parse WordPress sitemaps to find all content URLs."""
    urls: set[str] = set()

    def parse_sitemap(sitemap_url: str) -> None:
        resp = fetch(session, sitemap_url)
        if resp is None:
            return
        soup = BeautifulSoup(resp.text, "lxml-xml")

        # Nested sitemaps
        for loc in soup.find_all("loc"):
            loc_url = loc.get_text(strip=True)
            if loc_url.endswith(".xml"):
                parse_sitemap(loc_url)
            else:
                urls.add(loc_url)

    for sitemap_url in SITEMAP_URLS:
        log.info("Trying sitemap: %s", sitemap_url)
        parse_sitemap(sitemap_url)
        if urls:
            break

    log.info("Sitemap: discovered %d URLs", len(urls))
    return urls


def scrape_via_sitemap(session: requests.Session) -> list[Article]:
    """Discover URLs from sitemaps and scrape each page."""
    urls = discover_urls_from_sitemap(session)
    return scrape_urls(session, urls)


# ---------------------------------------------------------------------------
# Strategy 3: Recursive crawl
# ---------------------------------------------------------------------------


def discover_urls_by_crawl(session: requests.Session, max_pages: int = 500) -> set[str]:
    """Crawl internal links starting from the homepage."""
    visited: set[str] = set()
    to_visit: list[str] = [BASE_URL]
    discovered: set[str] = set()

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        resp = fetch(session, url)
        if resp is None:
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(BASE_URL, href)

            # Only follow internal links
            parsed = urlparse(full_url)
            if parsed.netloc and "bsh.ubc.ca" not in parsed.netloc:
                continue

            # Normalize
            full_url = full_url.split("#")[0].split("?")[0]
            if full_url.startswith(BASE_URL):
                discovered.add(full_url)
                if full_url not in visited:
                    to_visit.append(full_url)

    log.info("Crawl: discovered %d URLs (visited %d pages)", len(discovered), len(visited))
    return discovered


def scrape_via_crawl(session: requests.Session) -> list[Article]:
    """Discover URLs by crawling, then scrape each page."""
    urls = discover_urls_by_crawl(session)
    return scrape_urls(session, urls)


# ---------------------------------------------------------------------------
# Shared: scrape a set of URLs via HTML
# ---------------------------------------------------------------------------


def is_article_url(url: str) -> bool:
    """Heuristic: skip known structural pages."""
    path = urlparse(url).path.strip("/")
    if not path:
        return False
    slug = path.split("/")[-1]
    if slug in SKIP_SLUGS:
        return False
    # Skip files
    if re.search(r"\.(pdf|jpg|png|gif|css|js|xml|zip)$", path, re.I):
        return False
    # Skip author/tag/attachment pages
    if path.startswith(("tag/", "author/", "attachment/", "wp-content/", "wp-admin/")):
        return False
    return True


def scrape_urls(session: requests.Session, urls: set[str]) -> list[Article]:
    """Scrape article content from a set of discovered URLs."""
    article_urls = sorted(u for u in urls if is_article_url(u))
    log.info("Scraping %d candidate article URLs …", len(article_urls))

    articles: list[Article] = []
    for url in article_urls:
        resp = fetch(session, url)
        if resp is None:
            continue
        soup = BeautifulSoup(resp.text, "lxml")
        article = extract_article_from_html(soup, url)
        if article:
            log.info("  ✓ %s", article.title)
            articles.append(article)
        else:
            log.debug("  ✗ skipped %s", url)

    log.info("HTML scrape: collected %d articles", len(articles))
    return articles


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def deduplicate(articles: list[Article]) -> list[Article]:
    """Remove duplicates by URL."""
    seen: set[str] = set()
    unique: list[Article] = []
    for a in articles:
        key = a.url.rstrip("/")
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def save_articles(articles: list[Article]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSONL — one article per line (ideal for fine-tuning pipelines)
    with open(JSONL_FILE, "w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
    log.info("Saved %d articles to %s", len(articles), JSONL_FILE)

    # Also save as a single JSON array for convenience
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump([asdict(a) for a in articles], f, indent=2, ensure_ascii=False)
    log.info("Saved %d articles to %s", len(articles), JSON_FILE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGIES = {
    "api": scrape_via_api,
    "sitemap": scrape_via_sitemap,
    "crawl": scrape_via_crawl,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape articles from bsh.ubc.ca")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default=None,
        help="Scraping strategy (default: try all in order)",
    )
    args = parser.parse_args()

    session = get_session()
    all_articles: list[Article] = []

    if args.strategy:
        strategies_to_try = [args.strategy]
    else:
        strategies_to_try = ["api", "sitemap", "crawl"]

    for name in strategies_to_try:
        log.info("=== Strategy: %s ===", name)
        fn = STRATEGIES[name]
        articles = fn(session)
        if articles:
            log.info("Strategy '%s' returned %d articles", name, len(articles))
            all_articles.extend(articles)
            if args.strategy is None:
                # First successful strategy is enough; deduplicate covers overlap
                break
        else:
            log.warning("Strategy '%s' returned no articles, trying next …", name)

    if not all_articles:
        log.error("No articles collected from any strategy. Check your network connection.")
        return

    all_articles = deduplicate(all_articles)
    all_articles.sort(key=lambda a: a.date or "", reverse=True)
    save_articles(all_articles)

    log.info("Done — %d unique articles saved.", len(all_articles))


if __name__ == "__main__":
    main()
