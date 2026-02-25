# BSH Article Scraper

Scrape all articles from [bsh.ubc.ca](https://bsh.ubc.ca) (Balanced Supply of Housing — UBC Research Cluster) and prepare them for LLM fine-tuning.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Scrape articles

```bash
# Try all strategies (WordPress API → sitemap → crawl)
python scrape_bsh.py

# Or pick a specific strategy
python scrape_bsh.py --strategy api      # WordPress REST API (fastest, most reliable)
python scrape_bsh.py --strategy sitemap  # Parse sitemaps, then scrape HTML
python scrape_bsh.py --strategy crawl    # Recursive link crawl
```

### 2. Prepare fine-tuning data

```bash
# Generate both chat and completion formats
python prepare_finetune.py

# Options
python prepare_finetune.py --min-words 200    # skip short articles
python prepare_finetune.py --format chat       # OpenAI chat format only
python prepare_finetune.py --format completion  # prompt/completion format only
```

## Output

All output goes to `output/`:

| File | Description |
|------|-------------|
| `bsh_articles.jsonl` | Raw scraped articles (one JSON object per line) |
| `bsh_articles.json` | Same data as a single JSON array |
| `bsh_finetune_chat.jsonl` | Chat-format fine-tuning data (system/user/assistant messages) |
| `bsh_finetune_completion.jsonl` | Prompt/completion fine-tuning data |

### Article schema

```json
{
  "url": "https://bsh.ubc.ca/...",
  "title": "Article Title",
  "content": "Full plain-text body...",
  "date": "2025-01-15T10:00:00",
  "author": "Author Name",
  "categories": ["Research", "Housing Policy"],
  "excerpt": "Short summary..."
}
```

## Scraping strategies

1. **WordPress REST API** (`--strategy api`): Queries `/wp-json/wp/v2/posts` and `/wp-json/wp/v2/pages`. Returns structured data with metadata. Fastest and most complete.

2. **Sitemap** (`--strategy sitemap`): Parses `wp-sitemap.xml` to discover all URLs, then scrapes each page's HTML. Good fallback if the API is disabled.

3. **Crawl** (`--strategy crawl`): Starts from the homepage and follows all internal links. Slowest but works even without sitemaps.
