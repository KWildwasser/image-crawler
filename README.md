# image-crawler

Fetches images from the web and sorts them into folders by keyword:
`landscape`, `architecture`, `people`, `highkey`, or any other word.

```
output/
├── landscape/      pexels_3f2a…jpg
├── architecture/
├── people/
├── highkey/        (hard links, so one image can sit in several folders)
├── _unsorted/      matched none of the keywords confidently
└── metadata.jsonl  source, page URL, creator, license, scores per image
```

## How it works

1. **Search**: each keyword is sent as a query to the image sources
   (Openverse, Wikimedia Commons, Pexels), or a website you name is crawled.
2. **Download**: images are fetched in parallel, with size limits, a minimum
   resolution, and perceptual-hash deduplication across runs.
3. **Classify**: search results are noisy, so every image is checked, not trusted:
   - **Subject keywords** (landscape, architecture, people, …) use
     [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) zero-shot
     classification. The keywords compete with each other and with distractor
     prompts (screenshots, diagrams, logos …). Images below `--min-confidence`
     go to `_unsorted/`.
   - **Tonal keywords** (`highkey`, `lowkey`, `bw`) are measured from the
     luminance histogram and saturation, which is more reliable than asking a
     model about brightness.
4. **File**: the image goes to the best subject folder and is hard-linked into
   every tonal folder it matches. The source, license, and attribution go to
   `metadata.jsonl`.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in at least CRAWLER_CONTACT
```

The first run downloads the CLIP model (~600 MB). It runs on CPU.

## Usage

```bash
# search the APIs, 20 images per keyword per source
uv run image-crawler crawl landscape architecture people highkey

# choose sources and amount
uv run image-crawler crawl landscape night -s pexels -n 50 -o ~/Pictures/crawl

# only licenses that allow commercial use (Openverse)
uv run image-crawler crawl architecture --commercial

# crawl a website (same host, obeys robots.txt, 1 s between pages)
uv run image-crawler crawl landscape people --url https://example.com/gallery --max-pages 30

# sort images you already have
uv run image-crawler sort ~/Pictures/unsorted landscape architecture people highkey lowkey -o ~/Pictures/sorted

# keywords with tuned prompts (any other word works with generic prompts)
uv run image-crawler keywords
```

Useful options: `--min-confidence` (default 0.30; higher is stricter and sends more
to `_unsorted`), `--min-size` (shorter side in px, default 400), `--model`
(e.g. `openai/clip-vit-large-patch14` for better accuracy at ~3× the cost).

## Sources

| Source | Key needed | Notes |
|---|---|---|
| `openverse` | optional | CC / public domain. Anonymous requests from datacenter IPs may get a Cloudflare challenge, so register credentials (see `.env.example`). |
| `wikimedia` | no | Requires `CRAWLER_CONTACT` (email or URL) in the User-Agent by [policy](https://w.wiki/4wJS). |
| `pexels` | `PEXELS_API_KEY` | Free key, high-quality photography, Pexels license. |
| `--url` | n/a | Generic crawler for sites you have permission to crawl. |

## Licensing note

Images keep their original licenses. `metadata.jsonl` records creator, license,
and source page for each file. Check it before you publish anything, because CC
licenses usually require attribution.
