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
   By default the crawl runs until the sources run out or you press Ctrl+C;
   all keywords and sources take turns, so every folder fills up together.
   Website crawls are focused on the keywords: links that mention a keyword
   (in the URL, link text or title) are visited first and followed as deep as
   they go, while unrelated links are followed only a few hops (`--depth`).
   API results are also followed to the page they came from (e.g. the Flickr
   or museum page behind an Openverse hit), and that site gets a small crawl
   focused on the same keyword (`--follow-pages`, default 10 pages). Up to 4
   such pages are crawled at a time and at most 200 pages per host per run.
   Wikimedia Commons and Pexels pages aren't followed, because their APIs
   already cover them. Turn this off with `--no-follow`.
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

### macOS

Requires Apple Silicon and macOS 14 or newer. The locked PyTorch build has
no wheels for Intel Macs or older macOS.

```bash
# install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/KWildwasser/image-crawler.git
cd image-crawler

uv sync --managed-python   # uses uv's own Python 3.14, not Homebrew's
cp .env.example .env       # .env is git-ignored, so fill it in on each machine
```

**Error: torch "only has wheels for … `macosx_14_0_arm64`" on an Apple Silicon Mac.**
uv picked up an Intel (`x86_64`) Python, usually from an Intel Homebrew in
`/usr/local`, which is often carried over from an older Mac by Migration
Assistant. Apple Silicon Homebrew lives in `/opt/homebrew`. To fix it:

1. `uname -m` must print `arm64`. If it prints `x86_64`, your terminal runs
   under Rosetta: quit it, then in Finder open Applications → Utilities,
   choose Get Info on Terminal, untick **Open using Rosetta**, and reopen it.
2. `file "$(which uv)"` must say `arm64`. If it says `x86_64`, reinstall uv
   with the `curl` command above and open a new terminal.
3. `rm -rf .venv && uv sync --managed-python`

## Usage

```bash
# search the APIs until they run out of results (Ctrl+C to stop at any time)
uv run image-crawler crawl landscape architecture people highkey

# stop after saving 20 images per keyword per source (run again for 20 more)
uv run image-crawler crawl landscape architecture people highkey -n 20

# choose sources and amount
uv run image-crawler crawl landscape night -s pexels -n 50 -o ~/Pictures/crawl

# only licenses that allow commercial use (Openverse)
uv run image-crawler crawl architecture --commercial

# crawl a website (same host, obeys robots.txt, 1 s between pages),
# going deep along keyword-related links; --max-pages caps it (default: no cap)
uv run image-crawler crawl landscape people --url https://example.com/gallery --max-pages 300

# a quick test run: public-domain National Park Service pages, 20 images
uv run image-crawler crawl landscape architecture --url https://www.nps.gov/yose/learn/photosmultimedia/photogallery.htm -n 20 -o /tmp/crawl-test

# sort images you already have
uv run image-crawler sort ~/Pictures/unsorted landscape architecture people highkey lowkey -o ~/Pictures/sorted

# keywords with tuned prompts (any other word works with generic prompts)
uv run image-crawler keywords
```

Useful options: `--min-confidence` (default 0.30; higher is stricter and sends more
to `_unsorted`), `--min-size` (shorter side in px, default 400), `--model`
(e.g. `openai/clip-vit-large-patch14` for better accuracy at ~3× the cost).

**What `-n` counts:** only images that are actually saved. Images that are too
small (icons, logos), duplicates or already in the output folder don't count, so
a website full of thumbnails won't use up the limit.

**Progress:** while it runs, `crawl` shows a live line such as
`found 13 · saved 2 · too small 3  downloading from www.nps.gov`. If `saved`
stays at 0 for a while, the sources return images that get filtered out (too
small, failed downloads) or return nothing at all (`found 0`).

**Mature results:** Openverse hides results marked as mature by default. Add
`--mature` to include them. Wikimedia Commons doesn't filter, and Pexels has no
such content.

**If a run saves nothing**, read the warnings it prints. A site that refuses the
crawler (HTTP 403, robots.txt) is reported by name. Wikipedia and Wikimedia
Commons refuse every request without a contact address, so set
`CRAWLER_CONTACT` in `.env` for them.

## Where images are saved

Output folders are relative to the directory you run the command from:

| Command | Default folder | Change with |
|---|---|---|
| `crawl` | `./output/` | `-o ~/Pictures/crawl` |
| `sort` | `./sorted/` | `-o ~/Pictures/sorted` |

`crawl` prints the full path at the end of each run. The layout is shown at the
top of this README.

- **File names** are `<source>_<content hash>.<ext>`, e.g. `pexels_3f2a9c1e7b04.jpg`.
- **Multi-keyword images** are stored once on disk. Each extra folder gets a
  hard link, so it takes no extra space, and deleting the image from one folder
  doesn't remove it from the others.
- **Re-runs continue where the last one stopped.** With the same `-o` folder,
  `crawl` reads `metadata.jsonl`, skips images that are already saved (by URL,
  and by perceptual hash for the same picture at another URL), and keeps going
  through later search results (until it has saved `-n` images per keyword per
  source, if you set `-n`). It stops when a source runs out of results. Ctrl+C
  stops cleanly: everything saved so far is complete and recorded. The run summary
  shows skipped images as `already_have`. Each run still pages through the
  results it has already seen before it reaches new ones, so after many runs on
  the same keyword, Openverse without credentials may hit its page limit and stop
  returning new images. Register credentials (see `.env.example`) or switch
  sources.
- **`sort` copies by default.** Your originals stay where they are; `--move`
  deletes each original after sorting it.
- `output/` and `sorted/` are git-ignored, so images are never committed.
- The CLIP model is cached separately by Hugging Face in `~/.cache/huggingface/`.

## Sources

| Source | Key needed | Notes |
|---|---|---|
| `openverse` | optional | CC / public domain. Anonymous requests from datacenter IPs may get a Cloudflare challenge, so register credentials (see `.env.example`). |
| `wikimedia` | no | Requires `CRAWLER_CONTACT` (email or URL) in the User-Agent by [policy](https://w.wiki/4wJS). |
| `pexels` | `PEXELS_API_KEY` | Free key, high-quality photography, Pexels license. |
| `--url` | n/a | Generic crawler for sites you have permission to crawl. |
| followed pages | n/a | Saved as `followed_….jpg`, with `page_url`. On by default, `--no-follow` turns it off. |
