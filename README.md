# image-crawler

Searches the web for images that match a description and sorts them into a
folder per search. All words form **one** search (`green building in snow`
finds green buildings in snow, not everything green or snowy); separate
several searches with commas.

```
output/
├── green-building-in-snow/   web_3f2a…jpg
├── red-car/
├── highkey/        (hard links, so one image can sit in several folders)
├── _unsorted/      matched none of the searches confidently
└── metadata.jsonl  source, page URL, creator, license, scores per image
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv yet
git clone https://github.com/KWildwasser/image-crawler.git
cd image-crawler
uv sync --managed-python
cp .env.example .env   # set at least CRAWLER_CONTACT
```

The first run downloads the CLIP model (~600 MB). The classifier uses the Apple
GPU, an NVIDIA GPU or the CPU, whichever is available (`--device` forces one).
Update with `git pull`.

**macOS** needs Apple Silicon and macOS 14+. If uv complains that torch "only
has wheels for `macosx_14_0_arm64`", it picked up an Intel Python: check that
`uname -m` and `file "$(which uv)"` both say `arm64` (turn off Rosetta for
Terminal or reinstall uv if not), then `rm -rf .venv && uv sync --managed-python`.

## GUI

```bash
uv run image-crawler gui
```

Opens a page in your browser to enter searches, pick sources and watch images
arrive in a live gallery. On a Mac you can double-click **`Image Crawler.command`**
instead (first time: right-click → Open). Options: `--port` (default 8765),
`--out`, `--no-browser`. The page only listens on `127.0.0.1`; on a remote
machine, tunnel it with `ssh -N -L 8765:127.0.0.1:8765 user@host`.

## Command line

```bash
uv run image-crawler crawl green building in snow               # runs until Ctrl+C
uv run image-crawler crawl green building in snow, red car, highkey
uv run image-crawler crawl red car -n 20                        # 20 per search per source
uv run image-crawler crawl landscape -s pexels -o ~/Pictures/crawl
uv run image-crawler crawl landscape, people --url https://example.com/gallery
uv run image-crawler sort ~/Pictures/unsorted landscape, red car -o ~/Pictures/sorted
uv run image-crawler keywords                                   # keywords with tuned prompts
```

Useful options:

| Option | Default | |
|---|---|---|
| `-n` | none | images to save per search per source (in total with `--url`); only saved images count |
| `-s` | `web`, `openverse`, `wikimedia` | sources (repeatable), see below |
| `-o` | `./output` (`./sorted` for `sort`) | output folder |
| `--min-confidence` | 0.30 | higher is stricter, sends more to `_unsorted/` |
| `--min-size` / `--min-gif-size` | 400 / 100 px | shorter side |
| `--no-gifs` | | skip animated GIFs (kept animated by default) |
| `--no-follow`, `--follow-pages` | on, 10 | crawl the pages search results came from |
| `--url`, `--depth`, `--max-pages` | | crawl a website (same host, obeys robots.txt) |
| `--commercial` | | Openverse: only licenses that allow commercial use |
| `--model` | CLIP base | `openai/clip-vit-large-patch14` is more accurate, ~3× slower |
| `--move` | | `sort` deletes originals instead of copying |

Tips:

- **Wording matters.** "green house" usually means a *greenhouse*; say
  `green building`.
- **Re-runs continue** where the last one stopped: images already in the output
  folder are skipped (by URL and perceptual hash). Ctrl+C stops cleanly.
- **If nothing is saved**, read the printed warnings. Wikimedia refuses requests
  without `CRAWLER_CONTACT` in `.env`.
- File names are `<source>_<hash>.<ext>`; images in several folders are hard
  links, so they take no extra space.

## Sources

| Source | Key | Notes |
|---|---|---|
| `web` | no | Whole-web image search via Bing. Some sites refuse direct downloads (`failed`). |
| `openverse` | optional | CC / public domain. Datacenter IPs may need credentials (see `.env.example`). |
| `wikimedia` | no | Needs `CRAWLER_CONTACT` (email or URL) by [policy](https://w.wiki/4wJS). |
| `pexels` | `PEXELS_API_KEY` | Free key, high-quality photos. |
| `--url` | n/a | Any site you may crawl. |

## How it works

1. **Search** all sources (or crawl a site) with each search phrase. Searches
   and sources take turns, so all folders fill together. Result pages are
   followed and get a small crawl focused on the same search.
2. **Download** in parallel, with size limits and perceptual-hash deduplication.
3. **Classify** every image: subject searches with
   [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), competing against
   the other searches, distractors (screenshots, logos …) and near misses (the
   phrase with one word left out), so the whole description must match. Tonal
   searches (`highkey`, `lowkey`, `bw`) are measured from the histogram.
4. **File** into the best subject folder plus any matching tonal folders, with
   source, license and attribution in `metadata.jsonl`.
