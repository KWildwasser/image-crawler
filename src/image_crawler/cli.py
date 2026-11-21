"""Command line interface."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from . import __version__, sources
from .categories import get_category, known_keywords
from .classify import DEFAULT_MODEL, Classifier
from .pipeline import Store, run, sort_local

app = typer.Typer(add_completion=False, help="Fetch images from the web and sort them into keyword folders.")
console = Console()


class Source(str, Enum):
    openverse = "openverse"
    wikimedia = "wikimedia"
    pexels = "pexels"
    web = "web"


def _load_dotenv(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _client() -> httpx.Client:
    contact = os.environ.get("CRAWLER_CONTACT", "").strip()
    ua = f"image-crawler/{__version__} (hobby image sorter{'; ' + contact if contact else ''}) python-httpx"
    return httpx.Client(headers={"User-Agent": ua}, timeout=30, follow_redirects=True)


def _summary(stats) -> None:
    if not stats:
        console.print("[yellow]Nothing found.[/] Check the warnings above.")
        return
    t = Table(title="Summary", show_header=False)
    for k, v in sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])):
        t.add_row(k, str(v))
    console.print(t)


KeywordsArg = Annotated[list[str], typer.Argument(help="Keywords / categories, e.g. landscape architecture highkey people")]
OutOpt = Annotated[Path, typer.Option("--out", "-o", help="Output folder")]
ConfOpt = Annotated[float, typer.Option(help="Minimum CLIP probability for a semantic category")]
ModelOpt = Annotated[str, typer.Option(help="Hugging Face CLIP model id")]


@app.command()
def crawl(
    keywords: KeywordsArg,
    source: Annotated[list[Source], typer.Option("--source", "-s", help="Source(s) to search; 'web' is a whole-web image search (Bing, SafeSearch off)")] = [Source.web, Source.openverse, Source.wikimedia],
    url: Annotated[list[str], typer.Option("--url", "-u", help="Crawl these websites instead of the APIs (repeatable)")] = [],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Stop after saving this many images per keyword per source (or in total for --url). 0 = no limit, run until Ctrl+C or the sources run out")] = 0,
    out: OutOpt = Path("output"),
    min_size: Annotated[int, typer.Option(help="Skip images whose shorter side is below this (px)")] = 400,
    min_confidence: ConfOpt = 0.30,
    commercial: Annotated[bool, typer.Option(help="Openverse: only licenses allowing commercial use")] = False,
    mature: Annotated[bool, typer.Option(help="Openverse: include results marked as mature/explicit (--no-mature hides them)")] = True,
    max_pages: Annotated[int, typer.Option(help="--url: max pages to visit (0 = no limit)")] = 0,
    depth: Annotated[int, typer.Option(help="--url: max hops along links unrelated to the keywords (keyword-related links don't count)")] = 2,
    delay: Annotated[float, typer.Option(help="Seconds between web page requests (--url and --follow)")] = 1.0,
    follow: Annotated[bool, typer.Option(help="API mode: also crawl the web pages the API results come from, focused on the keyword")] = True,
    follow_pages: Annotated[int, typer.Option(help="--follow: pages to crawl around each source page")] = 10,
    workers: Annotated[int, typer.Option(help="Parallel image downloads")] = 4,
    model: ModelOpt = DEFAULT_MODEL,
) -> None:
    """Search the web for each keyword, download the images and sort them into folders."""
    _load_dotenv()
    cats = [get_category(k) for k in keywords]
    console.print(f"Categories: {', '.join(c.name + (' (tonal)' if c.tonal else '') for c in cats)}")
    if limit <= 0:
        console.print("No limit set: crawling until the sources run out. Press Ctrl+C to stop.")
    console.print("Loading classifier ..." if any(not c.tonal for c in cats) else "Tonal-only run, no CLIP model needed.")
    classifier = Classifier(cats, model_name=model, min_confidence=min_confidence)
    store = Store(out)

    with _client() as client:
        if url:
            terms = sorted({t for c in cats for t in sources.keyword_terms(c.name, c.query, not c.tonal)})
            site = sources.site(client, url, max_pages=max_pages, max_depth=depth, delay=delay, terms=terms)
            cands = sources.limit_new(sources.safe(site), limit, store.saved_count)
        else:
            gens = []
            for c in cats:
                if Source.openverse in source:
                    gens.append(("openverse", c.query, sources.openverse(client, c.query, commercial=commercial, mature=mature)))
                if Source.wikimedia in source:
                    gens.append(("wikimedia", c.query, sources.wikimedia(client, c.query)))
                if Source.pexels in source:
                    gens.append(("pexels", c.query, sources.pexels(client, c.query)))
                if Source.web in source:
                    gens.append(("web", c.query, sources.web(client, c.query)))
            gens = [
                sources.limit_new(sources.safe(g), limit, lambda s=src, q=q: store.saved_count(s, q))
                for src, q, g in gens
            ]
            if follow:
                by_query = {c.query: sources.keyword_terms(c.name, c.query, not c.tonal) for c in cats}
                cands = iter(sources.FollowMixer(
                    client, gens, lambda q: by_query.get(q, []), pages_per_source=follow_pages, delay=delay,
                ))
            else:
                cands = sources.interleave(gens)
        if limit <= 0:
            full = lambda cand: False
        elif url:
            full = lambda cand: store.saved_count() >= limit
        else:  # per keyword per source; images from followed pages have no quota
            full = lambda cand: cand.source != "followed" and store.saved_count(cand.source, cand.query) >= limit
        stats = run(cands, client, store, classifier, min_size=min_size, workers=workers, full=full)

    _summary(stats)
    console.print(f"Output: [bold]{out.resolve()}[/]  (attribution/licenses in metadata.jsonl)")


@app.command("sort")
def sort_cmd(
    folder: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Folder with images to sort")],
    keywords: KeywordsArg,
    out: OutOpt = Path("sorted"),
    min_confidence: ConfOpt = 0.30,
    move: Annotated[bool, typer.Option(help="Delete originals after sorting")] = False,
    model: ModelOpt = DEFAULT_MODEL,
) -> None:
    """Sort images you already have on disk into keyword folders."""
    cats = [get_category(k) for k in keywords]
    classifier = Classifier(cats, model_name=model, min_confidence=min_confidence)
    _summary(sort_local(folder, Store(out), classifier, move=move))


@app.command("keywords")
def keywords_cmd() -> None:
    """List keywords with tuned prompts (any other word works too)."""
    for k in known_keywords():
        c = get_category(k)
        console.print(f"{k:<14} {'tonal' if c.tonal else 'semantic':<9} search: {c.query!r}")
