"""One crawl run, shared by the command line and the GUI."""

from __future__ import annotations

import os
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import __version__, sources
from .categories import Category, get_category, parse_searches
from .classify import DEFAULT_MODEL, Classifier
from .pipeline import Store, run
from .sources import Candidate

SOURCES = ("web", "openverse", "wikimedia", "pexels")


@dataclass
class CrawlOptions:
    searches: list[str]  # e.g. ["green building in snow", "red car"]
    sources: list[str] = field(default_factory=lambda: ["web", "openverse", "wikimedia"])
    urls: list[str] = field(default_factory=list)  # crawl these sites instead of searching
    limit: int = 0  # images per search per source (total for urls); 0 = no limit
    out: Path = Path("output")
    min_size: int = 400
    gifs: bool = True
    min_gif_size: int = 100
    min_confidence: float = 0.30
    commercial: bool = False
    mature: bool = True
    max_pages: int = 0
    depth: int = 2
    delay: float = 1.0
    follow: bool = True
    follow_pages: int = 10
    workers: int = 4
    model: str = DEFAULT_MODEL
    device: str = "auto"


def load_dotenv(path: Path = Path(".env")) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def http_client() -> httpx.Client:
    contact = os.environ.get("CRAWLER_CONTACT", "").strip()
    ua = f"image-crawler/{__version__} (hobby image sorter{'; ' + contact if contact else ''}) python-httpx"
    return httpx.Client(headers={"User-Agent": ua}, timeout=30, follow_redirects=True)


def categories(searches: list[str]) -> list[Category]:
    """Searches -> categories, repeats dropped. Accepts raw words too ("red", "car,", "bike")."""
    phrases = parse_searches(searches)
    if not phrases:
        raise ValueError("give at least one search, e.g. green building in snow")
    return list({c.name: c for c in map(get_category, phrases)}.values())


def candidates(
    client: httpx.Client, cats: list[Category], store: Store, o: CrawlOptions
) -> tuple[Iterator[Candidate], Callable[[Candidate], bool]]:
    """The candidate stream for a run, and the ``full`` check that enforces ``limit``."""
    if o.urls:
        terms = sorted({t for c in cats for t in sources.keyword_terms(c.name, c.query, not c.tonal)})
        site = sources.site(client, o.urls, max_pages=o.max_pages, max_depth=o.depth, delay=o.delay, terms=terms)
        cands = sources.limit_new(sources.safe(site), o.limit, store.saved_count)
    else:
        gens = []
        for c in cats:
            if "openverse" in o.sources:
                gens.append(("openverse", c.query, sources.openverse(client, c.query, commercial=o.commercial, mature=o.mature)))
            if "wikimedia" in o.sources:
                gens.append(("wikimedia", c.query, sources.wikimedia(client, c.query, gifs=o.gifs)))
            if "pexels" in o.sources:
                gens.append(("pexels", c.query, sources.pexels(client, c.query)))
            if "web" in o.sources:
                gens.append(("web", c.query, sources.web(client, c.query, gifs=o.gifs)))
        gens = [
            sources.limit_new(sources.safe(g), o.limit, lambda s=src, q=q: store.saved_count(s, q))
            for src, q, g in gens
        ]
        if o.follow:
            by_query = {c.query: sources.keyword_terms(c.name, c.query, not c.tonal) for c in cats}
            cands = iter(sources.FollowMixer(
                client, gens, lambda q: by_query.get(q, []), pages_per_source=o.follow_pages, delay=o.delay,
            ))
        else:
            cands = sources.interleave(gens)

    if o.limit <= 0:
        full = lambda cand: False
    elif o.urls:
        full = lambda cand: store.saved_count() >= o.limit
    else:  # per search per source; images from followed pages have no quota
        full = lambda cand: cand.source != "followed" and store.saved_count(cand.source, cand.query) >= o.limit
    return cands, full


def crawl(
    o: CrawlOptions,
    classifier: Classifier,
    cats: list[Category],
    stop: threading.Event | None = None,
    stats: Counter | None = None,
    show_progress: bool = True,
) -> Counter:
    """Run one crawl with a loaded classifier. Returns the stats."""
    store = Store(o.out)
    with http_client() as client:
        cands, full = candidates(client, cats, store, o)
        return run(
            cands, client, store, classifier, min_size=o.min_size, workers=o.workers, full=full,
            # no GIFs: treat animations like photos, so the normal size limit applies
            min_gif_size=o.min_gif_size if o.gifs else o.min_size,
            stop=stop, stats=stats, show_progress=show_progress,
        )
