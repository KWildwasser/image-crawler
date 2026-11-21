"""Command line interface."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import job
from .categories import get_category, known_keywords
from .classify import DEFAULT_MODEL, DEVICE_NAMES, Classifier
from .pipeline import Store, sort_local

app = typer.Typer(add_completion=False, help="Fetch images from the web and sort them into keyword folders.")
console = Console()


class Source(str, Enum):
    openverse = "openverse"
    wikimedia = "wikimedia"
    pexels = "pexels"
    web = "web"


def _summary(stats) -> None:
    if not stats:
        console.print("[yellow]Nothing found.[/] Check the warnings above.")
        return
    t = Table(title="Summary", show_header=False)
    for k, v in sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])):
        t.add_row(k, str(v))
    console.print(t)


KeywordsArg = Annotated[
    list[str],
    typer.Argument(
        help="What to search for. All words form one search, e.g. green building in snow. "
        "Separate several searches with commas: green building in snow, red car",
        metavar="SEARCH",
    ),
]
OutOpt = Annotated[Path, typer.Option("--out", "-o", help="Output folder")]
ConfOpt = Annotated[float, typer.Option(help="Minimum CLIP probability for a semantic category")]
ModelOpt = Annotated[str, typer.Option(help="Hugging Face CLIP model id")]


class Device(str, Enum):
    auto = "auto"
    mps = "mps"
    cuda = "cuda"
    cpu = "cpu"


DeviceOpt = Annotated[Device, typer.Option(help="Where the classifier runs. auto: Apple GPU (mps) if available, else NVIDIA (cuda), else CPU")]


def _classifier(cats, model: str, min_confidence: float, device: Device) -> Classifier:
    try:
        clf = Classifier(cats, model_name=model, min_confidence=min_confidence, device=device.value)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    if clf.device != "none":
        console.print(f"Classifier runs on: {DEVICE_NAMES.get(clf.device, clf.device)}")
    return clf


def _categories(words: list[str]) -> list:
    try:
        return job.categories(words)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None


@app.command()
def crawl(
    keywords: KeywordsArg,
    source: Annotated[list[Source], typer.Option("--source", "-s", help="Source(s) to search; 'web' is a whole-web image search (Bing, SafeSearch off)")] = [Source.web, Source.openverse, Source.wikimedia],
    url: Annotated[list[str], typer.Option("--url", "-u", help="Crawl these websites instead of the APIs (repeatable)")] = [],
    limit: Annotated[int, typer.Option("--limit", "-n", help="Stop after saving this many images per keyword per source (or in total for --url). 0 = no limit, run until Ctrl+C or the sources run out")] = 0,
    out: OutOpt = Path("output"),
    min_size: Annotated[int, typer.Option(help="Skip images whose shorter side is below this (px)")] = 400,
    gifs: Annotated[bool, typer.Option(help="Also search for and keep animated GIFs")] = True,
    min_gif_size: Annotated[int, typer.Option(help="Like --min-size, for animated GIFs (they are usually small)")] = 100,
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
    device: DeviceOpt = Device.auto,
) -> None:
    """Search the web, download the images and sort them into one folder per search."""
    job.load_dotenv()
    opts = job.CrawlOptions(
        searches=keywords, sources=[s.value for s in source], urls=url, limit=limit, out=out,
        min_size=min_size, gifs=gifs, min_gif_size=min_gif_size, min_confidence=min_confidence,
        commercial=commercial, mature=mature, max_pages=max_pages, depth=depth, delay=delay,
        follow=follow, follow_pages=follow_pages, workers=workers, model=model, device=device.value,
    )
    cats = _categories(keywords)
    console.print(f"Searches: {', '.join(repr(c.query) + ' -> ' + c.name + '/' + (' (tonal)' if c.tonal else '') for c in cats)}")
    if limit <= 0:
        console.print("No limit set: crawling until the sources run out. Press Ctrl+C to stop.")
    console.print("Loading classifier ..." if any(not c.tonal for c in cats) else "Tonal-only run, no CLIP model needed.")
    classifier = _classifier(cats, model, min_confidence, device)
    stats = job.crawl(opts, classifier, cats)

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
    device: DeviceOpt = Device.auto,
) -> None:
    """Sort images you already have on disk into one folder per search phrase."""
    cats = _categories(keywords)
    classifier = _classifier(cats, model, min_confidence, device)
    _summary(sort_local(folder, Store(out), classifier, move=move))


@app.command()
def gui(
    out: Annotated[Path, typer.Option("--out", "-o", help="Default output folder (can be changed in the page)")] = Path("output"),
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1 (the next free one is used if taken)")] = 8765,
    browser: Annotated[bool, typer.Option(help="Open the page in the browser")] = True,
) -> None:
    """Open the graphical interface in your browser."""
    from .gui import serve

    serve(out, port=port, open_browser=browser)


@app.command("keywords")
def keywords_cmd() -> None:
    """List keywords with tuned prompts (any other word works too)."""
    for k in known_keywords():
        c = get_category(k)
        console.print(f"{k:<14} {'tonal' if c.tonal else 'semantic':<9} search: {c.query!r}")
