"""Download -> deduplicate -> classify -> file into category folders."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import imagehash
from PIL import Image, UnidentifiedImageError
from rich.console import Console

from .classify import Classifier
from .sources import Candidate

UNSORTED = "_unsorted"
_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tif"}

console = Console()


class Store:
    """Output folder with a metadata log and a perceptual-hash dedup index."""

    def __init__(self, root: Path, dedup_distance: int = 4) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = root / "metadata.jsonl"
        self.dedup_distance = dedup_distance
        self.hashes: list[imagehash.ImageHash] = []
        self.urls: set[str] = set()
        # Images saved in this run, per (source, query); used by ``sources.limit_new``.
        self.saved_by: Counter = Counter()
        self._lock = threading.Lock()
        if self.meta_path.exists():
            for line in self.meta_path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.urls.add(rec.get("url", ""))
                if rec.get("phash"):
                    self.hashes.append(imagehash.hex_to_hash(rec["phash"]))

    def saved_count(self, source: str | None = None, query: str | None = None) -> int:
        """Images saved in this run, for one source/query pair or (no arguments) in total."""
        if source is None:
            return sum(self.saved_by.values())
        return self.saved_by[(source, query)]

    def seen_url(self, url: str) -> bool:
        return url in self.urls

    def is_duplicate(self, h: imagehash.ImageHash) -> bool:
        return any(h - k <= self.dedup_distance for k in self.hashes)

    def save(self, data: bytes, ext: str, labels: list[str], record: dict) -> Path:
        name = f"{record['source']}_{hashlib.sha1(data).hexdigest()[:12]}.{ext}"
        folders = labels or [UNSORTED]
        first = self.root / folders[0] / name
        first.parent.mkdir(parents=True, exist_ok=True)
        first.write_bytes(data)
        # Additional labels (e.g. "landscape" + "highkey") share the same file via
        # hard links, so a multi-label image doesn't cost extra disk space.
        for f in folders[1:]:
            target = self.root / f / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(first, target)
                except OSError:
                    shutil.copy2(first, target)
        record["files"] = [f"{f}/{name}" for f in folders]
        with self._lock:
            with self.meta_path.open("a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.urls.add(record["url"])
            self.hashes.append(imagehash.hex_to_hash(record["phash"]))
            self.saved_by[(record["source"], record.get("query"))] += 1
        return first


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB, flattening transparency onto white (instead of random hidden colours)."""
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img if img.mode in ("RGB", "L") else img.convert("RGB")


def representative_frame(img: Image.Image) -> Image.Image:
    """The frame to classify, as RGB. For an animation that's a middle frame: the
    first one is often blank, a fade-in or a title card. The file itself is saved
    unchanged, so it stays animated."""
    if getattr(img, "is_animated", False):
        try:
            img.seek(img.n_frames // 2)
        except EOFError:
            img.seek(0)
    return to_rgb(img)


def download(client: httpx.Client, url: str, max_bytes: int) -> tuple[bytes, Image.Image, str] | None:
    try:
        with client.stream("GET", url) as r:
            if r.status_code != 200 or not r.headers.get("content-type", "image/").startswith("image/"):
                return None
            buf = bytearray()
            for chunk in r.iter_bytes():
                buf += chunk
                if len(buf) > max_bytes:
                    return None
    except httpx.HTTPError:
        return None
    data = bytes(buf)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return None
    ext = _EXT.get(img.format or "", None)
    if ext is None:
        return None
    return data, img, ext


def run(
    candidates: Iterable[Candidate],
    client: httpx.Client,
    store: Store,
    classifier: Classifier,
    min_size: int = 400,
    max_bytes: int = 25 * 1024 * 1024,
    workers: int = 4,
    full: Callable[[Candidate], bool] = lambda cand: False,
    min_gif_size: int = 100,
    stop: threading.Event | None = None,
    stats: Counter | None = None,
    show_progress: bool = True,
) -> Counter:
    """Download, classify and save ``candidates``.

    ``full(cand)`` says whether the candidate's ``-n`` quota is already used up.
    Several downloads run at once, so results that finish after the quota filled
    up are dropped instead of saved, which keeps ``-n`` exact.

    For the GUI: setting ``stop`` ends the run like Ctrl+C, and ``stats`` (updated
    live) lets another thread watch the progress.
    """
    stats = Counter() if stats is None else stats

    def handle(cand: Candidate, res: tuple[bytes, Image.Image, str] | None) -> None:
        if full(cand):
            return
        if res is None:
            stats["failed"] += 1
            return
        data, img, ext = res
        animated = getattr(img, "is_animated", False)
        # Animated GIFs are usually much smaller than photos, so they get their own limit.
        if min(img.size) < (min_gif_size if animated else min_size):
            stats["too_small"] += 1
            return
        img = representative_frame(img)
        h = imagehash.phash(img)
        if store.is_duplicate(h):
            stats["duplicate"] += 1
            return
        result = classifier.classify(img)
        record = {
            **asdict(cand),
            "width": img.size[0],
            "height": img.size[1],
            "phash": str(h),
            "primary": result.primary,
            "labels": result.labels,
            "semantic_scores": result.semantic_scores,
            "tonal_scores": result.tonal_scores,
            "tone": result.tone,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = store.save(data, ext, result.labels, record)
        for lbl in result.labels or [UNSORTED]:
            stats[lbl] += 1
        stats["saved"] += 1
        if animated:
            stats["animated"] += 1
        console.print(f"[green]saved[/] {', '.join(result.labels) or UNSORTED:<24} {path.name}  [dim]{cand.title or ''}[/]"[:180])

    pending: dict[Future, Candidate] = {}
    pool = ThreadPoolExecutor(max_workers=workers)
    found = 0

    def progress(doing: str) -> str:
        # A live line, so a run that hasn't saved anything yet doesn't look stuck.
        parts = [f"found {found}", f"saved {stats['saved']}"]
        parts += [f"{k.replace('_', ' ')} {stats[k]}" for k in ("already_have", "too_small", "duplicate", "failed") if stats[k]]
        return f"{' · '.join(parts)}  [dim]{doing}[/]"

    status = console.status(progress("searching ..."))
    if show_progress:
        status.start()
    try:
        for cand in candidates:
            if stop is not None and stop.is_set():
                stats["stopped"] = 1
                break
            found += 1
            stats["found"] = found
            if store.seen_url(cand.url):
                stats["already_have"] += 1
                status.update(progress("searching ..."))
                continue
            if full(cand):
                continue
            store.urls.add(cand.url)  # avoid fetching the same URL twice in one run
            pending[pool.submit(download, client, cand.url, max_bytes)] = cand
            status.update(progress(f"downloading from {urlparse(cand.url).netloc}"))
            if len(pending) >= workers * 2:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for f in done:
                    handle(pending.pop(f), f.result())
            status.update(progress("searching ..."))
        status.update(progress("finishing downloads ..."))
        for f in list(pending):
            if stop is not None and stop.is_set():
                break
            handle(pending.pop(f), f.result())
    except KeyboardInterrupt:
        # Ctrl+C ends an unlimited crawl. Every saved image is already complete on
        # disk and in metadata.jsonl, so unfinished downloads are just dropped.
        console.print("[yellow]Interrupted, stopping.[/]")
        stats["interrupted"] = 1
    finally:
        status.stop()
        pool.shutdown(wait=False, cancel_futures=True)
    return stats


def sort_local(
    src: Path, store: Store, classifier: Classifier, min_size: int = 0, move: bool = False
) -> Counter:
    stats: Counter = Counter()
    files = [p for p in sorted(src.rglob("*")) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff", ".bmp"}]
    for p in files:
        data = p.read_bytes()
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError):
            stats["failed"] += 1
            continue
        if min(img.size) < min_size:
            stats["too_small"] += 1
            continue
        img = representative_frame(img)  # middle frame for animations
        h = imagehash.phash(img)
        if store.is_duplicate(h):
            stats["duplicate"] += 1
            console.print(f"[dim]skip duplicate {p.name}[/]")
            continue
        result = classifier.classify(img)
        record = {
            "url": p.resolve().as_uri(),
            "source": "local",
            "title": p.name,
            "phash": str(h),
            "primary": result.primary,
            "labels": result.labels,
            "semantic_scores": result.semantic_scores,
            "tonal_scores": result.tonal_scores,
            "tone": result.tone,
        }
        ext = _EXT.get(img.format or "", p.suffix.lstrip(".").lower())
        out = store.save(data, ext, result.labels, record)
        if move:
            p.unlink()
        for lbl in result.labels or [UNSORTED]:
            stats[lbl] += 1
        stats["saved"] += 1
        console.print(f"[green]sorted[/] {', '.join(result.labels) or UNSORTED:<24} {p.name} -> {out.parent.name}/")
    return stats
