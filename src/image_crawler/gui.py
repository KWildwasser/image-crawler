"""A small local web GUI: ``image-crawler gui`` opens it in the browser.

Runs on Python's standard library HTTP server and only listens on 127.0.0.1.
The page polls ``/api/status`` and ``/api/images`` once a second while a crawl
runs; the crawl itself runs in a background thread with the same code as the CLI.
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import job, sources
from .classify import DEFAULT_MODEL, DEVICE_NAMES, Classifier

MODELS = {
    "openai/clip-vit-base-patch32": "Standard (fast)",
    "openai/clip-vit-large-patch14": "Accurate (slower, 1.7 GB download)",
}


class Crawler:
    """The one crawl the GUI runs at a time, plus its live state."""

    def __init__(self, out: Path) -> None:
        self.lock = threading.Lock()
        self.out = out.resolve()
        self.state = "idle"  # idle | loading | running | stopping | done | stopped | error
        self.message = ""
        self.device = ""
        self.searches: list[str] = []
        self.stats: Counter = Counter()
        self.warnings: list[str] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._clf_key: tuple | None = None
        self._clf: Classifier | None = None
        sources.warning_listeners.append(self._warning)

    def _warning(self, msg: str) -> None:
        with self.lock:
            if msg not in self.warnings:
                self.warnings.append(msg)
                del self.warnings[:-20]

    @property
    def busy(self) -> bool:
        return self.state in ("loading", "running", "stopping")

    def start(self, o: job.CrawlOptions) -> None:
        if self.busy:
            raise ValueError("a crawl is already running")
        cats = job.categories(o.searches)  # raises ValueError on empty input
        self.out = o.out.resolve()
        self.searches = [c.query for c in cats]
        self.stats = Counter()
        self.warnings = []
        self.stop_event = threading.Event()
        self.state, self.message = "loading", "Loading the classifier ..."
        self.thread = threading.Thread(target=self._run, args=(o, cats), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.busy:
            self.stop_event.set()
            self.state, self.message = "stopping", "Stopping after the current downloads ..."

    def _classifier(self, cats, o: job.CrawlOptions) -> Classifier:
        # Reuse the loaded model when only the output or the sources change.
        key = (tuple(c.name for c in cats), o.model, o.min_confidence, o.device)
        if key != self._clf_key:
            self._clf = Classifier(cats, model_name=o.model, min_confidence=o.min_confidence, device=o.device)
            self._clf_key = key
        return self._clf

    def _run(self, o: job.CrawlOptions, cats) -> None:
        try:
            clf = self._classifier(cats, o)
            self.device = DEVICE_NAMES.get(clf.device, "") if clf.device != "none" else "no model needed"
            if self.stop_event.is_set():
                self.state, self.message = "stopped", "Stopped."
                return
            self.state, self.message = "running", "Searching ..."
            job.crawl(o, clf, cats, stop=self.stop_event, stats=self.stats, show_progress=False)
            if self.stop_event.is_set():
                self.state, self.message = "stopped", "Stopped."
            else:
                self.state = "done"
                self.message = "Finished: the sources have no more results." if self.stats else "Nothing found. See the warnings."
        except Exception as e:  # show it in the page instead of killing the thread silently
            traceback.print_exc()
            self.state, self.message = "error", f"Error: {e}"

    def status(self) -> dict:
        with self.lock:
            warnings = list(self.warnings)
        return {
            "state": self.state,
            "message": self.message,
            "device": self.device,
            "searches": self.searches,
            "stats": dict(self.stats),
            "warnings": warnings,
            "out": str(self.out),
        }


def read_images(out: Path, offset: int) -> tuple[list[dict], int]:
    """Records appended to ``metadata.jsonl`` since byte ``offset``, and the new offset."""
    meta = out / "metadata.jsonl"
    if not meta.exists():
        return [], 0
    size = meta.stat().st_size
    if offset > size:  # file was replaced: start over
        offset = 0
    with meta.open("rb") as fh:
        fh.seek(offset)
        chunk = fh.read()
    # Only hand out complete lines; a line being written right now waits for the next poll.
    end = chunk.rfind(b"\n") + 1
    recs = []
    for line in chunk[:end].splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("files"):
            continue
        recs.append({
            "file": r["files"][0],
            "folders": [f.split("/", 1)[0] for f in r["files"]],
            "title": r.get("title"),
            "page_url": r.get("page_url"),
            "url": r.get("url"),
            "source": r.get("source"),
            "query": r.get("query"),
            "width": r.get("width"),
            "height": r.get("height"),
            "scores": r.get("semantic_scores") or {},
        })
    return recs, offset + end


@lru_cache(maxsize=3000)
def thumbnail(path: str, mtime: float, size: int = 360) -> bytes:
    """A small JPEG for the gallery grid (``mtime`` is only part of the cache key).

    Full-size files are often several MB (GIFs especially), and a grid of them
    would make the browser load and animate hundreds of MB at once.
    """
    from PIL import Image

    from .pipeline import representative_frame

    with Image.open(path) as im:
        frame = representative_frame(im)  # middle frame of a GIF, as RGB
        frame.thumbnail((size, size))
        buf = io.BytesIO()
        frame.convert("RGB").save(buf, "JPEG", quality=82)
        return buf.getvalue()


def open_in_file_manager(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(path)])


def options_from_json(d: dict, default_out: Path) -> job.CrawlOptions:
    def num(key, cast, default, lo=None, hi=None):
        try:
            v = cast(d.get(key, default))
        except (TypeError, ValueError):
            return default
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v

    srcs = [s for s in d.get("sources", []) if s in job.SOURCES]
    url = (d.get("url") or "").strip()
    out = Path(os.path.expanduser((d.get("out") or "").strip() or str(default_out)))
    model = d.get("model") if d.get("model") in MODELS else DEFAULT_MODEL
    return job.CrawlOptions(
        searches=[str(d.get("search", ""))],
        sources=srcs or ["web"],
        urls=[url] if url else [],
        limit=num("limit", int, 0, 0),
        out=out,
        gifs=bool(d.get("gifs", True)),
        follow=bool(d.get("follow", True)),
        mature=bool(d.get("mature", True)),
        min_confidence=num("min_confidence", float, 0.30, 0.05, 0.95),
        model=model,
    )


def make_handler(crawler: Crawler, page: bytes, default_out: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

        # -- helpers ---------------------------------------------------
        def _local(self) -> bool:
            # Refuse requests from other sites (DNS rebinding / cross-site posts).
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode(), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return {}

        # -- routes ------------------------------------------------------
        def do_GET(self) -> None:
            if not self._local():
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                return self._send(200, page, "text/html; charset=utf-8")
            if u.path == "/api/status":
                return self._json(crawler.status())
            if u.path == "/api/images":
                out = Path(os.path.expanduser(q.get("out", [str(crawler.out)])[0])).resolve()
                recs, offset = read_images(out, int(q.get("offset", ["0"])[0] or 0))
                return self._json({"images": recs, "offset": offset, "out": str(out)})
            if u.path.startswith(("/media/", "/thumb/")):
                kind, rel = u.path[1:].split("/", 1)
                out = Path(os.path.expanduser(q.get("out", [str(crawler.out)])[0])).resolve()
                target = (out / unquote(rel)).resolve()
                if not target.is_relative_to(out) or not target.is_file():
                    return self._send(404, b"not found", "text/plain")
                if kind == "thumb":
                    try:
                        data = thumbnail(str(target), target.stat().st_mtime)
                    except Exception:  # unreadable image: let the browser try the original
                        return self._send(302, b"", "text/plain", {"Location": "/media/" + rel + "?" + u.query})
                    return self._send(200, data, "image/jpeg", {"Cache-Control": "max-age=3600"})
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                return self._send(200, target.read_bytes(), ctype, {"Cache-Control": "max-age=3600"})
            return self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:
            # A JSON content type forces a CORS preflight for cross-site requests, which we never answer.
            if not self._local() or not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._send(403, b"forbidden", "text/plain")
            u = urlparse(self.path)
            d = self._body()
            if u.path == "/api/start":
                try:
                    crawler.start(options_from_json(d, default_out))
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)
                return self._json({"ok": True})
            if u.path == "/api/stop":
                crawler.stop()
                return self._json({"ok": True})
            if u.path == "/api/open":
                out = Path(os.path.expanduser(d.get("out") or str(crawler.out))).resolve()
                sub = (out / (d.get("folder") or "")).resolve()
                if not sub.is_relative_to(out):
                    return self._json({"error": "bad folder"}, 400)
                sub.mkdir(parents=True, exist_ok=True)
                open_in_file_manager(sub)
                return self._json({"ok": True})
            return self._send(404, b"not found", "text/plain")

    return Handler


def serve(out: Path = Path("output"), port: int = 8765, open_browser: bool = True) -> None:
    job.load_dotenv()
    out = out.resolve()
    crawler = Crawler(out)
    page = files("image_crawler").joinpath("gui.html").read_bytes()
    page = page.replace(b"__DEFAULT_OUT__", json.dumps(str(out)).encode()).replace(
        b"__MODELS__", json.dumps(MODELS).encode()
    )
    server = None
    for p in range(port, port + 20):  # next free port if this one is taken
        try:
            server = ThreadingHTTPServer(("127.0.0.1", p), make_handler(crawler, page, out))
            port = p
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"no free port between {port} and {port + 19}")
    server.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print(f"Image Crawler is running at {url}  (press Ctrl+C here to quit)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
        crawler.stop()
        time.sleep(0.2)
    finally:
        server.server_close()
