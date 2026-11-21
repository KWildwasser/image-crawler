"""Image sources. Each source yields ``Candidate`` objects (URL + attribution)."""

from __future__ import annotations

import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup


@dataclass
class Candidate:
    url: str
    source: str
    page_url: str | None = None
    title: str | None = None
    creator: str | None = None
    license: str | None = None
    license_url: str | None = None
    query: str | None = None


def _strip_html(s: str | None) -> str | None:
    return re.sub(r"<[^>]+>", "", s).strip() if s else s


def safe(gen: Iterator[Candidate]) -> Iterator[Candidate]:
    """Wrap a source so a failing/refusing API logs a warning instead of aborting the run."""
    try:
        yield from gen
    except (httpx.HTTPError, ValueError) as e:
        from rich.console import Console

        Console(stderr=True).print(f"[yellow]warning:[/] source {getattr(gen, '__name__', '?')} failed: {e}".splitlines()[0])


def limit_new(cands: Iterator[Candidate], limit: int, seen: Callable[[str], bool]) -> Iterator[Candidate]:
    """Pass candidates through until ``limit`` new ones have been yielded.

    Already-saved URLs (``seen``) are still yielded, so the pipeline can count them,
    but they don't use up the limit. Re-running a crawl therefore continues past the
    images you already have instead of re-checking the same first results. Sources
    are lazy, so the next result page is only requested when it's needed.
    """
    if limit <= 0:
        return
    new = 0
    for cand in cands:
        is_new = not seen(cand.url)  # check before yielding: the pipeline marks it seen
        yield cand
        if is_new:
            new += 1
            if new >= limit:
                return


# --------------------------------------------------------------------------
# Openverse (https://openverse.org) - CC-licensed / public domain images
# --------------------------------------------------------------------------
_OV = "https://api.openverse.org/v1"
_ov_token: str | None = None


def _openverse_headers(client: httpx.Client) -> dict[str, str]:
    """Use registered credentials if OPENVERSE_CLIENT_ID/SECRET are set (higher limits)."""
    global _ov_token
    cid, secret = os.environ.get("OPENVERSE_CLIENT_ID"), os.environ.get("OPENVERSE_CLIENT_SECRET")
    if not (cid and secret):
        return {}
    if _ov_token is None:
        r = client.post(
            f"{_OV}/auth_tokens/token/",
            data={"grant_type": "client_credentials", "client_id": cid, "client_secret": secret},
        )
        r.raise_for_status()
        _ov_token = r.json()["access_token"]
    return {"Authorization": f"Bearer {_ov_token}"}


def openverse(client: httpx.Client, query: str, commercial: bool = False) -> Iterator[Candidate]:
    page = 1
    headers = _openverse_headers(client)
    while True:
        params = {"q": query, "page_size": 20, "page": page, "mature": "false"}
        if commercial:
            params["license_type"] = "commercial"
        r = client.get(f"{_OV}/images/", params=params, headers=headers)
        if r.status_code in (400, 401, 429):  # anonymous page limit reached / throttled
            return
        if r.status_code == 403 and r.headers.get("cf-mitigated"):
            raise httpx.HTTPError(
                "Openverse returned a Cloudflare bot challenge (common from datacenter IPs). "
                "Register API credentials (see README) or use another source."
            )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return
        for it in results:
            yield Candidate(
                url=it["url"],
                source="openverse",
                page_url=it.get("foreign_landing_url"),
                title=it.get("title"),
                creator=it.get("creator"),
                license=f"{it.get('license', '')} {it.get('license_version') or ''}".strip().upper(),
                license_url=it.get("license_url"),
                query=query,
            )
        page += 1


# --------------------------------------------------------------------------
# Wikimedia Commons - freely licensed media
# --------------------------------------------------------------------------
def wikimedia(client: httpx.Client, query: str, width: int = 1600) -> Iterator[Candidate]:
    offset = 0
    while True:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": 30,
            "gsroffset": offset,
            "prop": "imageinfo",
            "iiprop": "url|mime|extmetadata",
            "iiurlwidth": width,
        }
        r = client.get("https://commons.wikimedia.org/w/api.php", params=params)
        if r.status_code == 403 and "robot policy" in r.text:
            raise httpx.HTTPError(
                "Wikimedia requires contact info in the User-Agent. Set CRAWLER_CONTACT "
                "(email or URL) in .env - see README."
            )
        r.raise_for_status()
        data = r.json()
        pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        if not pages:
            return
        for p in pages:
            info = (p.get("imageinfo") or [{}])[0]
            if info.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            meta = info.get("extmetadata", {})
            yield Candidate(
                url=info.get("thumburl") or info["url"],
                source="wikimedia",
                page_url=info.get("descriptionurl"),
                title=p.get("title", "").removeprefix("File:"),
                creator=_strip_html(meta.get("Artist", {}).get("value")),
                license=meta.get("LicenseShortName", {}).get("value"),
                license_url=meta.get("LicenseUrl", {}).get("value"),
                query=query,
            )
        cont = data.get("continue", {}).get("gsroffset")
        if cont is None:
            return
        offset = cont


# --------------------------------------------------------------------------
# Pexels (https://www.pexels.com/api/) - free API key, Pexels license
# --------------------------------------------------------------------------
def pexels(client: httpx.Client, query: str) -> Iterator[Candidate]:
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        raise httpx.HTTPError("PEXELS_API_KEY is not set (get a free key at https://www.pexels.com/api/)")
    page = 1
    while True:
        r = client.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": 80, "page": page},
            headers={"Authorization": key},
        )
        r.raise_for_status()
        photos = r.json().get("photos", [])
        if not photos:
            return
        for ph in photos:
            yield Candidate(
                url=ph["src"].get("large2x") or ph["src"]["original"],
                source="pexels",
                page_url=ph.get("url"),
                title=ph.get("alt"),
                creator=ph.get("photographer"),
                license="Pexels License",
                license_url="https://www.pexels.com/license/",
                query=query,
            )
        page += 1


# --------------------------------------------------------------------------
# Generic website crawler (same host, robots.txt-aware, rate limited)
# --------------------------------------------------------------------------
_SKIP_EXT = re.compile(r"\.(pdf|zip|gz|mp4|mp3|css|js|svg|ico|woff2?)(\?|$)", re.I)


def _largest_from_srcset(srcset: str) -> str | None:
    best, best_w = None, -1
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        w = 0
        if len(bits) > 1 and bits[1][:-1].replace(".", "").isdigit():
            w = float(bits[1][:-1])
        if w > best_w:
            best, best_w = bits[0], w
    return best


def site(
    client: httpx.Client,
    start_urls: list[str],
    max_pages: int = 50,
    max_depth: int = 2,
    delay: float = 1.0,
    user_agent: str = "*",
) -> Iterator[Candidate]:
    robots: dict[str, RobotFileParser] = {}

    def allowed(url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in robots:
            rp = RobotFileParser()
            try:
                r = client.get(f"{base}/robots.txt")
                rp.parse(r.text.splitlines() if r.status_code == 200 else [])
            except httpx.HTTPError:
                rp.parse([])
            robots[base] = rp
        return robots[base].can_fetch(user_agent, url)

    hosts = {urlparse(u).netloc for u in start_urls}
    queue = deque((u, 0) for u in start_urls)
    seen_pages: set[str] = set()
    seen_imgs: set[str] = set()
    pages = 0

    while queue and pages < max_pages:
        url, depth = queue.popleft()
        if url in seen_pages or not allowed(url):
            continue
        seen_pages.add(url)
        try:
            r = client.get(url)
        except httpx.HTTPError:
            continue
        pages += 1
        time.sleep(delay)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else None

        img_urls: list[tuple[str, str | None]] = []
        for meta in soup.select('meta[property="og:image"], meta[name="twitter:image"]'):
            if meta.get("content"):
                img_urls.append((meta["content"], title))
        for img in soup.find_all("img"):
            src = None
            if img.get("srcset"):
                src = _largest_from_srcset(img["srcset"])
            src = src or img.get("data-src") or img.get("src")
            if src and not src.startswith("data:"):
                img_urls.append((src, img.get("alt") or title))

        for src, alt in img_urls:
            full = urldefrag(urljoin(str(r.url), src))[0]
            if full in seen_imgs or _SKIP_EXT.search(full) or not allowed(full):
                continue
            seen_imgs.add(full)
            yield Candidate(url=full, source="site", page_url=str(r.url), title=alt)

        if depth < max_depth:
            for a in soup.find_all("a", href=True):
                nxt = urldefrag(urljoin(str(r.url), a["href"]))[0]
                if urlparse(nxt).netloc in hosts and nxt not in seen_pages and not _SKIP_EXT.search(nxt):
                    queue.append((nxt, depth + 1))
