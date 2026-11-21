"""Image sources. Each source yields ``Candidate`` objects (URL + attribution)."""

from __future__ import annotations

import heapq
import itertools
import os
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
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

    ``limit <= 0`` means no limit: everything the source produces is passed on.

    Already-saved URLs (``seen``) are still yielded, so the pipeline can count them,
    but they don't use up the limit. Re-running a crawl therefore continues past the
    images you already have instead of re-checking the same first results. Sources
    are lazy, so the next result page is only requested when it's needed.
    """
    if limit <= 0:
        yield from cands
        return
    new = 0
    for cand in cands:
        is_new = not seen(cand.url)  # check before yielding: the pipeline marks it seen
        yield cand
        if is_new:
            new += 1
            if new >= limit:
                return


def interleave(gens: list[Iterator[Candidate]]) -> Iterator[Candidate]:
    """Round-robin over several sources until all of them are exhausted.

    With no limit, chaining the sources would never get past the first one, so
    every keyword/source pair takes turns and all folders fill up together.
    """
    active = list(gens)
    while active:
        for g in list(active):
            try:
                yield next(g)
            except StopIteration:
                active.remove(g)


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


_GENERIC_TERMS = {"photo", "photos", "photograph", "photography", "image", "images", "picture", "pictures", "the", "and", "with", "of"}


def keyword_terms(name: str, query: str, split_words: bool = True) -> list[str]:
    """Lower-case terms used to judge whether a link is about one keyword.

    The keyword itself and its search phrase always count ("high key", "high-key").
    With ``split_words`` the phrase's single words count too; that's off for tonal
    keywords, where "key" or "low" alone would match almost anything.
    """
    words = [w for w in re.split(r"[^\w]+", query.lower()) if w and w not in _GENERIC_TERMS]
    terms = {name.lower()}
    if words:
        terms |= {" ".join(words), "-".join(words), "".join(words)}
        if split_words:
            terms |= {w for w in words if len(w) >= 3}
    return sorted(t for t in terms if len(t) >= 3)


def relevance(text: str, terms: list[str]) -> int:
    """Number of keyword terms that occur in ``text`` (substring match, so plurals count)."""
    text = text.lower()
    return sum(1 for t in terms if t in text)


@dataclass
class CrawlState:
    """State shared between website crawls in one run."""

    max_host_pages: int = 0  # 0 = no per-host budget
    robots: dict[str, RobotFileParser] = field(default_factory=dict)
    pages: set[str] = field(default_factory=set)
    images: set[str] = field(default_factory=set)
    host_pages: Counter = field(default_factory=Counter)


def site(
    client: httpx.Client,
    start_urls: list[str],
    max_pages: int = 0,
    max_depth: int = 2,
    delay: float = 1.0,
    user_agent: str = "*",
    terms: list[str] = (),
    source: str = "site",
    query: str | None = None,
    shared: CrawlState | None = None,
) -> Iterator[Candidate]:
    """Crawl the hosts of ``start_urls`` and yield every image found.

    The crawl is focused on ``terms``: links whose URL, anchor text or title mention
    a keyword are visited first, and following them does not use up ``max_depth``,
    so the crawler keeps going deeper along keyword-related pages. Unrelated links
    are still followed, but only up to ``max_depth`` hops from the last related page.
    ``max_pages <= 0`` means no page limit.

    ``shared`` lets several crawls share robots.txt rules, visited pages and a
    per-host page budget, so following many source pages on one host (e.g. Flickr)
    neither re-fetches pages nor hammers that host.
    """
    st = shared or CrawlState()
    robots = st.robots

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
    order = itertools.count()
    # (-relevance, depth, insertion order, url): most relevant first, then shallowest, then FIFO.
    queue: list[tuple[int, int, int, str]] = [(0, 0, next(order), u) for u in start_urls]
    queued: set[str] = set(start_urls)
    seen_imgs = st.images
    pages = 0

    while queue and (max_pages <= 0 or pages < max_pages):
        _, depth, _, url = heapq.heappop(queue)
        host = urlparse(url).netloc
        if url in st.pages or not allowed(url):
            continue
        if st.max_host_pages and st.host_pages[host] >= st.max_host_pages:
            continue
        st.pages.add(url)
        st.host_pages[host] += 1
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
            yield Candidate(url=full, source=source, page_url=str(r.url), title=alt, query=query)

        for a in soup.find_all("a", href=True):
            nxt = urldefrag(urljoin(str(r.url), a["href"]))[0]
            if nxt in queued or urlparse(nxt).netloc not in hosts or _SKIP_EXT.search(nxt):
                continue
            score = relevance(f"{nxt} {a.get_text(' ', strip=True)} {a.get('title', '')}", terms) if terms else 0
            # Keyword-related links don't use up depth, so the crawl goes deeper along them.
            nxt_depth = 0 if score else depth + 1
            if nxt_depth > max_depth:
                continue
            queued.add(nxt)
            heapq.heappush(queue, (-score, nxt_depth, next(order), nxt))


# --------------------------------------------------------------------------
# Following API results to their source pages
# --------------------------------------------------------------------------
# These hosts are already searched through their API (and Pexels doesn't allow
# scraping), so their HTML pages aren't crawled.
_NO_FOLLOW_HOSTS = {"commons.wikimedia.org", "pexels.com", "www.pexels.com", "openverse.org"}


class FollowMixer:
    """Round-robin over the API sources and crawls of the pages their results came from.

    Each API result's ``page_url`` (e.g. the Flickr or museum page behind an
    Openverse hit) is queued. Up to ``parallel`` of those pages are crawled at a
    time, each a small keyword-focused crawl of that page's site
    (``pages_per_source`` pages), taking turns with the API sources.
    """

    def __init__(
        self,
        client: httpx.Client,
        gens: list[Iterator[Candidate]],
        terms_for: Callable[[str | None], list[str]],
        pages_per_source: int = 10,
        max_depth: int = 1,
        delay: float = 1.0,
        parallel: int = 4,
        max_host_pages: int = 200,
    ) -> None:
        self.client = client
        self.gens = gens
        self.terms_for = terms_for
        self.pages_per_source = pages_per_source
        self.max_depth = max_depth
        self.delay = delay
        self.parallel = parallel
        self.state = CrawlState(max_host_pages=max_host_pages)
        self.backlog: list[tuple[str, str | None]] = []
        self.queued: set[str] = set()

    def _note(self, cand: Candidate) -> None:
        url = cand.page_url
        if not url or url in self.queued or not url.startswith(("http://", "https://")):
            return
        if urlparse(url).netloc.lower() in _NO_FOLLOW_HOSTS:
            return
        self.queued.add(url)
        self.backlog.append((url, cand.query))

    def _crawl(self, url: str, query: str | None) -> Iterator[Candidate]:
        return safe(site(
            self.client, [url], max_pages=self.pages_per_source, max_depth=self.max_depth,
            delay=self.delay, terms=self.terms_for(query), source="followed", query=query,
            shared=self.state,
        ))

    def __iter__(self) -> Iterator[Candidate]:
        apis = list(self.gens)
        crawls: list[Iterator[Candidate]] = []
        while apis or crawls or self.backlog:
            while self.backlog and len(crawls) < self.parallel:
                crawls.append(self._crawl(*self.backlog.pop(0)))
            for g in list(apis):
                try:
                    cand = next(g)
                except StopIteration:
                    apis.remove(g)
                    continue
                self._note(cand)
                yield cand
            for g in list(crawls):
                try:
                    yield next(g)
                except StopIteration:
                    crawls.remove(g)
