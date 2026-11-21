"""Keyword definitions: how to search for a keyword and how to recognise it.

Two kinds of keywords exist:

* **semantic** keywords (landscape, architecture, people, ...) describe the image
  content. They are classified with CLIP zero-shot prompts.
* **tonal** keywords (highkey, lowkey, bw) describe the light/tone of the image.
  They are measured directly from the pixel histogram, which is far more reliable
  than asking a vision model about brightness.

Any keyword not listed here is treated as semantic with generic prompts, so
``image-crawler crawl cats bridges`` works out of the box.
"""

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Category:
    name: str
    query: str  # search string sent to image APIs
    prompts: tuple[str, ...] = field(default_factory=tuple)  # CLIP prompts (semantic only)
    tonal: bool = False
    # CLIP prompts for "almost, but not quite" (the phrase with one word left out).
    # They compete with ``prompts``, so an image must match the whole phrase.
    contrasts: tuple[str, ...] = field(default_factory=tuple)


_SEMANTIC: dict[str, tuple[str, tuple[str, ...]]] = {
    "landscape": (
        "landscape photography",
        (
            "a landscape photograph of nature scenery",
            "a wide photo of mountains, hills, fields or a valley",
            "a scenic outdoor view with sky and horizon",
        ),
    ),
    "architecture": (
        "architecture photography",
        (
            "a photograph of architecture",
            "a photo of a building facade",
            "a photo of a modern or historic building interior",
        ),
    ),
    "people": (
        "people photography",
        (
            "a photo of a person",
            "a photo of people",
            "a photo of a woman",
            "a photo of a man",
            "a photo of a group of persons",
        ),
    ),
    "portrait": (
        "portrait photography",
        (
            "a close-up portrait photo of a person's face",
            "a studio portrait of a person",
        ),
    ),
    "animals": (
        "wildlife photography",
        ("a photo of an animal", "a wildlife photograph", "a photo of a pet"),
    ),
    "food": (
        "food photography",
        ("a photo of food on a plate", "a food photograph", "a photo of a meal"),
    ),
    "street": (
        "street photography",
        ("a street photograph in a city", "a candid photo of urban life"),
    ),
    "cityscape": (
        "cityscape skyline",
        ("a photo of a city skyline", "an aerial view of a city"),
    ),
    "macro": (
        "macro photography",
        ("an extreme close-up macro photo", "a macro photo of an insect or flower"),
    ),
    "night": (
        "night photography",
        ("a photo taken at night", "a night sky with stars", "city lights at night"),
    ),
    "seascape": (
        "seascape ocean",
        ("a photo of the sea or ocean", "a photo of a beach and waves"),
    ),
    "abstract": (
        "abstract photography",
        ("an abstract photograph of shapes and patterns", "an abstract texture"),
    ),
}

_TONAL: dict[str, str] = {
    "highkey": "high key photography",
    "lowkey": "low key photography",
    "bw": "black and white photography",
}

_ALIASES = {
    "high-key": "highkey",
    "high_key": "highkey",
    "low-key": "lowkey",
    "low_key": "lowkey",
    "blackandwhite": "bw",
    "monochrome": "bw",
    "person": "people",
    "buildings": "architecture",
    "building": "architecture",
    "landscapes": "landscape",
    "portraits": "portrait",
}

# Extra prompts that compete with the user's keywords so that images matching
# none of them land in "_unsorted" instead of being forced into a category.
DISTRACTOR_PROMPTS: tuple[str, ...] = (
    "a screenshot of text or a document",
    "a diagram, map or chart",
    "a logo or icon",
    "a drawing or painting",
    "a photo of an everyday object",
)


def parse_searches(words: list[str]) -> list[str]:
    """Turn command line words into searches: all words form one search, commas separate searches.

    ``["green", "house", "in", "snow,", "red", "car"]`` -> ``["green house in snow", "red car"]``
    """
    return [" ".join(part.split()) for part in " ".join(words).split(",") if part.strip()]


def normalize(keyword: str) -> str:
    """Folder name for a search: a known keyword ("high key" -> "highkey"), else a slug ("green-house-in-snow")."""
    compact = re.sub(r"[\s_-]+", "", keyword.lower())
    compact = _ALIASES.get(compact, compact)
    if compact in _TONAL or compact in _SEMANTIC:
        return compact
    return re.sub(r"[^\w]+", "-", keyword.lower()).strip("-") or "search"


def get_category(keyword: str) -> Category:
    """A search phrase as a category. The whole phrase is one search query and one CLIP
    prompt, so "green building in snow" means green buildings in a winter scene, not
    anything green or anything snowy."""
    name = normalize(keyword)
    if name in _TONAL:
        return Category(name=name, query=_TONAL[name], tonal=True)
    if name in _SEMANTIC:
        query, prompts = _SEMANTIC[name]
        return Category(name=name, query=query, prompts=prompts)
    raw = " ".join(keyword.split())
    return Category(
        name=name,
        query=raw,
        prompts=(f"a photo of {raw}", f"a photograph showing {raw}"),
        contrasts=tuple(f"a photo of {p}" for p in leave_one_out(raw)),
    )


_FILLER = {"a", "an", "the", "in", "on", "at", "of", "with", "and", "or", "by", "for", "from", "under", "over", "near", "during"}


def leave_one_out(phrase: str) -> list[str]:
    """The phrase with one content word dropped each time, filler words at the edges trimmed.

    "green house in snow" -> ["house in snow", "green in snow", "green house"]
    """
    words = phrase.split()
    content = [i for i, w in enumerate(words) if w.lower() not in _FILLER]
    if len(content) < 2:
        return []
    out = []
    for i in content:
        rest = words[:i] + words[i + 1 :]
        while rest and rest[0].lower() in _FILLER:
            rest.pop(0)
        while rest and rest[-1].lower() in _FILLER:
            rest.pop()
        if rest and " ".join(rest) not in out:
            out.append(" ".join(rest))
    return out


def known_keywords() -> list[str]:
    return sorted([*_SEMANTIC, *_TONAL])
