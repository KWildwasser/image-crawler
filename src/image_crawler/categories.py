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

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Category:
    name: str
    query: str  # search string sent to image APIs
    prompts: tuple[str, ...] = field(default_factory=tuple)  # CLIP prompts (semantic only)
    tonal: bool = False


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


def normalize(keyword: str) -> str:
    k = keyword.strip().lower().replace(" ", "")
    return _ALIASES.get(k, k)


def get_category(keyword: str) -> Category:
    name = normalize(keyword)
    if name in _TONAL:
        return Category(name=name, query=_TONAL[name], tonal=True)
    if name in _SEMANTIC:
        query, prompts = _SEMANTIC[name]
        return Category(name=name, query=query, prompts=prompts)
    raw = keyword.strip()
    return Category(name=name, query=raw, prompts=(f"a photo of {raw}", f"a photograph showing {raw}"))


def known_keywords() -> list[str]:
    return sorted([*_SEMANTIC, *_TONAL])
