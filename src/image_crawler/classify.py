"""Image classification: tonal analysis (histogram) + semantic analysis (CLIP)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .categories import DISTRACTOR_PROMPTS, Category

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class ToneStats:
    mean: float  # mean luminance 0..1
    median: float
    dark_frac: float  # share of pixels with luminance < 0.25
    bright_frac: float  # share of pixels with luminance > 0.70
    saturation: float  # mean HSV saturation 0..1

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def tone_stats(img: Image.Image) -> ToneStats:
    small = img.convert("RGB")
    small.thumbnail((512, 512))
    lum = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    sat = np.asarray(small.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
    return ToneStats(
        mean=float(lum.mean()),
        median=float(np.median(lum)),
        dark_frac=float((lum < 0.25).mean()),
        bright_frac=float((lum > 0.70).mean()),
        saturation=float(sat.mean()),
    )


def tonal_labels(s: ToneStats) -> dict[str, float]:
    """Return a 0..1 score per tonal keyword (>= 0.5 means the label applies)."""

    def clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    # High key: predominantly bright, very few deep shadows.
    highkey = clamp((s.median - 0.55) / 0.25) * clamp(1 - s.dark_frac / 0.12) * clamp(s.bright_frac / 0.5)
    # Low key: predominantly dark, with at least a bit of light (not just black).
    lowkey = clamp((0.40 - s.median) / 0.25) * clamp(s.dark_frac / 0.55)
    # Black & white: almost no colour saturation.
    bw = clamp(1 - s.saturation / 0.08)
    return {"highkey": round(highkey, 4), "lowkey": round(lowkey, 4), "bw": round(bw, 4)}


@dataclass
class Result:
    primary: str | None  # best semantic category, None -> unsorted
    labels: list[str]  # all categories the image is filed under
    semantic_scores: dict[str, float] = field(default_factory=dict)
    tonal_scores: dict[str, float] = field(default_factory=dict)
    tone: dict[str, float] = field(default_factory=dict)


class Classifier:
    def __init__(
        self,
        categories: list[Category],
        model_name: str = DEFAULT_MODEL,
        min_confidence: float = 0.30,
        tonal_threshold: float = 0.5,
    ) -> None:
        self.semantic = [c for c in categories if not c.tonal]
        self.tonal = [c for c in categories if c.tonal]
        self.min_confidence = min_confidence
        self.tonal_threshold = tonal_threshold
        self._model = None
        if self.semantic:
            self._load_clip(model_name)

    # -- CLIP -------------------------------------------------------------
    def _load_clip(self, model_name: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._model = CLIPModel.from_pretrained(model_name).eval()
        self._proc = CLIPProcessor.from_pretrained(model_name)

        # One averaged text embedding per category (prompt ensembling) plus
        # distractors so off-topic images don't get forced into a category.
        self._labels = [c.name for c in self.semantic] + [f"_distractor{i}" for i in range(len(DISTRACTOR_PROMPTS))]
        groups = [list(c.prompts) for c in self.semantic] + [[p] for p in DISTRACTOR_PROMPTS]
        embs = []
        with torch.no_grad():
            for prompts in groups:
                tok = self._proc(text=prompts, return_tensors="pt", padding=True)
                e = self._as_tensor(self._model.get_text_features(**tok))
                e = e / e.norm(dim=-1, keepdim=True)
                m = e.mean(dim=0)
                embs.append(m / m.norm())
        self._text = torch.stack(embs)
        self._scale = self._model.logit_scale.exp()

    @staticmethod
    def _as_tensor(out):
        # transformers 5 may return a model output object instead of a tensor
        return out if hasattr(out, "norm") else out.pooler_output

    def _semantic_scores(self, img: Image.Image) -> dict[str, float]:
        torch = self._torch
        with torch.no_grad():
            px = self._proc(images=img.convert("RGB"), return_tensors="pt")
            e = self._as_tensor(self._model.get_image_features(**px))
            e = e / e.norm(dim=-1, keepdim=True)
            probs = (self._scale * e @ self._text.T).softmax(dim=-1)[0].tolist()
        scores = dict(zip(self._labels, probs))
        distract = sum(v for k, v in scores.items() if k.startswith("_distractor"))
        out = {k: round(v, 4) for k, v in scores.items() if not k.startswith("_distractor")}
        out["_other"] = round(distract, 4)
        return out

    # -- public -----------------------------------------------------------
    def classify(self, img: Image.Image) -> Result:
        stats = tone_stats(img)
        res = Result(primary=None, labels=[], tone=stats.as_dict())

        if self.semantic:
            res.semantic_scores = self._semantic_scores(img)
            cand = {k: v for k, v in res.semantic_scores.items() if k != "_other"}
            best = max(cand, key=cand.get)
            if cand[best] >= self.min_confidence and cand[best] > res.semantic_scores["_other"]:
                res.primary = best
                res.labels.append(best)

        if self.tonal:
            all_tonal = tonal_labels(stats)
            res.tonal_scores = {c.name: all_tonal[c.name] for c in self.tonal}
            res.labels += [k for k, v in res.tonal_scores.items() if v >= self.tonal_threshold]

        return res
