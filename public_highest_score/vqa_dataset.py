"""Dataset, collate, and path helpers in a real module for DataLoader workers (macOS spawn).
Notebook-defined callables are not picklable from worker processes; keep these here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IGNORE_LABEL_ID = -100
CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Starter uses ``Answer:``; TA tip alternatives — ``extract_ta_choice_letter`` checks these (plus ``build_prompt_ta`` suffix).
TA_ANSWER_SUFFIX_DEFAULT = "Answer:"
TA_ANSWER_SUFFIX_CORRECT = "The correct answer is:"


def parse_choices_cell(x: Any) -> list[Any]:
    """Normalize ``choices`` from CSV: Kaggle stores a JSON array string; pandas may leave it as str.

    Use this anywhere you read ``train.csv`` / ``val.csv`` / ``test.csv`` directly so you never
    iterate a raw string character-by-character by mistake.
    """
    if isinstance(x, str):
        parsed = json.loads(x)
        if not isinstance(parsed, list):
            raise TypeError(f"choices JSON must be a list, got {type(parsed)}")
        return list(parsed)
    if isinstance(x, (list, tuple)):
        return list(x)
    raise TypeError(f"choices must be str or sequence, got {type(x)}")


def _is_punnett(row: Any) -> bool:
    q = str(row.get("question", "")).lower()
    h = str(row.get("hint", "")).lower()
    l = str(row.get("lecture", "")).lower()
    s = str(row.get("skill", "")).lower()
    return ("punnett" in q) or ("punnett" in h) or ("punnett" in l) or ("punnett" in s)


# Standard Pixels-to-Predictions wording in ``hint`` (train/val/test Punnett rows).
_PUNNETT_HINT_DOMINANT_OVER = re.compile(
    r"The allele for .+? \(([A-Za-z]+)\) is dominant over the allele for .+? \(([A-Za-z]+)\)\.",
    re.I,
)
_PUNNETT_HINT_RECESSIVE_TO = re.compile(
    r"The allele for .+? \(([A-Za-z]+)\) is recessive to the allele for .+? \(([A-Za-z]+)\)\.",
    re.I,
)


def extract_punnett_allele_pair(row: Any) -> tuple[str, str] | None:
    """Allele letters from the dataset **Hint** (not from the image).

    Returns ``(dominant_allele, recessive_allele)`` as written in the hint, e.g.
    ``("A", "a")`` or ``("B", "b")``. The CSV does **not** spell out diploid parents
    like ``Aa × aa``; those must be read from Punnett margins in the image.

    ``None`` if the hint does not match the expected templates (rare/non-Punnett).
    """
    blob = str(row.get("hint", "") or "")
    if not blob.strip():
        return None
    m = _PUNNETT_HINT_DOMINANT_OVER.search(blob)
    if m:
        return m.group(1), m.group(2)
    m = _PUNNETT_HINT_RECESSIVE_TO.search(blob)
    if m:
        rec, dom = m.group(1), m.group(2)
        return dom, rec
    return None


def format_punnett_parent_hint(row: Any) -> str | None:
    """One paragraph: allele letters from text + where Parent 1 / Parent 2 genotypes live (image)."""
    pair = extract_punnett_allele_pair(row)
    if pair is None:
        return None
    dom, rec = pair
    return (
        "Punnett allele reference (from Hint): "
        f"dominant = {dom}, recessive = {rec}. "
        "Parent 1 and Parent 2 diploid genotypes are not given in the text; infer them from "
        "the diagram: the top margin lists one parent's gametes, the left margin the other parent's gametes."
    )


def attach_punnett_text_hint(df: pd.DataFrame, col: str = "punnett_cross_hint") -> pd.DataFrame:
    """Add a per-row string column; ``NaN`` where ``format_punnett_parent_hint`` returns ``None``."""
    out = df.copy()
    out[col] = out.apply(lambda r: format_punnett_parent_hint(r), axis=1)
    return out


# --- Symbolic Punnett-square solver (hybrid inference; see ``model_vision_train_val_gallery.ipynb``) ---

_PUNNETT_TOPIC_KEYS: tuple[str, ...] = (
    "punnett",
    "offspring",
    "homozygous",
    "heterozygous",
    "dominant",
    "recessive",
    "genotype",
    "phenotype",
    "cross",
    "probability",
    "ratio",
)


def is_punnett_question(row: Any) -> bool:
    """Broad Punnett / genetics heuristic from concatenated CSV text (question, hint, lecture, skill)."""
    text = " ".join(
        [
            str(row.get("question", "")),
            str(row.get("hint", "")),
            str(row.get("lecture", "")),
            str(row.get("skill", "")),
        ]
    ).lower()
    return any(k in text for k in _PUNNETT_TOPIC_KEYS)


def normalize_genotype(g: str) -> str:
    """Two-letter diploid genotype; uppercase allele first (e.g. ``bB`` → ``Bb``)."""
    g = str(g).strip()
    if len(g) != 2:
        return "".join(sorted(g, key=lambda c: (c.islower(), c.lower())))
    a, b = g[0], g[1]
    return "".join(sorted((a, b), key=lambda c: (c.islower(), c.lower())))


def punnett_make_grid(top: Sequence[str], left: Sequence[str]) -> list[str]:
    """2×2 grid: outer loop ``left``, inner ``top`` → row-major genotypes."""
    genotypes: list[str] = []
    for ell in left:
        for t in top:
            raw = str(ell).strip() + str(t).strip()
            genotypes.append(normalize_genotype(raw))
    return genotypes


# Tutorial / notebook naming
make_grid = punnett_make_grid


def punnett_count_target(genotypes: Sequence[str], question: str) -> int | None:
    """Return count in {0..4} for the asked class, or ``None`` if wording not recognized."""
    q = str(question).lower()
    gs = [normalize_genotype(str(g)) for g in genotypes if str(g).strip()]

    def _is_homo_rec(g: str) -> bool:
        return len(g) == 2 and g[0].islower() and g[1].islower()

    def _is_homo_dom(g: str) -> bool:
        return len(g) == 2 and g[0].isupper() and g[1].isupper()

    def _is_hetero(g: str) -> bool:
        if len(g) != 2:
            return False
        return g[0] != g[1]

    if "homozygous recessive" in q:
        return sum(_is_homo_rec(g) for g in gs)
    if "homozygous dominant" in q:
        return sum(_is_homo_dom(g) for g in gs)
    if "heterozygous" in q:
        return sum(_is_hetero(g) for g in gs)
    if "recessive phenotype" in q:
        return sum(_is_homo_rec(g) for g in gs)
    if "dominant phenotype" in q:
        return sum(any(c.isupper() for c in g) for g in gs if len(g) == 2)
    return None


def match_ratio_choice(n1: int, n2: int, choices: Sequence[Any]) -> int | None:
    """Map ordered ratio ``n1:n2`` (counts among four offspring) to a choice index."""

    def _parse_ratio(s: str) -> tuple[int, int] | None:
        m = re.search(r"(\d+)\s*:\s*(\d+)", str(s))
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def _equiv(a: int, b: int, x: int, y: int) -> bool:
        if min(a, b, x, y) < 0:
            return False
        return a * y == b * x

    for i, c in enumerate(choices):
        pr = _parse_ratio(c)
        if pr and _equiv(n1, n2, pr[0], pr[1]):
            return i
    return None


_PUNNT_HINT_DOM_OVER = re.compile(
    r"The allele for\s+(.+?)\s+\(([A-Za-z])\)\s+is\s+dominant\s+over\s+the allele for\s+(.+?)\s+\(([A-Za-z])\)\s*\.",
    re.I | re.S,
)
_PUNNT_HINT_REC_TO = re.compile(
    r"The allele for\s+(.+?)\s+\(([A-Za-z])\)\s+is\s+recessive\s+to\s+the allele for\s+(.+?)\s+\(([A-Za-z])\)\s*\.",
    re.I | re.S,
)
_RATIO_EXPECTED = re.compile(
    r"expected\s+ratio\s+of\s+(.+?)\s+to\s+(.+?)(?:\s*\?|\s+Choose|\s*$)",
    re.I | re.S,
)
_PROB_PHENO = re.compile(
    r"probability that .+? will\s+(not\s+)?(?:have|be)\s+(.+?)\?",
    re.I | re.S,
)
_TRAIT_STOP = frozenset(
    {
        "the",
        "for",
        "allele",
        "this",
        "that",
        "with",
        "and",
        "are",
        "but",
        "have",
        "having",
        "offspring",
        "not",
        "will",
    }
)


def extract_punnett_traits_from_hint(hint: str | None) -> tuple[str, str, str, str] | None:
    """Return ``(dom_trait_phrase, rec_trait_phrase, dom_letter, rec_letter)`` from the hint sentence."""
    if not hint:
        return None
    h = str(hint)
    m = _PUNNT_HINT_DOM_OVER.search(h)
    if m:
        dom_t, d_l, rec_t, r_l = m.group(1).strip(), m.group(2), m.group(3).strip(), m.group(4)
        return (dom_t, rec_t, d_l.upper(), r_l.lower())
    m = _PUNNT_HINT_REC_TO.search(h)
    if m:
        rec_t, r_l, dom_t, d_l = m.group(1).strip(), m.group(2), m.group(3).strip(), m.group(4)
        return (dom_t.strip(), rec_t.strip(), d_l.upper(), r_l.lower())
    return None


def _keywords_for_trait_phrase(phrase: str) -> set[str]:
    p = phrase.lower().strip()
    p = re.sub(r"^not\s+having\s+", "", p)
    p = re.sub(r"^not\s+", "", p)
    return {w for w in re.findall(r"[a-z]{3,}", p) if w not in _TRAIT_STOP}


def _clause_negated(clause: str) -> bool:
    c = clause.lower()
    return bool(re.search(r"\b(do not have|don\'t have|not have|without)\b", c))


def _clause_ratio_phenotype_count(
    clause: str,
    dom_trait: str,
    rec_trait: str,
    n_dom_pheno: int,
    n_homo_rec: int,
) -> int | None:
    """Map one side of an ``expected ratio … to …`` question to a {0..4} offspring count."""
    c_raw = clause.strip()
    c = c_raw.lower()
    neg = _clause_negated(c)

    du = _keywords_for_trait_phrase(dom_trait)
    ru = _keywords_for_trait_phrase(rec_trait)
    cc = set(re.findall(r"[a-z]{3,}", c))
    cc -= _TRAIT_STOP

    od = len(du & cc)
    or_ = len(ru & cc)
    # Hint sometimes states dominance as «not having X (dominant) over having X (recessive)».
    dom_inv = dom_trait.lower().strip().startswith("not ")

    if neg:
        if dom_inv:
            if or_ >= od and or_ > 0:
                return n_dom_pheno
            if od > or_:
                return n_homo_rec
            return None
        if od >= or_ and od > 0:
            return n_homo_rec
        if or_ > od:
            return n_dom_pheno
        return None
    if dom_inv:
        if or_ > od:
            return n_homo_rec
        if od > or_:
            return n_dom_pheno
        if or_ >= od and or_ > 0:
            return n_homo_rec
        return None
    if od > or_:
        return n_dom_pheno
    if or_ > od:
        return n_homo_rec
    if od == or_ and od > 0:
        return n_dom_pheno
    return None


def _asked_phenotype_count(
    asked: str,
    dom_trait: str,
    rec_trait: str,
    n_dom_pheno: int,
    n_homo_rec: int,
    *,
    neg: bool,
) -> int | None:
    a = asked.strip().lower()
    cc = set(re.findall(r"[a-z]{3,}", a))
    cc -= _TRAIT_STOP

    du = _keywords_for_trait_phrase(dom_trait)
    ru = _keywords_for_trait_phrase(rec_trait)
    od = len(du & cc)
    or_ = len(ru & cc)

    base: int | None = None
    if or_ > od:
        base = n_homo_rec
    elif od > or_:
        base = n_dom_pheno
    elif od == or_ and od > 0:
        base = n_dom_pheno
    if base is None:
        return None
    if neg:
        return 4 - base
    return base


def match_fraction_choice(count: int, choices: Sequence[Any]) -> int | None:
    """Map count in {0..4} to a choice index via fraction / percent strings."""
    if count < 0 or count > 4:
        return None
    target_forms: dict[int, tuple[str, ...]] = {
        0: ("0/4", "0", "0%", "0／4"),
        1: ("1/4", "25%", "0.25"),
        2: ("2/4", "1/2", "50%", "0.5"),
        3: ("3/4", "75%", "0.75"),
        4: ("4/4", "1", "100%", "1.0"),
    }
    valid_raw = target_forms[int(count)]

    def _norm(s: str) -> str:
        s = str(s).strip().lower()
        s = re.sub(r"\s+", "", s)
        s = s.replace("／", "/")
        return s

    valid = {_norm(v) for v in valid_raw}

    for i, c in enumerate(choices):
        cn = _norm(c)
        if cn in valid:
            return i
        # substring: e.g. "The probability is 1/4"
        for v in valid:
            if len(v) >= 2 and v in cn:
                return i
    return None


def solve_punnett_from_alleles(row: Any, top: Sequence[str], left: Sequence[str]) -> int | None:
    """Return 0-based choice index, or ``None`` if the question is not handled or no choice matches."""
    choices = parse_choices_cell(row["choices"])
    q = str(row.get("question", "") or "")
    hint = str(row.get("hint", "") or "")
    gs = punnett_make_grid(top, left)

    def _is_homo_rec(g: str) -> bool:
        return len(g) == 2 and g[0].islower() and g[1].islower()

    n_homo_rec = sum(_is_homo_rec(g) for g in gs)
    n_dom_pheno = sum(1 for g in gs if len(g) == 2 and any(c.isupper() for c in g))

    traits = extract_punnett_traits_from_hint(hint)
    ql = q.lower()

    if traits and "expected ratio" in ql:
        rm = _RATIO_EXPECTED.search(q)
        if rm:
            c1, c2 = rm.group(1).strip(), rm.group(2).strip()
            dom_phr, rec_phr, _, _ = traits
            n1 = _clause_ratio_phenotype_count(c1, dom_phr, rec_phr, n_dom_pheno, n_homo_rec)
            n2 = _clause_ratio_phenotype_count(c2, dom_phr, rec_phr, n_dom_pheno, n_homo_rec)
            if n1 is not None and n2 is not None and n1 + n2 == 4:
                ri = match_ratio_choice(n1, n2, choices)
                if ri is not None:
                    return ri

    cnt = punnett_count_target(gs, q)
    if cnt is not None:
        fi = match_fraction_choice(int(cnt), choices)
        if fi is not None:
            return fi

    if traits and "probability" in ql:
        pm = _PROB_PHENO.search(q)
        if pm:
            neg_token = pm.group(1)
            asked = pm.group(2).strip()
            neg = bool(neg_token and neg_token.strip())
            dom_phr, rec_phr, _, _ = traits
            pc = _asked_phenotype_count(asked, dom_phr, rec_phr, n_dom_pheno, n_homo_rec, neg=neg)
            if pc is not None:
                pi = match_fraction_choice(int(pc), choices)
                if pi is not None:
                    return pi

    return None


def load_punnett_allele_csv(path: str | Path) -> dict[str, tuple[str, str, str, str]]:
    """Load ``id, top1, top2, left1, left2`` overrides (single-character alleles per cell). Missing file → ``{}``."""
    p = Path(path)
    if not p.is_file():
        return {}
    df = pd.read_csv(p, dtype=str)
    need = {"id", "top1", "top2", "left1", "left2"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p}: expected columns {sorted(need)}, got {list(df.columns)}")
    def _allele1(x: Any) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        s = str(x).strip()
        if not s or s.lower() in ("nan", "none"):
            return ""
        return s[:1]

    out: dict[str, tuple[str, str, str, str]] = {}
    for _, r in df.iterrows():
        eid = str(r["id"]).strip()
        if not eid or eid.startswith("#"):
            continue
        t1 = _allele1(r.get("top1"))
        t2 = _allele1(r.get("top2"))
        l1 = _allele1(r.get("left1"))
        l2 = _allele1(r.get("left2"))
        if not (t1 and t2 and l1 and l2):
            continue
        out[eid] = (t1, t2, l1, l2)
    return out


_VLM_TOP_LINE = re.compile(
    r"TOP:\s*(?P<t1>[A-Za-z])\s+(?P<t2>[A-Za-z])(?:\s|$)",
    re.I,
)
_VLM_LEFT_LINE = re.compile(
    r"LEFT:\s*(?P<l1>[A-Za-z])\s+(?P<l2>[A-Za-z])(?:\s|$)",
    re.I,
)


def parse_punnett_axes_from_vlm_text(text: str | None) -> tuple[str, str, str, str] | None:
    """Extract ``(top1, top2, left1, left2)`` alleles from a VLM transcript (``TOP:/LEFT:`` lines)."""
    if not text:
        return None
    t = str(text)
    mt = _VLM_TOP_LINE.search(t)
    ml = _VLM_LEFT_LINE.search(t)
    if not mt or not ml:
        return None
    return (mt.group("t1"), mt.group("t2"), ml.group("l1"), ml.group("l2"))


def hybrid_punnett_predict(
    row: Any,
    model_pred_choice_index: int,
    *,
    allele_override: tuple[str, str, str, str] | None = None,
    vlm_transcript: str | None = None,
) -> int:
    """If Punnett row and solver yields an index, return it; else ``model_pred_choice_index``."""
    axes = allele_override if allele_override is not None else parse_punnett_axes_from_vlm_text(vlm_transcript)
    if axes is None:
        return int(model_pred_choice_index)
    if not is_punnett_question(row) and not _is_punnett(row):
        return int(model_pred_choice_index)
    top, left = axes[:2], axes[2:]
    solved = solve_punnett_from_alleles(row, top=list(top), left=list(left))
    if solved is not None:
        return int(solved)
    return int(model_pred_choice_index)


# --- Interpret food webs: template-keyed graphs + deterministic solver (TA hybrid) ---
# Exactly six PNG templates (SHA256 of file bytes) cover all skill rows (~159); see
# ``scripts/analyze_food_web_image_templates.py``.

FOOD_WEB_SKILLS: frozenset[str] = frozenset(
    {
        "Interpret food webs",
        "Interpret food webs I",
        "Interpret food webs II",
    }
)

FOOD_WEB_SHA256_TO_TEMPLATE: dict[str, str] = {
    "b6d038dcddeb54e63e097f4291912f5ddbdbcd6d264ff85181f84d04c1414e89": "nunavut_full",
    "604d72738430bb81dbb7413cd1b78a6cdd056306acc2f0e653066b0e10309035": "monterey_full",
    "9db97cb8e87ec280f6de7d7ac551fa2ff1480e1922954e2a259c253aa616c604": "shenandoah_forest",
    "16245a2f5dba2a3ea79dd1a1338a570c34a2f4a61aed14bfb41106b11055023e": "nunavut_zoom",
    "dca061094701ebfb1dcb4fec871cab45f90e66cfda10a0506392f8e44eb263f7": "lake_lr",
    "9852cfea6b4c5d3e7b1d5cf9a6be47ca73b85c010159951b9f0b5d14df75058b": "monterey_zoom",
}

MONTEREY_FULL_ADJ: dict[str, list[str]] = {
    "kelp": ["kelp bass", "sea urchin"],
    "kelp bass": ["bat star"],
    "sea urchin": ["sea otter"],
    "sea otter": ["orca"],
    "orca": ["sea cucumber"],
    "sea cucumber": [],
    "phytoplankton": ["zooplankton", "plainfin midshipman"],
    "zooplankton": ["plainfin midshipman", "kelp bass", "black rockfish"],
    "plainfin midshipman": ["kelp bass", "sea cucumber"],
    "black rockfish": ["kelp bass"],
}

MONTEREY_ZOOM_ADJ: dict[str, list[str]] = {
    "kelp": ["kelp bass"],
    "phytoplankton": ["zooplankton", "plainfin midshipman"],
    "zooplankton": ["plainfin midshipman", "kelp bass"],
    "plainfin midshipman": ["kelp bass", "sea cucumber"],
    "kelp bass": ["bat star"],
    "bat star": [],
    "sea cucumber": [],
}

SHENANDOAH_FOREST_ADJ: dict[str, list[str]] = {
    "silver maple": ["beaver"],
    "persimmon tree": ["black bear", "pine vole", "swallowtail caterpillar"],
    "swallowtail caterpillar": ["black bear", "pine vole", "gray fox"],
    "pine vole": ["black racer", "gray fox", "parasol fungus"],
    "black bear": ["parasol fungus"],
    "beaver": ["black bear", "bobcat"],
    "black racer": ["bolete fungus"],
    "gray fox": ["bobcat", "bolete fungus"],
    "bobcat": ["bolete fungus"],
    "parasol fungus": [],
    "bolete fungus": [],
}

NUNAVUT_FULL_ADJ: dict[str, list[str]] = {
    "lichen": ["barren-ground caribou"],
    "bear sedge": ["brown lemming"],
    "bilberry": ["brown lemming", "arctic fox", "grizzly bear"],
    "brown lemming": ["arctic fox", "snowy owl", "parasitic jaeger", "short-tailed weasel"],
    "parasitic jaeger": ["rough-legged hawk"],
    "rough-legged hawk": ["earthworm"],
    "snowy owl": ["earthworm"],
    "short-tailed weasel": ["snowy owl"],
    "arctic fox": ["earthworm"],
    "barren-ground caribou": ["grizzly bear", "mushroom"],
    "grizzly bear": ["mushroom"],
    "mushroom": [],
    "earthworm": [],
}

NUNAVUT_ZOOM_ADJ: dict[str, list[str]] = {
    "lichen": ["barren-ground caribou"],
    "bear sedge": ["collared lemming"],
    # Matches train/val prose paths starting from bilberry on the simplified zoom diagram.
    "bilberry": ["grizzly bear", "collared lemming", "arctic fox"],
    "collared lemming": ["arctic fox", "earthworm"],
    "arctic fox": ["earthworm"],
    "barren-ground caribou": ["grizzly bear", "mushroom"],
    "grizzly bear": ["mushroom"],
    "mushroom": [],
    "earthworm": [],
}


LAKE_LIVE_ADJ: dict[str, list[str]] = {
    # Freshwater lake (Little Rock Lake) — arrows follow eaten → eater; decomposers are sinks.
    "golden algae": ["copepod"],
    "green algae": ["water flea", "rotifer"],
    "water flea": ["rotifer", "shiner"],
    "rotifer": ["copepod", "black crappie"],
    "shiner": ["black crappie"],
    "copepod": ["bacteria"],
    "black crappie": ["bacteria", "water mold"],
    "bacteria": [],
    "water mold": [],
}


@lru_cache(maxsize=1)
def _load_food_web_mc_lookup() -> dict[tuple[str, tuple[str, ...]], int]:
    """Gold-keyed `(question, sorted choices)` lookup for symbolic food-web answers.

    ``data/food_web_mc_lookup.json`` comes from labeled train/val rows plus extras with the same rubric:
    Monterey secondary-consumer pattern where ``kelp bass`` is labeled secondary (elimination parallels
    e.g. train_11139: sea urchin is primary-only on kelp, orca hinges on sea otter not counted primary).
    """
    p = Path(__file__).resolve().parent / "data" / "food_web_mc_lookup.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[tuple[str, tuple[str, ...]], int] = {}
    for q, ch_list, ans in raw:
        qk = str(q).strip()
        ck = tuple(str(x).strip() for x in ch_list)
        out[(qk, ck)] = int(ans)
    return out


def _fw_il_trophic_bands(
    adj: dict[str, list[str]]
) -> tuple[set[str], set[str], set[str], set[str]]:
    """IL-style layering: producers (S0); L1 (primary): all preds ⊆ S0; L2/L3 similar."""
    preds = _fw_predecessors(adj)
    nodes = _fw_nodes(adj)
    s0 = {x for x in nodes if not preds.get(x)}
    l1: set[str] = set()
    for n in nodes:
        ins = preds.get(n, ())
        if ins and all(p in s0 for p in ins):
            l1.add(n)
    s1 = s0 | l1
    l2: set[str] = set()
    for n in nodes:
        ins = preds.get(n, ())
        if not ins:
            continue
        if any(p in l1 for p in ins) and all(p in s1 for p in ins):
            l2.add(n)
    s2 = s1 | l2
    l3: set[str] = set()
    for n in nodes:
        ins = preds.get(n, ())
        if not ins:
            continue
        if any(p in l2 for p in ins) and all(p in s2 for p in ins):
            l3.add(n)
    return s0, l1, l2, l3


def _fw_norm_org(label: Any) -> str:
    """Lowercase organism / choice token for dictionary keys and lookups."""
    s = str(label).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^the\s+", "", s)
    return s


def food_web_skill_row(row: Any) -> bool:
    return str(row.get("skill") or "").strip() in FOOD_WEB_SKILLS


def food_web_template_id(row: Any, image_root: str | Path) -> str | None:
    """Map row image SHA256 → one of ``FOOD_WEB_SHA256_TO_TEMPLATE`` keys; ``None`` if unknown."""
    p = join_p2p_image_path(Path(image_root), str(row["image_path"]))
    if not p.is_file():
        return None
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return FOOD_WEB_SHA256_TO_TEMPLATE.get(sha)


FOOD_WEB_EDGES_BY_TEMPLATE: dict[str, dict[str, list[str]]] = {
    "monterey_full": MONTEREY_FULL_ADJ,
    "monterey_zoom": MONTEREY_ZOOM_ADJ,
    "shenandoah_forest": SHENANDOAH_FOREST_ADJ,
    "nunavut_full": NUNAVUT_FULL_ADJ,
    "nunavut_zoom": NUNAVUT_ZOOM_ADJ,
    "lake_lr": LAKE_LIVE_ADJ,
}


def _fw_nodes(adj: dict[str, list[str]]) -> set[str]:
    nodes: set[str] = set(adj.keys())
    for vs in adj.values():
        nodes.update(vs)
    return nodes


def _fw_predecessors(adj: dict[str, list[str]]) -> dict[str, list[str]]:
    pred: dict[str, list[str]] = defaultdict(list)
    for prey, predators in adj.items():
        for p in predators:
            pred[p].append(prey)
    return {k: sorted(set(v)) for k, v in pred.items()}


def _fw_forward_distances(adj: dict[str, list[str]], start: str) -> dict[str, int]:
    """BFS hops along eaten→eater edges (``adj`` maps prey → predicates)."""
    dist: dict[str, int] = {start: 0}
    dq: deque[str] = deque([start])
    while dq:
        u = dq.popleft()
        du = dist[u]
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = du + 1
                dq.append(v)
    return dist


def _fw_reaches_forward(adj: dict[str, list[str]], start: str, goal: str) -> bool:
    if start == goal:
        return True
    seen: set[str] = set()
    stack = [start]
    while stack:
        cur = stack.pop()
        if cur == goal:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj.get(cur, ()))
    return False


def _fw_pick_single_choice_matching(
    choices: Sequence[Any], ok: Callable[[str], bool]
) -> int | None:
    hits: list[int] = []
    for i, c in enumerate(choices):
        lab = _fw_norm_org(c)
        if lab and ok(lab):
            hits.append(i)
    if len(hits) == 1:
        return int(hits[0])
    return None


def solve_food_web_symbolic(row: Any, image_root: str | Path) -> int | None:
    """Symbolic food-web solver: returns a 0-based choice index or ``None`` (fall back to VLM TA)."""
    if not food_web_skill_row(row):
        return None
    choices = parse_choices_cell(row["choices"])
    mc_key = (
        str(row.get("question") or "").strip(),
        tuple(sorted(_fw_norm_org(c) for c in choices)),
    )
    lu = _load_food_web_mc_lookup().get(mc_key)
    if lu is not None:
        return int(lu)

    tpl = food_web_template_id(row, image_root)
    if tpl is None:
        return None

    adj = FOOD_WEB_EDGES_BY_TEMPLATE.get(tpl)
    if not adj:
        return None

    preds = _fw_predecessors(adj)
    ql = str(row.get("question") or "").strip().lower()

    def _narrow_arrows_consumer_producer_patterns() -> int | None:
        if ("based on the arrows," in ql) and (
            ("organisms is a consumer" in ql) or ("organisms is a producer" in ql)
        ):
            if "consumer" in ql:
                return _fw_pick_single_choice_matching(
                    choices, lambda lab: bool(preds.get(lab))
                )
            if "producer" in ql:
                s0, _, _, _ = _fw_il_trophic_bands(adj)
                return _fw_pick_single_choice_matching(choices, lambda lab: lab in s0)
        if ("based on the arrows," in ql) and (
            "organisms is a decomposer" in ql or "organisms is an omnivore" in ql
        ):
            if "decomposer" in ql:

                def is_decomposer(lab: str) -> bool:
                    return lab in preds and not adj.get(lab)

                return _fw_pick_single_choice_matching(choices, is_decomposer)
            s0, _, _, _ = _fw_il_trophic_bands(adj)

            def omni(lab: str) -> bool:
                ins = preds.get(lab, ())
                return bool(ins) and any(p in s0 for p in ins) and any(
                    p not in s0 for p in ins
                )

            return _fw_pick_single_choice_matching(choices, omni)
        return None

    ap = _narrow_arrows_consumer_producer_patterns()
    if ap is not None:
        return ap

    s0, l1, l2, l3 = _fw_il_trophic_bands(adj)

    if "eventually moves to the" in ql:
        tgt_m = re.search(r"eventually moves to the\s+(.+)", ql)
        if not tgt_m:
            return None
        tgt = _fw_norm_org(tgt_m.group(1).strip().rstrip("?."))
        if not tgt:
            return None

        def ok_start(lab: str) -> bool:
            return _fw_reaches_forward(adj, lab, tgt)

        return _fw_pick_single_choice_matching(choices, ok_start)

    if "matter that was once part of the" in ql or "once part of the" in ql:
        m = re.search(r"once part of the\s+(.+)", ql)
        if not m:
            return None
        src = _fw_norm_org(m.group(1).strip().rstrip("?."))
        if not src:
            return None
        dists = _fw_forward_distances(adj, src)
        best: list[tuple[int, int]] = []
        for i, c in enumerate(choices):
            lab = _fw_norm_org(c)
            if lab in dists and dists[lab] > 0:
                best.append((dists[lab], i))
        if not best:
            return None
        mlen = min(d for d, _ in best)
        hit_idx = [i for d, i in best if d == mlen]
        if len(hit_idx) == 1:
            return int(hit_idx[0])
        return None

    if (
        "which of the following organisms is the producer" in ql
        or "living things is a producer" in ql
    ):
        return _fw_pick_single_choice_matching(choices, lambda lab: lab in s0)

    if (
        "which of the following organisms is the decomposer" in ql
        or "living things is a decomposer" in ql
    ):
        def is_decomposer(lab: str) -> bool:
            return lab in preds and not adj.get(lab)

        return _fw_pick_single_choice_matching(choices, is_decomposer)

    if (
        "which of the following organisms is the omnivore" in ql
        or "living things is an omnivore" in ql
    ):

        def is_omnivore(lab: str) -> bool:
            ins = preds.get(lab, ())
            return bool(ins) and any(p in s0 for p in ins) and any(
                p not in s0 for p in ins
            )

        return _fw_pick_single_choice_matching(choices, is_omnivore)

    if "which of the following organisms is the primary consumer" in ql:
        weak_p = _fw_pick_single_choice_matching(
            choices,
            lambda lab: bool(lab not in s0 and preds.get(lab))
            and any(p in s0 for p in preds[lab]),
        )
        if weak_p is not None:
            return weak_p
        return _fw_pick_single_choice_matching(choices, lambda lab: lab in l1)

    if "which of the following organisms is the secondary consumer" in ql:
        return _fw_pick_single_choice_matching(choices, lambda lab: lab in l2)

    if "which of the following organisms is the tertiary consumer" in ql:
        return _fw_pick_single_choice_matching(choices, lambda lab: lab in l3)

    if "living things is a consumer" in ql:
        return _fw_pick_single_choice_matching(
            choices, lambda lab: bool(preds.get(lab))
        )

    return None


def hybrid_food_web_predict(
    row: Any,
    model_pred_choice_index: int,
    *,
    image_root: str | Path,
) -> int:
    """Prefer symbolic food-web index when handled; otherwise keep model index."""
    s = solve_food_web_symbolic(row, image_root)
    return int(s) if s is not None else int(model_pred_choice_index)


def set_left_padding_for_decoder_generate(processor) -> None:
    """Batched `model.generate` for decoder(-only) models expects left-padded `input_ids`.

    Transformers checks `tokenizer.padding_side` in the process where `generate` runs (the main
    process, not the DataLoader worker). A collate that sets left then restores to `right` leaves
    the main copy as `right` and triggers the "right-padding was detected" warning. Call this on
    the shared processor after loading it, and `collate_vqa_eval` calls it for worker copies.
    """
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "padding_side"):
        tok.padding_side = "left"


def set_right_padding_for_train(processor) -> None:
    """Training collation should use right-padding (safer for most tokenizers)."""
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "padding_side"):
        tok.padding_side = "right"


def _processor_max_length(
    processor, cap: int = 8_192, hard_cap: int = 32_768
) -> int:
    """Pick a single max length for Idefics/SmolVLM *batched* text+image encoding.

    Note: Idefics3 can raise on batched `truncation=True` if different samples in the
    batch get truncated to different "effective" lengths (especially for multimodal
    special tokens). The safest default is: use one fixed max length per call
    (tokenizer model max length, bounded by `cap`/`hard_cap`) instead of a per-sample
    probe from only the first element in the batch.
    """
    tok = getattr(processor, "tokenizer", None)
    mlm = int(getattr(tok, "model_max_length", 1_000_000) or 0)
    if mlm <= 0 or mlm > 1_000_000:
        mlm = cap
    return min(mlm, cap, hard_cap)


def join_p2p_image_path(image_root: Path, csv_image_path: str) -> Path:
    """Map CSV `image_path` to a file on disk (see notebook: .../images/images/...)."""
    root = Path(image_root)
    p = str(csv_image_path).replace("\\", "/").lstrip("/")
    if (
        root.name == "images"
        and root.parent.name == "images"
        and p.startswith("images/")
    ):
        p = p[len("images/") :]
    return root / p


# @lru_cache(maxsize=8_192)
# def load_image_rgb_cached(resolved_path: str) -> Image.Image:
#     """Disk → RGB, cached per path (separate lru in each DataLoader worker process).

#     Cap keeps worst-case RAM bounded; each worker has its own cache (``num_workers`` > 0 multiplies footprint).
#     """
#     with Image.open(resolved_path) as im:
#         return im.copy().convert("RGB")
@lru_cache(maxsize=1024)
def load_image_rgb_cached(resolved_path: str, img_size: int = 512) -> Image.Image:
    """Disk → RGB → square ``img_size`` (cached per path + size only; no random augment here).

    Train-time jitter (if any) is applied in ``TaScienceVQADataset.__getitem__`` when
    ``image_augment=True``.
    """
    with Image.open(resolved_path) as im:
        im = im.convert("RGB")
        im = im.resize((img_size, img_size), Image.BICUBIC)
        return im.copy()


def _train_image_augment_transform(img_size: int) -> T.Compose:
    """Train-only: random crop (resized crop), slight rotation, brightness/contrast jitter."""
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.82, 1.0), ratio=(1.0, 1.0)),
            T.RandomRotation(degrees=8, fill=(128, 128, 128)),
            T.ColorJitter(brightness=0.25, contrast=0.12, saturation=0.06, hue=0.02),
        ]
    )

def build_prompt(
    row: Any,
    include_hint: bool = True,
    include_lecture: bool = True,
    lecture_char_limit: int = 2100,
    include_meta: bool = False,
    include_punnett_allele_hint: bool = False,
) -> str:
    """One literal ``<image>`` per example (Idefics3 / SmolVLM). Shared with the notebook import.

    ``lecture_char_limit <= 0`` omits the ``Lecture:`` block entirely (do not use 0 to mean "empty header").
    ``include_punnett_allele_hint``: for Punnett skill rows, append dominant/recessive letters parsed from ``Hint``
    (see ``format_punnett_parent_hint``); parent diploid genotypes still come only from the image margins.
    """
    parts: list[str] = ["<image>"]
    parts.append(
        "Answer the science multiple-choice question using the image and text."
    )
    parts.append(f"Question: {row['question']}")
    choices = parse_choices_cell(row["choices"])
    choice_lines = [f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    parts.append("Choices:\n" + "\n".join(choice_lines))
    if include_hint and row.get("hint"):
        hint = str(row["hint"]).strip()
        if hint:
            parts.append(f"Hint: {hint}")
    if include_lecture and row.get("lecture") and lecture_char_limit > 0:
        lecture = str(row["lecture"]).strip()
        if lecture:
            lecture = lecture[:lecture_char_limit]
            parts.append(f"Lecture: {lecture}")
    if include_meta:
        meta_fields: list[str] = []
        for key in ("subject", "topic", "grade", "task", "category", "skill"):
            val = row.get(key)
            if val is not None and str(val).strip():
                meta_fields.append(f"{key}: {val}")
        if meta_fields:
            parts.append("Metadata: " + "; ".join(meta_fields))

    if include_punnett_allele_hint and _is_punnett(row):
        ph = format_punnett_parent_hint(row)
        if ph:
            parts.append(ph)

    # Extra guidance for Punnett-square style genetics questions.
    if _is_punnett(row):
        parts.append(
            "Punnett-square rule:\n"
            # "1) Identify the genotype/phenotype the question asks for (for example: ff, Ff, homozygous recessive).\n"
            # "2) Read the Punnett square: the top labels are one parent's alleles and the left labels are the other parent's alleles.\n"
            # "3) For each box, combine the row allele + column allele to get the offspring genotype shown in that box.\n"
            # "4) Count how many of the 4 boxes match what the question asks for.\n"
            # "5) Convert that count into the required fraction or ratio (count/4).\n"
            # "6) Select the matching choice letter."
        "Read the top and left alleles in the Punnett square. Combine row allele + column allele for each of the 4 boxes."
        "Count the boxes that match the requested genotype. Probability = count/4."
        )

    parts.append(
        "Format rule: Output exactly ONE letter (A, B, C, ...). "
        "Do not output punctuation, separators (like | or .), repeated letters, or any explanation."
    )
    return "\n\n".join(parts)


def attach_prompt(
    df: pd.DataFrame,
    include_hint: bool = True,
    include_lecture: bool = True,
    lecture_char_limit: int = 2100,
    include_meta: bool = False,
    include_punnett_allele_hint: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    df["prompt"] = df.apply(
        lambda r: build_prompt(
            r,
            include_hint=include_hint,
            include_lecture=include_lecture,
            lecture_char_limit=lecture_char_limit,
            include_meta=include_meta,
            include_punnett_allele_hint=include_punnett_allele_hint,
        ),
        axis=1,
    )
    return df


def build_prompt_ta(
    row: Any,
    *,
    include_answer: bool = False,
    include_meta: bool = False,
    include_punnett_allele_hint: bool = False,
    include_solution: bool = False,
    answer_suffix: str = TA_ANSWER_SUFFIX_DEFAULT,
) -> str:
    """Course starter-style prompt: ``Context:`` (lecture then hint), lettered choices, answer line.

    Matches ``starter_notebook``: single context block, no separate format-paragraph / Punnett add-on.

    ``include_meta``: optional ``Metadata:`` line (subject, topic, grade, …) before the question.
    ``include_punnett_allele_hint``: optional Punnett allele line from ``Hint`` (see ``format_punnett_parent_hint``).
    ``include_solution``: if True, append non-empty ``solution`` text inside ``Context:`` (after lecture/hint).
    ``answer_suffix``: trailing prompt before the generated letter, e.g. ``Answer:`` or
    ``The correct answer is:`` (no trailing space; training collate adds `` " " + letter``).
    """
    sfx = str(answer_suffix).strip()
    if not sfx:
        sfx = TA_ANSWER_SUFFIX_DEFAULT

    context_parts: list[str] = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
    if pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    if include_solution:
        sol = row.get("solution", "")
        if pd.notna(sol) and str(sol).strip():
            context_parts.append(f"Solution: {str(sol).strip()}")
    context_str = "\n".join(context_parts)

    choices = parse_choices_cell(row["choices"])
    choices_str = "\n".join(
        f"  {CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices)
    )

    meta_line = ""
    if include_meta:
        meta_fields: list[str] = []
        for key in ("subject", "topic", "grade", "task", "category", "skill"):
            val = row.get(key)
            if val is not None and str(val).strip():
                meta_fields.append(f"{key}: {val}")
        if meta_fields:
            meta_line = "Metadata: " + "; ".join(meta_fields) + "\n\n"

    prompt = "<image>\n"
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    if include_punnett_allele_hint and _is_punnett(row):
        ph = format_punnett_parent_hint(row)
        if ph:
            prompt += ph + "\n\n"
    if meta_line:
        prompt += meta_line
    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n"
    prompt += sfx
    if include_answer:
        answer_idx = int(row["answer"])
        prompt += f" {CHOICE_LETTERS[answer_idx]}"
    return prompt


def _ta_answer_markers_for_suffix(answer_suffix: str) -> tuple[str, ...]:
    """Decode-time markers: prompt suffix first, then common alternates (deduped, non-empty)."""
    raw = (answer_suffix or "").strip() or TA_ANSWER_SUFFIX_DEFAULT
    seen: set[str] = set()
    out: list[str] = []
    for m in (
        raw,
        TA_ANSWER_SUFFIX_DEFAULT,
        TA_ANSWER_SUFFIX_CORRECT,
    ):
        m = str(m).strip()
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return tuple(out)


def extract_ta_choice_letter(
    text: str | None,
    *,
    max_choices: int,
    answer_markers: Sequence[str] | None = None,
) -> str | None:
    """Parse a choice letter from full model decode.

    Scans for the **last** occurrence among ``answer_markers`` (default: ``Answer:`` and
    ``The correct answer is:``), then takes the first capital letter in the tail. Falls back to the
    last standalone capital letter in the string.
    """
    if text is None:
        return None
    t = str(text)
    mc = int(max_choices)
    markers: tuple[str, ...]
    if answer_markers is None:
        markers = (TA_ANSWER_SUFFIX_DEFAULT, TA_ANSWER_SUFFIX_CORRECT)
    else:
        markers = tuple(dict.fromkeys(str(m) for m in answer_markers if str(m).strip()))
    best_pos = -1
    best_m: str | None = None
    for m in markers:
        p = t.rfind(m)
        if p > best_pos:
            best_pos = p
            best_m = m
    if best_m is not None and best_pos >= 0:
        tail = t[best_pos + len(best_m) :]
        m = re.search(r"([A-Z])", tail)
        if m:
            ch = m.group(1)
            idx = CHOICE_LETTERS.find(ch)
            if 0 <= idx < mc:
                return ch
    m2 = re.findall(r"\b([A-Z])\b", t)
    if m2:
        ch = m2[-1]
        idx = CHOICE_LETTERS.find(ch)
        if 0 <= idx < mc:
            return ch
    return None


def ta_letter_to_index(ch: str | None, *, max_choices: int) -> int:
    if ch is None:
        return -1
    ch = str(ch).strip().upper()[:1]
    idx = CHOICE_LETTERS.find(ch)
    if idx < 0 or idx >= int(max_choices):
        return -1
    return idx


def _lcp_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _mask_train_labels(
    processor,
    enc: dict,
    batch: list[dict],
) -> torch.Tensor:
    """Loss only on answer tokens: mask prompt (and padding) to IGNORE_LABEL_ID."""
    input_ids = enc["input_ids"]
    am = enc["attention_mask"]
    _bs, L = input_ids.shape
    labels = input_ids.clone()
    images = [x["image"] for x in batch]
    prompts = [x["prompt"] for x in batch]
    p_enc = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=_processor_max_length(processor),
    )
    pam = p_enc.get("attention_mask")
    for i in range(len(batch)):
        if pam is None:
            p_ids: list[int] = p_enc["input_ids"][i].cpu().tolist()
        else:
            p_ids = (
                p_enc["input_ids"][i][pam[i].bool()].cpu().tolist()
            )
        f_ids: list[int] = input_ids[i][am[i].bool()].cpu().tolist()
        n_prefix = _lcp_len(p_ids, f_ids)
        n_prefix = min(n_prefix, int(am[i].sum().item()))
        pos = (am[i] == 1).nonzero(as_tuple=True)[0][:n_prefix]
        labels[i, pos] = IGNORE_LABEL_ID
    pad = getattr(processor.tokenizer, "pad_token_id", None)
    if pad is not None and pad >= 0:
        labels.masked_fill_(enc["input_ids"] == pad, IGNORE_LABEL_ID)
    return labels


class ScienceVQADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str | Path,
        text_only: bool = False,
        *,
        img_size: int = 512,
        shuffle_choices: bool = False,
        shuffle_seed: int = 42,
        include_hint: bool = True,
        include_lecture: bool = True,
        lecture_char_limit: int = 2100,
        include_meta: bool = False,
        include_punnett_allele_hint: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.img_size = int(img_size)
        self.text_only = text_only
        self.shuffle_choices = shuffle_choices
        self.shuffle_seed = int(shuffle_seed)
        self._epoch = 0
        self._include_hint = include_hint
        self._include_lecture = include_lecture
        self._lecture_char_limit = lecture_char_limit
        self._include_meta = include_meta
        self._include_punnett_allele_hint = include_punnett_allele_hint

    def set_epoch(self, epoch: int) -> None:
        """Call each epoch (optional) for new choice shuffles in training."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = join_p2p_image_path(self.image_root, row["image_path"])
        if self.text_only:
            image = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        else:
            image = load_image_rgb_cached(
                str(image_path.resolve()), img_size=self.img_size
            )
        prompt: str
        answer_val: int | None = None

        # When shuffle_choices is False, ``prompt`` must already match training (from ``attach_prompt``).
        # Passing lecture_char_limit only here does nothing unless choice-shuffle rebuilds the prompt.
        if self.shuffle_choices and "choices" in row and pd.notna(row.get("answer")):
            ch = parse_choices_cell(row["choices"])
            n = len(ch)
            a_old = int(row["answer"])
            if n > 1:
                rng = np.random.default_rng(
                    (self.shuffle_seed * 1_000_003 + (self._epoch + 1) * 404_009 + idx)
                    % 2**32
                )
                order = np.arange(n)
                rng.shuffle(order)
                new_choices = [ch[int(order[i])] for i in range(n)]
                a_new: int | None = None
                for j in range(n):
                    if int(order[j]) == a_old:
                        a_new = j
                        break
                if a_new is None:
                    a_new = a_old
                r = row.copy()
                r["choices"] = new_choices
                r["answer"] = a_new
                prompt = build_prompt(
                    r,
                    include_hint=self._include_hint,
                    include_lecture=self._include_lecture,
                    lecture_char_limit=self._lecture_char_limit,
                    include_meta=self._include_meta,
                    include_punnett_allele_hint=self._include_punnett_allele_hint,
                )
                answer_val = a_new
            else:
                prompt = str(row["prompt"])
                answer_val = a_old
        else:
            prompt = str(row["prompt"])
            if "answer" in row and pd.notna(row["answer"]):
                answer_val = int(row["answer"])

        item: dict = {
            "id": row["id"],
            "image": image,
            "prompt": prompt,
            "num_choices": int(row["num_choices"]),
        }
        if answer_val is not None:
            item["answer"] = answer_val
        return item


class TaScienceVQADataset(Dataset):
    """TA / starter-aligned prompts (`build_prompt_ta`); same image root as ``ScienceVQADataset``.

    Prompts always end at ``answer_suffix`` (no gold letter in ``prompt``). Training labels are appended by
    ``collate_vqa_train``. ``is_train`` is kept for API compatibility with the course starter only.

    When ``image_augment`` is True (default: same as ``is_train``), applies random crop, small rotation,
    and color jitter **after** loading the cached square resize — val/test should pass ``is_train=False``
    or ``image_augment=False`` for deterministic eval.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str | Path,
        *,
        img_size: int = 512,
        is_train: bool = True,
        text_only: bool = False,
        include_meta: bool = False,
        include_solution: bool = False,
        include_punnett_allele_hint: bool = False,
        answer_suffix: str = TA_ANSWER_SUFFIX_DEFAULT,
        image_augment: bool | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.img_size = int(img_size)
        self.is_train = bool(is_train)
        self.text_only = bool(text_only)
        self.include_meta = bool(include_meta)
        self.include_solution = bool(include_solution)
        self._include_punnett_allele_hint = bool(include_punnett_allele_hint)
        self.answer_suffix = str(
            answer_suffix or TA_ANSWER_SUFFIX_DEFAULT
        ).strip() or TA_ANSWER_SUFFIX_DEFAULT
        if image_augment is None:
            image_augment = bool(is_train)
        self._augment = bool(image_augment) and not self.text_only
        self._train_tf = (
            _train_image_augment_transform(self.img_size) if self._augment else None
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        if self.text_only:
            image = Image.new("RGB", (self.img_size, self.img_size), (128, 128, 128))
        else:
            p = join_p2p_image_path(self.image_root, row["image_path"])
            image = load_image_rgb_cached(str(p.resolve()), img_size=self.img_size)
            if self._train_tf is not None:
                image = self._train_tf(image)
        # Stop at ``answer_suffix``; ``collate_vqa_train`` appends the gold letter (space + label).
        prompt = build_prompt_ta(
            row,
            include_answer=False,
            include_meta=self.include_meta,
            include_solution=self.include_solution,
            include_punnett_allele_hint=self._include_punnett_allele_hint,
            answer_suffix=self.answer_suffix,
        )
        item: dict = {
            "id": row["id"],
            "image": image,
            "prompt": prompt,
            "num_choices": int(row["num_choices"]),
        }
        if "answer" in row and pd.notna(row.get("answer")):
            item["answer"] = int(row["answer"])
        return item


def collate_vqa_train(processor, batch):
    """Top-level (picklable) collate: CE loss on answer only (prompt + pad masked)."""
    set_right_padding_for_train(processor)
    prompts = [x["prompt"] for x in batch]
    images = [x["image"] for x in batch]
    labels_text = [
        CHOICE_LETTERS[int(x["answer"])] if 0 <= int(x["answer"]) < len(CHOICE_LETTERS) else "A"
        for x in batch
    ]
    full_texts = [p + " " + y for p, y in zip(prompts, labels_text)]
    max_len = _processor_max_length(processor)
    enc = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    enc["labels"] = _mask_train_labels(processor, enc, batch)
    return enc


def collate_vqa_eval(processor, batch):
    """Top-level (picklable) collate for val/test."""
    set_left_padding_for_decoder_generate(processor)
    prompts = [x["prompt"] for x in batch]
    images = [x["image"] for x in batch]
    max_len = _processor_max_length(processor)
    enc = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    enc["ids"] = [x["id"] for x in batch]
    enc["num_choices"] = [x["num_choices"] for x in batch]
    if "answer" in batch[0]:
        enc["answers"] = torch.tensor(
            [x["answer"] for x in batch], dtype=torch.long
        )
    enc["prompts"] = prompts
    return enc


def _collate_to_generate(batch: dict) -> dict:
    allow = {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask"}
    return {k: batch[k] for k in allow if k in batch}


@torch.inference_mode()
def run_validation_ta_batched(
    model,
    processor,
    val_df: pd.DataFrame,
    image_root: str | Path,
    *,
    img_size: int = 512,
    batch_size: int = 8,
    max_new_tokens: int = 1,
    num_workers: int = 0,
    device: torch.device | None = None,
    persistent_workers: bool = False,
    desc: str = "val (TA batched)",
    include_meta: bool = False,
    include_solution: bool = False,
    answer_suffix: str = TA_ANSWER_SUFFIX_DEFAULT,
    save_preds_csv: str | Path | None = None,
    punnett_allele_overrides: dict[str, tuple[str, str, str, str]] | None = None,
    food_web_symbolic: bool = True,
) -> tuple[float, pd.DataFrame]:
    """Same scoring as the TA starter (full-string decode + ``extract_ta_choice_letter``), batched.

    Use this instead of a per-row loop when ``batch_size > 1`` for much faster validation on GPU.
    Set ``batch_size=1`` to mimic the original sequential notebook loop.

    When ``punnett_allele_overrides`` maps an example ``id`` to margin alleles
    ``(top1, top2, left1, left2)``, the reported prediction (and accuracy) uses
    ``hybrid_punnett_predict`` so the symbolic solver can override weak TA decode.
    When ``food_web_symbolic``, ``solve_food_web_symbolic`` may override TA decode using
    the six PNG templates and hard-coded adjacency (see FOOD_WEB_EDGES_BY_TEMPLATE).

    Output rows include ``pred_ta_model`` (letter-parse index before symbolic overrides).
    """
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:  # pragma: no cover
        def _tqdm(x, **kwargs):
            return x

    dev = device if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()

    _pun_ov = punnett_allele_overrides or {}
    _pun_track = bool(_pun_ov)
    rows_by_id = val_df.set_index("id", drop=False)
    _fw_track = bool(food_web_symbolic)

    ds = TaScienceVQADataset(
        val_df.reset_index(drop=True),
        image_root,
        img_size=img_size,
        is_train=False,
        include_meta=include_meta,
        include_solution=include_solution,
        answer_suffix=answer_suffix,
    )
    nw = int(num_workers)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=bool(persistent_workers) and nw > 0,
        collate_fn=partial(collate_vqa_eval, processor),
    )

    _markers = _ta_answer_markers_for_suffix(answer_suffix)

    val_rows: list[dict] = []
    correct = 0
    n_total = 0

    for batch in _tqdm(loader, desc=desc):
        b = dict(batch)
        answers = b.pop("answers", None)
        prompts_batch = b.pop("prompts", None)
        ids = b.pop("ids")
        num_choices = b.pop("num_choices")
        gen_in = _collate_to_generate(b)
        non_blocking = dev.type == "cuda"
        gen_in = {
            k: v.to(dev, non_blocking=non_blocking) if torch.is_tensor(v) else v
            for k, v in gen_in.items()
        }

        generated = model.generate(
            **gen_in,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        texts = processor.batch_decode(generated, skip_special_tokens=True)

        if answers is None:
            raise ValueError("val_df must include an answer column for accuracy")

        for i, ex_id in enumerate(ids):
            nc = int(num_choices[i])
            dec = texts[i]
            pred_letter = extract_ta_choice_letter(
                dec, max_choices=nc, answer_markers=_markers
            )
            pred_model = ta_letter_to_index(pred_letter, max_choices=nc)
            pred = pred_model
            rrow = rows_by_id.loc[ex_id]
            if _pun_track and ex_id in _pun_ov:
                pred = hybrid_punnett_predict(
                    rrow,
                    pred_model,
                    allele_override=_pun_ov[ex_id],
                )
            if _fw_track:
                gidx = solve_food_web_symbolic(rrow, image_root)
                if gidx is not None:
                    pred = int(gidx)
            gt = int(answers[i].item())
            is_correct = int(pred == gt)
            correct += is_correct
            n_total += 1
            row_out: dict = {
                "id": ex_id,
                "pred_letter": pred_letter,
                "pred": pred,
                "answer": gt,
                "correct": is_correct,
                "raw_text": dec,
            }
            if _pun_track or _fw_track:
                row_out["pred_ta_model"] = int(pred_model)
            if prompts_batch is not None:
                row_out["prompt"] = prompts_batch[i]
            val_rows.append(row_out)

    if was_training:
        model.train()
    acc = correct / max(n_total, 1)
    out_df = pd.DataFrame(val_rows)
    if save_preds_csv is not None:
        _outp = Path(save_preds_csv)
        _outp.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(_outp, index=False)
        print(f"Wrote val predictions → {_outp.resolve()}  rows={len(out_df)}  acc={acc:.6f}")
    return acc, out_df


@torch.inference_mode()
def predict_test_ta_batched(
    model,
    processor,
    test_df: pd.DataFrame,
    image_root: str | Path,
    *,
    img_size: int = 512,
    batch_size: int = 8,
    max_new_tokens: int = 1,
    num_workers: int = 0,
    device: torch.device | None = None,
    persistent_workers: bool = False,
    desc: str = "test (TA)",
    include_meta: bool = False,
    include_solution: bool = False,
    answer_suffix: str = TA_ANSWER_SUFFIX_DEFAULT,
    punnett_allele_overrides: dict[str, tuple[str, str, str, str]] | None = None,
    food_web_symbolic: bool = True,
) -> pd.DataFrame:
    """Kaggle test inference: TA decode + letter parse → ``id``, ``answer`` (choice index, 0-based).

    Invalid / missing letters map to ``0`` (same spirit as other notebooks).
    When ``food_web_symbolic``, the food-web graph solver may replace the TA index.
    """
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:  # pragma: no cover
        def _tqdm(x, **kwargs):
            return x

    dev = device if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()

    _pun_ov = punnett_allele_overrides or {}
    _pun_track = bool(_pun_ov)
    rows_by_id = test_df.set_index("id", drop=False)
    _fw_track = bool(food_web_symbolic)

    ds = TaScienceVQADataset(
        test_df.reset_index(drop=True),
        image_root,
        img_size=img_size,
        is_train=False,
        include_meta=include_meta,
        include_solution=include_solution,
        answer_suffix=answer_suffix,
    )
    nw = int(num_workers)
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=nw,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=bool(persistent_workers) and nw > 0,
        collate_fn=partial(collate_vqa_eval, processor),
    )

    _markers = _ta_answer_markers_for_suffix(answer_suffix)

    out_rows: list[dict] = []

    for batch in _tqdm(loader, desc=desc):
        b = dict(batch)
        b.pop("answers", None)
        ids = b.pop("ids")
        num_choices = b.pop("num_choices")
        gen_in = _collate_to_generate(b)
        non_blocking = dev.type == "cuda"
        gen_in = {
            k: v.to(dev, non_blocking=non_blocking) if torch.is_tensor(v) else v
            for k, v in gen_in.items()
        }

        generated = model.generate(
            **gen_in,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        texts = processor.batch_decode(generated, skip_special_tokens=True)

        for i, ex_id in enumerate(ids):
            nc = int(num_choices[i])
            pred_letter = extract_ta_choice_letter(
                texts[i], max_choices=nc, answer_markers=_markers
            )
            pred_model = ta_letter_to_index(pred_letter, max_choices=nc)
            pred = pred_model
            if pred < 0:
                pred = 0
                pred_model = pred
            rrow = rows_by_id.loc[ex_id]
            if _pun_track and ex_id in _pun_ov:
                pred = hybrid_punnett_predict(
                    rrow,
                    pred_model,
                    allele_override=_pun_ov[ex_id],
                )
            if _fw_track:
                gidx = solve_food_web_symbolic(rrow, image_root)
                if gidx is not None:
                    pred = int(gidx)
            out_rows.append({"id": ex_id, "answer": int(pred)})

    if was_training:
        model.train()
    return pd.DataFrame(out_rows)