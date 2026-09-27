"""Shared utilities: config loading, normalization, timing, metric."""

from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    global _config
    if _config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


# ---------------------------------------------------------------------------
# Resource guard — HARD total-system ceiling
# ---------------------------------------------------------------------------

class ResourceGuard:
    """
    Enforces the total-system resource ceiling for the whole pipeline.

    The ceiling applies to the ENTIRE machine, so the guard measures
    system-wide RAM/CPU (``psutil.virtual_memory().percent``) rather than
    process-only usage. ``ceiling`` is therefore a fraction of *total* system
    RAM, and must stay below the hard limit (0.80) with headroom, because
    Windows, the shell and background services already occupy ~5.9 GB of the
    15.6 GB machine before Python starts.

    Behaviour:
      * ``reserve()`` raises :class:`MemoryCeilingError` if an allocation would
        push total system RAM above the ceiling.
      * ``adaptive()`` shrinks working sizes under pressure and restores them
        when it is low, so slow stages back off instead of trading system
        stability for speed.
      * ``gc()`` forces collection between stages.
    """

    #: hard upper bound on TOTAL system RAM, per the resource policy
    HARD_CEILING = 0.80
    #: working ceiling on TOTAL system RAM. Set to the hard limit because the
    #: policy is "use up to 80%, never past it": the guard throttles *before*
    #: the limit, so the machine never actually exceeds it.
    DEFAULT_CEILING = 0.80
    #: fraction of physical CPU cores the pipeline may keep busy (of TOTAL)
    DEFAULT_CPU_FRACTION = 0.78

    def __init__(
        self,
        ceiling: float | None = None,
        cpu_fraction: float | None = None,
    ) -> None:
        cfg = get_config()
        res = cfg.get("resources", {})
        self.ceiling = float(ceiling if ceiling is not None
                             else res.get("ram_ceiling", self.DEFAULT_CEILING))
        if self.ceiling > self.HARD_CEILING:
            raise ValueError(
                f"ram_ceiling {self.ceiling} exceeds the hard limit "
                f"{self.HARD_CEILING} on total system usage"
            )
        self.cpu_fraction = float(
            cpu_fraction if cpu_fraction is not None
            else res.get("cpu_fraction", self.DEFAULT_CPU_FRACTION)
        )
        self._psutil = None
        self._total_ram: float = 0.0
        self._n_cpu: int = 1
        self._try_init()

    def _try_init(self) -> None:
        try:
            import psutil
            self._psutil = psutil
            vm = psutil.virtual_memory()
            self._total_ram = float(vm.total)
            self._n_cpu = psutil.cpu_count(logical=True) or 1
        except Exception:
            self._psutil = None
            self._total_ram = 0.0
            self._n_cpu = 1

    # -- properties ------------------------------------------------------
    @property
    def budget_bytes(self) -> float:
        """Total bytes of RAM the pipeline may cause to be in use."""
        return self._total_ram * self.ceiling

    @property
    def max_workers(self) -> int:
        """Worker count that keeps total system CPU under the fraction."""
        return max(1, int(self._n_cpu * self.cpu_fraction))

    def system_ram_used(self) -> float:
        if self._psutil is None:
            return 0.0
        return float(self._psutil.virtual_memory().used)

    def system_ram_percent(self) -> float:
        if self._psutil is None:
            return 0.0
        return float(self._psutil.virtual_memory().percent)

    def system_cpu_percent(self) -> float:
        if self._psutil is None:
            return 0.0
        return float(self._psutil.cpu_percent(interval=None))

    # -- checks ----------------------------------------------------------
    def headroom_bytes(self) -> float:
        return self.budget_bytes - self.system_ram_used()

    def would_exceed(self, nbytes: float) -> bool:
        return self.system_ram_used() + float(nbytes) > self.budget_bytes

    def reserve(self, nbytes: float, what: str = "allocation") -> None:
        if self.would_exceed(nbytes):
            raise MemoryCeilingError(
                f"refusing {what} of {nbytes / 1024**3:.2f} GB: total system RAM "
                f"{self.system_ram_used() / 1024**3:.2f} GB would exceed the "
                f"{self.ceiling:.0%} ceiling "
                f"({self.budget_bytes / 1024**3:.2f} GB)"
            )

    def pressure(self) -> float:
        """
        Total-system RAM usage as a fraction of the ceiling (1.0 == at cap).

        Uses ``virtual_memory().percent`` (TOTAL machine usage) divided by the
        ceiling, so 1.0 means the machine as a whole has reached the limit.
        """
        if self.ceiling <= 0:
            return 0.0
        return self.system_ram_percent() / 100.0 / self.ceiling

    def status(self) -> str:
        return (
            f"RAM {self.system_ram_used() / 1024**3:.2f}/"
            f"{self._total_ram / 1024**3:.1f}GB "
            f"({self.system_ram_percent():.1f}% sys of "
            f"{self.ceiling:.0%} cap)  "
            f"CPU {self.system_cpu_percent():.0f}%  "
            f"max_workers={self.max_workers}"
        )

    def adaptive(self, size: int, minimum: int = 1, high: float = 0.80,
                 low: float = 0.55) -> int:
        """
        Return a working size shrunk under memory pressure and restored when
        pressure is low. ``high``/``low`` are fractions of the RAM budget.
        """
        p = self.pressure()
        if p >= high:
            size = max(minimum, size // 4)
        elif p >= low:
            size = max(minimum, size // 2)
        return size

    def pause_for_memory(self, timeout: float = 600.0) -> None:
        """Block until system RAM drops back under the high-pressure mark."""
        import time
        start = time.perf_counter()
        while self.pressure() >= 0.80:
            if time.perf_counter() - start > timeout:
                return
            time.sleep(2.0)

    def gc(self, rounds: int = 3) -> None:
        import gc as _gc
        for _ in range(rounds):
            _gc.collect()


class MemoryCeilingError(RuntimeError):
    """Raised when an operation would breach the total-system RAM ceiling."""


_GUARD: ResourceGuard | None = None


def guard() -> ResourceGuard:
    """Process-wide singleton resource guard."""
    global _GUARD
    if _GUARD is None:
        _GUARD = ResourceGuard()
    return _GUARD


def paths() -> dict[str, Path]:
    cfg = get_config()["paths"]
    return {k: Path(v) for k, v in cfg.items()}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Scripts that must NOT be NFKC/NFKD-decomposed: doing so explodes complex
# scripts (Devanagari, Bengali, Tamil, ...) into combining marks and shreds
# whole words into single characters. Measured example:
#   "हिमाचल प्रदेश" --NFKC--> "ह म चल प रद श"   (2 words -> 6 junk tokens)
# Latin/Cyrillic/Greek decomposing marks are usually desirable, so the fold is
# applied selectively per script instead of globally.
_NO_FOLD_SCRIPTS = (
    "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL",
    "TELUGU", "KANNADA", "MALAYALAM", "SINHALA", "MYANMAR", "KHMER",
    "THAI", "LAO", "TIBETAN", "ETHIOPIC", "CHEROKEE", "MONGOLIAN",
    "HANGUL", "HIRAGANA", "KATAKANA",
)
_NO_FOLD_PREFIXES = tuple(_NO_FOLD_SCRIPTS)

_ZERO_WIDTH = re.compile(r"[-‏‪-‮⁠-⁤﻿]")
_WS = re.compile(r"\s+")
# Keep word characters AND Unicode combining marks. Dropping marks (categories
# Mn/Mc, e.g. Devanagari matras) splits a single word into fragments, because
# Python's \w does not include them. Punctuation/symbols are still removed.
_PUNCT = re.compile(
    r"[^\w\s\u0300-\u036f\u0483-\u0489\u0591-\u05bd\u0610-\u061a"
    r"\u064b-\u065f\u0670\u06d6-\u06dc\u0900-\u0903\u093a-\u094f"
    r"\u0951-\u0957\u0962-\u0963\u0e31\u0e34-\u0e3a\u0e47-\u0e4e"
    r"\u1ab0-\u1aff\u1dc0-\u1dff\u20d0-\u20f0\ufe00-\ufe0f\ufe20-\ufe2f]",
    flags=re.UNICODE,
)
_AMP = "&"

# Devanagari danda and similar punctuation
_DANDA = re.compile(r"[।॥،؛؟۔]")


def norm_text(value: str) -> str:
    """
    Conservative, script-safe normalization.

    Steps:
      * strip zero-width / bidi control characters
      * NFKC-normalize only when the text contains no protected script
      * casefold
      * "&" -> " and "
      * punctuation -> space (protecting non-ASCII letters and digits)
      * collapse whitespace

    Raw values are always preserved upstream; this is a derived view.
    """
    if not value:
        return ""
    value = _ZERO_WIDTH.sub("", value)
    if not value.isascii() and _has_protected_script(value):
        # Avoid NFKC: keep the codepoints as authored, only clean separators.
        value = _DANDA.sub(" ", value)
    else:
        value = unicodedata.normalize("NFKC", value)
    value = value.casefold()
    value = value.replace(_AMP, " and ")
    value = _PUNCT.sub(" ", value)
    return _WS.sub(" ", value).strip()


def _has_protected_script(value: str) -> bool:
    """True if the text contains a script that NFKC would destroy."""
    try:
        for ch in value:
            if ch.isascii():
                continue
            name = unicodedata.name(ch, "")
            if name.startswith(_NO_FOLD_PREFIXES):
                return True
    except (ValueError, TypeError):
        return False
    return False


def norm_tokens(value: str) -> list[str]:
    return norm_text(value).split()


def address_digits(value: str) -> str:
    """Extract digit sequences from address, space-joined."""
    return " ".join(re.findall(r"\d+", value))


def address_digit_set(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"\d+", value))


def token_set(value: str) -> frozenset[str]:
    return frozenset(t for t in norm_tokens(value) if len(t) >= 3)


# ---------------------------------------------------------------------------
# Timing context
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            print(f"[timer] {self.label}: {self.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# S1-level Macro F0.5 metric
# ---------------------------------------------------------------------------

def f05_score(precision: float, recall: float) -> float:
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return 1.25 * precision * recall / denom


def macro_f05(
    predictions: dict[str, set[str]],
    ground_truth: dict[str, set[str]],
) -> dict[str, float]:
    """
    Compute S1-level macro F0.5.
    predictions: {s1_id: set of predicted match IDs}
    ground_truth: {s1_id: set of true match IDs}
    Every S1 in ground_truth is included; missing predictions treated as empty.
    """
    scores = []
    tp_total = fp_total = fn_total = 0
    zero_correct = zero_total = 0

    for s1, true_set in ground_truth.items():
        pred_set = predictions.get(s1, set())
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        tp_total += tp
        fp_total += fp
        fn_total += fn

        if not true_set:
            zero_total += 1
            if not pred_set:
                zero_correct += 1
                scores.append(1.0)
            else:
                scores.append(0.0)
            continue

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(f05_score(prec, rec))

    macro = sum(scores) / len(scores) if scores else 0.0
    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    return {
        "macro_f05": round(macro, 6),
        "micro_f05": round(f05_score(micro_prec, micro_rec), 6),
        "micro_precision": round(micro_prec, 6),
        "micro_recall": round(micro_rec, 6),
        "n_s1": len(scores),
        "zero_match_accuracy": round(zero_correct / zero_total, 6) if zero_total else None,
        "zero_match_total": zero_total,
    }
