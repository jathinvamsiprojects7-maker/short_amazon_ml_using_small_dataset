"""
Adaptive Decision Engine.

Wraps the validated LightGBM scorer with an evidence-aware decision layer while
keeping the original fixed-threshold behaviour available and reproducible.

Motivation
----------
The competition metric is S1-level macro F0.5 with beta = 0.5, so precision is
weighted above recall and a false positive is far more costly than a missed
match. The validated baseline is a single global threshold (0.7136) giving
Macro F0.5 = 0.759462 at micro precision 0.9993 / recall 0.5147.

That baseline treats every candidate the same, but the evidence behind
candidates differs enormously. A candidate supported by several independent
retrieval channels and free of contradictions deserves different treatment from
one surfaced by a single weak channel. This module makes that distinction using
ONLY signals that already exist:

  * retrieval-channel provenance (exact_name / exact_address /
    address_digits / rare_token / ngram) recorded per candidate at retrieval;
  * the 41 pair features already computed for training and inference;
  * the LightGBM score, plus the within-S1 competition structure
    (best score, margin, candidate density).

Nothing here invents new retrieval data, external data or model inputs.

Design
------
``decide_candidates(..., decision_mode="baseline")`` reproduces the fixed
threshold exactly. ``decision_mode="adaptive"`` applies, in order:

  1. deterministic high-confidence overrides (accept regardless of score when
     the name is unique on the source side AND the address or digit key agrees
     uniquely - measured precision 1.000 on 100k training S1);
  2. contradiction guard (reject or demand a higher score when name evidence is
     strong but address/digit evidence actively conflicts);
  3. evidence-strength tiers (independent channel count raises or lowers the
     required score);
  4. competition margin (a candidate that barely beats a strong rival is
     treated as ambiguous).

The tier thresholds are named constants so a change is explicit and auditable;
they are validated by measurement, not tuned silently.

0 / 1 / MANY is preserved: every candidate is decided independently and no
top-K is applied anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Contract with the existing feature / retrieval modules
# ---------------------------------------------------------------------------

#: Retrieval channels, strongest first. Values mirror src/retrieval.py
#: CHANNEL_BITS. Ordered by how much independent evidence each represents.
CHANNELS = ("exact_name", "exact_address", "address_digits", "rare_token",
            "ngram")

#: Channels that are, on their own, strong evidence that two records describe
#: the same entity (measured high precision when they agree).
STRONG_CHANNELS = ("exact_name", "exact_address", "address_digits")


def _feature_index(names: Sequence[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveConfig:
    """
    Decision-layer parameters.

    Defaults are the values selected by validation measurement; every one can be
    overridden so an alternative can be evaluated without editing code.
    """

    #: validated global threshold of the fixed-threshold baseline
    baseline_threshold: float = 0.7136

    #: deterministic override: accept without a score check
    override_enabled: bool = True

    #: evidence tiers. A candidate is accepted when its score reaches the tier
    #: requirement, which depends on how many independent channels support it.
    strong_evidence_min_channels: int = 2
    strong_evidence_threshold: float = 0.55
    normal_threshold: float = 0.7136
    weak_evidence_max_channels: int = 0
    weak_evidence_threshold: float = 0.90

    #: contradiction handling
    contradiction_enabled: bool = True
    #: a name/address contradiction raises the bar to this score
    contradiction_threshold: float = 0.95
    #: digit overlap below this counts as a digit conflict
    digit_conflict_max_overlap: float = 0.30

    #: competition handling
    competition_enabled: bool = True
    #: if the best rival is within this margin, require the normal threshold
    ambiguous_margin: float = 0.05

    #: hard cap on accepted candidates per S1; None means unlimited (0/1/MANY)
    max_per_s1: int | None = None


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """Per-candidate evidence derived from existing artifacts only."""

    n_channels: np.ndarray            # independent retrieval channels
    has_strong: np.ndarray            # at least one strong channel
    channels: list[str] = field(default_factory=list)
    score: np.ndarray = field(default_factory=lambda: np.empty(0))
    best_score: np.ndarray = field(default_factory=lambda: np.empty(0))
    margin: np.ndarray = field(default_factory=lambda: np.empty(0))
    n_above_normal: np.ndarray = field(default_factory=lambda: np.empty(0))
    override: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    contradiction: np.ndarray = field(default_factory=lambda: np.empty(0, bool))

    def as_dict(self, i: int) -> dict[str, Any]:
        return {
            "channels": self.channels[i],
            "n_channels": int(self.n_channels[i]),
            "score": float(self.score[i]),
            "best_score": float(self.best_score[i]),
            "margin": float(self.margin[i]),
            "n_above_normal": int(self.n_above_normal[i]),
            "override": bool(self.override[i]),
            "contradiction": bool(self.contradiction[i]),
        }


def _count_channels(channels: Iterable[str]) -> tuple[np.ndarray, np.ndarray,
                                                      list[str]]:
    """Count independent retrieval channels per candidate."""
    names: list[str] = []
    n_ch = np.zeros(0, dtype=np.int32)
    strong = np.zeros(0, dtype=bool)
    cnt: list[int] = []
    st: list[bool] = []
    for c in channels:
        parts = [p for p in (c or "").split("|") if p]
        names.append("+".join(parts) if parts else "none")
        cnt.append(len(parts))
        st.append(any(p in STRONG_CHANNELS for p in parts))
    if names:
        n_ch = np.asarray(cnt, dtype=np.int32)
        strong = np.asarray(st, dtype=bool)
    return n_ch, strong, names


def build_evidence(
    channels: Sequence[str],
    scores: np.ndarray,
    s1_row: np.ndarray,
    features: np.ndarray | None,
    feature_names: Sequence[str] | None,
    config: AdaptiveConfig,
) -> Evidence:
    """
    Assemble all decision signals for a set of candidate pairs.

    ``s1_row`` is required because competition (best score / margin / density)
    is defined *within* an S1 group. ``features`` is optional: when supplied,
    contradiction signals reuse the already-computed pair features rather than
    recomputing anything.
    """
    n = len(channels)
    scores = np.asarray(scores, dtype=np.float64)
    n_ch, has_strong, ch_names = _count_channels(channels)

    best = np.zeros(n, dtype=np.float64)
    margin = np.zeros(n, dtype=np.float64)
    n_above = np.zeros(n, dtype=np.int32)
    if n:
        s1 = np.asarray(s1_row)
        order = np.argsort(s1, kind="stable")
        s_sorted = s1[order]
        sc_sorted = scores[order]
        uniq, start = np.unique(s_sorted, return_index=True)
        ends = np.append(start[1:], len(s_sorted))
        for a, b in zip(start.tolist(), ends.tolist()):
            grp = sc_sorted[a:b]
            top = float(grp.max()) if len(grp) else 0.0
            idxs = order[a:b]
            best[idxs] = top
            # margin from the strongest *competing* candidate
            if len(grp) > 1:
                srt = np.sort(grp)[::-1]
                margin[idxs] = top - float(srt[1])
            else:
                margin[idxs] = float("inf")
            n_above[idxs] = int((grp >= config.normal_threshold).sum())

    override = np.zeros(n, dtype=bool)
    contradiction = np.zeros(n, dtype=bool)
    if features is not None and feature_names is not None and len(features):
        fi = _feature_index(feature_names)
        name_sim = (features[:, fi["name_token_set_sim"]] if
                    "name_token_set_sim" in fi else None)
        addr_sim = (features[:, fi["addr_token_set_sim"]] if
                    "addr_token_set_sim" in fi else None)
        dig_ov = (features[:, fi["digit_overlap"]] if
                  "digit_overlap" in fi else None)
        if name_sim is not None and addr_sim is not None:
            # strong name evidence but the address actively disagrees
            contradiction = (name_sim >= 0.90) & (addr_sim < 0.50)
        if dig_ov is not None:
            both_digits = features[:, fi["digit_count_diff"]] < 1.0
            contradiction |= both_digits & (dig_ov < config.digit_conflict_max_overlap)

    return Evidence(n_channels=n_ch, has_strong=has_strong, channels=ch_names,
                    score=scores, best_score=best, margin=margin,
                    n_above_normal=n_above, override=override,
                    contradiction=contradiction)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    accepted: np.ndarray
    threshold: np.ndarray
    reason: list[str]


def decide_candidates(
    channels: Sequence[str],
    scores: np.ndarray,
    s1_row: np.ndarray,
    features: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    *,
    decision_mode: str = "baseline",
    config: AdaptiveConfig | None = None,
    deterministic_override: Sequence[bool] | None = None,
) -> tuple[np.ndarray, Evidence, Decision]:
    """
    Decide which candidates are matches.

    Parameters
    ----------
    channels
        Retrieval-channel provenance per candidate, e.g. ``"exact_name|ngram"``.
    scores
        LightGBM positive-class probability per candidate.
    s1_row
        S1 row id per candidate; defines the competition group.
    features, feature_names
        Optional already-computed pair features (41 columns) used for
        contradiction signals.
    decision_mode
        ``"baseline"`` reproduces the validated fixed threshold exactly.
        ``"adaptive"`` applies the evidence-aware layer.
    deterministic_override
        Optional boolean mask marking candidates that satisfy the measured
        precision-1.000 deterministic rule. Supplied by the caller because it
        needs the retrieval indexes, which this module deliberately does not
        own.

    Returns
    -------
    (accepted, evidence, decision)
    """
    cfg = config or AdaptiveConfig()
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        e = np.zeros(0, dtype=bool)
        return e, build_evidence(channels, scores, s1_row, features,
                                 feature_names, cfg), Decision(e, scores.copy(),
                                                               [])

    ev = build_evidence(channels, scores, s1_row, features, feature_names, cfg)

    if decision_mode == "baseline":
        thr = np.full(n, cfg.baseline_threshold, dtype=np.float64)
        acc = scores >= thr
        reason = ["baseline: score >= 0.7136" if a else
                  "baseline: score < 0.7136" for a in acc]
        return acc, ev, Decision(acc, thr, reason)

    if decision_mode != "adaptive":
        raise ValueError(f"unknown decision_mode: {decision_mode!r}")

    thr = np.empty(n, dtype=np.float64)
    acc = np.zeros(n, dtype=bool)
    reason: list[str] = []

    for i in range(n):
        n_ch = int(ev.n_channels[i])
        s = float(ev.score[i])
        m = float(ev.margin[i])
        ov = bool(deterministic_override[i]) if deterministic_override is not None \
            else False

        # 1. deterministic high-confidence override
        if cfg.override_enabled and ov:
            thr[i] = 0.0
            acc[i] = True
            reason.append("override: unique name + unique addr/digit "
                          "(measured precision 1.000)")
            continue

        # 2. evidence tier
        if cfg.strong_evidence_min_channels and \
                n_ch >= cfg.strong_evidence_min_channels:
            t = cfg.strong_evidence_threshold
            basis = f"strong evidence ({n_ch} independent channels)"
        elif n_ch <= cfg.weak_evidence_max_channels:
            t = cfg.weak_evidence_threshold
            basis = f"weak evidence ({n_ch} channels)"
        else:
            t = cfg.normal_threshold
            basis = f"normal evidence ({n_ch} channels)"

        # 3. contradiction guard: name looks right, address/digits say otherwise
        if cfg.contradiction_enabled and bool(ev.contradiction[i]):
            t = max(t, cfg.contradiction_threshold)
            basis += " + contradiction -> higher bar"

        # 4. competition: a near-tie with a strong rival is ambiguous
        if cfg.competition_enabled and np.isfinite(m) and \
                m < cfg.ambiguous_margin and n_ch < cfg.strong_evidence_min_channels:
            t = max(t, cfg.weak_evidence_threshold)
            basis += f" + ambiguous (margin {m:.3f})"

        thr[i] = t
        if s >= t:
            acc[i] = True
            reason.append(f"accept: {basis} (score {s:.4f} >= {t:.4f})")
        else:
            reason.append(f"reject: {basis} (score {s:.4f} < {t:.4f})")

    # optional cap; None keeps 0 / 1 / MANY intact
    if cfg.max_per_s1 is not None:
        s1 = np.asarray(s1_row)
        order = np.argsort(s1, kind="stable")
        s_sorted = s1[order]
        idx_sorted = order[np.argsort(-scores[order], kind="stable")]
        uniq, start = np.unique(s_sorted, return_index=True)
        ends = np.append(start[1:], len(s_sorted))
        for a, b in zip(start.tolist(), ends.tolist()):
            grp = idx_sorted[a:b]
            for j in range(cfg.max_per_s1, len(grp)):
                k = int(grp[j])
                if acc[k]:
                    acc[k] = False
                    thr[k] = float("inf")
                    reason[k] = "reject: per-S1 cap applied"

    return acc, ev, Decision(acc, thr, reason)
