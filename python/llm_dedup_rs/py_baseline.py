"""Pure-Python reference baseline implementing the exact same LLM dataset
cleaning, quality filtering, exact deduplication, and MinHash + LSH pipeline.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

_MERSENNE_PRIME = 2_147_483_647
_SHINGLE_BYTES = 5
_SEED_A_BASE = 6_364_136_223_846_793_005
_SEED_B_BASE = 1_442_695_040_888_963_407
_MASK_64 = 0xFFFF_FFFF_FFFF_FFFF


@dataclass(slots=True)
class PyPipelineStats:
    """Statistics produced by the pure-Python preprocessing pipeline."""

    total_docs: int
    filtered_quality: int
    exact_duplicates_removed: int
    fuzzy_duplicates_removed: int
    kept_docs: int
    retention_pct: float
    elapsed_ms: float
    throughput_docs_per_sec: float


def clean_text_py(text: str) -> str:
    """Strip HTML tags, lowercase text, and normalize whitespace."""
    no_html = _HTML_TAG_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", no_html).strip().lower()


def passes_quality_filter_py(
    text: str, min_chars: int = 20, min_alpha_ratio: float = 0.50
) -> bool:
    """Check minimum length and alphabetic character ratio."""
    total_chars = len(text)
    if total_chars < min_chars or total_chars == 0:
        return False
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    return (alpha_chars / total_chars) >= min_alpha_ratio


def _generate_permutations(num_perm: int) -> list[tuple[int, int]]:
    perms: list[tuple[int, int]] = []
    state = 0x9E37_79B9_7F4A_7C15
    for _ in range(num_perm):
        state = (state * _SEED_A_BASE + _SEED_B_BASE) & _MASK_64
        a = ((state >> 33) % (_MERSENNE_PRIME - 1)) + 1
        state = (state * _SEED_A_BASE + _SEED_B_BASE) & _MASK_64
        b = (state >> 33) % _MERSENNE_PRIME
        perms.append((a, b))
    return perms


def _compute_minhash_signature(
    text: str, perms: list[tuple[int, int]]
) -> list[int]:
    raw = text.encode("utf-8")
    if len(raw) < _SHINGLE_BYTES:
        shingles = [hash(raw) % _MERSENNE_PRIME]
    else:
        shingles = [
            hash(raw[i : i + _SHINGLE_BYTES]) % _MERSENNE_PRIME
            for i in range(len(raw) - _SHINGLE_BYTES + 1)
        ]

    # Deduplicate shingle hashes once before computing permutations
    unique_shingles = set(shingles)
    prime = _MERSENNE_PRIME
    return [min((a * h + b) % prime for h in unique_shingles) for a, b in perms]


def run_full_pipeline_py(
    docs: list[str],
    min_chars: int = 20,
    min_alpha_ratio: float = 0.50,
    threshold: float = 0.80,
    num_perm: int = 128,
    bands: int = 16,
) -> tuple[list[str], PyPipelineStats]:
    """Run the full LLM cleaning and MinHash + LSH deduplication pipeline in pure Python."""
    if not (0.0 <= min_alpha_ratio <= 1.0):
        raise ValueError("min_alpha_ratio must be between 0.0 and 1.0")
    if not (0.0 < threshold <= 1.0):
        raise ValueError("threshold must be in the interval (0.0, 1.0]")
    if num_perm <= 0 or bands <= 0 or num_perm % bands != 0:
        raise ValueError("num_perm and bands must be > 0 and divisible")

    start = time.perf_counter()
    total_docs = len(docs)

    # Step 1 & 2: Clean, quality filter, and exact deduplication
    seen_exact: set[str] = set()
    after_exact: list[str] = []
    passed_quality = 0

    for raw in docs:
        cleaned = clean_text_py(raw)
        if passes_quality_filter_py(cleaned, min_chars, min_alpha_ratio):
            passed_quality += 1
            if cleaned not in seen_exact:
                seen_exact.add(cleaned)
                after_exact.append(cleaned)

    filtered_quality = total_docs - passed_quality
    exact_duplicates_removed = passed_quality - len(after_exact)

    # Step 3: MinHash + LSH fuzzy deduplication
    n = len(after_exact)
    if n <= 1:
        kept_docs_list = after_exact
    else:
        rows = num_perm // bands
        perms = _generate_permutations(num_perm)
        signatures = [_compute_minhash_signature(doc, perms) for doc in after_exact]

        band_tables: list[dict[tuple[int, ...], list[int]]] = [
            defaultdict(list) for _ in range(bands)
        ]
        for idx, sig in enumerate(signatures):
            for b in range(bands):
                band_key = tuple(sig[b * rows : (b + 1) * rows])
                band_tables[b][band_key].append(idx)

        removed = [False] * n
        kept_indices: list[int] = []

        for i in range(n):
            if removed[i]:
                continue
            kept_indices.append(i)
            sig_i = signatures[i]
            candidates: set[int] = set()
            for b in range(bands):
                band_key = tuple(sig_i[b * rows : (b + 1) * rows])
                for j in band_tables[b][band_key]:
                    if j > i and not removed[j]:
                        candidates.add(j)

            for j in candidates:
                sig_j = signatures[j]
                matches = sum(1 for x, y in zip(sig_i, sig_j) if x == y)
                if (matches / num_perm) >= threshold:
                    removed[j] = True

        kept_docs_list = [after_exact[idx] for idx in kept_indices]

    fuzzy_duplicates_removed = len(after_exact) - len(kept_docs_list)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    kept_docs = len(kept_docs_list)
    retention_pct = (kept_docs / total_docs * 100.0) if total_docs > 0 else 0.0
    throughput = (total_docs / (elapsed_ms / 1000.0)) if elapsed_ms > 0.0 else 0.0

    stats = PyPipelineStats(
        total_docs=total_docs,
        filtered_quality=filtered_quality,
        exact_duplicates_removed=exact_duplicates_removed,
        fuzzy_duplicates_removed=fuzzy_duplicates_removed,
        kept_docs=kept_docs,
        retention_pct=retention_pct,
        elapsed_ms=elapsed_ms,
        throughput_docs_per_sec=throughput,
    )
    return kept_docs_list, stats
