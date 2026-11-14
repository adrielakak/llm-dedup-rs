"""Integration and equivalence tests comparing Rust PyO3 pipeline with pure Python baseline."""

from __future__ import annotations

import pytest

import llm_dedup_rs
from llm_dedup_rs.py_baseline import run_full_pipeline_py


def test_clean_corpus_strips_html_and_normalizes_whitespace() -> None:
    raw = [
        "  <div>Hello   <b>LLM</b>\nWorld!</div>  ",
        "Plain   Text\t\tTest",
    ]
    cleaned = llm_dedup_rs.clean_corpus(raw)
    assert cleaned == ["hello llm world!", "plain text test"]


def test_deduplicate_exact_preserves_first_occurrence() -> None:
    docs = ["doc alpha", "doc beta", "doc alpha", "doc gamma", "doc beta"]
    assert llm_dedup_rs.deduplicate_exact(docs) == ["doc alpha", "doc beta", "doc gamma"]


def test_deduplicate_minhash_removes_near_duplicates() -> None:
    docs = [
        "Large language models benefit enormously from aggressive deduplication of training corpora.",
        "Large language models benefit enormously from aggressive deduplication of training corpus.",
        "Rust provides zero-cost abstractions, memory safety, and thread safety without garbage collection.",
    ]
    kept = llm_dedup_rs.deduplicate_minhash(docs, threshold=0.75, num_perm=128, bands=16)
    assert len(kept) == 2
    assert kept[0] == docs[0]
    assert kept[1] == docs[2]


def test_full_pipeline_matches_python_baseline_behavior() -> None:
    dataset = [
        "1234567890 !!!",  # noise -> quality filter
        "short",  # too short -> quality filter
        "<p>Fine-tuning open-weight language models requires high-quality curated datasets without repetitive web boilerplate.</p>",
        "Fine-tuning open-weight language models requires high-quality curated datasets without repetitive web boilerplate.",  # exact duplicate
        "Fine-tuning open-weight language models requires high-quality curated datasets without repetitive web boilerplates.",  # fuzzy near-duplicate
        "Distributed systems engineered in Rust eliminate data races at compile time via ownership and borrowing rules.",
    ]

    rs_kept, rs_stats = llm_dedup_rs.run_full_pipeline(
        dataset, min_chars=20, min_alpha_ratio=0.50, threshold=0.80
    )
    py_kept, py_stats = run_full_pipeline_py(
        dataset, min_chars=20, min_alpha_ratio=0.50, threshold=0.80
    )

    assert rs_kept == py_kept
    assert rs_stats.total_docs == py_stats.total_docs == 6
    assert rs_stats.filtered_quality == py_stats.filtered_quality == 2
    assert rs_stats.exact_duplicates_removed == py_stats.exact_duplicates_removed == 1
    assert rs_stats.fuzzy_duplicates_removed == py_stats.fuzzy_duplicates_removed == 1
    assert rs_stats.kept_docs == py_stats.kept_docs == 2


def test_full_pipeline_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        llm_dedup_rs.run_full_pipeline(["valid sample text for testing"], threshold=1.5)

    with pytest.raises(ValueError):
        llm_dedup_rs.run_full_pipeline(["valid sample text for testing"], num_perm=100, bands=16)

    with pytest.raises(ValueError):
        llm_dedup_rs.run_full_pipeline(["valid sample text for testing"], num_threads=0)
