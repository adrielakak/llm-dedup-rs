"""High-performance LLM dataset cleaning and MinHash LSH deduplication in Rust + PyO3."""

from llm_dedup_rs._core import (
    PipelineStats,
    clean_corpus,
    cpu_count,
    deduplicate_exact,
    deduplicate_minhash,
    run_full_pipeline,
)

__all__ = [
    "PipelineStats",
    "clean_corpus",
    "cpu_count",
    "deduplicate_exact",
    "deduplicate_minhash",
    "run_full_pipeline",
]
