"""Benchmark runner comparing pure Python vs Rust (PyO3 + Rayon) on synthetic LLM corpora.

Measures execution time, throughput, multi-core thread scaling, and true OS peak RAM (RSS)
via isolated subprocesses, exports `benchmarks/results.json`, and renders dark-mode SVG charts
with percentage gains in `assets/`.
"""

from __future__ import annotations

import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
RESULTS_PATH = ROOT_DIR / "benchmarks" / "results.json"

VOCAB_NOUNS = [
    "transformer", "tokenizer", "gradient", "embedding", "attention", "throughput",
    "latency", "pipeline", "dataset", "corpus", "sharding", "quantization", "kernel",
    "scheduler", "allocator", "checkpoint", "inference", "distillation", "alignment",
    "benchmark", "telemetry", "serialization", "compression", "partition", "concurrency",
    "backpropagation", "regularization", "normalization", "vectorization", "indexing",
    "clustering", "retrieval", "ranking", "filtering", "deduplication", "hashing",
    "protocol", "compiler", "runtime", "bytecode", "register", "buffer", "cache",
    "socket", "worker", "cluster", "replica", "snapshot", "manifest", "metadata",
]

VOCAB_ADJS = [
    "distributed", "parallel", "deterministic", "zero-copy", "asynchronous", "scalable",
    "memory-safe", "lock-free", "vectorized", "contiguous", "low-latency", "reproducible",
    "lightweight", "stateless", "idempotent", "fault-tolerant", "high-bandwidth", "compact",
    "incremental", "adaptive", "modular", "portable", "verifiable", "predictable",
]

VOCAB_VERBS = [
    "accelerates", "optimizes", "validates", "transforms", "aggregates", "compresses",
    "partitions", "evaluates", "calibrates", "stabilizes", "streams", "encodes",
    "indexes", "extracts", "normalizes", "dispatches", "synchronizes", "verifies",
]


def generate_synthetic_llm_corpus(n_docs: int, seed: int = 42) -> list[str]:
    """Generate a realistic synthetic LLM corpus with noise, exact duplicates, and near-duplicates."""
    rng = random.Random(seed)
    n_unique = max(1, int(n_docs * 0.60))
    n_exact = int(n_docs * 0.20)
    n_fuzzy = int(n_docs * 0.15)
    n_noisy = n_docs - n_unique - n_exact - n_fuzzy

    unique_docs: list[str] = []
    for i in range(n_unique):
        words = [
            rng.choice(VOCAB_ADJS),
            rng.choice(VOCAB_NOUNS),
            rng.choice(VOCAB_VERBS),
            rng.choice(VOCAB_ADJS),
            rng.choice(VOCAB_NOUNS),
            "across",
            rng.choice(VOCAB_NOUNS),
            "and",
            rng.choice(VOCAB_ADJS),
            rng.choice(VOCAB_NOUNS),
            "with",
            rng.choice(VOCAB_NOUNS),
            rng.choice(VOCAB_VERBS),
            rng.choice(VOCAB_NOUNS),
            "for",
            rng.choice(VOCAB_ADJS),
            rng.choice(VOCAB_NOUNS),
        ]
        doc = f"<article>Entry {i:06d}: " + " ".join(words) + ".</article>"
        unique_docs.append(doc)

    corpus = list(unique_docs)

    # Exact duplicates (wrapped in messy HTML/whitespace that cleaning collapses)
    for i in range(n_exact):
        base = unique_docs[i % len(unique_docs)]
        corpus.append(f"  <div>  {base.upper()}  </div> \n")

    # Fuzzy near-duplicates (small edit on distinct base docs -> ~15% fuzzy dup rate)
    for i in range(n_fuzzy):
        base = unique_docs[(n_unique - 1 - i) % len(unique_docs)]
        near = base.replace("Entry ", "Sample ").replace(".</article>", " updated.</article>")
        corpus.append(near)

    # Low-quality / short / non-alphabetic noise
    noise_templates = [
        "<div>123456 7890 !!! ???</div>",
        "   ok   ",
        "--- 404 --- 999 ---",
        "<br/>   $$$ ### @@@   ",
    ]
    for i in range(n_noisy):
        corpus.append(noise_templates[i % len(noise_templates)])

    rng.shuffle(corpus)
    return corpus


def measure_peak_rss_mb(engine: str, n_docs: int) -> float:
    """Measure isolated process peak memory (RSS in MiB) for a given engine and corpus size."""
    code = f"""
import resource, sys
from benchmarks.run_benchmarks import generate_synthetic_llm_corpus
docs = generate_synthetic_llm_corpus({n_docs})
if "{engine}" == "python":
    from llm_dedup_rs.py_baseline import run_full_pipeline_py
    run_full_pipeline_py(docs)
else:
    import llm_dedup_rs
    llm_dedup_rs.run_full_pipeline(docs)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
div = (1024 * 1024) if sys.platform == "darwin" else 1024
print(rss / div)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT_DIR)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT_DIR),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip().splitlines()[-1])


def measure_baseline_rss_mb(n_docs: int) -> float:
    """Measure baseline process RSS (MiB) after loading corpus without running deduplication."""
    code = f"""
import resource, sys
from benchmarks.run_benchmarks import generate_synthetic_llm_corpus
docs = generate_synthetic_llm_corpus({n_docs})
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
div = (1024 * 1024) if sys.platform == "darwin" else 1024
print(rss / div)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT_DIR)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT_DIR),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return float(proc.stdout.strip().splitlines()[-1])


def _nice_max(val: float) -> float:
    if val <= 0:
        return 1.0
    mag = 10 ** int(f"{val:.0e}".split("e")[1])
    for mult in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if mult * mag >= val:
            return mult * mag
    return 10 * mag


def render_execution_time_svg(scaling: list[dict[str, Any]], out_path: Path) -> None:
    """Render SVG line chart comparing Python vs Rust 1-Thread vs Rust Multi-Core execution time."""
    width, height = 880, 465
    pad_l, pad_r, pad_t, pad_b = 85, 45, 92, 75
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    max_y = _nice_max(max(row["python_ms"] for row in scaling) * 1.08)
    n_pts = len(scaling)

    def x_pos(idx: int) -> float:
        return pad_l + (idx / max(1, n_pts - 1)) * plot_w

    def y_pos(val: float) -> float:
        return pad_t + plot_h - (val / max_y) * plot_h

    def polyline(vals: list[float]) -> str:
        return " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(vals))

    py_vals = [r["python_ms"] for r in scaling]
    rs1_vals = [r["rust_1t_ms"] for r in scaling]
    rsm_vals = [r["rust_multi_ms"] for r in scaling]

    last = scaling[-1]
    speedup_pct = last["speedup_pct"]
    time_reduction_pct = last["time_reduction_pct"]

    grid_lines = []
    for k in range(5):
        v = max_y * k / 4
        y = y_pos(v)
        label = f"{v / 1000:.1f}s" if max_y >= 2000 else f"{v:.0f}ms"
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#262c36" stroke-width="1"/>'
            f'<text x="{pad_l - 12}" y="{y + 4:.1f}" text-anchor="end" fill="#8b949e" font-size="12" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{label}</text>'
        )

    x_labels = []
    dots = []
    for i, row in enumerate(scaling):
        x = x_pos(i)
        n_label = f"{row['n_docs'] / 1000:g}k" if row["n_docs"] >= 1000 else str(row["n_docs"])
        x_labels.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 24}" text-anchor="middle" fill="#8b949e" font-size="12" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{n_label} docs</text>'
        )
        dots.append(f'<circle cx="{x:.1f}" cy="{y_pos(row["python_ms"]):.1f}" r="4.5" fill="#f85149" stroke="#0d1117" stroke-width="1.5"/>')
        dots.append(f'<circle cx="{x:.1f}" cy="{y_pos(row["rust_1t_ms"]):.1f}" r="4.5" fill="#f0883e" stroke="#0d1117" stroke-width="1.5"/>')
        dots.append(f'<circle cx="{x:.1f}" cy="{y_pos(row["rust_multi_ms"]):.1f}" r="5" fill="#3fb950" stroke="#0d1117" stroke-width="1.5"/>')

    # Value annotation on the final point
    end_x = x_pos(n_pts - 1)
    py_end_y = y_pos(last["python_ms"])
    rs_end_y = y_pos(last["rust_multi_ms"])

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <rect width="{width}" height="{height}" rx="12" fill="#0d1117" stroke="#30363d" stroke-width="1"/>
  <text x="{pad_l}" y="34" fill="#f0f6fc" font-size="18" font-weight="700" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">Temps d'exécution : Python Pur vs Rust (PyO3 + Rayon)</text>
  <text x="{pad_l}" y="58" fill="#8b949e" font-size="13" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">Pipeline complet : Nettoyage HTML + Filtre qualité + Dédup exacte + MinHash LSH (128 perm.)</text>

  <rect x="{width - 265}" y="14" width="220" height="28" rx="14" fill="#132a1c" stroke="#2ea043" stroke-width="1"/>
  <text x="{width - 155}" y="32" text-anchor="middle" fill="#3fb950" font-size="12.5" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">-{time_reduction_pct:.1f}% temps (+{speedup_pct:.0f}%)</text>

  {"".join(grid_lines)}
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="#30363d" stroke-width="1.5"/>
  {"".join(x_labels)}

  <polyline fill="none" stroke="#f85149" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{polyline(py_vals)}"/>
  <polyline fill="none" stroke="#f0883e" stroke-width="2.5" stroke-dasharray="6 4" stroke-linecap="round" stroke-linejoin="round" points="{polyline(rs1_vals)}"/>
  <polyline fill="none" stroke="#3fb950" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{polyline(rsm_vals)}"/>

  {"".join(dots)}

  <text x="{end_x - 8:.1f}" y="{py_end_y - 10:.1f}" text-anchor="end" fill="#ff7b72" font-size="12" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{last["python_ms"]:.0f} ms</text>
  <text x="{end_x - 8:.1f}" y="{rs_end_y - 12:.1f}" text-anchor="end" fill="#3fb950" font-size="12" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{last["rust_multi_ms"]:.0f} ms ({last["speedup_x"]:.1f}x faster)</text>

  <!-- Legend -->
  <g transform="translate({pad_l}, {height - 26})">
    <circle cx="6" cy="0" r="5" fill="#f85149"/>
    <text x="18" y="4" fill="#c9d1d9" font-size="12" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Python Pur (GIL)</text>
    <circle cx="170" cy="0" r="5" fill="#f0883e"/>
    <text x="182" y="4" fill="#c9d1d9" font-size="12" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Rust PyO3 (1 Thread)</text>
    <circle cx="360" cy="0" r="5" fill="#3fb950"/>
    <text x="372" y="4" fill="#c9d1d9" font-size="12" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Rust PyO3 + Rayon ({last["threads"]} Threads)</text>
  </g>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def render_thread_scaling_svg(thread_scaling: list[dict[str, Any]], out_path: Path) -> None:
    """Render SVG chart of Rust throughput (docs/sec) and scaling efficiency % across thread counts."""
    width, height = 880, 425
    pad_l, pad_r, pad_t, pad_b = 85, 45, 92, 70
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    max_y = _nice_max(max(r["throughput_docs_per_sec"] for r in thread_scaling) * 1.15)
    n_pts = len(thread_scaling)

    def x_pos(idx: int) -> float:
        return pad_l + (idx / max(1, n_pts - 1)) * plot_w

    def y_pos(val: float) -> float:
        return pad_t + plot_h - (val / max_y) * plot_h

    pts = " ".join(
        f"{x_pos(i):.1f},{y_pos(r['throughput_docs_per_sec']):.1f}"
        for i, r in enumerate(thread_scaling)
    )
    area_pts = (
        f"{x_pos(0):.1f},{pad_t + plot_h:.1f} "
        + pts
        + f" {x_pos(n_pts - 1):.1f},{pad_t + plot_h:.1f}"
    )

    grid = []
    for k in range(5):
        v = max_y * k / 4
        y = y_pos(v)
        label = f"{v / 1000:.0f}k/s" if v >= 1000 else f"{v:.0f}/s"
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#262c36" stroke-width="1"/>'
            f'<text x="{pad_l - 12}" y="{y + 4:.1f}" text-anchor="end" fill="#8b949e" font-size="12" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{label}</text>'
        )

    labels_and_dots = []
    for i, r in enumerate(thread_scaling):
        x = x_pos(i)
        y = y_pos(r["throughput_docs_per_sec"])
        gain_pct = r["scaling_gain_pct"]
        badge = "baseline" if i == 0 else f"+{gain_pct:.0f}%"
        anchor = "start" if i == 0 else ("end" if i == n_pts - 1 else "middle")
        labels_and_dots.append(
            f'<text x="{x:.1f}" y="{height - pad_b + 24}" text-anchor="middle" fill="#8b949e" font-size="12" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{r["threads"]} thread{"s" if r["threads"] > 1 else ""}</text>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#58a6ff" stroke="#0d1117" stroke-width="2"/>'
            f'<text x="{x:.1f}" y="{y - 14:.1f}" text-anchor="{anchor}" fill="#79c0ff" font-size="11.5" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{r["throughput_docs_per_sec"]:,.0f} docs/s ({badge})</text>'
        )

    max_gain = thread_scaling[-1]["scaling_gain_pct"]
    n_docs_tested = thread_scaling[0]["n_docs"]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <linearGradient id="areaBlue" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#58a6ff" stop-opacity="0.30"/>
      <stop offset="100%" stop-color="#58a6ff" stop-opacity="0.02"/>
    </linearGradient>
  </defs>
  <rect width="{width}" height="{height}" rx="12" fill="#0d1117" stroke="#30363d" stroke-width="1"/>
  <text x="{pad_l}" y="34" fill="#f0f6fc" font-size="18" font-weight="700" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">Scalabilité Multi-Cœurs (Rayon sans GIL)</text>
  <text x="{pad_l}" y="58" fill="#8b949e" font-size="13" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">Débit mesuré (documents/seconde) sur {n_docs_tested:,} documents selon le nombre de threads</text>

  <rect x="{width - 245}" y="14" width="200" height="28" rx="14" fill="#11223a" stroke="#388bfd" stroke-width="1"/>
  <text x="{width - 145}" y="32" text-anchor="middle" fill="#79c0ff" font-size="12.5" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">+{max_gain:.0f}% débit multi-cœurs</text>

  {"".join(grid)}
  <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{width - pad_r}" y2="{pad_t + plot_h}" stroke="#30363d" stroke-width="1.5"/>

  <polygon fill="url(#areaBlue)" points="{area_pts}"/>
  <polyline fill="none" stroke="#58a6ff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="{pts}"/>
  {"".join(labels_and_dots)}
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def render_memory_svg(mem: dict[str, Any], out_path: Path) -> None:
    """Render SVG bar chart comparing peak memory overhead (MiB) of Python vs Rust."""
    width, height = 880, 345
    pad_l, pad_r, pad_t = 210, 260, 100
    bar_area_w = width - pad_l - pad_r

    py_delta = mem["python_net_mb"]
    rs_delta = mem["rust_net_mb"]
    max_mb = max(py_delta, rs_delta, 1.0) * 1.10

    py_w = max(8.0, (py_delta / max_mb) * bar_area_w)
    rs_w = max(8.0, (rs_delta / max_mb) * bar_area_w)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <rect width="{width}" height="{height}" rx="12" fill="#0d1117" stroke="#30363d" stroke-width="1"/>
  <text x="45" y="34" fill="#f0f6fc" font-size="18" font-weight="700" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">Empreinte Mémoire Nette (Pic RAM durant l'exécution)</text>
  <text x="45" y="58" fill="#8b949e" font-size="13" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica, Arial, sans-serif">Surcoût RAM mesuré par processus isolé (ru_maxrss) sur {mem["n_docs"]:,} documents</text>

  <rect x="{width - 250}" y="14" width="205" height="28" rx="14" fill="#132a1c" stroke="#2ea043" stroke-width="1"/>
  <text x="{width - 147}" y="32" text-anchor="middle" fill="#3fb950" font-size="12.5" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">-{mem["ram_reduction_pct"]:.1f}% de RAM</text>

  <!-- Python Bar -->
  <text x="{pad_l - 16}" y="{pad_t + 26}" text-anchor="end" fill="#c9d1d9" font-size="14" font-weight="600" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Python Pur (list/dict/int)</text>
  <rect x="{pad_l}" y="{pad_t}" width="{py_w:.1f}" height="40" rx="6" fill="#f85149" fill-opacity="0.85"/>
  <text x="{pad_l + py_w + 12:.1f}" y="{pad_t + 25}" fill="#ff7b72" font-size="13" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{py_delta:.1f} MiB (pic total: {mem["python_peak_rss_mb"]:.1f} MiB)</text>

  <!-- Rust Bar -->
  <text x="{pad_l - 16}" y="{pad_t + 106}" text-anchor="end" fill="#c9d1d9" font-size="14" font-weight="600" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Rust PyO3 (flat Vec&lt;u32&gt;)</text>
  <rect x="{pad_l}" y="{pad_t + 80}" width="{rs_w:.1f}" height="40" rx="6" fill="#3fb950" fill-opacity="0.90"/>
  <text x="{pad_l + rs_w + 12:.1f}" y="{pad_t + 105}" fill="#3fb950" font-size="13" font-weight="700" font-family="ui-monospace, SFMono-Regular, Menlo, monospace">{rs_delta:.1f} MiB (pic total: {mem["rust_peak_rss_mb"]:.1f} MiB)</text>

  <text x="45" y="303" fill="#8b949e" font-size="12" font-family="-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">Baseline corpus seul : {mem["baseline_rss_mb"]:.1f} MiB • Rust stocke toutes les signatures MinHash dans un buffer contigu sans objets heap Python.</text>
</svg>
"""
    out_path.write_text(svg, encoding="utf-8")


def main() -> None:
    """Execute the full benchmark suite, save JSON metrics, and render SVG plots."""
    import llm_dedup_rs
    from llm_dedup_rs.py_baseline import run_full_pipeline_py

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    detected_cores = llm_dedup_rs.cpu_count()
    print(f"Detected Rayon threads: {detected_cores}")

    # Warmup
    warmup_docs = generate_synthetic_llm_corpus(400)
    llm_dedup_rs.run_full_pipeline(warmup_docs)
    run_full_pipeline_py(warmup_docs)

    # Probe pure Python speed on 2,000 docs to keep full suite under ~35s
    probe_docs = generate_synthetic_llm_corpus(2_000)
    t0 = time.perf_counter()
    run_full_pipeline_py(probe_docs)
    probe_sec = time.perf_counter() - t0
    if probe_sec > 2.2:
        sizes = [1_000, 2_500, 5_000, 10_000, 20_000]
    else:
        sizes = [1_000, 5_000, 10_000, 25_000, 50_000]

    print(f"Benchmark dataset sizes: {sizes}")

    scaling_results: list[dict[str, Any]] = []
    last_stats_summary: dict[str, Any] = {}

    for n_docs in sizes:
        docs = generate_synthetic_llm_corpus(n_docs)

        t_py0 = time.perf_counter()
        _, py_stats = run_full_pipeline_py(docs)
        py_ms = (time.perf_counter() - t_py0) * 1000.0

        t_rs1_0 = time.perf_counter()
        _, rs1_stats = llm_dedup_rs.run_full_pipeline(docs, num_threads=1)
        rs1_ms = (time.perf_counter() - t_rs1_0) * 1000.0

        t_rsm_0 = time.perf_counter()
        _, rsm_stats = llm_dedup_rs.run_full_pipeline(docs, num_threads=None)
        rsm_ms = (time.perf_counter() - t_rsm_0) * 1000.0

        speedup_x = py_ms / rsm_ms if rsm_ms > 0 else 0.0
        speedup_pct = (speedup_x - 1.0) * 100.0
        time_reduction_pct = ((py_ms - rsm_ms) / py_ms * 100.0) if py_ms > 0 else 0.0

        row = {
            "n_docs": n_docs,
            "threads": detected_cores,
            "python_ms": round(py_ms, 2),
            "rust_1t_ms": round(rs1_ms, 2),
            "rust_multi_ms": round(rsm_ms, 2),
            "python_throughput": round(n_docs / (py_ms / 1000.0), 1),
            "rust_multi_throughput": round(n_docs / (rsm_ms / 1000.0), 1),
            "speedup_x": round(speedup_x, 2),
            "speedup_pct": round(speedup_pct, 1),
            "time_reduction_pct": round(time_reduction_pct, 2),
        }
        scaling_results.append(row)
        print(
            f"[{n_docs:>6} docs] Python: {py_ms:8.1f} ms | Rust 1T: {rs1_ms:6.1f} ms | "
            f"Rust {detected_cores}T: {rsm_ms:6.1f} ms | Speedup: {speedup_x:.1f}x (-{time_reduction_pct:.1f}%)"
        )

        if n_docs == sizes[-1]:
            last_stats_summary = {
                "total_docs": rsm_stats.total_docs,
                "filtered_quality": rsm_stats.filtered_quality,
                "filtered_quality_pct": round(rsm_stats.filtered_quality / rsm_stats.total_docs * 100.0, 2),
                "exact_duplicates_removed": rsm_stats.exact_duplicates_removed,
                "exact_duplicates_pct": round(rsm_stats.exact_duplicates_removed / rsm_stats.total_docs * 100.0, 2),
                "fuzzy_duplicates_removed": rsm_stats.fuzzy_duplicates_removed,
                "fuzzy_duplicates_pct": round(rsm_stats.fuzzy_duplicates_removed / rsm_stats.total_docs * 100.0, 2),
                "kept_docs": rsm_stats.kept_docs,
                "retention_pct": round(rsm_stats.retention_pct, 2),
                "py_kept_docs": py_stats.kept_docs,
            }

    # Multi-core thread scaling on a larger corpus (50,000 docs) for crisp Rust measurements
    thread_bench_size = 50_000
    thread_docs = generate_synthetic_llm_corpus(thread_bench_size)
    thread_counts = sorted({t for t in (1, 2, 4, 8, detected_cores) if t <= detected_cores})
    thread_scaling: list[dict[str, Any]] = []
    base_throughput = 0.0

    for tc in thread_counts:
        # Best of 3 runs for steady multi-core measurement
        best_ms = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            llm_dedup_rs.run_full_pipeline(thread_docs, num_threads=tc)
            best_ms = min(best_ms, (time.perf_counter() - t0) * 1000.0)
        tp = thread_bench_size / (best_ms / 1000.0)
        if tc == 1:
            base_throughput = tp
        gain_pct = ((tp - base_throughput) / base_throughput * 100.0) if base_throughput > 0 else 0.0
        thread_scaling.append(
            {
                "n_docs": thread_bench_size,
                "threads": tc,
                "elapsed_ms": round(best_ms, 2),
                "throughput_docs_per_sec": round(tp, 1),
                "scaling_gain_pct": round(gain_pct, 1),
            }
        )
        print(f"[Thread scaling {tc:>2}T on {thread_bench_size} docs] {best_ms:6.1f} ms | {tp:,.0f} docs/s (+{gain_pct:.1f}%)")

    # Isolated RSS memory benchmark on max tested size
    mem_docs_count = sizes[-1]
    base_rss = measure_baseline_rss_mb(mem_docs_count)
    py_rss = measure_peak_rss_mb("python", mem_docs_count)
    rs_rss = measure_peak_rss_mb("rust", mem_docs_count)
    py_net = max(0.1, py_rss - base_rss)
    rs_net = max(0.1, rs_rss - base_rss)
    ram_reduction_pct = ((py_net - rs_net) / py_net * 100.0) if py_net > rs_net else 0.0
    total_rss_reduction_pct = ((py_rss - rs_rss) / py_rss * 100.0) if py_rss > rs_rss else 0.0

    memory_summary = {
        "n_docs": mem_docs_count,
        "baseline_rss_mb": round(base_rss, 2),
        "python_peak_rss_mb": round(py_rss, 2),
        "rust_peak_rss_mb": round(rs_rss, 2),
        "python_net_mb": round(py_net, 2),
        "rust_net_mb": round(rs_net, 2),
        "ram_reduction_pct": round(ram_reduction_pct, 1),
        "total_rss_reduction_pct": round(total_rss_reduction_pct, 1),
    }
    print(
        f"[Memory @ {mem_docs_count} docs] Python net: {py_net:.1f} MiB | Rust net: {rs_net:.1f} MiB | "
        f"Reduction: -{ram_reduction_pct:.1f}%"
    )

    results = {
        "cpu_threads": detected_cores,
        "scaling": scaling_results,
        "thread_scaling": thread_scaling,
        "memory": memory_summary,
        "corpus_stats": last_stats_summary,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    render_execution_time_svg(scaling_results, ASSETS_DIR / "execution_time_curve.svg")
    render_thread_scaling_svg(thread_scaling, ASSETS_DIR / "thread_scaling_curve.svg")
    render_memory_svg(memory_summary, ASSETS_DIR / "memory_comparison.svg")
    print(f"Saved results to {RESULTS_PATH} and SVGs to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
