//! High-performance LLM dataset cleaning and MinHash + LSH deduplication
//! implemented in Rust with PyO3 bindings and Rayon data parallelism.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

/// Mersenne prime M31 (2^31 - 1) used for fast universal MinHash permutations.
const MERSENNE_PRIME: u64 = 2_147_483_647;

/// Default n-gram shingle size in bytes.
const SHINGLE_BYTES: usize = 5;

/// Deterministic LCG seed constants for generating universal hash coefficients.
const SEED_A_BASE: u64 = 6_364_136_223_846_793_005;
const SEED_B_BASE: u64 = 1_442_695_040_888_963_407;

/// Statistics returned after running the full LLM preprocessing and deduplication pipeline.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Total number of raw input documents.
    #[pyo3(get)]
    pub total_docs: usize,
    /// Number of documents discarded by length or alphabetic quality filtering.
    #[pyo3(get)]
    pub filtered_quality: usize,
    /// Number of exact duplicates removed via 64-bit FxHash.
    #[pyo3(get)]
    pub exact_duplicates_removed: usize,
    /// Number of near-duplicates removed via MinHash + LSH.
    #[pyo3(get)]
    pub fuzzy_duplicates_removed: usize,
    /// Final number of unique, cleaned documents retained.
    #[pyo3(get)]
    pub kept_docs: usize,
    /// Percentage of the original corpus retained (0.0 - 100.0).
    #[pyo3(get)]
    pub retention_pct: f64,
    /// Total pipeline execution time inside Rust in milliseconds.
    #[pyo3(get)]
    pub elapsed_ms: f64,
    /// Processing throughput in input documents per second.
    #[pyo3(get)]
    pub throughput_docs_per_sec: f64,
}

#[pymethods]
impl PipelineStats {
    fn __repr__(&self) -> String {
        format!(
            "PipelineStats(total={}, kept={}, retention={:.1}%, exact_removed={}, fuzzy_removed={}, elapsed_ms={:.2}, throughput={:.0} docs/s)",
            self.total_docs,
            self.kept_docs,
            self.retention_pct,
            self.exact_duplicates_removed,
            self.fuzzy_duplicates_removed,
            self.elapsed_ms,
            self.throughput_docs_per_sec
        )
    }
}

/// Clean a single document: strip HTML tags (`<...>`), lowercase ASCII/Unicode,
/// and collapse consecutive whitespace into single spaces.
pub fn clean_text_core(input: &str) -> String {
    let mut stripped = String::with_capacity(input.len());
    let mut rest = input;

    while let Some(open_idx) = rest.find('<') {
        stripped.push_str(&rest[..open_idx]);
        let after_open = &rest[open_idx + 1..];
        if let Some(close_idx) = after_open.find('>') {
            if close_idx > 0 {
                stripped.push(' ');
                rest = &after_open[close_idx + 1..];
                continue;
            }
        }
        stripped.push('<');
        rest = after_open;
    }
    stripped.push_str(rest);

    let mut normalized = String::with_capacity(stripped.len());
    let mut prev_space = true;

    for ch in stripped.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                normalized.push(' ');
                prev_space = true;
            }
        } else {
            for lower in ch.to_lowercase() {
                normalized.push(lower);
            }
            prev_space = false;
        }
    }

    if normalized.ends_with(' ') {
        normalized.pop();
    }

    normalized
}

/// Check whether a cleaned document passes minimum length and alphabetic ratio criteria.
#[inline]
fn passes_quality_filter(text: &str, min_chars: usize, min_alpha_ratio: f64) -> bool {
    let mut total_chars = 0usize;
    let mut alpha_chars = 0usize;

    for ch in text.chars() {
        total_chars += 1;
        if ch.is_alphabetic() {
            alpha_chars += 1;
        }
    }

    if total_chars < min_chars || total_chars == 0 {
        return false;
    }

    (alpha_chars as f64 / total_chars as f64) >= min_alpha_ratio
}

/// Compute a fast 64-bit FxHash for a string slice.
#[inline]
fn fx_hash_str(s: &str) -> u64 {
    let mut hasher = FxHasher::default();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Compute a fast 64-bit FxHash for a byte slice (shingle).
#[inline]
fn fx_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// Generate deterministic universal hash pairs `(a, b)` for `num_perm` MinHash permutations.
fn generate_permutations(num_perm: usize) -> Vec<(u64, u64)> {
    let mut perms = Vec::with_capacity(num_perm);
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;

    for _ in 0..num_perm {
        state = state.wrapping_mul(SEED_A_BASE).wrapping_add(SEED_B_BASE);
        let a = ((state >> 33) % (MERSENNE_PRIME - 1)) + 1;
        state = state.wrapping_mul(SEED_A_BASE).wrapping_add(SEED_B_BASE);
        let b = (state >> 33) % MERSENNE_PRIME;
        perms.push((a, b));
    }

    perms
}

/// Compute the MinHash signature of a single document into `out_sig`.
fn compute_minhash_signature(text: &str, perms: &[(u64, u64)], out_sig: &mut [u32]) {
    out_sig.fill(u32::MAX);
    let bytes = text.as_bytes();

    if bytes.len() < SHINGLE_BYTES {
        let h = fx_hash_bytes(bytes) % MERSENNE_PRIME;
        for (slot, &(a, b)) in out_sig.iter_mut().zip(perms.iter()) {
            let val = ((a * h + b) % MERSENNE_PRIME) as u32;
            if val < *slot {
                *slot = val;
            }
        }
        return;
    }

    for window in bytes.windows(SHINGLE_BYTES) {
        let h = fx_hash_bytes(window) % MERSENNE_PRIME;
        for (slot, &(a, b)) in out_sig.iter_mut().zip(perms.iter()) {
            let val = ((a * h + b) % MERSENNE_PRIME) as u32;
            if val < *slot {
                *slot = val;
            }
        }
    }
}

/// Compute the exact MinHash Jaccard similarity estimate between two equal-length signatures.
#[inline]
fn signature_similarity(sig_a: &[u32], sig_b: &[u32]) -> f64 {
    let matches = sig_a
        .iter()
        .zip(sig_b.iter())
        .filter(|(&x, &y)| x == y)
        .count();
    matches as f64 / sig_a.len() as f64
}

/// Hash a signature band slice into a 64-bit bucket identifier.
#[inline]
fn hash_band(band_slice: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    band_slice.hash(&mut hasher);
    hasher.finish()
}

/// Run exact deduplication in parallel (hashes computed on Rayon threads, first occurrence kept).
pub fn deduplicate_exact_core(docs: &[String]) -> Vec<String> {
    let hashes: Vec<u64> = docs.par_iter().map(|d| fx_hash_str(d)).collect();
    let mut seen = FxHashSet::with_capacity_and_hasher(docs.len(), Default::default());
    let mut unique = Vec::with_capacity(docs.len());

    for (doc, hash) in docs.iter().zip(hashes) {
        if seen.insert(hash) {
            unique.push(doc.clone());
        }
    }

    unique
}

/// Run MinHash + LSH fuzzy deduplication across `docs` and return the kept indices.
fn find_fuzzy_kept_indices(
    docs: &[String],
    threshold: f64,
    num_perm: usize,
    bands: usize,
) -> Vec<usize> {
    let n = docs.len();
    if n <= 1 {
        return (0..n).collect();
    }

    let rows = num_perm / bands;
    let perms = generate_permutations(num_perm);

    // Single flat allocation for all signatures improves cache locality and reduces heap pressure.
    let mut signatures = vec![u32::MAX; n * num_perm];
    signatures
        .par_chunks_mut(num_perm)
        .zip(docs.par_iter())
        .for_each(|(sig_slice, doc)| {
            compute_minhash_signature(doc, &perms, sig_slice);
        });

    // Build LSH band tables in parallel (one FxHashMap per band).
    let band_tables: Vec<FxHashMap<u64, Vec<usize>>> = (0..bands)
        .into_par_iter()
        .map(|b| {
            let mut map: FxHashMap<u64, Vec<usize>> =
                FxHashMap::with_capacity_and_hasher(n / 2, Default::default());
            let start = b * rows;
            let end = start + rows;
            for i in 0..n {
                let sig = &signatures[i * num_perm..(i + 1) * num_perm];
                let bucket_key = hash_band(&sig[start..end]);
                map.entry(bucket_key).or_default().push(i);
            }
            map
        })
        .collect();

    let mut removed = vec![false; n];
    let mut kept = Vec::with_capacity(n);
    let mut candidates = Vec::with_capacity(64);

    for i in 0..n {
        if removed[i] {
            continue;
        }
        kept.push(i);

        candidates.clear();
        let sig_i = &signatures[i * num_perm..(i + 1) * num_perm];

        for (b, table) in band_tables.iter().enumerate() {
            let start = b * rows;
            let bucket_key = hash_band(&sig_i[start..start + rows]);
            if let Some(bucket) = table.get(&bucket_key) {
                for &j in bucket {
                    if j > i && !removed[j] {
                        candidates.push(j);
                    }
                }
            }
        }

        if candidates.is_empty() {
            continue;
        }

        candidates.sort_unstable();
        candidates.dedup();

        for &j in &candidates {
            let sig_j = &signatures[j * num_perm..(j + 1) * num_perm];
            if signature_similarity(sig_i, sig_j) >= threshold {
                removed[j] = true;
            }
        }
    }

    kept
}

/// Core end-to-end LLM dataset cleaning, quality filtering, exact dedup, and MinHash LSH dedup.
pub fn run_pipeline_core(
    docs: &[String],
    min_chars: usize,
    min_alpha_ratio: f64,
    threshold: f64,
    num_perm: usize,
    bands: usize,
) -> (Vec<String>, PipelineStats) {
    let start = Instant::now();
    let total_docs = docs.len();

    // Step 1: Parallel cleaning + quality filtering + exact hashing
    let cleaned_and_hashed: Vec<Option<(String, u64)>> = docs
        .par_iter()
        .map(|raw| {
            let cleaned = clean_text_core(raw);
            if passes_quality_filter(&cleaned, min_chars, min_alpha_ratio) {
                let hash = fx_hash_str(&cleaned);
                Some((cleaned, hash))
            } else {
                None
            }
        })
        .collect();

    // Step 2: Exact deduplication preserving first occurrence
    let mut seen_hashes = FxHashSet::with_capacity_and_hasher(total_docs, Default::default());
    let mut after_exact = Vec::with_capacity(total_docs);
    let mut passed_quality_count = 0usize;

    for item in cleaned_and_hashed.into_iter().flatten() {
        passed_quality_count += 1;
        if seen_hashes.insert(item.1) {
            after_exact.push(item.0);
        }
    }

    let filtered_quality = total_docs.saturating_sub(passed_quality_count);
    let exact_duplicates_removed = passed_quality_count.saturating_sub(after_exact.len());

    // Step 3: Parallel MinHash + LSH fuzzy deduplication
    let kept_indices = find_fuzzy_kept_indices(&after_exact, threshold, num_perm, bands);
    let fuzzy_duplicates_removed = after_exact.len().saturating_sub(kept_indices.len());

    let final_docs: Vec<String> = kept_indices
        .into_iter()
        .map(|idx| std::mem::take(&mut after_exact[idx]))
        .collect();

    let kept_docs = final_docs.len();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let retention_pct = if total_docs == 0 {
        0.0
    } else {
        (kept_docs as f64 / total_docs as f64) * 100.0
    };
    let throughput_docs_per_sec = if elapsed_ms <= 0.0 {
        0.0
    } else {
        total_docs as f64 / (elapsed_ms / 1000.0)
    };

    let stats = PipelineStats {
        total_docs,
        filtered_quality,
        exact_duplicates_removed,
        fuzzy_duplicates_removed,
        kept_docs,
        retention_pct,
        elapsed_ms,
        throughput_docs_per_sec,
    };

    (final_docs, stats)
}

fn validate_pipeline_params(
    min_alpha_ratio: f64,
    threshold: f64,
    num_perm: usize,
    bands: usize,
) -> PyResult<()> {
    if !(0.0..=1.0).contains(&min_alpha_ratio) {
        return Err(PyValueError::new_err(
            "min_alpha_ratio must be between 0.0 and 1.0",
        ));
    }
    if !(0.0..=1.0).contains(&threshold) || threshold == 0.0 {
        return Err(PyValueError::new_err(
            "threshold must be in the interval (0.0, 1.0]",
        ));
    }
    if num_perm == 0 || bands == 0 || !num_perm.is_multiple_of(bands) {
        return Err(PyValueError::new_err(
            "num_perm and bands must be > 0 and num_perm must be divisible by bands",
        ));
    }
    Ok(())
}

/// Clean and normalize a list of raw documents in parallel using Rayon.
#[pyfunction]
fn clean_corpus(py: Python<'_>, docs: Vec<String>) -> Vec<String> {
    py.allow_threads(|| docs.par_iter().map(|d| clean_text_core(d)).collect())
}

/// Remove exact duplicates from a list of documents while preserving first-seen order.
#[pyfunction]
fn deduplicate_exact(py: Python<'_>, docs: Vec<String>) -> Vec<String> {
    py.allow_threads(|| deduplicate_exact_core(&docs))
}

/// Remove near-duplicate documents using parallel MinHash signatures and LSH banding.
#[pyfunction]
#[pyo3(signature = (docs, threshold = 0.80, num_perm = 128, bands = 16))]
fn deduplicate_minhash(
    py: Python<'_>,
    docs: Vec<String>,
    threshold: f64,
    num_perm: usize,
    bands: usize,
) -> PyResult<Vec<String>> {
    validate_pipeline_params(0.0, threshold, num_perm, bands)?;

    let kept = py.allow_threads(|| {
        let mut owned = docs;
        let indices = find_fuzzy_kept_indices(&owned, threshold, num_perm, bands);
        indices
            .into_iter()
            .map(|idx| std::mem::take(&mut owned[idx]))
            .collect()
    });

    Ok(kept)
}

/// Run the full LLM preprocessing pipeline (HTML cleaning, quality filter, exact dedup,
/// and MinHash LSH fuzzy dedup). Optionally pin Rayon's thread pool size for scaling studies.
#[pyfunction]
#[pyo3(signature = (docs, min_chars = 20, min_alpha_ratio = 0.50, threshold = 0.80, num_perm = 128, bands = 16, num_threads = None))]
#[expect(
    clippy::too_many_arguments,
    reason = "PyO3 Python API exposes optional keyword parameters directly"
)]
fn run_full_pipeline(
    py: Python<'_>,
    docs: Vec<String>,
    min_chars: usize,
    min_alpha_ratio: f64,
    threshold: f64,
    num_perm: usize,
    bands: usize,
    num_threads: Option<usize>,
) -> PyResult<(Vec<String>, PipelineStats)> {
    validate_pipeline_params(min_alpha_ratio, threshold, num_perm, bands)?;

    py.allow_threads(|| {
        if let Some(threads) = num_threads {
            if threads == 0 {
                return Err(PyValueError::new_err("num_threads must be >= 1"));
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| PyValueError::new_err(format!("Failed to build thread pool: {e}")))?;
            Ok(pool.install(|| {
                run_pipeline_core(
                    &docs,
                    min_chars,
                    min_alpha_ratio,
                    threshold,
                    num_perm,
                    bands,
                )
            }))
        } else {
            Ok(run_pipeline_core(
                &docs,
                min_chars,
                min_alpha_ratio,
                threshold,
                num_perm,
                bands,
            ))
        }
    })
}

/// Return the number of logical CPU cores visible to Rayon.
#[pyfunction]
fn cpu_count() -> usize {
    rayon::current_num_threads()
}

/// Python extension module `llm_dedup_rs._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PipelineStats>()?;
    m.add_function(wrap_pyfunction!(clean_corpus, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_exact, m)?)?;
    m.add_function(wrap_pyfunction!(deduplicate_minhash, m)?)?;
    m.add_function(wrap_pyfunction!(run_full_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_count, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_strips_html_and_normalizes_whitespace() {
        let raw = "  <p>Bonjour   <b>LLM</b>  Dataset!</p>\n\t  Test  ";
        assert_eq!(clean_text_core(raw), "bonjour llm dataset! test");
    }

    #[test]
    fn exact_dedup_preserves_first_occurrence_order() {
        let docs = vec![
            "alpha beta gamma".to_string(),
            "delta epsilon".to_string(),
            "alpha beta gamma".to_string(),
            "zeta".to_string(),
        ];
        let deduped = deduplicate_exact_core(&docs);
        assert_eq!(deduped, vec!["alpha beta gamma", "delta epsilon", "zeta"]);
    }

    #[test]
    fn minhash_lsh_detects_near_duplicate_documents() {
        let docs = vec![
            "Large language models require clean and deduplicated training corpora to avoid memorization and improve generalization across benchmarks.".to_string(),
            "Large language models require clean and deduplicated training corpora to avoid memorization and improve generalization across benchmarks!".to_string(),
            "Completely unrelated recipe about baking sourdough bread with organic flour, sea salt, and wild yeast starter.".to_string(),
        ];

        let kept = find_fuzzy_kept_indices(&docs, 0.75, 128, 16);
        assert_eq!(kept, vec![0, 2]);
    }

    #[test]
    fn full_pipeline_tracks_quality_exact_and_fuzzy_counts() {
        let docs = vec![
            "12345 !!! ???".to_string(), // filtered by quality
            "<p>Transformer architectures use self-attention mechanisms over tokenized sequences for language modeling tasks.</p>".to_string(),
            "Transformer architectures use self-attention mechanisms over tokenized sequences for language modeling tasks.".to_string(), // exact dup after clean
            "Transformer architectures use self-attention mechanisms over tokenized sequences for language modeling task!".to_string(), // near dup
            "Rust provides zero-cost abstractions and fearless concurrency without a garbage collector.".to_string(),
        ];

        let (kept, stats) = run_pipeline_core(&docs, 20, 0.50, 0.80, 128, 16);
        assert_eq!(stats.total_docs, 5);
        assert_eq!(stats.filtered_quality, 1);
        assert_eq!(stats.exact_duplicates_removed, 1);
        assert_eq!(stats.fuzzy_duplicates_removed, 1);
        assert_eq!(stats.kept_docs, 2);
        assert_eq!(kept.len(), 2);
    }
}
