# Omem Semantic Retrieval Benchmarks

Measured on:

- CPU: `Intel(R) Core(TM) i7-3632QM CPU @ 2.20GHz`
- OS/Arch: `linux/amd64`
- Command style: `go test -run '^$' -bench ... -benchmem -benchtime=1x`
- Date: `2026-03-09`

## Semantic Search

Direct `FactStore.SemanticSearch` benchmarks using the current Omem implementation.

| Mode | Dataset | Config | Time/op | Memory/op | Allocs/op |
|------|---------|--------|---------|-----------|-----------|
| Brute-force | 1,000 facts | exact scan | 110.59 ms | 8.22 MB | 30,748 |
| Brute-force | 5,000 facts | exact scan | 4094.57 ms | 42.43 MB | 154,836 |
| ANN | 5,000 facts | `nlist=32`, `nprobe=4`, `pq=16x4` | 6138.71 ms | 66.39 MB | 38,300 |
| ANN | 5,000 facts | `nlist=64`, `nprobe=6`, `pq=24x4` | 4264.32 ms | 66.50 MB | 39,855 |

## ANN Tuning

Initial tuning sweep on the current synthetic 10k-fact benchmark:

| Sweep | Result |
|-------|--------|
| `nprobe=2` | 434.60 ms/op, 129.86 MB/op, 73,216 allocs/op |

Additional `nprobe` points are benchmark-ready in `internal/context/memory/omem/semantic_search_benchmark_test.go` and can be run individually.

## Recall Validation

ANN quality is validated against brute-force retrieval on the same synthetic dataset. The current gating metric is top-1 agreement, with overlap metrics exported for inspection.

Source snapshot: `internal/context/memory/omem/testdata/benchmarks/semantic_search_ann_recall.json`

| Metric | Result |
|--------|--------|
| Facts | 10,000 |
| Top-K | 10 |
| ANN Recall@K | 1.00 |
| Average Overlap vs Brute-force | 1.00 |
| Top-1 Accuracy vs Brute-force | 1.00 |
| Config | `nlist=64`, `nprobe=8`, `pq_subvectors=24`, `pq_bits=4` |

## Current Takeaways

- The current ANN path is correct and exact-reranked.
- Quality validation remains strong in the current synthetic benchmark, with `1.00` top-1 agreement against brute-force at 10k facts.
- Latency results are still noisy across runs on this host, so ANN speedups should be treated as provisional until more sweeps are collected.
- ANN reduces allocation count substantially versus brute-force in the 5k benchmark, but still uses more bytes/op in this shape.

## Commands Used

```bash
go test -run '^$' -bench 'BenchmarkSemanticSearchBruteForce' ./internal/context/memory/omem -benchmem -benchtime=1x
go test -run '^$' -bench '^BenchmarkSemanticSearchANN/facts=5000/nlist=32_nprobe=4_pq=16x4$' ./internal/context/memory/omem -benchmem -benchtime=1x
go test -run '^$' -bench '^BenchmarkSemanticSearchANN/facts=5000/nlist=64_nprobe=6_pq=24x4$' ./internal/context/memory/omem -benchmem -benchtime=1x
go test -run '^$' -bench '^BenchmarkSemanticSearchANNTuning/nprobe=2$' ./internal/context/memory/omem -benchmem -benchtime=1x
go test -run 'TestSemanticSearchANNRecallAgainstBruteForce' ./internal/context/memory/omem
```
