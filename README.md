# Faster-Protenix

Infrastructure acceleration experiments for [Protenix](https://github.com/bytedance/Protenix) training, inspired by [ResearchBoost-Agent](https://github.com/AI4AI-for-Anything/ResearchBoost-Agent). This repository records the optimized implementation, bounded experiment metadata, and the observed 128-update reference–candidate–reference comparison.

**Status: external exploratory case; the frozen numerical judge rejected it.** The candidate completed the fixed workload and passed every training-trajectory comparison, but 3 of 138 evaluation comparisons exceeded the frozen tolerance for one EMA pLDDT rank-1 metric. One of those failures was the reference-versus-reference control. The result is useful performance evidence, not a certified equivalence or convergence claim.

## Observed result

The experiment ran sequentially on one host with **8 NVIDIA H20 GPUs**. All arms used the same 128 optimizer updates, crop size 384, diffusion batch 48, maximum recycle count 4, effective batch 8, BF16 outer model with the original FP32 diffusion exclusions, Adam, seeds, inputs, learning-rate schedule, and evaluation workload.

| Measurement | Reference 1 | Candidate | Reference 2 | Ratio |
| --- | ---: | ---: | ---: | ---: |
| Whole process, seconds | 1517.671 | 968.829 | 1490.307 | **1.55×** |
| Mean max-rank step time, steps 16–127, seconds | 10.310 | 6.071 | 10.168 | **1.69×** |

The ratios use the mean of the two reference measurements divided by the candidate. Whole-process timing includes startup, cold native compilation, training, checkpointing, and evaluation. The steady measurement is narrower and excludes the first 16 updates.

The candidate used 87.92 GB peak allocated and 100.26 GB peak reserved memory per rank. All six final normal/EMA checkpoints were finite and passed the saved build/checkpoint audit. The training trajectory passed 216 of 216 frozen comparisons. Evaluation passed 135 of 138 comparisons.

The three failed comparisons all concern `ema0.999_lddt/complex/plddt.rank1.avg`:

- reference 1 versus candidate: delta 0.001859, limit 0.001425;
- reference 2 versus candidate: delta 0.001309, limit 0.001267;
- reference 1 versus reference 2: delta 0.003168, limit 0.001425.

Because the frozen judge has no waiver for a reference-repeat failure, `accepted=false` remains unchanged. The 1.55× ratio should not be read as a ResearchBoost certified result or as evidence of full-corpus convergence.

## What changed

The combined candidate retains selected Pairformer, MSA, and diffusion blocks instead of recomputing them; caches repeated conditioning and sampling projections; uses a guarded pair-layout transpose path; fuses guarded attention softmax and head-gate backward operations; uses native CUDA paths for selected layer-normalization/affine work; parallelizes cold native builds and model initialization; and moves discrete symmetric-residue metadata preparation to integer CPU operations. Precision, optimizer, effective batch, inputs, seeds, and workload stay fixed.

The implementation is under [`optimization/`](optimization/). The final switch composition is recorded in [`optimization/performance_policies.json`](optimization/performance_policies.json). The full-run adapter is [`optimization/training_entry.py`](optimization/training_entry.py), layered over [`optimization/native_entry.py`](optimization/native_entry.py). These files preserve the experiment's guards and receipts rather than presenting the changes as a generic drop-in patch.

## Evidence

- [`results/summary.json`](results/summary.json): compact, publication-facing result and limitation record.
- [`results/FINDINGS.json`](results/FINDINGS.json): complete reviewed full-run findings.
- [`results/PREPARATION.json`](results/PREPARATION.json): frozen scope and threshold.
- [`results/CLOSURE.json`](results/CLOSURE.json): clean controller/GPU closure and archive binding.
- [`results/AUDIT_PIPELINE.json`](results/AUDIT_PIPELINE.json): checkpoint and native-build audit pipeline.

Validate the publication-facing arithmetic locally:

```bash
python3 scripts/check_evidence.py
```

This check verifies the reported ratios, pass counts, failed metric, and selected values against `results/FINDINGS.json`. It does not rerun training.

## Reproduction scope

The source snapshot used Protenix 2.0.0 and is identified by [`SOURCE_MANIFEST.sha256`](SOURCE_MANIFEST.sha256); an upstream Git commit was not recorded, so this repository does not invent one. The exact 128-update fixture and training data are not redistributed. Re-running the experiment therefore requires a compatible Protenix training dataset and adapting the absolute asset/data paths in the bounded launcher to the local environment.

The included implementation and records are intended for audit and porting. They are not a one-command public reproduction of the original infrastructure run. Some infrastructure adapters retain historical absolute paths from the recorded experiment; replace those paths before reuse. No model weights, generated checkpoints, private data, credentials, or access tokens are published.

## License

Protenix and the derived implementation files are provided under the included Apache-2.0 license. This is an independent external contribution, not an official ByteDance or ResearchBoost-Agent release.
