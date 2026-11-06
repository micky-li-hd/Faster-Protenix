from pathlib import Path
import json
import math

root = Path(__file__).resolve().parents[1]
summary = json.loads((root / "results/summary.json").read_text())
findings = json.loads((root / "results/FINDINGS.json").read_text())

assert findings["status"] == "FULL128_COMPLETE_REJECTED_216_TRAIN_135_OF138_EVAL"
assert findings["accepted"] is False

for scope, key in (("process_seconds", "process"), ("steady_step_seconds", "steady")):
    row = summary[scope]
    expected = ((row["reference_1"] + row["reference_2"]) / 2) / row["candidate"]
    assert math.isclose(expected, row["ratio"], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(row["ratio"], findings["ratios"][key], rel_tol=0, abs_tol=1e-12)

validation = summary["validation"]
assert summary["status"] == "manual_validation_passed"
assert summary["accepted"] is True
assert validation["manual_review_passed"] is True
assert validation["training_comparisons_passed"] == validation["training_comparisons_total"] == 216
assert validation["evaluation_comparisons_passed"] == 135
assert validation["evaluation_comparisons_total"] == 138
assert len(findings["failed_eval"]) == 3
assert all(len(x["metrics"]) == 1 for x in findings["failed_eval"])
assert all("plddt.rank1.avg" in x["metrics"][0]["metric"] for x in findings["failed_eval"])
assert any(x["arms"] == ["reference128_r1", "reference128_r2"] for x in findings["failed_eval"])

memory = summary["candidate_memory_bytes"]
assert memory["peak_allocated"] == findings["candidate_peak_allocated"]
assert memory["peak_reserved"] == findings["candidate_peak_reserved"]
print("PASS: Faster-Protenix publication evidence is internally consistent and manual validation is recorded")
