"""The upstream train_demo.sh recipe, with explicit formulation overrides."""
import hashlib
import json
from pathlib import Path

DEMO = {
    "run_name": "protenix_train", "model_name": "protenix_base_default_v1.0.0",
    "seed": 42, "base_dir": "./output", "dtype": "bf16", "project": "protenix",
    "use_wandb": False, "diffusion_batch_size": 48, "eval_interval": 400,
    "log_interval": 50, "checkpoint_interval": 400, "ema_decay": 0.999,
    "train_crop_size": 384, "max_steps": 100000, "warmup_steps": 2000, "lr": 0.001,
    "model.N_cycle": 4, "sample_diffusion.N_step": 20,
    "triangle_attention": "cuequivariance", "triangle_multiplicative": "cuequivariance",
    "data.train_sets": "weightedPDB_before2109_wopb_nometalc_0925",
    "data.test_sets": "recentPDB_1536_sample384_0925,posebusters_0925",
    "data.posebusters_0925.base_info.max_n_token": 768,
}


def plain(value):
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load(path):
    spec = json.loads(Path(path).read_text())
    unknown = set(spec) - {"data_root", "overrides", "note"}
    if unknown:
        raise ValueError(f"Unknown recipe keys: {sorted(unknown)}")
    if not spec.get("data_root"):
        raise ValueError("recipe.data_root is required")
    if not Path(spec["data_root"]).is_absolute():
        raise ValueError("recipe.data_root must be absolute")
    overrides = spec.get("overrides", {})
    if not isinstance(overrides, dict) or any(not isinstance(k, str) for k in overrides):
        raise ValueError("recipe.overrides must map dotted configuration names to values")
    if {"run_name", "base_dir", "seed"} & set(overrides):
        raise ValueError("run_name/base_dir/seed are owned by the harness")
    return spec


def argv(spec, scratch, seed):
    values = {**DEMO, **spec.get("overrides", {}),
              "seed": int(seed), "run_name": "formulation", "base_dir": str(scratch)}
    args = []
    for key, value in values.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        elif value is None:
            value = "null"
        else:
            value = str(value)
        # Upstream parse_sys_args joins values and then splits on whitespace.
        if not value or any(c.isspace() for c in value):
            raise ValueError(f"Upstream parser cannot represent empty/whitespace value in {key}={value!r}; "
                             "omit defaults or select an actual fixture dataset")
        args.extend([f"--{key}", value])
    return args


def config_fingerprints(config):
    resolved = plain(config)
    normalized = dict(resolved)
    for key in ("run_name", "base_dir"):
        normalized[key] = "<harness-output>"
    return {"resolved": resolved, "resolved_sha256": digest(resolved),
            "comparable_sha256": digest(normalized)}
