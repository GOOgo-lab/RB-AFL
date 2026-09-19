from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import PreparedTripletDataset, file_sha256, identities_for_split
from .protocol import ABLATIONS, AblationSpec


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    return requested


class GeometryEncoder(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 256) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embedding_dim = int(embedding_dim)
        layers: List[nn.Module] = []
        widths = (32, 64, 128, 192, 256)
        current = self.in_channels
        for width in widths:
            layers.extend(
                [
                    nn.Conv2d(current, width, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(num_groups=min(8, width), num_channels=width),
                    nn.SiLU(inplace=False),
                ]
            )
            current = width
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)
        self.projection = nn.Linear(widths[-1], self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        z = self.projection(h)
        return F.normalize(z, p=2, dim=1)


class TrainingModel(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = GeometryEncoder(in_channels, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    embedding_dim: int = 256
    triplet_margin: float = 0.8
    validation_per_identity: int = 1
    num_workers: int = 0
    seed: int = 20260730
    device: str = "auto"
    preload_tensors_to_ram: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4


def _one_epoch(
    model: TrainingModel,
    loader: DataLoader,
    device: str,
    spec: AblationSpec,
    margin: float,
    optimizer: Optional[torch.optim.Optimizer],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "classification": 0.0, "consistency": 0.0, "triplet": 0.0}
    correct = 0
    seen = 0
    batches = 0
    for batch in loader:
        non_blocking = device == "cuda"
        anchor = batch["anchor"].to(device, non_blocking=non_blocking).float()
        positive = batch["positive"].to(device, non_blocking=non_blocking).float()
        negative = batch["negative"].to(device, non_blocking=non_blocking).float()
        labels = batch["class_id"].to(device, non_blocking=non_blocking)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            z_anchor = model.encoder(anchor)
            z_positive = model.encoder(positive)
            z_negative = model.encoder(negative)
            logits_anchor = model.classifier(z_anchor)
            logits_positive = model.classifier(z_positive)
            classification = 0.5 * (
                F.cross_entropy(logits_anchor, labels) + F.cross_entropy(logits_positive, labels)
            )
            consistency = (1.0 - F.cosine_similarity(z_anchor, z_positive, dim=1)).mean()
            triplet = F.triplet_margin_loss(
                z_anchor,
                z_positive,
                z_negative,
                margin=float(margin),
                p=2,
            )
            total = (
                classification
                + float(spec.lambda_consistency) * consistency
                + float(spec.lambda_triplet) * triplet
            )
            if training:
                total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        sums["total"] += float(total.detach().cpu())
        sums["classification"] += float(classification.detach().cpu())
        sums["consistency"] += float(consistency.detach().cpu())
        sums["triplet"] += float(triplet.detach().cpu())
        correct += int((logits_anchor.argmax(dim=1) == labels).sum().detach().cpu())
        seen += int(labels.numel())
        batches += 1
    denominator = max(1, batches)
    return {
        "loss": sums["total"] / denominator,
        "loss_classification": sums["classification"] / denominator,
        "loss_consistency": sums["consistency"] / denominator,
        "loss_triplet": sums["triplet"] / denominator,
        "accuracy": correct / max(1, seen),
    }


def train_ablation(
    prepared_root: str | Path,
    output_root: str | Path,
    spec: AblationSpec,
    config: TrainingConfig,
    identity_split_path: str | Path | None = None,
    study_train_split: str = "train",
    reuse_existing: bool = True,
) -> Path:
    set_global_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(output_root).expanduser().resolve() / f"{spec.exp_id}_{spec.exp_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "best.pt"
    allowed_identities = (
        identities_for_split(identity_split_path, study_train_split)
        if identity_split_path is not None
        else None
    )
    split_sha256 = file_sha256(identity_split_path) if identity_split_path is not None else ""
    prepared_manifest_path = Path(prepared_root).expanduser().resolve() / "manifest.csv"
    prepared_manifest_sha256 = file_sha256(prepared_manifest_path)
    expected_training_config = asdict(config)

    if reuse_existing and checkpoint_path.is_file():
        try:
            existing = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            old_seed = int(existing.get("training_config", {}).get("seed", -1))
            old_split = str(existing.get("study_split", {}).get("identity_split_sha256", ""))
            old_exp = str(existing.get("experiment", {}).get("exp_id", ""))
            old_prepared = str(existing.get("study_split", {}).get("prepared_manifest_sha256", ""))
            old_train_split = str(existing.get("study_split", {}).get("study_train_split", ""))
            old_classes = sorted(str(value) for value in existing.get("class_names", []))
            expected_classes = sorted(str(value) for value in (allowed_identities or old_classes))
            if (
                old_seed == int(config.seed)
                and existing.get("training_config") == expected_training_config
                and old_split == split_sha256
                and old_prepared == prepared_manifest_sha256
                and old_train_split == study_train_split
                and old_classes == expected_classes
                and old_exp == spec.exp_id
                and existing.get("format") == "rbafl_geometry_encoder_v1.1.0"
            ):
                print(f"[{spec.exp_id}] reuse existing checkpoint: {checkpoint_path}", flush=True)
                return checkpoint_path
        except Exception:
            pass

    shared_tensor_cache: Dict[str, np.ndarray] = {}
    train_ds = PreparedTripletDataset(
        prepared_root,
        spec.channels,
        "train",
        validation_per_identity=config.validation_per_identity,
        seed=config.seed,
        allowed_identities=allowed_identities,
        tensor_cache=shared_tensor_cache,
        preload_to_ram=bool(config.preload_tensors_to_ram),
    )
    val_ds = PreparedTripletDataset(
        prepared_root,
        spec.channels,
        "val",
        validation_per_identity=config.validation_per_identity,
        seed=config.seed,
        allowed_identities=allowed_identities,
        tensor_cache=shared_tensor_cache,
        preload_to_ram=bool(config.preload_tensors_to_ram),
    )
    if config.preload_tensors_to_ram:
        gib = sum(int(x.nbytes) for x in shared_tensor_cache.values()) / (1024.0**3)
        print(f"[{spec.exp_id}] RAM tensor cache: {len(shared_tensor_cache)} tensors, {gib:.2f} GiB", flush=True)
    generator = torch.Generator().manual_seed(config.seed)
    pin_memory = bool(config.pin_memory and device == "cuda")
    loader_common = {
        "num_workers": int(config.num_workers),
        "pin_memory": pin_memory,
    }
    if int(config.num_workers) > 0:
        loader_common.update(
            persistent_workers=bool(config.persistent_workers),
            prefetch_factor=max(2, int(config.prefetch_factor)),
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )
    model = TrainingModel(
        in_channels=len(spec.channels),
        embedding_dim=config.embedding_dim,
        num_classes=len(train_ds.class_names),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, object]] = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    start_all = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_stats = _one_epoch(
            model, train_loader, device, spec, config.triplet_margin, optimizer
        )
        val_stats = _one_epoch(model, val_loader, device, spec, config.triplet_margin, None)
        record = {
            "epoch": epoch,
            "elapsed_s": time.perf_counter() - start,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(record)
        print(
            f"[{spec.exp_id}] epoch={epoch:03d}/{config.epochs} "
            f"train={train_stats['loss']:.5f} val={val_stats['loss']:.5f} "
            f"val_acc={val_stats['accuracy']:.3f}",
            flush=True,
        )
        if val_stats["loss"] < best_val:
            best_val = float(val_stats["loss"])
            best_epoch = epoch
            best_state = copy.deepcopy(model.encoder.state_dict())

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    checkpoint = {
        "format": "rbafl_geometry_encoder_v1.1.0",
        "experiment": spec.to_dict(),
        "training_config": asdict(config),
        "device_used": device,
        "in_channels": len(spec.channels),
        "embedding_dim": config.embedding_dim,
        "class_names": train_ds.class_names,
        "study_split": {
            "identity_split_file": Path(identity_split_path).name
            if identity_split_path is not None
            else "",
            "identity_split_sha256": split_sha256,
            "prepared_manifest_sha256": prepared_manifest_sha256,
            "study_train_split": study_train_split if identity_split_path is not None else "",
            "training_identity_count": len(train_ds.class_names),
            "internal_validation_per_identity": int(config.validation_per_identity),
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "encoder": best_state,
    }
    torch.save(checkpoint, checkpoint_path)
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "exp_id": spec.exp_id,
        "exp_name": spec.exp_name,
        "channels": list(spec.channels),
        "lambda_consistency": spec.lambda_consistency,
        "lambda_triplet": spec.lambda_triplet,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "training_time_s": time.perf_counter() - start_all,
        "checkpoint": checkpoint_path.name,
        "parameter_count": sum(p.numel() for p in model.encoder.parameters()),
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checkpoint_path


def train_all_ablations(
    prepared_root: str | Path,
    output_root: str | Path,
    config: TrainingConfig,
    selected: Optional[Sequence[str]] = None,
    identity_split_path: str | Path | None = None,
    study_train_split: str = "train",
    reuse_existing: bool = True,
) -> List[Path]:
    # Normalize selection defensively. Older launchers/users may accidentally pass
    # [None], empty strings, or mixed values. Only non-empty strings are valid.
    if selected:
        invalid = [x for x in selected if x is not None and not isinstance(x, str)]
        if invalid:
            raise TypeError(
                "selected must contain experiment names/IDs as strings, "
                f"got invalid values: {invalid!r}"
            )
        keys = {
            x.strip().lower()
            for x in selected
            if isinstance(x, str) and x.strip()
        }
        keys = keys or None
    else:
        keys = None
    checkpoints: List[Path] = []
    for spec in ABLATIONS:
        if keys and spec.exp_id.lower() not in keys and spec.exp_name.lower() not in keys:
            continue
        checkpoints.append(
            train_ablation(
                prepared_root,
                output_root,
                spec,
                config,
                identity_split_path=identity_split_path,
                study_train_split=study_train_split,
                reuse_existing=reuse_existing,
            )
        )
    if not checkpoints:
        raise RuntimeError("No ablation experiment was selected")
    return checkpoints


def load_encoder(checkpoint_path: str | Path, device: str = "auto"):
    resolved = resolve_device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=resolved, weights_only=False)
    if checkpoint.get("format") != "rbafl_geometry_encoder_v1.1.0":
        raise ValueError("Unsupported checkpoint format; retrain with this public pipeline")
    encoder = GeometryEncoder(
        in_channels=int(checkpoint["in_channels"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
    ).to(resolved)
    encoder.load_state_dict(checkpoint["encoder"], strict=True)
    encoder.eval()
    return encoder, checkpoint, resolved


def extract_embedding(
    encoder: GeometryEncoder,
    tensor: np.ndarray,
    channel_indices: Sequence[int],
    device: str,
) -> np.ndarray:
    selected = np.ascontiguousarray(np.asarray(tensor, dtype=np.float32)[list(channel_indices)])
    batch = torch.from_numpy(selected[None]).to(device)
    with torch.inference_mode():
        feature = encoder(batch)[0].detach().cpu().numpy().astype(np.float32)
    return feature


def feature_to_bits(feature: np.ndarray, bit_length: int, threshold_mode: str = "median") -> np.ndarray:
    values = np.asarray(feature, dtype=np.float32).reshape(-1)
    if values.size != bit_length or bit_length <= 0:
        raise ValueError("Feature dimension must equal watermark length; interpolation is not allowed")
    if not np.isfinite(values).all():
        raise ValueError("Feature vector must be finite")
    if threshold_mode == "balanced_topk":
        if bit_length % 2:
            raise ValueError("balanced_topk requires an even bit_length")
        # Stable ranking yields exactly bit_length/2 one bits even when feature
        # values contain ties.  This fixed-Hamming-weight rule prevents all-zero,
        # near-zero, or all-one feature codes from biasing cosine NC.
        order = np.argsort(values, kind="stable")
        bits = np.zeros(bit_length, dtype=np.uint8)
        bits[order[bit_length // 2 :]] = 1
        return bits
    if threshold_mode == "median":
        threshold = float(np.median(values))
    elif threshold_mode == "mean":
        threshold = float(np.mean(values))
    elif threshold_mode == "zero":
        threshold = 0.0
    else:
        raise ValueError("threshold_mode must be balanced_topk, median, mean or zero")
    return (values >= threshold).astype(np.uint8)


def encoder_parameter_bytes(encoder: nn.Module) -> int:
    return int(sum(p.nelement() * p.element_size() for p in encoder.parameters()))


def checkpoint_index(model_root: str | Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    root = Path(model_root).expanduser().resolve()
    for path in sorted(root.glob("E*_*/best.pt")):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        exp = ckpt["experiment"]
        rows.append(
            {
                "exp_id": exp["exp_id"],
                "exp_name": exp["exp_name"],
                "checkpoint": str(path),
                "channels": ",".join(exp["channels"]),
            }
        )
    if not rows:
        raise RuntimeError(f"No compatible RB-AFL checkpoints found below {root}")
    return pd.DataFrame(rows).sort_values("exp_id").reset_index(drop=True)
