from __future__ import annotations

import json
import hashlib
import math
import os
import pickle
try:
    import resource
except ImportError:  # Windows
    resource = None
import shutil
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .data import base_manifest
from .evaluation import (
    _checkpoint_rows,
    _experiment_complete,
    _load_completed_experiment,
    _median_encode_time,
    _timed_call,
    _uniqueness_matrices,
)
from .model import (
    encoder_parameter_bytes,
    extract_embedding,
    feature_to_bits,
    load_encoder,
)
from .protocol import NC_THRESHOLD, attack_plan, protocol_dict
from .vector import (
    apply_attack, build_four_channels, prepare_coordinate_noise_index, read_vector,
    stable_seed, warmup_v13_fastpath,
)
from .watermark import (
    ber_score,
    bit_sha256,
    create_registry_record,
    file_sha256,
    nc_score,
    recover_watermark,
    save_registry,
    watermark_image_to_bits,
)

CACHE_VERSION = "rbafl-manuscript-1.1-equation-fields-fixed-attacks"
SOURCE_CACHE_VERSION = "rbafl-public-1.0.0-sanitized-geometry"

_PROGRESS_STAGE_CN = {
    "job.start": "任务启动",
    "load.start": "载入缓存几何",
    "load.done": "载入完成",
    "case.start": "攻击样本开始",
    "attack.prepare": "坐标扰动准备",
    "attack.extract_coordinates": "提取全部坐标",
    "attack.extract_coordinates_done": "坐标提取完成",
    "attack.generate_offsets": "生成随机扰动偏移",
    "attack.offsets_ready": "扰动坐标计算完成",
    "attack.set_coordinates": "写回扰动坐标",
    "attack.sanitize": "几何有效性检查 / MakeValid",
    "attack.resume_checkpoint": "恢复攻击阶段检查点",
    "attack.sanitize_skipped": "几何有效性检查跳过",
    "attack.fallback_legacy": "快速路径异常，回退旧实现",
    "attack.done": "攻击生成完成",
    "channel.canonicalize": "四通道：规范化几何",
    "channel.canonicalize_done": "四通道：规范化完成",
    "channel.rasterize": "四通道：栅格化占用通道",
    "channel.distance_transform": "四通道：距离变换",
    "channel.collect_segments": "四通道：提取线段与密度种子",
    "channel.collect_segments_done": "四通道：线段提取完成",
    "channel.orientation_accumulate": "四通道：方向场并行累积",
    "channel.orientation_propagate": "四通道：方向场传播",
    "channel.density_filter": "四通道：密度高斯滤波",
    "channel.stack": "四通道：堆叠输出张量",
    "channel.done": "四通道构建完成",
    "save.tensor": "保存 attacks shard",
    "save.metadata": "保存 metadata",
    "save.marker": "写入 complete 标记",
    "job.done": "任务完成",
    "job.error": "任务异常",
}


def _progress_path_for_job(job: Dict[str, object]) -> Path:
    cache_dir = Path(str(job["cache_dir"]))
    a = int(job["start_idx"]); b = int(job["stop_idx"])
    pdir = cache_dir / "progress_v13_0_8"
    pdir.mkdir(parents=True, exist_ok=True)
    return pdir / f"progress_{a:03d}_{b:03d}.json"


def _make_progress_writer(path: Path, base: Dict[str, object]):
    """Atomic, thread-safe worker progress telemetry; best effort only."""
    lock = threading.Lock()
    job_start = time.perf_counter()
    state = {"stage": None, "stage_start": job_start}

    def emit(*, stage: str, **detail):
        now = time.perf_counter()
        wall = time.time()
        with lock:
            if stage != state.get("stage"):
                state["stage"] = stage
                state["stage_start"] = now
            done = detail.pop("done", None)
            total = detail.pop("total", None)
            threads = detail.pop("threads", None)
            record = dict(base)
            record.update({
                "pid": int(os.getpid()),
                "stage": str(stage),
                "stage_cn": _PROGRESS_STAGE_CN.get(str(stage), str(stage)),
                "job_elapsed_s": float(now - job_start),
                "stage_elapsed_s": float(now - float(state["stage_start"])),
                "updated_unix": float(wall),
                "updated_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall)),
                "rss_gb": float(_max_rss_gb()),
            })
            if done is not None:
                record["done"] = int(done)
            if total is not None:
                record["total"] = int(total)
                record["percent"] = float(100.0 * int(done or 0) / max(1, int(total)))
            if threads is not None:
                record["threads"] = int(threads)
            record.update(detail)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
                tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp, path)
            except Exception:
                pass
    return emit


def _read_progress_snapshot(job: Dict[str, object]) -> Optional[Dict[str, object]]:
    try:
        p = _progress_path_for_job(job)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_progress_snapshot(job: Dict[str, object]) -> str:
    identity = str(job.get("identity", "?"))
    a = int(job.get("start_idx", 0)); b = int(job.get("stop_idx", 0))
    snap = _read_progress_snapshot(job)
    if not snap:
        return f"{identity} {a:03d}:{b:03d} | waiting for first progress heartbeat"
    stage = str(snap.get("stage_cn") or snap.get("stage") or "?")
    detail = str(snap.get("detail") or "")
    if "total" in snap:
        frac = f" {int(snap.get('done',0))}/{int(snap.get('total',0))} ({float(snap.get('percent',0)):.1f}%)"
    else:
        frac = ""
    strength = snap.get("strength")
    attack = snap.get("attack")
    case_text = f" {attack}={strength}" if attack is not None else ""
    age = max(0.0, time.time() - float(snap.get("updated_unix", time.time())))
    return (
        f"{identity} {a:03d}:{b:03d}{case_text} | {stage}{frac} | "
        f"job={float(snap.get('job_elapsed_s',0))/60:.1f}m stage={float(snap.get('stage_elapsed_s',0))/60:.1f}m "
        f"rss={float(snap.get('rss_gb',0)):.1f}GB heartbeat_age={age:.0f}s"
        + (f" | {detail}" if detail else "")
    )



def _available_memory_gb() -> float:
    """Linux available RAM without adding a psutil dependency."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return float("inf")


def _prepare_noise_index_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """Persist the coordinate duplicate map once for all six noise strengths."""
    pickle_path = Path(str(payload["pickle_path"]))
    inverse_path = Path(str(payload["inverse_path"]))
    meta_path = Path(str(payload["meta_path"]))
    source_path = str(payload["source_path"])
    expected_signature = payload.get("signature")
    if inverse_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            inv = np.load(inverse_path, mmap_mode="r", allow_pickle=False)
            if (
                meta.get("version") == "coordinate-noise-index-v13.0.2"
                and meta.get("signature") == expected_signature
                and int(meta.get("coordinate_count", -1)) == int(len(inv))
                and int(meta.get("unique_count", -1)) >= 0
            ):
                return {
                    "source_path": source_path, "status": "reused",
                    "inverse_path": str(inverse_path),
                    "unique_count": int(meta["unique_count"]),
                    "coordinate_count": int(len(inv)),
                }
        except Exception:
            pass
    t0 = time.perf_counter()
    with pickle_path.open("rb") as fh:
        gdf = pickle.load(fh)
    inverse, unique_count = prepare_coordinate_noise_index(gdf)
    inverse_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = inverse_path.with_name(inverse_path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as fh:
        np.save(fh, np.asarray(inverse), allow_pickle=False)
    os.replace(tmp, inverse_path)
    meta = {
        "version": "coordinate-noise-index-v13.0.2",
        "signature": expected_signature,
        "coordinate_count": int(len(inverse)),
        "unique_count": int(unique_count),
    }
    tmp_meta = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    return {
        "source_path": source_path, "status": "built",
        "inverse_path": str(inverse_path), "unique_count": int(unique_count),
        "coordinate_count": int(len(inverse)), "elapsed_s": time.perf_counter() - t0,
    }


def _job_is_heavy(job: Dict[str, object]) -> bool:
    """Return True only for genuinely large source layers.

    V13.0.3 marked *every* coordinate-noise case as heavy, even a layer with only
    a few thousand coordinates.  With ``heavy_cap=13`` this artificially limited
    an 80-core host to about 13 active Python processes during the final six noise
    attacks of each identity.  V13.0.4 makes heaviness a memory/geometry-size
    property: small coordinate-noise jobs are normal one-core jobs, while the
    million-coordinate water networks remain protected by the heavy admission cap.
    """
    threshold = int(os.environ.get("RB_AFL_LARGE_SOURCE_COORD_THRESHOLD", "200000"))
    return int(job.get("source_coordinate_count", 0) or 0) >= max(1, threshold)


def _job_priority(job: Dict[str, object]) -> Tuple[int, int, str, int]:
    """Longest/high-risk work first so expensive tasks never become tail stragglers."""
    heavy = 1 if _job_is_heavy(job) else 0
    try:
        source_bytes = Path(str(job.get("source_pickle") or job.get("source_path"))).stat().st_size
    except Exception:
        source_bytes = 0
    # descending heavy/source size is achieved by negative fields
    return (-heavy, -int(source_bytes), str(job.get("identity", "")), int(job["start_idx"]))


def _threads_for_job(
    job: Dict[str, object],
    *,
    cpu_budget: int,
    heavy_remaining: int,
    normal_remaining: int,
    max_heavy_inflight: int,
) -> int:
    """Return *channel* threads; scheduler slot cost is handled separately.

    Profiling on the real 80-core host showed that a water worker spends most of
    its lifetime in single-threaded GEOS/raster/canonicalisation work. Reserving
    four or six CPU slots for that process therefore left dozens of physical cores
    idle.  V13.0.4 runs many independent attacks as one-process/one-slot work.
    Extra channel threads are enabled only in the true tail, when very few jobs
    remain and there are otherwise no independent attacks available.
    """
    cpu_budget = max(1, int(cpu_budget))
    outstanding = max(1, int(heavy_remaining) + int(normal_remaining))
    if outstanding > 12:
        return 1
    # Tail acceleration: ThreadPool/Numba helps only the orientation accumulator,
    # so keep this moderate and do not reduce process-level concurrency.
    return max(1, min(24, cpu_budget // outstanding))



def _tail_job_key(job: Dict[str, object]) -> Tuple[str, int, int]:
    return (str(job.get("identity", "")), int(job.get("start_idx", 0)), int(job.get("stop_idx", 0)))


def _tail_job_weight(job: Dict[str, object]) -> float:
    """Relative work estimate for the last few huge coordinate-noise cases.

    Coordinate count is the dominant geometry-size term.  A mild strength factor
    reflects the empirical fact that stronger jitter tends to create more GEOS
    repair work without allowing one case to monopolize all cores.
    """
    coords = max(1, int(job.get("source_coordinate_count", 1) or 1))
    strength = 0.0
    try:
        a, b = int(job["start_idx"]), int(job["stop_idx"])
        if b == a + 1:
            case = attack_plan()[a]
            if str(case.attack) == "coordinate_noise":
                strength = max(0.0, float(case.strength))
    except Exception:
        pass
    return math.sqrt(float(coords)) * (1.0 + 20.0 * strength)


def _tail_thread_plan(jobs: Sequence[Dict[str, object]], physical_cores: Optional[int] = None) -> Dict[Tuple[str,int,int], int]:
    """Weighted software-thread plan for <=3 huge tail jobs on the 80-core host.

    Real measurements showed 24 GEOS threads delivering only ~15 runnable cores
    per worker. V13.0.8 therefore uses controlled oversubscription (default 1.4x)
    while keeping the outer attack count at three.  This does *not* change any
    geometry/RNG operation; it only changes how many independent feature chunks
    are allowed to execute concurrently.
    """
    jobs = list(jobs)
    if not jobs:
        return {}
    if physical_cores is None:
        physical_cores = int(os.environ.get("RB_AFL_PHYSICAL_CPU_CORES", "80"))
    physical_cores = max(1, int(physical_cores))
    default_total = max(1, int(round(physical_cores * float(os.environ.get("RB_AFL_TAIL_OVERSUBSCRIBE", "1.40")))))
    total_threads = max(1, int(os.environ.get("RB_AFL_TAIL_TOTAL_THREADS", str(default_total))))
    min_per = max(1, int(os.environ.get("RB_AFL_TAIL_MIN_THREADS_PER_JOB", "20")))
    max_per = max(min_per, int(os.environ.get("RB_AFL_TAIL_MAX_THREADS_PER_JOB", "40")))

    n = len(jobs)
    min_per = min(min_per, max(1, total_threads // n))
    total_threads = min(total_threads, max_per * n)
    base = [min_per] * n
    remaining = max(0, total_threads - min_per * n)
    weights = [_tail_job_weight(j) for j in jobs]
    # Weighted fair allocation with per-job caps. One software thread at a
    # time goes to the job with the highest weight/current-allocation ratio.
    # For the user's current last-three workload this naturally gives roughly
    # 35 / 37 / 40 threads (Sichuan-0.002 / Sichuan-0.005 / Shandong-0.005).
    alloc = base[:]
    while remaining > 0:
        eligible = [i for i in range(n) if alloc[i] < max_per]
        if not eligible:
            break
        i = max(eligible, key=lambda k: weights[k] / float(alloc[k] + 1))
        alloc[i] += 1
        remaining -= 1
    return {_tail_job_key(job): int(t) for job, t in zip(jobs, alloc)}


def _source_signature(path: str | Path) -> Dict[str, object]:
    """Signature geometry source and common shapefile sidecars for cache invalidation."""
    path = Path(path).expanduser().resolve()
    files = [path]
    if path.suffix.lower() == ".shp":
        for ext in (".shx", ".prj", ".cpg", ".dbf"):
            q = path.with_suffix(ext)
            if q.is_file():
                files.append(q)
    def sha256_file(item: Path) -> str:
        digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return {
        "source_name": path.name,
        "files": [
            {
                "name": q.name,
                "size": int(q.stat().st_size),
                "mtime_ns": int(q.stat().st_mtime_ns),
                "sha256": sha256_file(q),
            }
            for q in files if q.is_file()
        ],
    }


def _attack_plan_sha256() -> str:
    payload = [
        {
            "case_id": case.case_id,
            "attack": case.attack,
            "strength": float(case.strength),
            "repeat_index": int(case.repeat_index),
        }
        for case in attack_plan()
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_source_pickle_worker(payload: Dict[str, object]) -> Dict[str, object]:
    source_path = Path(str(payload["source_path"])).expanduser().resolve()
    pickle_path = Path(str(payload["pickle_path"]))
    meta_path = Path(str(payload["meta_path"]))
    expected = {"version": SOURCE_CACHE_VERSION, "signature": _source_signature(source_path)}
    if pickle_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta == expected:
                return {"source_path": str(source_path), "pickle_path": str(pickle_path), "status": "reused"}
        except Exception:
            pass
    t0 = time.perf_counter()
    gdf = read_vector(source_path)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_pickle = pickle_path.with_name(pickle_path.name + f".tmp.{os.getpid()}")
    tmp_meta = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
    with tmp_pickle.open("wb") as fh:
        pickle.dump(gdf, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_meta.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_pickle, pickle_path)
    os.replace(tmp_meta, meta_path)
    return {
        "source_name": source_path.name,
        "pickle_path": str(pickle_path),
        "status": "built",
        "elapsed_s": time.perf_counter() - t0,
        "features": int(len(gdf)),
    }


def _load_cached_vector(source_path: str | Path, pickle_path: str | Path | None):
    # Local import avoids adding GeoPandas to module annotations at import time.
    if pickle_path:
        p = Path(str(pickle_path))
        if p.is_file():
            with p.open("rb") as fh:
                return pickle.load(fh)
    return read_vector(source_path)


def _max_rss_gb() -> float:
    try:
        # Linux ru_maxrss is KiB.
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _completed_cache_status(payload: Dict[str, object]) -> Tuple[bool, str]:
    """Validate one identity cache without modifying it.

    This pre-scan is deliberately read-only so a resumed study run never
    touches identities that were already completed by V12.2.7.
    """
    cache_dir = Path(str(payload["cache_dir"]))
    grid_size = int(payload["grid_size"])
    density_sigma = float(payload["density_sigma"])
    fixed_attack_seed = int(payload["fixed_attack_seed"])
    plan = attack_plan()
    complete_path = cache_dir / "complete.json"
    attacks_path = cache_dir / "attacks.npy"
    meta_path = cache_dir / "attack_metadata.csv"
    clean_path = cache_dir / "clean.npy"
    if not all(x.is_file() for x in (complete_path, attacks_path, meta_path, clean_path)):
        return False, "missing_files"
    try:
        info = json.loads(complete_path.read_text(encoding="utf-8"))
        if info.get("cache_version") != CACHE_VERSION:
            return False, "cache_version"
        if int(info.get("fixed_attack_seed", -1)) != fixed_attack_seed:
            return False, "seed_mismatch"
        if not math.isclose(
            float(info.get("density_sigma", float("nan"))),
            density_sigma,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False, "density_sigma_mismatch"
        if info.get("attack_plan_sha256") != payload.get("attack_plan_sha256"):
            return False, "attack_plan_mismatch"
        if info.get("source_signature") != payload.get("source_signature"):
            return False, "source_signature_mismatch"
        if info.get("external_signature") != payload.get("external_signature"):
            return False, "external_signature_mismatch"
        arr = np.load(attacks_path, mmap_mode="r", allow_pickle=False)
        if tuple(arr.shape) != (len(plan), 4, grid_size, grid_size):
            return False, f"shape={tuple(arr.shape)}"
        meta = pd.read_csv(meta_path).fillna("")
        if len(meta) != len(plan):
            return False, f"metadata_rows={len(meta)}"
        required = {"case_id", "attack", "strength", "repeat_index"}
        if not required.issubset(meta.columns):
            return False, "metadata_schema"
        for index, case in enumerate(plan):
            row = meta.iloc[index]
            if (
                str(row["case_id"]) != case.case_id
                or str(row["attack"]) != case.attack
                or not math.isclose(
                    float(row["strength"]), float(case.strength), rel_tol=0.0, abs_tol=1e-12
                )
                or int(row["repeat_index"]) != int(case.repeat_index)
            ):
                return False, f"metadata_plan_mismatch_at_{index}"
        clean = np.load(clean_path, mmap_mode="r", allow_pickle=False)
        if tuple(clean.shape) != (4, grid_size, grid_size):
            return False, f"clean_shape={tuple(clean.shape)}"
        return True, "complete"
    except Exception as exc:
        return False, f"invalid:{type(exc).__name__}"


def _apply_source_map(base: pd.DataFrame, source_path_map: str | Path | None) -> pd.DataFrame:
    base = base.copy()
    if source_path_map is None:
        return base
    mapping_path = Path(source_path_map).expanduser().resolve()
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Source-path mapping not found: {mapping_path}")
    mapping_df = pd.read_csv(mapping_path).fillna("")
    required = {"identity", "resolved_source_path", "status"}
    missing = required.difference(mapping_df.columns)
    if missing:
        raise ValueError(f"Source-path mapping is missing columns: {sorted(missing)}")
    resolved = mapping_df.set_index(mapping_df["identity"].astype(str))["resolved_source_path"].to_dict()
    status = mapping_df.set_index(mapping_df["identity"].astype(str))["status"].to_dict()
    base["source_path_original"] = base["source_path"].astype(str)
    for idx in base.index:
        identity = str(base.at[idx, "identity"])
        if str(status.get(identity, "")) == "resolved" and str(resolved.get(identity, "")).strip():
            base.at[idx, "source_path"] = str(resolved[identity])
    return base


def _safe_case_filename(case_id: str) -> str:
    return case_id.replace("/", "_").replace("\\", "_")


def _cache_identity_worker(payload: Dict[str, object]) -> Dict[str, object]:
    identity = str(payload["identity"])
    source_path = Path(str(payload["source_path"]))
    cache_dir = Path(str(payload["cache_dir"]))
    external_path = str(payload.get("external_path") or "")
    grid_size = int(payload["grid_size"])
    density_sigma = float(payload["density_sigma"])
    fixed_attack_seed = int(payload["fixed_attack_seed"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    complete_path = cache_dir / "complete.json"
    attacks_path = cache_dir / "attacks.npy"
    meta_path = cache_dir / "attack_metadata.csv"
    clean_path = cache_dir / "clean.npy"
    plan = attack_plan()

    reusable, _ = _completed_cache_status(payload)
    if reusable:
        return {"identity": identity, "status": "reused", "elapsed_s": 0.0, "rows": len(plan)}

    start_all = time.perf_counter()
    t_load = time.perf_counter()
    gdf = _load_cached_vector(source_path, payload.get("source_pickle"))
    external = _load_cached_vector(external_path, payload.get("external_pickle")) if external_path else None
    load_s = time.perf_counter() - t_load
    clean_tensor = build_four_channels(gdf, grid_size=grid_size, density_sigma=density_sigma)[0]
    np.save(clean_path, np.asarray(clean_tensor, dtype=np.float32), allow_pickle=False)

    mmap = np.lib.format.open_memmap(
        attacks_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(plan), 4, grid_size, grid_size),
    )
    rows: List[Dict[str, object]] = []
    for idx, case in enumerate(plan):
        attack_seed = stable_seed(
            fixed_attack_seed,
            identity,
            case.attack,
            case.strength,
            case.repeat_index,
        )
        t0 = time.perf_counter()
        attacked, metadata = apply_attack(
            gdf,
            case.attack,
            case.strength,
            attack_seed,
            external,
        )
        attack_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        tensor = build_four_channels(
            attacked,
            grid_size=grid_size,
            density_sigma=density_sigma,
        )[0]
        channel_s = time.perf_counter() - t1
        mmap[idx] = np.asarray(tensor, dtype=np.float32)
        rows.append(
            {
                "identity": identity,
                "case_index": idx,
                "case_id": case.case_id,
                "attack": case.attack,
                "strength": case.strength,
                "repeat_index": int(case.repeat_index),
                "attack_seed": int(attack_seed),
                "attack_time_s": float(attack_s),
                "channel_build_time_s": float(channel_s),
                "attack_metadata": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            }
        )
    mmap.flush()
    del mmap
    pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")
    info = {
        "cache_version": CACHE_VERSION,
        "identity": identity,
        "source_name": source_path.name,
        "fixed_attack_seed": fixed_attack_seed,
        "grid_size": grid_size,
        "density_sigma": density_sigma,
        "attack_plan_sha256": str(payload["attack_plan_sha256"]),
        "source_signature": payload["source_signature"],
        "external_signature": payload["external_signature"],
        "attack_count": len(plan),
        "attack_seed_definition": "stable_seed(fixed_attack_seed, identity, attack, strength, repeat_index)",
        "note": "Attack realizations are fixed across all ablations and all training seeds.",
    }
    complete_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "identity": identity,
        "status": "built",
        "elapsed_s": time.perf_counter() - start_all,
        "load_s": float(load_s),
        "rows": len(plan),
        "max_rss_gb": _max_rss_gb(),
    }




def _cache_shard_worker(payload: Dict[str, object]) -> Dict[str, object]:
    """Build a deterministic contiguous shard of one identity's attack cases.

    Sharding is only used for identities whose legacy cache is incomplete.  It
    reduces the long-tail problem on very large water-system vector maps while
    preserving exactly the same attack seed and tensor construction functions.
    """
    # Emit a Python traceback on native faults (e.g. GEOS/Numba SIGSEGV) before
    # the process disappears.  This is especially useful when a ProcessPool is
    # intentionally restarted after one abrupt worker death.
    try:
        import faulthandler
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    identity = str(payload["identity"])
    source_path = Path(str(payload["source_path"]))
    cache_dir = Path(str(payload["cache_dir"]))
    external_path = str(payload.get("external_path") or "")
    grid_size = int(payload["grid_size"])
    density_sigma = float(payload["density_sigma"])
    fixed_attack_seed = int(payload["fixed_attack_seed"])
    start_idx = int(payload["start_idx"])
    stop_idx = int(payload["stop_idx"])
    plan = attack_plan()
    if not (0 <= start_idx < stop_idx <= len(plan)):
        raise ValueError(f"Invalid attack shard range: {start_idx}:{stop_idx}")

    shard_dir = cache_dir / "shards_v2"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{start_idx:03d}_{stop_idx:03d}"
    tensor_path = shard_dir / f"attacks_{stem}.npy"
    meta_path = shard_dir / f"metadata_{stem}.csv"
    done_path = shard_dir / f"complete_{stem}.json"
    expected_shape = (stop_idx - start_idx, 4, grid_size, grid_size)
    progress_path = _progress_path_for_job(payload)
    progress_base = {
        "identity": identity,
        "start_idx": start_idx,
        "stop_idx": stop_idx,
        "channel_threads": int(payload.get("channel_threads", 1)),
    }
    emit_progress = _make_progress_writer(progress_path, progress_base)
    emit_progress(stage="job.start", detail=f"range={start_idx:03d}:{stop_idx:03d}")

    if tensor_path.is_file() and meta_path.is_file() and done_path.is_file():
        try:
            info = json.loads(done_path.read_text(encoding="utf-8"))
            arr = np.load(tensor_path, mmap_mode="r", allow_pickle=False)
            meta = pd.read_csv(meta_path)
            if (
                info.get("cache_version") == CACHE_VERSION
                and int(info.get("fixed_attack_seed", -1)) == fixed_attack_seed
                and tuple(arr.shape) == expected_shape
                and len(meta) == stop_idx - start_idx
            ):
                return {
                    "identity": identity,
                    "status": "reused_shard",
                    "elapsed_s": 0.0,
                    "start_idx": start_idx,
                    "stop_idx": stop_idx,
                }
        except Exception:
            pass

    t_all = time.perf_counter()
    emit_progress(stage="load.start", detail=str(source_path.name))
    t_load = time.perf_counter()
    gdf = _load_cached_vector(source_path, payload.get("source_pickle"))
    external = _load_cached_vector(external_path, payload.get("external_pickle")) if external_path else None
    noise_inverse = None
    noise_unique_count = None
    noise_index_path = str(payload.get("noise_inverse_path") or "")
    if noise_index_path and Path(noise_index_path).is_file():
        try:
            noise_inverse = np.load(noise_index_path, mmap_mode="r", allow_pickle=False)
            noise_unique_count = int(payload.get("noise_unique_count", 0))
        except Exception:
            noise_inverse = None
            noise_unique_count = None
    load_s = time.perf_counter() - t_load
    emit_progress(stage="load.done", detail=f"features={len(gdf):,} load_s={load_s:.2f}")
    out = np.empty(expected_shape, dtype=np.float32)
    rows: List[Dict[str, object]] = []
    attack_total_s = 0.0
    channel_total_s = 0.0
    for local_idx, case_idx in enumerate(range(start_idx, stop_idx)):
        case = plan[case_idx]
        attack_seed = stable_seed(
            fixed_attack_seed,
            identity,
            case.attack,
            case.strength,
            case.repeat_index,
        )
        progress_base.update({
            "case_index": int(case_idx),
            "case_id": str(case.case_id),
            "attack": str(case.attack),
            "strength": float(case.strength),
            "repeat_index": int(case.repeat_index),
            "attack_seed": int(attack_seed),
        })
        emit_progress(stage="case.start", detail=f"{case.attack} strength={case.strength}")
        t0 = time.perf_counter()
        case_checkpoint_dir = None
        case_checkpoint_signature = None
        if case.attack == "coordinate_noise":
            case_checkpoint_dir = cache_dir / "checkpoint_v13_0_8" / f"case_{case_idx:03d}"
            case_checkpoint_signature = json.dumps({
                "version": "v13.0.8-coordinate-noise-checkpoint-v1",
                "identity": identity,
                "case_index": int(case_idx),
                "attack": str(case.attack),
                "strength": float(case.strength),
                "attack_seed": int(attack_seed),
                "source_coordinate_count": int(payload.get("source_coordinate_count", 0) or 0),
                "source_path": str(source_path),
                "source_pickle_size": int(Path(str(payload.get("source_pickle"))).stat().st_size) if payload.get("source_pickle") and Path(str(payload.get("source_pickle"))).is_file() else -1,
                "source_pickle_mtime_ns": int(Path(str(payload.get("source_pickle"))).stat().st_mtime_ns) if payload.get("source_pickle") and Path(str(payload.get("source_pickle"))).is_file() else -1,
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        attacked, metadata = apply_attack(
            gdf, case.attack, case.strength, attack_seed, external,
            coordinate_noise_inverse=noise_inverse if case.attack == "coordinate_noise" else None,
            coordinate_noise_unique_count=noise_unique_count if case.attack == "coordinate_noise" else None,
            source_is_sanitized=True,
            coordinate_noise_threads=max(1, int(payload.get("channel_threads", 1))) if case.attack == "coordinate_noise" else 1,
            progress_callback=emit_progress,
            coordinate_noise_checkpoint_dir=case_checkpoint_dir,
            coordinate_noise_checkpoint_signature=case_checkpoint_signature,
        )
        attack_s = time.perf_counter() - t0
        attack_total_s += attack_s
        t1 = time.perf_counter()
        tensor = build_four_channels(
            attacked,
            grid_size=grid_size,
            density_sigma=density_sigma,
            parallel_threads=max(1, int(payload.get("channel_threads", 1))),
            assume_clean=(case.attack == "coordinate_noise"),
            progress_callback=emit_progress,
        )[0]
        channel_s = time.perf_counter() - t1
        channel_total_s += channel_s
        out[local_idx] = np.asarray(tensor, dtype=np.float32)
        rows.append(
            {
                "identity": identity,
                "case_index": case_idx,
                "case_id": case.case_id,
                "attack": case.attack,
                "strength": case.strength,
                "repeat_index": int(case.repeat_index),
                "attack_seed": int(attack_seed),
                "attack_time_s": float(attack_s),
                "channel_build_time_s": float(channel_s),
                "attack_metadata": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            }
        )
    emit_progress(stage="save.tensor", detail=str(tensor_path.name))
    np.save(tensor_path, out, allow_pickle=False)
    emit_progress(stage="save.metadata", detail=str(meta_path.name))
    pd.DataFrame(rows).to_csv(meta_path, index=False, encoding="utf-8-sig")
    emit_progress(stage="save.marker", detail=str(done_path.name))
    done_path.write_text(
        json.dumps(
            {
                "cache_version": CACHE_VERSION,
                "identity": identity,
                "fixed_attack_seed": fixed_attack_seed,
                "grid_size": grid_size,
                "density_sigma": density_sigma,
                "attack_plan_sha256": str(payload["attack_plan_sha256"]),
                "source_signature": payload["source_signature"],
                "external_signature": payload["external_signature"],
                "start_idx": start_idx,
                "stop_idx": stop_idx,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    # Final shard persistence is authoritative; attack-stage checkpoints are only
    # needed until the shard complete marker exists. Reclaim disk afterwards.
    try:
        cp_root = cache_dir / "checkpoint_v13_0_8"
        for case_idx in range(start_idx, stop_idx):
            shutil.rmtree(cp_root / f"case_{case_idx:03d}", ignore_errors=True)
    except Exception:
        pass
    emit_progress(stage="job.done", done=1, total=1, detail=f"elapsed={time.perf_counter()-t_all:.1f}s")
    return {
        "identity": identity,
        "status": "built_shard",
        "elapsed_s": time.perf_counter() - t_all,
        "load_s": float(load_s),
        "attack_s": float(attack_total_s),
        "channel_s": float(channel_total_s),
        "max_rss_gb": _max_rss_gb(),
        "start_idx": start_idx,
        "stop_idx": stop_idx,
    }



def _valid_shard_ranges(payload: Dict[str, object]) -> List[Tuple[int, int]]:
    """Return all valid persisted shard ranges for an unfinished identity.

    V12.3.2 uses this to resume at case granularity.  Completed ranges from an
    earlier run are kept read-only; only uncovered attack cases are submitted.
    """
    cache_dir = Path(str(payload["cache_dir"]))
    grid_size = int(payload["grid_size"])
    density_sigma = float(payload["density_sigma"])
    fixed_attack_seed = int(payload["fixed_attack_seed"])
    plan = attack_plan()
    plan_len = len(plan)
    shard_dir = cache_dir / "shards_v2"
    if not shard_dir.is_dir():
        return []
    valid: List[Tuple[int, int]] = []
    for done_path in sorted(shard_dir.glob("complete_*.json")):
        stem = done_path.stem.replace("complete_", "")
        try:
            a_s, b_s = stem.split("_", 1)
            a, b = int(a_s), int(b_s)
        except Exception:
            continue
        if not (0 <= a < b <= plan_len):
            continue
        tensor_path = shard_dir / f"attacks_{stem}.npy"
        meta_path = shard_dir / f"metadata_{stem}.csv"
        if not (tensor_path.is_file() and meta_path.is_file()):
            continue
        try:
            info = json.loads(done_path.read_text(encoding="utf-8"))
            if info.get("cache_version") != CACHE_VERSION:
                continue
            if int(info.get("fixed_attack_seed", -1)) != fixed_attack_seed:
                continue
            if not math.isclose(
                float(info.get("density_sigma", float("nan"))),
                density_sigma,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                continue
            if info.get("attack_plan_sha256") != payload.get("attack_plan_sha256"):
                continue
            if info.get("source_signature") != payload.get("source_signature"):
                continue
            if info.get("external_signature") != payload.get("external_signature"):
                continue
            arr = np.load(tensor_path, mmap_mode="r", allow_pickle=False)
            if tuple(arr.shape) != (b-a, 4, grid_size, grid_size):
                continue
            meta = pd.read_csv(meta_path).fillna("")
            if len(meta) != b-a:
                continue
            required = {"case_id", "attack", "strength", "repeat_index"}
            if not required.issubset(meta.columns):
                continue
            mismatch = False
            for local_index, case in enumerate(plan[a:b]):
                row = meta.iloc[local_index]
                if (
                    str(row["case_id"]) != case.case_id
                    or str(row["attack"]) != case.attack
                    or not math.isclose(
                        float(row["strength"]),
                        float(case.strength),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or int(row["repeat_index"]) != int(case.repeat_index)
                ):
                    mismatch = True
                    break
            if mismatch:
                continue
            valid.append((a,b))
        except Exception:
            continue
    return valid


def _coverage_from_ranges(ranges: Sequence[Tuple[int, int]], n_cases: int) -> np.ndarray:
    covered = np.zeros(n_cases, dtype=bool)
    for a,b in ranges:
        covered[max(0,a):min(n_cases,b)] = True
    return covered


def _assemble_identity_from_available_shards(payload: Dict[str, object]) -> None:
    """Assemble a complete identity from any mixture of old and new shards.

    Overlapping shards are allowed. For each case index, the first valid shard
    covering that case is used. This lets a V12.3.1 run with 16 broad shards be
    resumed by V12.3.2 with one-case long-tail shards without rebuilding work.
    """
    identity = str(payload["identity"])
    source_path = Path(str(payload["source_path"]))
    cache_dir = Path(str(payload["cache_dir"]))
    grid_size = int(payload["grid_size"])
    density_sigma = float(payload["density_sigma"])
    fixed_attack_seed = int(payload["fixed_attack_seed"])
    plan = attack_plan()
    shard_dir = cache_dir / "shards_v2"
    ranges = _valid_shard_ranges(payload)
    if not ranges:
        raise RuntimeError(f"No valid shards available for {identity}")

    # Map each case to one persisted shard. Prefer the narrowest range, which
    # naturally prefers fine-grained resume shards over older broad shards.
    case_source: Dict[int, Tuple[int,int]] = {}
    for a,b in sorted(ranges, key=lambda x: ((x[1]-x[0]), x[0])):
        for idx in range(a,b):
            case_source.setdefault(idx, (a,b))
    missing = [i for i in range(len(plan)) if i not in case_source]
    if missing:
        raise RuntimeError(f"Missing attack cases for {identity}: {missing[:10]}")

    attacks_path = cache_dir / "attacks.npy"
    meta_path = cache_dir / "attack_metadata.csv"
    clean_path = cache_dir / "clean.npy"
    complete_path = cache_dir / "complete.json"
    mmap = np.lib.format.open_memmap(attacks_path, mode="w+", dtype=np.float32,
                                     shape=(len(plan),4,grid_size,grid_size))
    meta_rows = []
    opened = {}
    meta_cache = {}
    for idx in range(len(plan)):
        a,b = case_source[idx]
        stem=f"{a:03d}_{b:03d}"
        if stem not in opened:
            opened[stem]=np.load(shard_dir/f"attacks_{stem}.npy", mmap_mode="r", allow_pickle=False)
            df=pd.read_csv(shard_dir/f"metadata_{stem}.csv")
            meta_cache[stem]={int(r["case_index"]):r for _,r in df.iterrows()}
        mmap[idx]=opened[stem][idx-a]
        meta_rows.append(meta_cache[stem][idx])
    mmap.flush(); del mmap; opened.clear()
    pd.DataFrame(meta_rows).sort_values("case_index").to_csv(meta_path,index=False,encoding="utf-8-sig")

    clean_ok=False
    if clean_path.is_file():
        try:
            clean=np.load(clean_path,mmap_mode="r",allow_pickle=False)
            clean_ok=tuple(clean.shape)==(4,grid_size,grid_size)
        except Exception:
            clean_ok=False
    if not clean_ok:
        gdf=_load_cached_vector(source_path, payload.get("source_pickle"))
        clean_tensor=build_four_channels(gdf,grid_size=grid_size,density_sigma=density_sigma)[0]
        np.save(clean_path,np.asarray(clean_tensor,dtype=np.float32),allow_pickle=False)

    info={
        "cache_version": CACHE_VERSION,
        "identity": identity,
        "source_name": source_path.name,
        "fixed_attack_seed": fixed_attack_seed,
        "grid_size": grid_size,
        "density_sigma": density_sigma,
        "attack_plan_sha256": str(payload["attack_plan_sha256"]),
        "source_signature": payload["source_signature"],
        "external_signature": payload["external_signature"],
        "attack_count": len(plan),
        "attack_seed_definition": "stable_seed(fixed_attack_seed, identity, attack, strength, repeat_index)",
        "note": "Attack realizations are fixed across all ablations and all training seeds.",
        "builder": "mixed-shard-resume-v3",
        "persisted_shards": [[int(a),int(b)] for a,b in sorted(set(ranges))],
    }
    complete_path.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")


def _assemble_sharded_identity_cache(payload: Dict[str, object], ranges: Sequence[Tuple[int, int]]) -> None:
    identity = str(payload["identity"])
    source_path = Path(str(payload["source_path"]))
    cache_dir = Path(str(payload["cache_dir"]))
    grid_size = int(payload["grid_size"])
    density_sigma = float(payload["density_sigma"])
    fixed_attack_seed = int(payload["fixed_attack_seed"])
    plan = attack_plan()
    shard_dir = cache_dir / "shards_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)

    attacks_path = cache_dir / "attacks.npy"
    meta_path = cache_dir / "attack_metadata.csv"
    clean_path = cache_dir / "clean.npy"
    complete_path = cache_dir / "complete.json"

    mmap = np.lib.format.open_memmap(
        attacks_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(plan), 4, grid_size, grid_size),
    )
    metas: List[pd.DataFrame] = []
    for start_idx, stop_idx in ranges:
        stem = f"{start_idx:03d}_{stop_idx:03d}"
        tensor_path = shard_dir / f"attacks_{stem}.npy"
        shard_meta_path = shard_dir / f"metadata_{stem}.csv"
        done_path = shard_dir / f"complete_{stem}.json"
        if not (tensor_path.is_file() and shard_meta_path.is_file() and done_path.is_file()):
            raise RuntimeError(f"Missing shard for {identity}: {stem}")
        arr = np.load(tensor_path, mmap_mode="r", allow_pickle=False)
        if tuple(arr.shape) != (stop_idx - start_idx, 4, grid_size, grid_size):
            raise RuntimeError(f"Shard shape mismatch for {identity} {stem}: {arr.shape}")
        mmap[start_idx:stop_idx] = arr
        metas.append(pd.read_csv(shard_meta_path))
    mmap.flush()
    del mmap
    meta = pd.concat(metas, ignore_index=True).sort_values("case_index").reset_index(drop=True)
    if len(meta) != len(plan) or meta["case_index"].astype(int).tolist() != list(range(len(plan))):
        raise RuntimeError(f"Assembled metadata incomplete for {identity}")
    meta.to_csv(meta_path, index=False, encoding="utf-8-sig")

    clean_ok = False
    if clean_path.is_file():
        try:
            clean = np.load(clean_path, mmap_mode="r", allow_pickle=False)
            clean_ok = tuple(clean.shape) == (4, grid_size, grid_size)
        except Exception:
            clean_ok = False
    if not clean_ok:
        gdf = _load_cached_vector(source_path, payload.get("source_pickle"))
        clean_tensor = build_four_channels(gdf, grid_size=grid_size, density_sigma=density_sigma)[0]
        np.save(clean_path, np.asarray(clean_tensor, dtype=np.float32), allow_pickle=False)

    info = {
        "cache_version": CACHE_VERSION,
        "identity": identity,
        "source_path": str(source_path),
        "fixed_attack_seed": fixed_attack_seed,
        "grid_size": grid_size,
        "density_sigma": density_sigma,
        "attack_plan_sha256": str(payload["attack_plan_sha256"]),
        "source_signature": payload["source_signature"],
        "external_signature": payload["external_signature"],
        "attack_count": len(plan),
        "attack_seed_definition": "stable_seed(fixed_attack_seed, identity, attack, strength, repeat_index)",
        "note": "Attack realizations are fixed across all ablations and all training seeds.",
        "builder": "sharded-v2",
        "shard_ranges": [[int(a), int(b)] for a, b in ranges],
    }
    complete_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")




def _legacy_cache_candidate(identity: str, roots: Sequence[str | Path], plan_len: int, grid_size: int, fixed_attack_seed: int):
    """Return a scientifically compatible legacy complete identity cache, if any."""
    plan = attack_plan()
    for root in roots:
        d = Path(root).expanduser().resolve() / identity
        attacks = d / "attacks.npy"
        meta_path = d / "attack_metadata.csv"
        complete = d / "complete.json"
        clean = d / "clean.npy"
        if not (attacks.is_file() and meta_path.is_file() and complete.is_file() and clean.is_file()):
            continue
        try:
            info = json.loads(complete.read_text(encoding="utf-8"))
            # Seed equality is non-negotiable for a paired study experiment.
            if int(info.get("fixed_attack_seed", -1)) != int(fixed_attack_seed):
                continue
            arr = np.load(attacks, mmap_mode="r", allow_pickle=False)
            if tuple(arr.shape) != (plan_len, 4, grid_size, grid_size):
                continue
            cln = np.load(clean, mmap_mode="r", allow_pickle=False)
            if tuple(cln.shape) != (4, grid_size, grid_size):
                continue
            meta = pd.read_csv(meta_path)
            if len(meta) != plan_len:
                continue
            if "attack" in meta.columns and "strength" in meta.columns:
                for i, case in enumerate(plan):
                    if str(meta.iloc[i]["attack"]) != case.attack:
                        raise ValueError("attack plan mismatch")
                    if not math.isclose(float(meta.iloc[i]["strength"]), float(case.strength), rel_tol=0.0, abs_tol=1e-12):
                        raise ValueError("attack strength mismatch")
                    stored_repeat = int(meta.iloc[i].get("repeat_index", 0))
                    if stored_repeat != int(case.repeat_index):
                        raise ValueError("attack repeat-index mismatch")
            return {
                "root": str(Path(root).expanduser().resolve()),
                "dir": str(d),
                "attacks": str(attacks),
                "meta": str(meta_path),
                "complete": str(complete),
                "clean": str(clean),
                "legacy_cache_version": str(info.get("cache_version", "unknown")),
            }
        except Exception:
            continue
    return None


def _load_v13_case_tensor(job: Dict[str, object], case_idx: int) -> Optional[np.ndarray]:
    cache_dir = Path(str(job["cache_dir"]))
    shard_dir = cache_dir / "shards_v2"
    if not shard_dir.is_dir():
        return None
    for a, b in _valid_shard_ranges(job):
        if a <= case_idx < b:
            p = shard_dir / f"attacks_{a:03d}_{b:03d}.npy"
            try:
                arr = np.load(p, mmap_mode="r", allow_pickle=False)
                return np.asarray(arr[case_idx - a], dtype=np.float32)
            except Exception:
                return None
    return None


def _try_import_legacy_complete_caches(
    pending: Sequence[Dict[str, object]],
    legacy_roots: Sequence[str | Path],
    *,
    grid_size: int,
    fixed_attack_seed: int,
    tolerance: float = 2e-6,
    min_comparisons: int = 8,
    min_attack_families: int = 5,
) -> Dict[str, object]:
    """Validate old full caches against already-built V13 shards, then reuse them.

    The import is deliberately conservative: legacy tensors are adopted only
    after real overlapping V13 tensors demonstrate numerical equivalence within
    a strict tolerance.  The comparison/report is persisted beside the V13 cache.
    """
    if not legacy_roots:
        return {"compatible": False, "reason": "no_legacy_roots", "imported_identities": 0}
    plan = attack_plan()
    candidates = {}
    comparisons = []
    preferred = [83, 84, 85, 86, 87, 88, 76, 50, 20, 0, 93]
    for job in pending:
        identity = str(job["identity"])
        cand = _legacy_cache_candidate(identity, legacy_roots, len(plan), grid_size, fixed_attack_seed)
        if cand is None:
            continue
        candidates[identity] = cand
        old_arr = np.load(cand["attacks"], mmap_mode="r", allow_pickle=False)
        covered = _coverage_from_ranges(_valid_shard_ranges(job), len(plan))
        covered_indices = [int(i) for i in np.flatnonzero(covered).tolist()]
        # Compare across diverse attack families first, then favor the known
        # long-tail/noise cases. This makes whole-identity legacy adoption a
        # stronger scientific equivalence audit rather than a spot check.
        family_first = []
        seen_families = set()
        for i in covered_indices:
            fam = plan[i].attack
            if fam not in seen_families:
                family_first.append(i)
                seen_families.add(fam)
        indices = family_first
        indices += [i for i in preferred if i < len(plan) and covered[i] and i not in indices]
        indices += [i for i in covered_indices if i not in indices]
        for i in indices[:16]:
            new = _load_v13_case_tensor(job, i)
            if new is None:
                continue
            old = np.asarray(old_arr[i], dtype=np.float32)
            diff = np.abs(new.astype(np.float64) - old.astype(np.float64))
            comparisons.append({
                "identity": identity,
                "case_index": int(i),
                "attack": plan[i].attack,
                "strength": float(plan[i].strength),
                "max_abs_diff": float(diff.max(initial=0.0)),
                "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
            })
            if len(comparisons) >= 24:
                break
        if len(comparisons) >= 24:
            break

    max_diff = max((x["max_abs_diff"] for x in comparisons), default=float("inf"))
    attack_families = sorted({str(x["attack"]) for x in comparisons})
    compatible = (
        len(comparisons) >= int(min_comparisons)
        and len(attack_families) >= int(min_attack_families)
        and max_diff <= float(tolerance)
    )
    imported = []
    if compatible:
        import shutil
        for job in pending:
            identity = str(job["identity"])
            cand = candidates.get(identity)
            if cand is None:
                continue
            cache_dir = Path(str(job["cache_dir"]))
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Whole-file reuse is far cheaper and cleaner than rewriting all one-case shards.
            shutil.copy2(cand["attacks"], cache_dir / "attacks.npy")
            shutil.copy2(cand["meta"], cache_dir / "attack_metadata.csv")
            shutil.copy2(cand["clean"], cache_dir / "clean.npy")
            info = {
                "cache_version": CACHE_VERSION,
                "identity": identity,
                "source_path": str(job["source_path"]),
                "fixed_attack_seed": int(fixed_attack_seed),
                "grid_size": int(grid_size),
                "density_sigma": float(job["density_sigma"]),
                "attack_count": len(plan),
                "attack_seed_definition": "stable_seed(fixed_attack_seed, identity, attack, strength, repeat_index)",
                "note": "Numerically validated legacy cache imported by V13.0.2; no scientific definition changed.",
                "builder": "legacy-compatible-import-v13.0.2",
                "legacy_source": cand["dir"],
                "legacy_cache_version": cand["legacy_cache_version"],
                "compatibility_tolerance": float(tolerance),
                "compatibility_max_abs_diff": float(max_diff),
                "compatibility_comparisons": int(len(comparisons)),
            }
            (cache_dir / "complete.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            imported.append(identity)

    return {
        "compatible": bool(compatible),
        "reason": "validated" if compatible else "insufficient_or_mismatched_overlap",
        "legacy_roots": [str(Path(x).expanduser().resolve()) for x in legacy_roots],
        "candidate_identities": sorted(candidates),
        "comparisons": comparisons,
        "comparison_count": int(len(comparisons)),
        "attack_families_compared": attack_families,
        "attack_family_count": int(len(attack_families)),
        "minimum_attack_families_required": int(min_attack_families),
        "max_abs_diff": None if not comparisons else float(max_diff),
        "tolerance": float(tolerance),
        "imported_identities": int(len(imported)),
        "imported_identity_names": imported,
    }

def build_fixed_test_attack_cache(
    prepared_root: str | Path,
    cache_root: str | Path,
    *,
    identity_split_path: str | Path,
    source_path_map: str | Path,
    study_split_name: str = "test",
    external_vector_path: str | Path | None = None,
    grid_size: int = 256,
    density_sigma: float = 3.0,
    fixed_attack_seed: int = 20260910,
    workers: int = 8,
    shards_per_identity: int = 1,
    max_shards_per_identity: int = 12,
    longtail_cases_per_shard: int = 1,
    source_cache_workers: int = 8,
    legacy_cache_roots: Optional[Sequence[str | Path]] = None,
    import_legacy_cache: bool = False,
    legacy_compat_tolerance: float = 2e-6,
    noise_index_workers: int = 6,
    min_available_memory_gb: float = 20.0,
    max_heavy_inflight: int = 10,
) -> Path:
    """Build model-independent test attacks exactly once.

    This is both faster and a cleaner paired experimental design: the attacked
    test input is identical for E1-E7 and for every training seed.  Only the
    trained model changes across comparisons.
    """
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = base_manifest(prepared_root, identity_split_path, study_split_name)
    base = _apply_source_map(base, source_path_map)
    missing = [str(p) for p in base["source_path"].astype(str) if not Path(str(p)).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw test vector: {missing[0]}")
    if external_vector_path and not Path(external_vector_path).expanduser().is_file():
        raise FileNotFoundError(f"External vector not found: {external_vector_path}")

    plan = attack_plan()
    attack_plan_sha256 = _attack_plan_sha256()
    external_signature = (
        _source_signature(external_vector_path) if external_vector_path else None
    )
    source_signatures = {
        str(row["identity"]): _source_signature(str(row["source_path"]))
        for row in base.to_dict("records")
    }
    manifest_path = root / "cache_manifest.json"
    cache_meta = {
        "cache_version": CACHE_VERSION,
        "fixed_attack_seed": int(fixed_attack_seed),
        "grid_size": int(grid_size),
        "density_sigma": float(density_sigma),
        "identity_split_sha256": hashlib.sha256(
            Path(identity_split_path).expanduser().resolve().read_bytes()
        ).hexdigest(),
        "attack_plan_sha256": attack_plan_sha256,
        "source_signatures": source_signatures,
        "external_signature": external_signature,
        "identity_count": int(len(base)),
        "attack_count": int(len(plan)),
        "total_attacked_tensors": int(len(base) * len(plan)),
        "attack_seed_definition": "stable_seed(fixed_attack_seed, identity, attack, strength, repeat_index)",
        "comparison_design": "same attacked input for every E1-E7 model and every training seed",
        "external_vector_name": Path(external_vector_path).name if external_vector_path else "",
    }
    manifest_path.write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    jobs: List[Dict[str, object]] = []
    for row in base.to_dict("records"):
        identity = str(row["identity"])
        jobs.append(
            {
                "identity": identity,
                "source_path": str(row["source_path"]),
                "cache_dir": str(root / identity),
                "external_path": str(external_vector_path or ""),
                "grid_size": int(grid_size),
                "density_sigma": float(density_sigma),
                "fixed_attack_seed": int(fixed_attack_seed),
                "attack_plan_sha256": attack_plan_sha256,
                "source_signature": source_signatures[identity],
                "external_signature": external_signature,
            }
        )

    # Read-only pre-scan: completed V12.2.7 identity caches are never submitted
    # to workers again.  This is important for long water-system maps that may
    # each take hours to build.
    reusable: List[Dict[str, object]] = []
    pending: List[Dict[str, object]] = []
    pending_reasons: Dict[str, str] = {}
    for job in jobs:
        ok, reason = _completed_cache_status(job)
        if ok:
            reusable.append(job)
        else:
            pending.append(job)
            pending_reasons[str(job["identity"])] = reason

    # Build a persistent, sanitized geometry-only pickle once per pending source.
    # Every attack process then loads this compact cache instead of repeatedly
    # reparsing SHP/GPKG + attributes + geometry validity from disk.
    if pending:
        source_cache_root = root / "_source_geometry_cache_v1"
        source_cache_root.mkdir(parents=True, exist_ok=True)
        prep_payloads: List[Dict[str, object]] = []
        path_to_cache: Dict[str, str] = {}
        unique_paths = sorted(
            set(str(Path(str(job["source_path"])).expanduser().resolve()) for job in pending)
            | ({str(Path(external_vector_path).expanduser().resolve())} if external_vector_path else set())
        )
        import hashlib
        for src in unique_paths:
            key = hashlib.sha256(src.encode("utf-8")).hexdigest()[:20]
            stem = Path(src).stem.replace(" ", "_")[:48]
            pkl = source_cache_root / f"{stem}_{key}.pkl"
            meta = source_cache_root / f"{stem}_{key}.json"
            prep_payloads.append({"source_path": src, "pickle_path": str(pkl), "meta_path": str(meta)})
            path_to_cache[src] = str(pkl)
        prep_workers = max(1, min(int(source_cache_workers), len(prep_payloads)))
        print(
            f"[SOURCE GEOMETRY CACHE] sources={len(prep_payloads)} workers={prep_workers} "
            f"root={source_cache_root}",
            flush=True,
        )
        if prep_workers == 1:
            prep_results = [_prepare_source_pickle_worker(x) for x in prep_payloads]
        else:
            with ProcessPoolExecutor(max_workers=prep_workers, mp_context=get_context("spawn")) as pool:
                prep_results = list(pool.map(_prepare_source_pickle_worker, prep_payloads))
        for result in prep_results:
            print(
                f"  [SOURCE CACHE] {Path(result['source_path']).name} {result['status']} "
                f"{float(result.get('elapsed_s', 0.0)):.1f}s",
                flush=True,
            )
        external_resolved = str(Path(external_vector_path).expanduser().resolve()) if external_vector_path else ""
        for job in pending:
            src = str(Path(str(job["source_path"])).expanduser().resolve())
            job["source_pickle"] = path_to_cache.get(src, "")
            job["external_pickle"] = path_to_cache.get(external_resolved, "") if external_resolved else ""


    # V13.0.2: reuse scientifically equivalent complete caches from earlier runs.
    # Auto-detect the user's prior fixed cache next to the V13 cache, but never
    # adopt it blindly: at least three overlapping V13 tensors must agree within
    # the configured numerical tolerance first.
    if pending and import_legacy_cache:
        roots = list(legacy_cache_roots or [])
        auto_old = root.parent / "test_attack_cache_fixed_v1"
        if auto_old.is_dir() and all(Path(x).expanduser().resolve() != auto_old.resolve() for x in roots):
            roots.append(auto_old)
        roots = [Path(x).expanduser().resolve() for x in roots if Path(x).expanduser().is_dir()]
        if roots:
            report = _try_import_legacy_complete_caches(
                pending, roots, grid_size=grid_size,
                fixed_attack_seed=fixed_attack_seed,
                tolerance=float(legacy_compat_tolerance),
            )
            (root / "legacy_cache_import_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"[LEGACY CACHE AUDIT] compatible={report['compatible']} "
                f"comparisons={report.get('comparison_count',0)} "
                f"max_abs_diff={report.get('max_abs_diff')} "
                f"imported_identities={report.get('imported_identities',0)}",
                flush=True,
            )
            # Re-scan after any whole-identity imports. Imported caches now carry
            # the V13 cache marker and are treated exactly like native completed caches.
            reusable, pending, pending_reasons = [], [], {}
            for job in jobs:
                ok, reason = _completed_cache_status(job)
                if ok:
                    reusable.append(job)
                else:
                    pending.append(job)
                    pending_reasons[str(job["identity"])] = reason

    # V13.0.2: build the O(N log N) coordinate duplicate index once for every
    # still-pending source, then mmap it in all six coordinate-noise tasks.
    if pending:
        noise_root = root / "_coordinate_noise_index_v1"
        noise_root.mkdir(parents=True, exist_ok=True)
        noise_payloads = []
        noise_by_source: Dict[str, Dict[str, object]] = {}
        import hashlib
        seen_sources = set()
        for job in pending:
            src = str(Path(str(job["source_path"])).expanduser().resolve())
            if src in seen_sources:
                continue
            seen_sources.add(src)
            source_pickle = str(job.get("source_pickle") or "")
            if not source_pickle or not Path(source_pickle).is_file():
                continue
            key = hashlib.sha256(src.encode("utf-8")).hexdigest()[:20]
            stem = Path(src).stem.replace(" ", "_")[:48]
            inv = noise_root / f"{stem}_{key}_inverse.npy"
            meta = noise_root / f"{stem}_{key}.json"
            noise_payloads.append({
                "source_path": src,
                "pickle_path": source_pickle,
                "inverse_path": str(inv),
                "meta_path": str(meta),
                "signature": _source_signature(src),
            })
        nw = max(1, min(int(noise_index_workers), len(noise_payloads))) if noise_payloads else 0
        if noise_payloads:
            print(
                f"[COORDINATE NOISE INDEX] sources={len(noise_payloads)} workers={nw} root={noise_root}",
                flush=True,
            )
            if nw == 1:
                noise_results = [_prepare_noise_index_worker(x) for x in noise_payloads]
            else:
                with ProcessPoolExecutor(max_workers=nw, mp_context=get_context("spawn")) as pool:
                    noise_results = list(pool.map(_prepare_noise_index_worker, noise_payloads))
            for result in noise_results:
                src = str(Path(str(result["source_path"])).expanduser().resolve())
                noise_by_source[src] = result
                coord_count = int(result.get("coordinate_count", 0))
                large_threshold = int(os.environ.get("RB_AFL_LARGE_SOURCE_COORD_THRESHOLD", "200000"))
                water_tag = " WATER-HEAVY" if coord_count >= max(1, large_threshold) else ""
                print(
                    f"  [NOISE INDEX] {Path(src).name} {result['status']} "
                    f"coords={coord_count:,} unique={int(result.get('unique_count',0)):,} "
                    f"{float(result.get('elapsed_s',0.0)):.1f}s{water_tag}",
                    flush=True,
                )
            for job in pending:
                src = str(Path(str(job["source_path"])).expanduser().resolve())
                result = noise_by_source.get(src)
                if result:
                    job["noise_inverse_path"] = str(result["inverse_path"])
                    job["noise_unique_count"] = int(result["unique_count"])
                    # Reuse the already-computed source complexity for scheduling.
                    # Large water-system layers then run as fewer multi-threaded
                    # channel builders instead of dozens of contending 1-thread jobs.
                    job["source_coordinate_count"] = int(result.get("coordinate_count", 0))

    print(
        f"\n[ATTACK CACHE] identities={len(jobs)} attacks={len(plan)} "
        f"total_tensors={len(jobs) * len(plan):,}",
        flush=True,
    )
    print(
        f"[ATTACK CACHE RESUME] reusable={len(reusable)} pending={len(pending)}; "
        "completed identities are read-only and will NOT be rebuilt.",
        flush=True,
    )
    for job in reusable:
        print(f"  [REUSED] {job['identity']}", flush=True)
    worker_budget = max(1, int(workers))
    if pending:
        # Keep the full worker budget for attack sharding. The old implementation
        # reduced workers to len(pending) before creating shards, which meant that
        # 2 unfinished identities could never use more than 2 CPU processes.
        identity_workers = max(1, min(worker_budget, len(pending)))
        print(
            f"[ATTACK CACHE BUILD] pending identities={len(pending)} "
            f"identity_workers={identity_workers} worker_budget={worker_budget}; "
            f"new expensive builds at most={len(pending) * len(plan):,}",
            flush=True,
        )
        for job in pending:
            print(
                f"  [PENDING] {job['identity']} reason={pending_reasons[str(job['identity'])]}",
                flush=True,
            )
    else:
        identity_workers = 0
        print("[ATTACK CACHE BUILD] nothing to build; all identities reused.", flush=True)

    # Precompile/load the compiled line-rasterization kernel before spawning the
    # large worker pool. This avoids dozens of children compiling simultaneously.
    warmup_v13_fastpath()

    started = time.perf_counter()
    completed = len(reusable)
    requested_shards = max(1, int(shards_per_identity))
    safe_shard_cap = max(1, int(max_shards_per_identity))
    if pending:
        # V13: build many small globally schedulable shards even when many
        # identities are pending.  Geometry-only source caches make reloading
        # inexpensive, while 24-48 shards/identity prevent one heavy attack
        # from pinning a whole identity-sized worker for hours.
        effective_shards = min(requested_shards, safe_shard_cap, len(plan))
    else:
        effective_shards = 1
    if pending and requested_shards != effective_shards:
        print(
            f"[ATTACK CACHE SHARD ADAPT] requested={requested_shards} -> effective={effective_shards} "
            f"per identity (worker_budget={worker_budget}, pending={len(pending)}, safe_cap={safe_shard_cap})",
            flush=True,
        )

    if pending and effective_shards > 1:
        # V12.3.2: resume from ANY valid persisted shard ranges.  If broad
        # shards from an earlier run already cover part of an identity, only
        # uncovered attack cases are scheduled.  Long-tail gaps are split into
        # tiny deterministic shards so idle CPU cores can help instead of
        # waiting for 5-6 serial attacks inside one old shard.
        shard_jobs: List[Dict[str, object]] = []
        assembly_mode: Dict[str, str] = {}
        for job in pending:
            identity = str(job["identity"])
            existing_ranges = _valid_shard_ranges(job)
            covered = _coverage_from_ranges(existing_ranges, len(plan))
            missing_indices = [i for i in range(len(plan)) if not covered[i]]
            if existing_ranges:
                assembly_mode[identity] = "mixed"
                print(
                    f"  [SHARD RESUME] {identity} persisted_ranges={len(existing_ranges)} "
                    f"covered_cases={int(covered.sum())}/{len(plan)} missing_cases={len(missing_indices)}",
                    flush=True,
                )
                chunk = max(1, int(longtail_cases_per_shard))
                # Group only consecutive missing cases, never cross a completed
                # range. With chunk=1 each remaining attack becomes independently
                # schedulable and immediately persistent.
                k = 0
                while k < len(missing_indices):
                    a = missing_indices[k]
                    b = a + 1
                    k += 1
                    while (
                        k < len(missing_indices)
                        and missing_indices[k] == b
                        and (b - a) < chunk
                    ):
                        b += 1
                        k += 1
                    shard = dict(job)
                    shard["start_idx"] = int(a)
                    shard["stop_idx"] = int(b)
                    shard_jobs.append(shard)
            else:
                assembly_mode[identity] = "fresh"
                n_shards = min(effective_shards, len(plan))
                edges = np.linspace(0, len(plan), n_shards + 1, dtype=int)
                for i in range(n_shards):
                    a, b = int(edges[i]), int(edges[i + 1])
                    if a >= b:
                        continue
                    # V13.0.2: coordinate-noise cases are never bundled together.
                    # They are the dominant long-running cases on huge water maps
                    # and must be independently schedulable across the whole host.
                    cursor = a
                    while cursor < b:
                        if plan[cursor].attack == "coordinate_noise":
                            shard = dict(job)
                            shard["start_idx"] = cursor
                            shard["stop_idx"] = cursor + 1
                            shard_jobs.append(shard)
                            cursor += 1
                            continue
                        end = cursor + 1
                        while end < b and plan[end].attack != "coordinate_noise":
                            end += 1
                        shard = dict(job)
                        shard["start_idx"] = cursor
                        shard["stop_idx"] = end
                        shard_jobs.append(shard)
                        cursor = end

        # V13.0.14: keep all remaining heavy tail shards runnable.
        # FORCE_SINGLE was a debugging fallback in V13.0.13 and is disabled
        # by default. Existing checkpoints remain reusable.
        total_workers = max(
            1,
            min(worker_budget, len(shard_jobs))
        ) if shard_jobs else 0
        # V13.0.4: schedule expensive work early (LPT-like), but classify
        # heaviness by source size rather than attack name and account one CPU
        # scheduler slot per independent attack process.  Every completed future
        # immediately opens a slot for another task: there is NO wave barrier.
        shard_jobs.sort(key=_job_priority)

        # V13.0.14: automatic 1-3 heavy tail parallel mode.
        tail_thread_plan = {}
        if (
            1 <= len(shard_jobs) <= 3
            and all(_job_is_heavy(j) for j in shard_jobs)
        ):
            tail_thread_plan = _tail_thread_plan(shard_jobs)
            print(
                f"[TAIL RESOURCE PLAN] physical_cores={int(os.environ.get('RB_AFL_PHYSICAL_CPU_CORES','80'))} "
                f"software_threads={sum(tail_thread_plan.values())} jobs={len(shard_jobs)} "
                f"oversubscribe={os.environ.get('RB_AFL_TAIL_OVERSUBSCRIBE','1.40')}x",
                flush=True,
            )
            for j in shard_jobs:
                key = _tail_job_key(j)
                print(
                    f"  [TAIL THREADS] {key[0]} {key[1]:03d}:{key[2]:03d} "
                    f"coords={int(j.get('source_coordinate_count',0) or 0):,} threads={tail_thread_plan.get(key,1)}",
                    flush=True,
                )
        print(
            f"[ATTACK CACHE SHARDED] pending_identities={len(pending)} "
            f"new_shard_jobs={len(shard_jobs)} max_processes={total_workers} "
            f"scheduler=rolling-resource-aware-v13.0.14-final-tail-parallel heavy_cap={max(1,int(max_heavy_inflight))} "
            f"cpu_budget={worker_budget} min_available_ram={float(min_available_memory_gb):.1f}GB "
            f"large_source_threshold={int(os.environ.get('RB_AFL_LARGE_SOURCE_COORD_THRESHOLD','200000')):,}",
            flush=True,
        )
        shard_done = 0
        if shard_jobs and total_workers == 1:
            for job in shard_jobs:
                j = dict(job)
                j["channel_threads"] = max(1, min(worker_budget, 32)) if _job_is_heavy(j) else 1
                result = _cache_shard_worker(j)
                shard_done += 1
                print(
                    f"  [SHARD {shard_done:03d}/{len(shard_jobs):03d}] {result['identity']} "
                    f"{result['start_idx']:03d}:{result['stop_idx']:03d} "
                    f"{result['status']} {result['elapsed_s']:.1f}s "
                    f"[load={float(result.get('load_s',0)):.1f} attack={float(result.get('attack_s',0)):.1f} "
                    f"channel={float(result.get('channel_s',0)):.1f} rss={float(result.get('max_rss_gb',0)):.1f}GB]",
                    flush=True,
                )
        elif shard_jobs:
            queue: List[Dict[str, object]] = list(shard_jobs)
            active_workers = int(total_workers)
            recovery_no = 0
            # A pool is kept alive continuously; only a native worker crash
            # recreates it. The queue itself survives every recovery.
            while queue:
                pool = ProcessPoolExecutor(max_workers=active_workers, mp_context=get_context("spawn"))
                futures: Dict[object, Dict[str, object]] = {}
                cpu_slots_used = 0
                heavy_inflight = 0
                broken = False
                print(
                    f"  [ROLLING POOL] workers={active_workers} queued={len(queue)} "
                    f"available_ram={_available_memory_gb():.1f}GB recovery={recovery_no}",
                    flush=True,
                )
                try:
                    while queue or futures:
                        # Keep filling free CPU/process slots. If the first heavy
                        # job is temporarily blocked, scan for a normal job so
                        # CPUs never idle behind the RAM/heavy admission control.
                        submitted = True
                        while queue and submitted and len(futures) < active_workers:
                            submitted = False
                            chosen_idx = None
                            chosen_threads = None
                            for qi, candidate in enumerate(queue):
                                heavy = _job_is_heavy(candidate)
                                if heavy and heavy_inflight >= max(1, int(max_heavy_inflight)):
                                    continue
                                heavy_remaining = heavy_inflight + sum(
                                    1 for q in queue if _job_is_heavy(q)
                                )
                                normal_remaining = (len(futures) - heavy_inflight) + sum(
                                    1 for q in queue if not _job_is_heavy(q)
                                )
                                key = _tail_job_key(candidate)
                                if key in tail_thread_plan:
                                    threads = int(tail_thread_plan[key])
                                else:
                                    threads = _threads_for_job(
                                        candidate,
                                        cpu_budget=worker_budget,
                                        heavy_remaining=heavy_remaining,
                                        normal_remaining=normal_remaining,
                                        max_heavy_inflight=max_heavy_inflight,
                                    )
                                # Every submitted attack occupies one scheduler slot.
                                # channel_threads may transiently use extra cores only in
                                # the true tail; it must not block 3-5 independent jobs
                                # from being submitted while the attack/canonicalisation
                                # phase is effectively single-threaded.
                                slot_cost = 1
                                if cpu_slots_used + slot_cost > worker_budget:
                                    continue
                                # RAM admission guard applies to heavy jobs only;
                                # normal 0.8-1.1GB tasks can continue filling slots.
                                if heavy and futures and _available_memory_gb() < float(min_available_memory_gb):
                                    continue
                                chosen_idx = qi
                                chosen_threads = threads
                                break
                            if chosen_idx is None:
                                # Avoid a deadlock when the queue contains only
                                # heavy work and all current futures have drained.
                                if not futures and queue:
                                    chosen_idx = 0
                                    heavy_remaining = sum(1 for q in queue if _job_is_heavy(q))
                                    normal_remaining = len(queue) - heavy_remaining
                                    key0 = _tail_job_key(queue[0])
                                    if key0 in tail_thread_plan:
                                        chosen_threads = int(tail_thread_plan[key0])
                                    else:
                                        chosen_threads = max(1, min(
                                            worker_budget,
                                            _threads_for_job(
                                                queue[0],
                                                cpu_budget=worker_budget,
                                                heavy_remaining=heavy_remaining,
                                                normal_remaining=normal_remaining,
                                                max_heavy_inflight=max_heavy_inflight,
                                            ),
                                        ))
                                else:
                                    break
                            job = queue.pop(chosen_idx)
                            run_job = dict(job)
                            run_job["channel_threads"] = int(chosen_threads)
                            fut = pool.submit(_cache_shard_worker, run_job)
                            futures[fut] = {
                                "job": job,
                                "threads": int(chosen_threads),
                                "heavy": bool(_job_is_heavy(job)),
                            }
                            cpu_slots_used += 1
                            if _job_is_heavy(job):
                                heavy_inflight += 1
                            submitted = True

                        if not futures:
                            continue
                        done, _ = wait(tuple(futures.keys()), timeout=30.0, return_when=FIRST_COMPLETED)
                        if not done:
                            print("  [LIVE PROGRESS] no shard finished in last 30s; active internal stages:", flush=True)
                            for inf in futures.values():
                                print("    " + _format_progress_snapshot(inf["job"]), flush=True)
                            continue
                        for future in done:
                            info = futures.pop(future)
                            job = info["job"]
                            cpu_slots_used = max(0, cpu_slots_used - 1)
                            if info["heavy"]:
                                heavy_inflight = max(0, heavy_inflight - 1)
                            try:
                                result = future.result()
                            except BrokenProcessPool:
                                broken = True
                                # Requeue this and every in-flight job. A task
                                # that actually reached its done marker will be
                                # returned instantly as reused_shard after restart.
                                queue.insert(0, job)
                                for inf in futures.values():
                                    queue.insert(0, inf["job"])
                                print(
                                    f"  [POOL BROKEN] abrupt worker exit candidate={job['identity']} "
                                    f"{int(job['start_idx']):03d}:{int(job['stop_idx']):03d}; "
                                    f"requeue={len(futures)+1}",
                                    flush=True,
                                )
                                futures.clear()
                                break
                            except BaseException as exc:
                                for f in futures:
                                    f.cancel()
                                pool.shutdown(wait=False, cancel_futures=True)
                                raise RuntimeError(
                                    f"Shard failed: {job['identity']} "
                                    f"{int(job['start_idx']):03d}:{int(job['stop_idx']):03d}"
                                ) from exc

                            shard_done += 1
                            print(
                                f"  [SHARD {shard_done:03d}/{len(shard_jobs):03d}] {result['identity']} "
                                f"{result['start_idx']:03d}:{result['stop_idx']:03d} "
                                f"{result['status']} {result['elapsed_s']:.1f}s "
                                f"[thr={int(info['threads'])} load={float(result.get('load_s',0)):.1f} "
                                f"attack={float(result.get('attack_s',0)):.1f} "
                                f"channel={float(result.get('channel_s',0)):.1f} "
                                f"rss={float(result.get('max_rss_gb',0)):.1f}GB "
                                f"ram_avail={_available_memory_gb():.1f}GB]",
                                flush=True,
                            )
                        if broken:
                            break
                    if not broken:
                        pool.shutdown(wait=True)
                        queue.clear()
                    else:
                        for f in futures:
                            f.cancel()
                        pool.shutdown(wait=False, cancel_futures=True)
                        recovery_no += 1
                        if active_workers <= 1:
                            raise RuntimeError(
                                "A shard worker exited abruptly even with one process. "
                                "Inspect faulthandler output above."
                            )
                        old_workers = active_workers
                        active_workers = max(1, active_workers // 2)
                        # De-duplicate requeued jobs by identity/range.
                        dedup = {}
                        for j in queue:
                            dedup[(str(j['identity']), int(j['start_idx']), int(j['stop_idx']))] = j
                        queue = sorted(dedup.values(), key=_job_priority)
                        print(
                            f"  [POOL RECOVERY] rolling pool workers={active_workers} "
                            f"(was {old_workers}); queued={len(queue)}; completed shards remain persistent.",
                            flush=True,
                        )
                        time.sleep(1.0)
                finally:
                    pass

        print("[ATTACK CACHE ASSEMBLE] assembling incomplete identities from all persisted shards ...", flush=True)
        for job in pending:
            identity = str(job["identity"])
            _assemble_identity_from_available_shards(job)
            completed += 1
            print(f"  [ASSEMBLED {completed:02d}/{len(jobs)}] {identity}", flush=True)
    elif pending and identity_workers == 1:
        results = [_cache_identity_worker(job) for job in pending]
        for result in results:
            completed += 1
            elapsed = time.perf_counter() - started
            built_done = completed - len(reusable)
            eta = elapsed / built_done * (len(pending) - built_done) if built_done else 0.0
            print(
                f"  [CACHE {completed:02d}/{len(jobs)}] {result['identity']} "
                f"{result['status']} {result['elapsed_s']:.1f}s ETA~{eta/60:.1f}m",
                flush=True,
            )
    elif pending:
        # Spawn avoids inheriting GEOS/GDAL state into workers.
        with ProcessPoolExecutor(max_workers=identity_workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_cache_identity_worker, job): job for job in pending}
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                elapsed = time.perf_counter() - started
                built_done = completed - len(reusable)
                eta = elapsed / built_done * (len(pending) - built_done) if built_done else 0.0
                print(
                    f"  [CACHE {completed:02d}/{len(jobs)}] {result['identity']} "
                    f"{result['status']} {result['elapsed_s']:.1f}s ETA~{eta/60:.1f}m",
                    flush=True,
                )

    # Final integrity check and lightweight artifact binding. Large tensor files
    # are fully hashed later by the run-manifest stage.
    identity_artifacts: Dict[str, object] = {}
    for job in jobs:
        d = Path(str(job["cache_dir"]))
        a = d / "attacks.npy"
        m = d / "attack_metadata.csv"
        c = d / "clean.npy"
        done = d / "complete.json"
        valid, reason = _completed_cache_status(job)
        if not valid:
            raise RuntimeError(f"Attack cache invalid for {job['identity']}: {reason}")
        identity_artifacts[str(job["identity"])] = {
            "attacks_bytes": int(a.stat().st_size),
            "clean_bytes": int(c.stat().st_size),
            "metadata_sha256": file_sha256(m),
            "complete_marker_sha256": file_sha256(done),
        }
    cache_meta["identity_artifacts"] = identity_artifacts
    manifest_path.write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[ATTACK CACHE READY] {len(jobs) * len(plan):,} attacked tensors; "
        f"elapsed={(time.perf_counter()-started)/60:.1f} min; root={root}",
        flush=True,
    )
    return root


def _batch_embeddings(
    encoder,
    tensors: np.ndarray,
    channel_indices: Sequence[int],
    device: str,
    batch_size: int,
) -> Tuple[np.ndarray, float]:
    channel_indices = tuple(int(x) for x in channel_indices)
    outputs: List[np.ndarray] = []
    total_elapsed = 0.0
    n = int(tensors.shape[0])
    for start in range(0, n, max(1, int(batch_size))):
        stop = min(n, start + max(1, int(batch_size)))
        selected = np.ascontiguousarray(
            np.asarray(tensors[start:stop], dtype=np.float32)[:, list(channel_indices)]
        )
        batch = torch.from_numpy(selected).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            features = encoder(batch).detach().cpu().numpy().astype(np.float32)
        if device == "cuda":
            torch.cuda.synchronize()
        total_elapsed += time.perf_counter() - t0
        outputs.append(features)
    return np.concatenate(outputs, axis=0), total_elapsed


def evaluate_all_cached(
    prepared_root: str | Path,
    models_root: str | Path,
    watermark_path: str | Path,
    output_root: str | Path,
    cache_root: str | Path,
    *,
    selected: Optional[Sequence[str]] = None,
    grid_size: int = 256,
    density_sigma: float = 3.0,
    bit_length: int = 256,
    threshold_mode: str = "median",
    nc_threshold: Optional[float] = NC_THRESHOLD,
    seed: int = 20260730,
    device: str = "auto",
    timing_repeats: int = 10,
    identity_split_path: str | Path,
    study_split_name: str = "test",
    source_path_map: str | Path | None = None,
    eval_batch_size: int = 16,
    fixed_attack_seed: int = 20260910,
    resume: bool = True,
) -> Dict[str, Path]:
    if nc_threshold is None:
        raise ValueError("nc_threshold is required; load the frozen calibration artifact")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_root).expanduser().resolve()

    base = base_manifest(prepared_root, identity_split_path, study_split_name)
    base = _apply_source_map(base, source_path_map)
    identities = base["identity"].astype(str).tolist()
    source_paths = base["source_path"].astype(str).tolist()
    if len(identities) < 2:
        raise RuntimeError("Evaluation requires at least two identities")

    plan = attack_plan()
    cache_manifest_path = cache_root / "cache_manifest.json"
    if not cache_manifest_path.is_file():
        raise RuntimeError(f"Missing attack-cache manifest: {cache_manifest_path}")
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    expected_split_sha256 = hashlib.sha256(
        Path(identity_split_path).expanduser().resolve().read_bytes()
    ).hexdigest()
    if (
        cache_manifest.get("cache_version") != CACHE_VERSION
        or int(cache_manifest.get("fixed_attack_seed", -1)) != int(fixed_attack_seed)
        or int(cache_manifest.get("grid_size", -1)) != int(grid_size)
        or not math.isclose(
            float(cache_manifest.get("density_sigma", float("nan"))),
            float(density_sigma),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or cache_manifest.get("identity_split_sha256") != expected_split_sha256
        or cache_manifest.get("attack_plan_sha256") != _attack_plan_sha256()
    ):
        raise RuntimeError("Attack-cache manifest does not match the frozen evaluation protocol")
    external_signature = cache_manifest.get("external_signature")
    for identity in identities:
        d = cache_root / identity
        if not (d / "complete.json").is_file():
            raise RuntimeError(f"Missing fixed attack cache for {identity}: {d}")
        info = json.loads((d / "complete.json").read_text(encoding="utf-8"))
        if int(info.get("fixed_attack_seed", -1)) != int(fixed_attack_seed):
            raise RuntimeError(f"Attack cache seed mismatch for {identity}")
    for identity, source_path in zip(identities, source_paths):
        valid, reason = _completed_cache_status(
            {
                "identity": identity,
                "source_path": source_path,
                "cache_dir": str(cache_root / identity),
                "grid_size": int(grid_size),
                "density_sigma": float(density_sigma),
                "fixed_attack_seed": int(fixed_attack_seed),
                "attack_plan_sha256": _attack_plan_sha256(),
                "source_signature": _source_signature(source_path),
                "external_signature": external_signature,
            }
        )
        if not valid:
            raise RuntimeError(f"Attack cache integrity check failed for {identity}: {reason}")

    protocol_payload = protocol_dict()
    if isinstance(protocol_payload.get("decision_rule"), dict):
        protocol_payload["decision_rule"]["threshold"] = float(nc_threshold)
        protocol_payload["decision_rule"]["threshold_source"] = "held-out calibration split"
    protocol_payload["evaluation"] = {
        "seed": int(seed),
        "nc_threshold": float(nc_threshold),
        "identity_split_file": Path(identity_split_path).name,
        "study_split_name": study_split_name,
        "source_path_map_file": Path(source_path_map).name if source_path_map else "",
        "attack_cache_version": CACHE_VERSION,
        "fixed_attack_seed": int(fixed_attack_seed),
        "attack_seed_definition": "stable_seed(fixed_attack_seed, identity, attack, strength, repeat_index)",
        "paired_test_design": "same attack realization across experiments and training seeds",
        "eval_batch_size": int(eval_batch_size),
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    watermark_bits, watermark_width, watermark_height = watermark_image_to_bits(watermark_path, bit_length)
    if np.all(watermark_bits == watermark_bits[0]):
        raise ValueError("Degenerate copyright watermark: all bits are identical")
    model_index = _checkpoint_rows(models_root, selected)

    # Measure model-independent clean vector read + four-channel construction once
    # per identity for this evaluation seed. Reusing these timings across E1-E7
    # avoids repeating GIS work while preserving the original end-to-end timing
    # definition used by the publication tables.
    clean_preprocess_timing: Dict[str, Tuple[float, float]] = {}
    print("[FAST TIMING] measuring clean vector read/channel build once per test identity ...", flush=True)
    for identity, source_path in zip(identities, source_paths):
        gdf, read_s = _timed_call(lambda p=source_path: read_vector(p))
        _, channel_s = _timed_call(
            lambda g=gdf: build_four_channels(
                g,
                grid_size=int(grid_size),
                density_sigma=float(density_sigma),
            )
        )
        clean_preprocess_timing[identity] = (float(read_s), float(channel_s))

    all_ablation_rows: List[Dict[str, object]] = []
    all_timing_rows: List[Dict[str, object]] = []
    all_robust_rows: List[Dict[str, object]] = []
    all_unique_rows: List[Dict[str, object]] = []
    all_zero_watermark_rows: List[Dict[str, object]] = []

    for model_idx, model_row in model_index.iterrows():
        exp_id = str(model_row["exp_id"])
        exp_name = str(model_row["exp_name"])
        checkpoint_path = Path(str(model_row["checkpoint"]))
        exp_dir = output / f"{exp_id}_{exp_name}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        if resume and _experiment_complete(exp_dir, len(identities), len(plan), nc_threshold):
            print(f"[FAST EVALUATE] {exp_id} {exp_name}: completed -> reuse", flush=True)
            summary_row, timing_old, robust_old, unique_old = _load_completed_experiment(exp_dir)
            all_ablation_rows.append(summary_row)
            all_timing_rows.extend(timing_old)
            all_robust_rows.extend(robust_old)
            all_unique_rows.extend(unique_old)
            zero_stats_path = exp_dir / "zero_watermark_stats.csv"
            if zero_stats_path.is_file():
                all_zero_watermark_rows.extend(pd.read_csv(zero_stats_path).to_dict("records"))
            continue

        print(
            f"[FAST EVALUATE] model {model_idx+1}/{len(model_index)} {exp_id} {exp_name}",
            flush=True,
        )
        encoder, checkpoint, resolved = load_encoder(checkpoint_path, device)
        experiment = checkpoint["experiment"]
        channel_indices = tuple(int(x) for x in experiment["channel_indices"])
        parameter_bytes = encoder_parameter_bytes(encoder)

        # Clean registration/uniqueness. Clean tensors are model-independent and cached.
        clean_bits: List[np.ndarray] = []
        records: List[Dict[str, object]] = []
        timing_rows: List[Dict[str, object]] = []
        for identity, source_path in zip(identities, source_paths):
            tensor = np.load(cache_root / identity / "clean.npy", allow_pickle=False)
            embedding, encode_once_s = _timed_call(
                lambda: extract_embedding(encoder, tensor, channel_indices, resolved), resolved
            )
            median_encode_s = _median_encode_time(
                encoder, tensor, channel_indices, resolved, timing_repeats
            )
            bits, quantize_s = _timed_call(
                lambda: feature_to_bits(embedding, bit_length, threshold_mode)
            )
            clean_bits.append(bits)
            selected_tensor_bytes = int(tensor[list(channel_indices)].nbytes)
            space = {
                "selected_channel_tensor_bytes": selected_tensor_bytes,
                "encoder_parameter_bytes": parameter_bytes,
                "embedding_bytes": int(embedding.nbytes),
                "registered_zero_watermark_payload_bytes": int(math.ceil(bit_length / 8)),
                "working_memory_bytes_excluding_model": selected_tensor_bytes + int(embedding.nbytes) + int(bits.nbytes),
            }
            t0 = time.perf_counter()
            record = create_registry_record(
                identity=identity,
                source_path=source_path,
                checkpoint_path=checkpoint_path,
                watermark_bits=watermark_bits,
                feature_bits=bits,
                experiment=experiment,
                channel_config={
                    "grid_size": int(tensor.shape[-1]),
                    "density_sigma": float(density_sigma),
                    "canonicalization": "centroid + PCA dominant axis + isotropic extent",
                    "watermark_width": watermark_width,
                    "watermark_height": watermark_height,
                    "source": "fixed clean tensor cache",
                },
                threshold_mode=threshold_mode,
                space=space,
            )
            xor_s = time.perf_counter() - t0
            record_json_bytes = len(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            space["registry_record_json_bytes"] = record_json_bytes
            space["additional_storage_bytes_actual_per_record"] = record_json_bytes
            # Clean vector read/channel construction is model-independent, so it is
            # measured once per identity and reused across E1-E7. Model inference
            # remains measured separately for every checkpoint.
            read_s, channel_s = clean_preprocess_timing[identity]
            timing = {
                "vector_read_time_s": read_s,
                "channel_build_time_s": channel_s,
                "feature_extract_time_single_s": encode_once_s,
                "feature_extract_time_median_s": median_encode_s,
                "feature_quantization_time_s": quantize_s,
                "xor_and_record_build_time_s": xor_s,
                "zero_watermark_generation_time_s_excluding_read": channel_s + median_encode_s + quantize_s + xor_s,
                "zero_watermark_generation_time_s_including_read": read_s + channel_s + median_encode_s + quantize_s + xor_s,
            }
            record["timing"] = timing
            record["space"] = space
            records.append(record)
            timing_rows.append(
                {
                    "evaluation_seed": int(seed),
                    "study_split": study_split_name,
                    "exp_id": exp_id,
                    "exp_name": exp_name,
                    "identity": identity,
                    "channels": ",".join(experiment["channels"]),
                    **timing,
                    **space,
                }
            )

        registry_path = save_registry(exp_dir / "registry.json", records)
        zero_rows: List[Dict[str, object]] = []
        for record in records:
            zstats = dict(record.get("zero_watermark_stats", {}))
            fstats = dict(record.get("feature_bit_stats", {}))
            wstats = dict(record.get("copyright_watermark_stats", {}))
            zero_rows.append({
                "evaluation_seed": int(seed),
                "study_split": study_split_name,
                "exp_id": exp_id,
                "exp_name": exp_name,
                "identity": str(record.get("identity", "")),
                "zero_watermark_zero_bits": zstats.get("zero_bits"),
                "zero_watermark_one_bits": zstats.get("one_bits"),
                "zero_watermark_one_ratio": zstats.get("one_ratio"),
                "zero_watermark_entropy_bits": zstats.get("binary_entropy_bits"),
                "zero_watermark_minority_bit_ratio": zstats.get("minority_bit_ratio"),
                "zero_watermark_is_all_zero": zstats.get("is_all_zero"),
                "zero_watermark_is_all_one": zstats.get("is_all_one"),
                "feature_one_ratio": fstats.get("one_ratio"),
                "feature_entropy_bits": fstats.get("binary_entropy_bits"),
                "copyright_one_ratio": wstats.get("one_ratio"),
                "copyright_entropy_bits": wstats.get("binary_entropy_bits"),
            })
        pd.DataFrame(zero_rows).to_csv(
            exp_dir / "zero_watermark_stats.csv", index=False, encoding="utf-8-sig"
        )
        all_zero_watermark_rows.extend(zero_rows)
        timing_df = pd.DataFrame(timing_rows)
        timing_df.to_csv(exp_dir / "timing_space.csv", index=False, encoding="utf-8-sig")
        all_timing_rows.extend(timing_rows)

        nc_matrix, ber_matrix, unique_summary, unique_rows = _uniqueness_matrices(
            identities, records, clean_bits, watermark_bits, nc_threshold
        )
        nc_matrix.to_csv(exp_dir / "uniqueness_nc_matrix.csv", encoding="utf-8-sig")
        ber_matrix.to_csv(exp_dir / "uniqueness_ber_matrix.csv", encoding="utf-8-sig")
        for row in unique_rows:
            row.update({
                "evaluation_seed": int(seed),
                "study_split": study_split_name,
                "exp_id": exp_id,
                "exp_name": exp_name,
            })
        pd.DataFrame(unique_rows).to_csv(exp_dir / "uniqueness_rows.csv", index=False, encoding="utf-8-sig")
        all_unique_rows.extend(unique_rows)

        robust_path_child = exp_dir / "robustness_rows.csv"
        robust_map: Dict[Tuple[str, str], Dict[str, object]] = {}
        if resume and robust_path_child.is_file():
            try:
                old = pd.read_csv(robust_path_child)
                for row in old.to_dict("records"):
                    if str(row.get("status", "")) == "ok" and int(row.get("fixed_attack_seed", -1)) == int(fixed_attack_seed):
                        robust_map[(str(row.get("identity", "")), str(row.get("case_id", "")))] = row
            except Exception:
                robust_map = {}

        model_start = time.perf_counter()
        for identity_idx, (identity, record) in enumerate(zip(identities, records), start=1):
            attack_tensor = np.load(cache_root / identity / "attacks.npy", mmap_mode="r", allow_pickle=False)
            meta_df = pd.read_csv(cache_root / identity / "attack_metadata.csv").fillna("")
            missing_indices = [i for i, case in enumerate(plan) if (identity, case.case_id) not in robust_map]
            if missing_indices:
                # Use contiguous groups where possible; at restart, evaluate only missing rows.
                for batch_start in range(0, len(missing_indices), max(1, int(eval_batch_size))):
                    idxs = missing_indices[batch_start: batch_start + max(1, int(eval_batch_size))]
                    tensors = np.asarray(attack_tensor[idxs], dtype=np.float32)
                    features, batch_elapsed = _batch_embeddings(
                        encoder, tensors, channel_indices, resolved, eval_batch_size
                    )
                    per_feature_s = batch_elapsed / max(1, len(idxs))
                    for local_i, case_i in enumerate(idxs):
                        case = plan[case_i]
                        m = meta_df.iloc[case_i]
                        t0 = time.perf_counter()
                        bits = feature_to_bits(features[local_i], bit_length, threshold_mode)
                        quantize_s = time.perf_counter() - t0
                        t1 = time.perf_counter()
                        recovered = recover_watermark(record, bits)
                        recover_s = time.perf_counter() - t1
                        nc = nc_score(watermark_bits, recovered)
                        ber = ber_score(watermark_bits, recovered)
                        attack_s = float(m["attack_time_s"])
                        channel_s = float(m["channel_build_time_s"])
                        robust_map[(identity, case.case_id)] = {
                            "evaluation_seed": int(seed),
                            "study_split": study_split_name,
                            "exp_id": exp_id,
                            "exp_name": exp_name,
                            "identity": identity,
                            "attack": case.attack,
                            "strength": case.strength,
                            "repeat_index": int(case.repeat_index),
                            "case_id": case.case_id,
                            "attack_seed": int(m["attack_seed"]),
                            "fixed_attack_seed": int(fixed_attack_seed),
                            "status": "ok",
                            "nc": float(nc),
                            "ber": float(ber),
                            "bit_accuracy": float(1.0 - ber),
                            "passed": bool(nc >= nc_threshold),
                            "threshold": float(nc_threshold),
                            "attack_time_s": attack_s,
                            "channel_build_time_s": channel_s,
                            "feature_extract_time_s": float(per_feature_s),
                            "feature_quantization_time_s": float(quantize_s),
                            "watermark_recovery_time_s": float(recover_s),
                            "verification_time_s": float(attack_s + channel_s + per_feature_s + quantize_s + recover_s),
                            "attack_metadata": str(m["attack_metadata"]),
                            "error": "",
                            "evaluation_mode": "fixed_attack_cache_batched_inference",
                        }
            pd.DataFrame(list(robust_map.values())).to_csv(
                robust_path_child, index=False, encoding="utf-8-sig"
            )
            elapsed = time.perf_counter() - model_start
            eta = elapsed / identity_idx * (len(identities) - identity_idx)
            print(
                f"  [{exp_id}] identity {identity_idx:02d}/{len(identities)} {identity} "
                f"rows={len(robust_map):,}/{len(identities)*len(plan):,} ETA~{eta/60:.1f}m",
                flush=True,
            )

        ordered_rows = [robust_map[(identity, case.case_id)] for identity in identities for case in plan]
        robust_df = pd.DataFrame(ordered_rows)
        robust_df.to_csv(robust_path_child, index=False, encoding="utf-8-sig")
        ok = robust_df[robust_df["status"] == "ok"].copy()
        summary = (
            ok.groupby(["attack", "strength"], as_index=False)
            .agg(
                sample_count=("nc", "count"),
                nc_mean=("nc", "mean"),
                nc_std=("nc", "std"),
                nc_min=("nc", "min"),
                ber_mean=("ber", "mean"),
                pass_rate=("passed", "mean"),
                verification_time_mean_s=("verification_time_s", "mean"),
            )
            .fillna({"nc_std": 0.0})
        )
        summary.to_csv(exp_dir / "robustness_summary.csv", index=False, encoding="utf-8-sig")
        all_robust_rows.extend(ordered_rows)

        # Random deletion masks are repeated ten times. Reduce those repeats to
        # one identity-by-attack-strength unit before computing an overall
        # ablation metric, so deletion is not weighted ten times more heavily
        # than every deterministic setting.
        robust_units = (
            ok.groupby(["identity", "attack", "strength"], as_index=False)
            .agg(
                nc=("nc", "mean"),
                ber=("ber", "mean"),
                passed=("passed", "mean"),
            )
        )
        time_means = timing_df.mean(numeric_only=True)
        ablation_row: Dict[str, object] = {
            "evaluation_seed": int(seed),
            "study_split": study_split_name,
            "threshold": float(nc_threshold),
            "exp_id": exp_id,
            "exp_name": exp_name,
            "description": experiment["description"],
            "channels": ",".join(experiment["channels"]),
            "num_channels": len(experiment["channels"]),
            "lambda_consistency": experiment["lambda_consistency"],
            "lambda_triplet": experiment["lambda_triplet"],
            "identity_count": len(identities),
            "robust_nc_mean": float(robust_units["nc"].mean()),
            "robust_nc_std": float(robust_units["nc"].std(ddof=1)) if len(robust_units) > 1 else 0.0,
            "robust_nc_min": float(robust_units["nc"].min()),
            "robust_ber_mean": float(robust_units["ber"].mean()),
            "robust_pass_rate": float(robust_units["passed"].mean()),
            "robustness_aggregation_unit_count": int(len(robust_units)),
            "robustness_aggregation_unit": "all identity x attack family x strength conditions, including clean; masks averaged first",
            "completed_attack_evaluations": int(len(ok)),
            "failed_attack_evaluations": 0,
            "skipped_attack_evaluations": 0,
            **unique_summary,
            "zero_watermark_time_including_read_mean_s": float(time_means["zero_watermark_generation_time_s_including_read"]),
            "zero_watermark_time_excluding_read_mean_s": float(time_means["zero_watermark_generation_time_s_excluding_read"]),
            "feature_extract_time_median_mean_s": float(time_means["feature_extract_time_median_s"]),
            "selected_channel_tensor_bytes_mean": float(time_means["selected_channel_tensor_bytes"]),
            "encoder_parameter_bytes": parameter_bytes,
            "registered_zero_watermark_payload_bytes": int(math.ceil(bit_length / 8)),
            "registry_file_bytes": registry_path.stat().st_size,
            "zero_watermark_entropy_mean": float(pd.DataFrame(zero_rows)["zero_watermark_entropy_bits"].astype(float).mean()),
            "zero_watermark_entropy_min": float(pd.DataFrame(zero_rows)["zero_watermark_entropy_bits"].astype(float).min()),
            "zero_watermark_one_ratio_min": float(pd.DataFrame(zero_rows)["zero_watermark_one_ratio"].astype(float).min()),
            "zero_watermark_one_ratio_max": float(pd.DataFrame(zero_rows)["zero_watermark_one_ratio"].astype(float).max()),
            "zero_watermark_degenerate_count": int(
                pd.DataFrame(zero_rows)[["zero_watermark_is_all_zero", "zero_watermark_is_all_one"]]
                .fillna(False).astype(bool).any(axis=1).sum()
            ),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "watermark_sha256": bit_sha256(watermark_bits),
            "fixed_attack_seed": int(fixed_attack_seed),
            "evaluation_mode": "fixed_attack_cache_batched_inference",
        }
        all_ablation_rows.append(ablation_row)
        (exp_dir / "summary.json").write_text(json.dumps(ablation_row, ensure_ascii=False, indent=2), encoding="utf-8")

    ablation_path = output / "ablation_results.csv"
    pd.DataFrame(all_ablation_rows).sort_values("exp_id").to_csv(ablation_path, index=False, encoding="utf-8-sig")
    timing_path = output / "timing_space_all.csv"
    pd.DataFrame(all_timing_rows).to_csv(timing_path, index=False, encoding="utf-8-sig")
    robustness_path = output / "robustness_rows_all.csv"
    pd.DataFrame(all_robust_rows).to_csv(robustness_path, index=False, encoding="utf-8-sig")
    uniqueness_path = output / "uniqueness_rows_all.csv"
    pd.DataFrame(all_unique_rows).to_csv(uniqueness_path, index=False, encoding="utf-8-sig")
    zero_stats_path = output / "zero_watermark_stats_all.csv"
    pd.DataFrame(all_zero_watermark_rows).to_csv(zero_stats_path, index=False, encoding="utf-8-sig")
    return {
        "ablation_results": ablation_path,
        "timing_space": timing_path,
        "robustness_rows": robustness_path,
        "uniqueness_rows": uniqueness_path,
        "zero_watermark_stats": zero_stats_path,
    }
