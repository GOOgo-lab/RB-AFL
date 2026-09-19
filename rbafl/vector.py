# V13.0.12_FAST_GEOM_ACTIVE
from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import shapely


# ===============================
# V13.0.12 FINAL FAST GEOMETRY ENGINE
# Geometry-aware MakeValid optimization V13.0.12
# ===============================

FAST_MAKEVALID_BATCH = int(os.environ.get("RB_AFL_MAKEVALID_BATCH", "2048"))

_MAKEVALID_PROFILE = {
    "total": 0,
    "line_skip": 0,
    "point_skip": 0,
    "polygon_checked": 0,
    "polygon_repaired": 0,
    "failed": 0,
    "time": 0.0,
}


def fast_make_valid(arr):
    """
    V13.0.10:
    - LineString/MultiLineString: keep directly
    - Point/MultiPoint: keep directly
    - Polygon/MultiPolygon: strict MakeValid

    Only changes geometry repair implementation.
    Attack generation and evaluation logic remain unchanged.
    """
    import numpy as np
    import shapely
    import time

    t0 = time.time()

    arr = np.asarray(arr, dtype=object)

    if len(arr) == 0:
        return arr

    result = []

    for geom in arr:
        _MAKEVALID_PROFILE["total"] += 1

        try:
            if geom is None or geom.is_empty:
                result.append(None)
                continue

            gtype = geom.geom_type

            if gtype in ("LineString", "MultiLineString"):
                _MAKEVALID_PROFILE["line_skip"] += 1
                result.append(geom)
                continue

            if gtype in ("Point", "MultiPoint"):
                _MAKEVALID_PROFILE["point_skip"] += 1
                result.append(geom)
                continue

            if gtype in ("Polygon", "MultiPolygon"):
                _MAKEVALID_PROFILE["polygon_checked"] += 1

                if geom.is_valid:
                    result.append(geom)
                else:
                    result.append(shapely.make_valid(geom))
                    _MAKEVALID_PROFILE["polygon_repaired"] += 1
                continue

            result.append(shapely.make_valid(geom))

        except Exception:
            _MAKEVALID_PROFILE["failed"] += 1
            try:
                result.append(geom.buffer(0))
            except Exception:
                result.append(None)

    _MAKEVALID_PROFILE["time"] += time.time() - t0

    return np.asarray(result, dtype=object)


def print_makevalid_profile():
    print("\n========== MAKEVALID PROFILE ==========")
    for k, v in _MAKEVALID_PROFILE.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.3f}s")
        else:
            print(f"{k:25s}: {v}")
    print("=======================================\n")

try:  # Only the explicitly named legacy raster builders need rasterio.
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
except ImportError:
    rasterize = from_bounds = None
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely import affinity
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False

from shapely.geometry import (
    GeometryCollection,
    LineString,
    LinearRing,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    box,
)


VECTOR_SUFFIXES = {".shp", ".geojson", ".json", ".gpkg"}


def _emit_progress(progress_callback, stage: str, **detail) -> None:
    """Best-effort progress hook; never changes scientific computation."""
    if progress_callback is None:
        return
    try:
        progress_callback(stage=stage, **detail)
    except Exception:
        # Telemetry must never be able to fail an experiment.
        pass



@dataclass(frozen=True)
class VectorSample:
    identity: str
    path: Path


def stable_seed(*parts: object) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & 0x7FFFFFFF


def discover_vector_files(source_root: str | Path) -> List[VectorSample]:
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in VECTOR_SUFFIXES)
    if not files:
        raise RuntimeError(f"No vector files found below {root}")
    samples: List[VectorSample] = []
    used: Dict[str, int] = {}
    for path in files:
        rel = path.relative_to(root).with_suffix("")
        base = "__".join(rel.parts)
        base = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in base).strip("_")
        base = base or path.stem
        count = used.get(base, 0)
        used[base] = count + 1
        identity = base if count == 0 else f"{base}_{count + 1}"
        samples.append(VectorSample(identity=identity, path=path))
    return samples


def read_vector(path: str | Path) -> gpd.GeoDataFrame:
    """Read only geometry, because all attribute columns are discarded downstream.

    Pyogrio's geometry-only path materially reduces I/O and memory on very large
    water-system layers.  The fallback preserves compatibility with older
    GeoPandas/Fiona installations.
    """
    path = Path(path)
    try:
        gdf = gpd.read_file(path, columns=[], engine="pyogrio")
    except Exception:
        try:
            gdf = gpd.read_file(path, columns=[])
        except Exception:
            gdf = gpd.read_file(path)
    return sanitize_gdf(gdf)


def _valid_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    try:
        if not geom.is_valid:
            geom = fast_make_valid(np.asarray([geom], dtype=object))[0]
        if geom is None or geom.is_empty:
            return None
        return geom
    except Exception:
        try:
            repaired = geom.buffer(0)
            return None if repaired.is_empty else repaired
        except Exception:
            return None


def _sanitize_gdf_legacy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reference implementation retained for exact-equivalence tests/fallback."""
    if gdf is None or "geometry" not in gdf:
        raise ValueError("A GeoDataFrame with a geometry column is required")
    crs = gdf.crs
    geoms = [_valid_geometry(g) for g in gdf.geometry]
    clean = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    clean = clean[clean.geometry.notna() & ~clean.geometry.is_empty].reset_index(drop=True)
    if clean.empty:
        raise RuntimeError("The vector dataset contains no usable geometry")
    return clean


def sanitize_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Vectorized geometry validation with the same geometry semantics as legacy.

    The old implementation called ``is_valid`` once per Python object.  Shapely
    2.x can validate/make-valid an object array in compiled code, which is much
    faster on water-system layers containing tens/hundreds of thousands of
    features.  Any vectorized failure falls back to the legacy implementation.
    """
    if gdf is None or "geometry" not in gdf:
        raise ValueError("A GeoDataFrame with a geometry column is required")
    crs = gdf.crs
    try:
        arr = np.asarray(gdf.geometry.array, dtype=object).copy()
        if arr.size == 0:
            raise RuntimeError("The vector dataset contains no usable geometry")
        # Shapely 2.x handles missing/empty checks in compiled vectorized code.
        # Avoid Python iteration over hundreds of thousands of water features.
        try:
            present = ~np.asarray(shapely.is_missing(arr), dtype=bool)
        except Exception:
            present = np.fromiter((g is not None for g in arr), dtype=bool, count=len(arr))
        active = present.copy()
        if np.any(active):
            empty = np.zeros(len(arr), dtype=bool)
            empty[active] = np.asarray(shapely.is_empty(arr[active]), dtype=bool)
            active &= ~empty
        if np.any(active):
            valid = np.ones(len(arr), dtype=bool)
            valid[active] = np.asarray(shapely.is_valid(arr[active]), dtype=bool)
            bad = active & ~valid
            if np.any(bad):
                arr[bad] = fast_make_valid(arr[bad])
        try:
            keep = (~np.asarray(shapely.is_missing(arr), dtype=bool)) & (~np.asarray(shapely.is_empty(arr), dtype=bool))
        except Exception:
            keep = np.fromiter(
                (g is not None and not g.is_empty for g in arr),
                dtype=bool,
                count=len(arr),
            )
        clean = gpd.GeoDataFrame(geometry=arr[keep], crs=crs).reset_index(drop=True)
        if clean.empty:
            raise RuntimeError("The vector dataset contains no usable geometry")
        return clean
    except RuntimeError:
        raise
    except Exception:
        return _sanitize_gdf_legacy(gdf)



def _sanitize_gdf_parallel(
    gdf: gpd.GeoDataFrame,
    threads: int = 1,
    progress_callback=None,
    checkpoint_dir: Optional[str | Path] = None,
    checkpoint_signature: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Fine-grained threaded sanitize with resumable micro-chunk checkpoints.

    Scientific semantics are unchanged: each feature is validated independently
    using the same :func:`sanitize_gdf` implementation, and cleaned chunks are
    concatenated in original feature order. V13.0.8 adds two engineering layers:

    * many coordinate-balanced micro-chunks for better GEOS load balancing;
    * atomic WKB checkpoints for every completed micro-chunk, so an interrupted
      million-vertex coordinate-noise case can resume inside MakeValid instead
      of repeating hours of already-completed work.

    WKB stores IEEE-754 coordinates losslessly for the geometries used here and
    is used only as a local persistence format; no RNG or geometry operation is
    changed by the checkpoint mechanism.
    """
    threads = max(1, int(threads))
    if threads <= 1 or len(gdf) < 512:
        return sanitize_gdf(gdf)

    crs = gdf.crs
    arr = np.asarray(gdf.geometry.array, dtype=object)
    if arr.size == 0:
        raise RuntimeError("The vector dataset contains no usable geometry")

    micro_factor = max(2, int(os.environ.get("RB_AFL_SANITIZE_MICROCHUNKS_PER_THREAD", "8")))
    target_chunks = min(len(arr), max(threads, threads * micro_factor))

    try:
        coord_counts = np.asarray(shapely.get_num_coordinates(arr), dtype=np.int64)
        total_coords = int(coord_counts.sum(dtype=np.int64))
    except Exception:
        coord_counts = np.ones(len(arr), dtype=np.int64)
        total_coords = int(len(arr))

    if total_coords > 0:
        ends = np.cumsum(coord_counts, dtype=np.int64)
        targets = np.linspace(0, total_coords, target_chunks + 1)
        boundaries = [0]
        for t in targets[1:-1]:
            j = int(np.searchsorted(ends, t, side="left") + 1)
            j = max(boundaries[-1] + 1, min(j, len(arr) - 1))
            boundaries.append(j)
        boundaries.append(len(arr))
        b2 = [boundaries[0]]
        for x in boundaries[1:]:
            if x > b2[-1]:
                b2.append(x)
        boundaries = b2
    else:
        edges = np.linspace(0, len(arr), target_chunks + 1, dtype=np.int64)
        boundaries = [int(edges[0])]
        for x in edges[1:]:
            x = int(x)
            if x > boundaries[-1]:
                boundaries.append(x)
        if boundaries[-1] != len(arr):
            boundaries.append(len(arr))

    pairs = [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]
    total_features = int(len(arr))
    pair_coord_counts = [int(coord_counts[a:b].sum(dtype=np.int64)) for a, b in pairs]

    # Checkpoint layout is deterministic for the exact case + boundaries.
    cp_dir = Path(checkpoint_dir) if checkpoint_dir else None
    cp_enabled = cp_dir is not None and bool(checkpoint_signature)
    manifest_payload = {
        "version": "v13.0.8-sanitize-microchunks-v1",
        "signature": str(checkpoint_signature or ""),
        "feature_count": total_features,
        "coordinate_count": total_coords,
        "boundaries": [int(x) for x in boundaries],
        "micro_factor": int(micro_factor),
    }
    if cp_enabled:
        cp_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cp_dir / "sanitize_manifest.json"
        compatible = False
        if manifest_path.is_file():
            try:
                compatible = json.loads(manifest_path.read_text(encoding="utf-8")) == manifest_payload
            except Exception:
                compatible = False
        if not compatible:
            for q in cp_dir.glob("sanitize_chunk_*.npy"):
                try:
                    q.unlink()
                except Exception:
                    pass
            tmp = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
            tmp.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, manifest_path)

    def _chunk_path(i: int, a: int, b: int) -> Optional[Path]:
        if not cp_enabled:
            return None
        return cp_dir / f"sanitize_chunk_{i:04d}_{a}_{b}.npy"

    def _load_checkpoint_chunk(i: int, a: int, b: int):
        path = _chunk_path(i, a, b)
        if path is None or not path.is_file():
            return None
        try:
            wkb = np.load(path, allow_pickle=True)
            geoms = np.asarray(shapely.from_wkb(wkb), dtype=object)
            return a, b, geoms
        except Exception:
            try:
                path.unlink()
            except Exception:
                pass
            return None

    def _save_checkpoint_chunk(i: int, a: int, b: int, cleaned: np.ndarray) -> None:
        path = _chunk_path(i, a, b)
        if path is None:
            return
        wkb = np.asarray(shapely.to_wkb(np.asarray(cleaned, dtype=object)), dtype=object)
        tmp = path.with_name(path.name + f".tmp.{os.getpid()}.{threading.get_ident()}")
        with tmp.open("wb") as fh:
            np.save(fh, wkb, allow_pickle=True)
        os.replace(tmp, path)

    def _clean_slice(item):
        i, a, b = map(int, item)
        chunk = gpd.GeoDataFrame(
            geometry=np.asarray(arr[a:b], dtype=object).copy(), crs=crs
        )
        out = sanitize_gdf(chunk)
        cleaned = np.asarray(out.geometry.array, dtype=object)
        _save_checkpoint_chunk(i, a, b, cleaned)
        return i, a, b, cleaned

    done_count = 0
    done_features = 0
    done_coords = 0
    fresh_done_coords = 0
    results = []
    pending_items = []
    resumed_chunks = 0
    for i, (a, b) in enumerate(pairs):
        loaded = _load_checkpoint_chunk(i, a, b)
        if loaded is not None:
            aa, bb, cleaned = loaded
            results.append((i, aa, bb, cleaned))
            done_count += 1
            done_features += int(b - a)
            done_coords += int(pair_coord_counts[i])
            resumed_chunks += 1
        else:
            pending_items.append((i, a, b))

    sanitize_started = time.perf_counter()

    def _detail_text(pending_count: int) -> str:
        coord_pct = (100.0 * done_coords / max(1, total_coords)) if total_coords else 100.0
        feat_pct = 100.0 * done_features / max(1, total_features)
        active = min(threads, max(0, pending_count))
        elapsed = max(1e-9, time.perf_counter() - sanitize_started)
        if fresh_done_coords > 0 and total_coords > done_coords:
            rate = fresh_done_coords / elapsed
            eta_s = (total_coords - done_coords) / max(rate, 1e-9)
            eta_text = f"; rate={rate:,.0f} coords/s; ETA~{eta_s/60:.1f}m"
        elif done_coords >= total_coords and total_coords > 0:
            eta_text = "; ETA~0.0m"
        else:
            eta_text = "; ETA=warming-up"
        cp_text = f"; checkpoint={resumed_chunks}/{len(pairs)} reused" if cp_enabled else ""
        return (
            f"microchunks={done_count}/{len(pairs)}; "
            f"features={done_features:,}/{total_features:,} ({feat_pct:.1f}%); "
            f"coords={done_coords:,}/{total_coords:,} ({coord_pct:.1f}%); "
            f"active_workers~{active}; pool_threads={threads}{cp_text}" + eta_text
        )

    _emit_progress(
        progress_callback,
        "attack.sanitize",
        done=done_count,
        total=len(pairs),
        threads=threads,
        detail=_detail_text(len(pending_items)),
    )

    if pending_items:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            future_to_meta = {}
            for item in pending_items:
                i, a, b = item
                fut = ex.submit(_clean_slice, item)
                future_to_meta[fut] = (i, a, b, pair_coord_counts[i])
            pending = set(future_to_meta)

            while pending:
                completed, pending = wait(pending, timeout=5.0, return_when=FIRST_COMPLETED)
                if completed:
                    ready = set(completed)
                    ready.update(f for f in list(pending) if f.done())
                    pending.difference_update(ready)
                    for fut in ready:
                        i, a, b, ncoords = future_to_meta[fut]
                        try:
                            ii, aa, bb, cleaned = fut.result()
                        except Exception:
                            for other in pending:
                                other.cancel()
                            raise
                        results.append((ii, aa, bb, cleaned))
                        done_count += 1
                        done_features += int(b - a)
                        done_coords += int(ncoords)
                        fresh_done_coords += int(ncoords)

                _emit_progress(
                    progress_callback,
                    "attack.sanitize",
                    done=done_count,
                    total=len(pairs),
                    threads=threads,
                    detail=_detail_text(len(pending)),
                )

    results.sort(key=lambda x: x[1])
    merged = (
        np.concatenate([x[3] for x in results])
        if results
        else np.empty((0,), dtype=object)
    )
    clean = gpd.GeoDataFrame(geometry=merged, crs=crs).reset_index(drop=True)
    if clean.empty:
        raise RuntimeError("The vector dataset contains no usable geometry")
    return clean

def _iter_parts(geom) -> Iterator:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
        for part in geom.geoms:
            yield from _iter_parts(part)
    else:
        yield geom


def _iter_coordinate_sequences(geom) -> Iterator[List[Tuple[float, float]]]:
    for part in _iter_parts(geom):
        if isinstance(part, Point):
            yield [(float(part.x), float(part.y))]
        elif isinstance(part, (LineString, LinearRing)):
            yield [(float(x), float(y)) for x, y, *_ in part.coords]
        elif isinstance(part, Polygon):
            yield [(float(x), float(y)) for x, y, *_ in part.exterior.coords]
            for ring in part.interiors:
                yield [(float(x), float(y)) for x, y, *_ in ring.coords]


def _all_coordinates_legacy(gdf: gpd.GeoDataFrame, max_points: int = 250_000) -> np.ndarray:
    pieces: List[np.ndarray] = []
    count = 0
    for geom in gdf.geometry:
        for seq in _iter_coordinate_sequences(geom):
            arr = np.asarray(seq, dtype=np.float64)
            if arr.size:
                pieces.append(arr[:, :2])
                count += arr.shape[0]
    if not pieces:
        raise RuntimeError("No coordinates found")
    coords = np.concatenate(pieces, axis=0)
    if count > max_points:
        idx = np.linspace(0, count - 1, max_points, dtype=np.int64)
        coords = coords[idx]
    return coords


def all_coordinates(gdf: gpd.GeoDataFrame, max_points: int = 250_000) -> np.ndarray:
    """Return coordinates in storage order using Shapely's compiled bulk path.

    Water-system layers may contain more than a million vertices.  Iterating
    every geometry/coordinate in Python was one of the main V13.0.2 bottlenecks
    because canonicalization is repeated for every attack.  Shapely 2.x exposes
    all coordinates from an object array in compiled code and preserves geometry
    storage order, including polygon rings and multipart members.  The same
    deterministic linspace down-sampling as the reference path is retained.
    """
    try:
        arr = np.asarray(gdf.geometry.array, dtype=object)
        coords = np.asarray(shapely.get_coordinates(arr, include_z=False), dtype=np.float64)
        if coords.size == 0:
            raise RuntimeError("No coordinates found")
        coords = coords[:, :2]
        count = int(len(coords))
        if count > max_points:
            idx = np.linspace(0, count - 1, max_points, dtype=np.int64)
            coords = coords[idx]
        return coords
    except RuntimeError:
        raise
    except Exception:
        return _all_coordinates_legacy(gdf, max_points=max_points)


def dataset_extent(gdf: gpd.GeoDataFrame) -> Tuple[float, float, float, float, float]:
    minx, miny, maxx, maxy = (float(x) for x in gdf.total_bounds)
    width = max(maxx - minx, 1e-12)
    height = max(maxy - miny, 1e-12)
    return minx, miny, maxx, maxy, max(width, height)


def _canonicalize_geometry_legacy(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Dict[str, float]]:
    """Remove translation, uniform scale and dominant-axis rotation deterministically.

    All parts are transformed in one shared frame. This fixes the previous bug where
    each MultiGeometry child was normalized independently and spatial layout was lost.
    """
    clean = sanitize_gdf(gdf)
    coords = all_coordinates(clean)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    principal = vecs[:, int(np.argmax(vals))]
    angle_deg = math.degrees(math.atan2(float(principal[1]), float(principal[0])))

    rotated_coords = _rotate_xy(centered, -angle_deg)
    skew_x = float(np.mean(rotated_coords[:, 0] ** 3))
    if abs(skew_x) < 1e-12:
        far = rotated_coords[int(np.argmax(np.sum(rotated_coords**2, axis=1)))]
        flip = bool(far[0] < 0 or (abs(far[0]) < 1e-12 and far[1] < 0))
    else:
        flip = bool(skew_x < 0)
    if flip:
        angle_deg += 180.0

    geoms = []
    for geom in clean.geometry:
        g = affinity.translate(geom, xoff=-float(center[0]), yoff=-float(center[1]))
        g = affinity.rotate(g, -angle_deg, origin=(0.0, 0.0), use_radians=False)
        geoms.append(g)
    normalized = gpd.GeoDataFrame(geometry=geoms, crs=clean.crs)
    _, _, _, _, span = dataset_extent(normalized)
    scale = 2.0 / span
    normalized.geometry = normalized.geometry.map(
        lambda g: affinity.scale(g, xfact=scale, yfact=scale, origin=(0.0, 0.0))
    )
    return normalized, {
        "canonical_center_x": float(center[0]),
        "canonical_center_y": float(center[1]),
        "canonical_rotation_deg": float(angle_deg),
        "canonical_scale": float(scale),
    }



def canonicalize_geometry(
    gdf: gpd.GeoDataFrame, *, assume_clean: bool = False
) -> Tuple[gpd.GeoDataFrame, Dict[str, float]]:
    """V13 vectorized canonicalization, numerically equivalent to the legacy frame.

    The legacy path applied translate -> rotate -> scale separately to every
    Shapely object.  V13 computes the same global frame once and applies a single
    vectorized coordinate transformation with ``shapely.transform``.  This keeps
    the scientific definition unchanged while removing most Python-object loops.
    """
    clean = gdf if assume_clean else sanitize_gdf(gdf)
    coords = all_coordinates(clean)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    principal = vecs[:, int(np.argmax(vals))]
    angle_deg = math.degrees(math.atan2(float(principal[1]), float(principal[0])))

    rotated_coords = _rotate_xy(centered, -angle_deg)
    skew_x = float(np.mean(rotated_coords[:, 0] ** 3))
    if abs(skew_x) < 1e-12:
        far = rotated_coords[int(np.argmax(np.sum(rotated_coords**2, axis=1)))]
        flip = bool(far[0] < 0 or (abs(far[0]) < 1e-12 and far[1] < 0))
    else:
        flip = bool(skew_x < 0)
    if flip:
        angle_deg += 180.0

    # The span after rotation is computed from the same coordinates used by the
    # legacy normalization.  Because affine transforms reach extrema at vertices,
    # this matches GeoDataFrame.total_bounds for the transformed geometries.
    rad = math.radians(-angle_deg)
    c, sn = math.cos(rad), math.sin(rad)
    rx = centered[:, 0] * c - centered[:, 1] * sn
    ry = centered[:, 0] * sn + centered[:, 1] * c
    span = max(float(rx.max() - rx.min()), float(ry.max() - ry.min()), 1e-12)
    scale = 2.0 / span

    def transform_xy(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        x = xy[:, 0] - float(center[0])
        y = xy[:, 1] - float(center[1])
        out = np.empty_like(xy, dtype=np.float64)
        out[:, 0] = (x * c - y * sn) * scale
        out[:, 1] = (x * sn + y * c) * scale
        if xy.shape[1] > 2:
            out[:, 2:] = xy[:, 2:]
        return out

    try:
        arr = np.asarray(clean.geometry.array, dtype=object)
        transformed = shapely.transform(arr, transform_xy, include_z=False)
        normalized = gpd.GeoDataFrame(geometry=transformed, crs=clean.crs)
    except Exception:
        # Exact compatibility fallback for older Shapely installations.
        return _canonicalize_geometry_legacy(gdf)

    return normalized, {
        "canonical_center_x": float(center[0]),
        "canonical_center_y": float(center[1]),
        "canonical_rotation_deg": float(angle_deg),
        "canonical_scale": float(scale),
    }

def _rotate_xy(coords: np.ndarray, degrees: float) -> np.ndarray:
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return coords @ rot.T


def _pixel_xy(x: float, y: float, grid_size: int, limit: float = 1.05) -> Tuple[int, int]:
    px = int(np.clip(round((x + limit) / (2.0 * limit) * (grid_size - 1)), 0, grid_size - 1))
    py = int(np.clip(round((limit - y) / (2.0 * limit) * (grid_size - 1)), 0, grid_size - 1))
    return px, py


def _line_pixels(
    p0: Tuple[int, int], p1: Tuple[int, int], grid_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1])) + 1
    xs = np.clip(np.rint(np.linspace(p0[0], p1[0], n)).astype(int), 0, grid_size - 1)
    ys = np.clip(np.rint(np.linspace(p0[1], p1[1], n)).astype(int), 0, grid_size - 1)
    return xs, ys


def _normalise01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _build_four_channels_legacy(
    gdf: gpd.GeoDataFrame,
    grid_size: int = 256,
    density_sigma: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    canonical, canonical_meta = _canonicalize_geometry_legacy(gdf)
    limit = 1.05
    transform = from_bounds(-limit, -limit, limit, limit, grid_size, grid_size)
    shapes = [(geom, 1.0) for geom in canonical.geometry if geom is not None and not geom.is_empty]
    occ = rasterize(
        shapes,
        out_shape=(grid_size, grid_size),
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype="float32",
    )
    occ = np.clip(occ, 0.0, 1.0).astype(np.float32)

    if np.any(occ > 0):
        dist_px = distance_transform_edt(1.0 - occ).astype(np.float32)
        dist = np.exp(-dist_px / max(1.0, grid_size / 32.0)).astype(np.float32)
    else:
        dist = np.zeros_like(occ)

    cos2 = np.zeros_like(occ)
    sin2 = np.zeros_like(occ)
    weight = np.zeros_like(occ)
    density_seed = np.zeros_like(occ)

    for geom in canonical.geometry:
        if geom is None or geom.is_empty:
            continue
        for part in _iter_parts(geom):
            try:
                centroid = part.centroid
                cx, cy = _pixel_xy(float(centroid.x), float(centroid.y), grid_size, limit)
                density_seed[cy, cx] += 1.0
            except Exception:
                pass
            for seq in _iter_coordinate_sequences(part):
                for (x0, y0), (x1, y1) in zip(seq[:-1], seq[1:]):
                    dx, dy = x1 - x0, y1 - y0
                    if abs(dx) + abs(dy) <= 1e-15:
                        continue
                    theta = math.atan2(dy, dx) % math.pi
                    p0 = _pixel_xy(x0, y0, grid_size, limit)
                    p1 = _pixel_xy(x1, y1, grid_size, limit)
                    xs, ys = _line_pixels(p0, p1, grid_size)
                    seg_weight = max(math.hypot(dx, dy), 1e-8)
                    cos2[ys, xs] += math.cos(2.0 * theta) * seg_weight
                    sin2[ys, xs] += math.sin(2.0 * theta) * seg_weight
                    weight[ys, xs] += seg_weight

    orient_mask = weight > 0
    orientation = np.zeros_like(occ)
    if np.any(orient_mask):
        local_angle = (0.5 * np.arctan2(sin2, cos2)) % math.pi
        nearest = distance_transform_edt(~orient_mask, return_distances=False, return_indices=True)
        orientation = (local_angle[nearest[0], nearest[1]] / math.pi).astype(np.float32)

    density = gaussian_filter(density_seed, sigma=max(0.0, float(density_sigma))).astype(np.float32)
    density = _normalise01(density)
    tensor = np.stack((occ, dist, orientation, density), axis=0).astype(np.float32)
    meta: Dict[str, float] = {
        **canonical_meta,
        "feature_count": float(len(canonical)),
        "grid_size": float(grid_size),
        "occ_fraction": float(occ.mean()),
        "dist_mean": float(dist.mean()),
        "orient_mean": float(orientation.mean()),
        "density_mean": float(density.mean()),
    }
    return tensor, meta



def _collect_orientation_segments_and_density_legacy(
    canonical: gpd.GeoDataFrame, grid_size: int, limit: float
):
    """Reference Python-object traversal retained for compatibility tests/fallback."""
    starts, stops = [], []
    density_seed = np.zeros((grid_size, grid_size), dtype=np.float32)
    for geom in canonical.geometry:
        if geom is None or geom.is_empty:
            continue
        for part in _iter_parts(geom):
            try:
                centroid = part.centroid
                cx, cy = _pixel_xy(float(centroid.x), float(centroid.y), grid_size, limit)
                density_seed[cy, cx] += 1.0
            except Exception:
                pass
            for seq in _iter_coordinate_sequences(part):
                arr = np.asarray(seq, dtype=np.float64)
                if len(arr) >= 2:
                    starts.append(arr[:-1, :2])
                    stops.append(arr[1:, :2])
    if not starts:
        empty = np.empty((0, 2), dtype=np.float64)
        return empty, empty, density_seed
    return np.concatenate(starts, axis=0), np.concatenate(stops, axis=0), density_seed


def _flatten_simple_parts_vectorized(geoms: np.ndarray) -> np.ndarray:
    """Flatten multipart/collection geometries without per-feature Python traversal.

    ``shapely.get_parts`` expands an entire object array in compiled code.  A
    second/third pass is only needed for unusual nested GeometryCollections.
    Normal water-system MultiLineStrings flatten in a single pass.
    """
    arr = np.asarray(geoms, dtype=object)
    if arr.size == 0:
        return arr
    parts = np.asarray(shapely.get_parts(arr), dtype=object)
    for _ in range(8):
        if parts.size == 0:
            return parts
        tids = np.asarray(shapely.get_type_id(parts), dtype=np.int8)
        nested = tids >= 4  # MultiPoint/MultiLineString/MultiPolygon/GeometryCollection
        if not np.any(nested):
            return parts
        # Nested collections are rare in the datasets; preserve all simple parts
        # and expand the nested subset in bulk. Accumulation is order-insensitive.
        simple = parts[~nested]
        expanded = np.asarray(shapely.get_parts(parts[nested]), dtype=object)
        parts = np.concatenate((simple, expanded)) if simple.size else expanded
    # Extremely deeply nested collection: fall back rather than change semantics.
    if np.any(np.asarray(shapely.get_type_id(parts), dtype=np.int8) >= 4):
        raise RuntimeError("Nested geometry collection exceeds vectorized flatten depth")
    return parts


def _collect_orientation_segments_and_density(canonical: gpd.GeoDataFrame, grid_size: int, limit: float):
    """V13.0.3 bulk segment/density extraction for large water-system layers.

    The previous fast path still iterated every feature, multipart child and
    coordinate sequence in Python.  On Shandong/Sichuan water networks this
    traversal dominated the channel time for every one of the 94 attacks.
    This implementation performs multipart expansion, centroids and coordinate
    extraction in Shapely's compiled vectorized kernels, then creates segment
    boundaries with a single NumPy mask.  Scientific definitions are unchanged.
    """
    try:
        geoms = np.asarray(canonical.geometry.array, dtype=object)
        parts = _flatten_simple_parts_vectorized(geoms)
        density_seed = np.zeros((grid_size, grid_size), dtype=np.float32)
        if parts.size == 0:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty, density_seed

        # Density uses one centroid for each simple part, exactly as _iter_parts.
        cent = np.asarray(shapely.centroid(parts), dtype=object)
        cxy = np.asarray(shapely.get_coordinates(cent, include_z=False), dtype=np.float64)
        if cxy.size:
            pix = _pixel_xy_array(cxy[:, :2], grid_size, limit)
            np.add.at(density_seed, (pix[:, 1], pix[:, 0]), 1.0)

        tids = np.asarray(shapely.get_type_id(parts), dtype=np.int8)
        line_parts = parts[(tids == 1) | (tids == 2)]  # LineString / LinearRing
        polygon_parts = parts[tids == 3]
        if polygon_parts.size:
            rings = np.asarray(shapely.get_rings(polygon_parts), dtype=object)
            seq_geoms = (
                np.concatenate((line_parts, rings))
                if line_parts.size and rings.size else (line_parts if line_parts.size else rings)
            )
        else:
            seq_geoms = line_parts

        if seq_geoms.size == 0:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty, density_seed

        coords, seq_idx = shapely.get_coordinates(seq_geoms, include_z=False, return_index=True)
        coords = np.asarray(coords, dtype=np.float64)[:, :2]
        seq_idx = np.asarray(seq_idx)
        if len(coords) < 2:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, empty, density_seed
        same_seq = seq_idx[:-1] == seq_idx[1:]
        starts = coords[:-1][same_seq]
        stops = coords[1:][same_seq]
        return starts, stops, density_seed
    except Exception:
        return _collect_orientation_segments_and_density_legacy(canonical, grid_size, limit)


def _pixel_xy_array(coords: np.ndarray, grid_size: int, limit: float) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.int64)
    px = np.rint((coords[:, 0] + limit) / (2.0 * limit) * (grid_size - 1))
    py = np.rint((limit - coords[:, 1]) / (2.0 * limit) * (grid_size - 1))
    out = np.empty((len(coords), 2), dtype=np.int64)
    out[:, 0] = np.clip(px, 0, grid_size - 1).astype(np.int64)
    out[:, 1] = np.clip(py, 0, grid_size - 1).astype(np.int64)
    return out


if _NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True)
    def _accumulate_segments_numba(p0, p1, cvals, svals, wvals, grid_size):
        cos2 = np.zeros((grid_size, grid_size), dtype=np.float64)
        sin2 = np.zeros((grid_size, grid_size), dtype=np.float64)
        weight = np.zeros((grid_size, grid_size), dtype=np.float64)
        for i in range(p0.shape[0]):
            x0 = int(p0[i, 0]); y0 = int(p0[i, 1])
            x1 = int(p1[i, 0]); y1 = int(p1[i, 1])
            adx = abs(x1 - x0); ady = abs(y1 - y0)
            n = max(adx, ady) + 1
            if n <= 1:
                cos2[y0, x0] += cvals[i]
                sin2[y0, x0] += svals[i]
                weight[y0, x0] += wvals[i]
                continue
            den = float(n - 1)
            for k in range(n):
                # np.rint has the same ties-to-even semantics as the legacy path.
                x = int(np.rint(x0 + (x1 - x0) * (k / den)))
                y = int(np.rint(y0 + (y1 - y0) * (k / den)))
                if x < 0: x = 0
                elif x >= grid_size: x = grid_size - 1
                if y < 0: y = 0
                elif y >= grid_size: y = grid_size - 1
                cos2[y, x] += cvals[i]
                sin2[y, x] += svals[i]
                weight[y, x] += wvals[i]
        return cos2, sin2, weight
else:
    _accumulate_segments_numba = None


def _accumulate_segments_python(p0, p1, cvals, svals, wvals, grid_size):
    cos2 = np.zeros((grid_size, grid_size), dtype=np.float64)
    sin2 = np.zeros((grid_size, grid_size), dtype=np.float64)
    weight = np.zeros((grid_size, grid_size), dtype=np.float64)
    for i in range(len(p0)):
        xs, ys = _line_pixels((int(p0[i,0]), int(p0[i,1])), (int(p1[i,0]), int(p1[i,1])), grid_size)
        cos2[ys, xs] += float(cvals[i])
        sin2[ys, xs] += float(svals[i])
        weight[ys, xs] += float(wvals[i])
    return cos2, sin2, weight


def _orientation_accumulators(starts, stops, grid_size: int, limit: float, threads: int, progress_callback=None):
    if len(starts) == 0:
        z = np.zeros((grid_size, grid_size), dtype=np.float32)
        return z.copy(), z.copy(), z.copy()
    delta = stops - starts
    valid = (np.abs(delta[:, 0]) + np.abs(delta[:, 1])) > 1e-15
    starts = starts[valid]; stops = stops[valid]; delta = delta[valid]
    if len(starts) == 0:
        z = np.zeros((grid_size, grid_size), dtype=np.float32)
        return z.copy(), z.copy(), z.copy()
    theta = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), math.pi)
    segw = np.maximum(np.hypot(delta[:, 0], delta[:, 1]), 1e-8).astype(np.float64)
    cvals = np.cos(2.0 * theta) * segw
    svals = np.sin(2.0 * theta) * segw
    p0 = _pixel_xy_array(starts, grid_size, limit)
    p1 = _pixel_xy_array(stops, grid_size, limit)
    fn = _accumulate_segments_numba if _NUMBA_AVAILABLE else _accumulate_segments_python
    threads = max(1, int(threads))
    if threads <= 1 or len(p0) < 10_000:
        c, s, w = fn(p0, p1, cvals, svals, segw, grid_size)
        return c.astype(np.float32), s.astype(np.float32), w.astype(np.float32)
    # Thread-level parallelism is safe because every worker writes to a private
    # 256x256 accumulator; reduction happens only after all chunks complete.
    edges = np.linspace(0, len(p0), min(threads, len(p0)) + 1, dtype=np.int64)
    args = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            args.append((p0[a:b], p1[a:b], cvals[a:b], svals[a:b], segw[a:b], grid_size))
    _emit_progress(progress_callback, "channel.orientation_accumulate", done=0, total=len(args), threads=len(args), detail=f"segments={len(p0):,}")
    counter = {"done": 0}
    lock = threading.Lock()

    def _acc_one(x):
        result = fn(*x)
        with lock:
            counter["done"] += 1
            done = counter["done"]
        _emit_progress(progress_callback, "channel.orientation_accumulate", done=done, total=len(args), threads=len(args), detail=f"segments={len(p0):,}")
        return result

    with ThreadPoolExecutor(max_workers=len(args)) as ex:
        parts = list(ex.map(_acc_one, args))
    c = np.sum([x[0] for x in parts], axis=0, dtype=np.float64)
    s = np.sum([x[1] for x in parts], axis=0, dtype=np.float64)
    w = np.sum([x[2] for x in parts], axis=0, dtype=np.float64)
    return c.astype(np.float32), s.astype(np.float32), w.astype(np.float32)


def warmup_v13_fastpath() -> None:
    """Compile/load the optional Numba rasterizer once in the parent process.

    With 50-70 spawned cache workers, letting every child race to compile the
    same kernel wastes CPU and can create cache-lock contention.  This tiny call
    populates Numba's on-disk cache before workers start.
    """
    if not _NUMBA_AVAILABLE:
        return
    p0 = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
    p1 = np.asarray([[3, 3], [4, 1]], dtype=np.int64)
    v = np.asarray([1.0, 0.5], dtype=np.float64)
    _accumulate_segments_numba(p0, p1, v, v, v, 8)


def _build_four_channels_previous_release(
    gdf: gpd.GeoDataFrame,
    grid_size: int = 256,
    density_sigma: float = 3.0,
    parallel_threads: Optional[int] = None,
    assume_clean: bool = False,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """V13 high-performance four-channel builder.

    The representation remains occupancy / distance / orientation / density.
    Only the implementation is accelerated.  ``parallel_threads`` controls the
    orientation accumulator and is intended for long-tail cases when only a few
    attacks remain.  Normal massively parallel attack batches should use 1.
    """
    _emit_progress(progress_callback, "channel.canonicalize", detail=f"features={len(gdf):,}")
    canonical, canonical_meta = canonicalize_geometry(gdf, assume_clean=bool(assume_clean))
    _emit_progress(progress_callback, "channel.canonicalize_done", detail=f"features={len(canonical):,}")
    limit = 1.05
    transform = from_bounds(-limit, -limit, limit, limit, grid_size, grid_size)
    shapes = [(geom, 1.0) for geom in canonical.geometry if geom is not None and not geom.is_empty]
    _emit_progress(progress_callback, "channel.rasterize", detail=f"shapes={len(shapes):,}")
    occ = rasterize(
        shapes,
        out_shape=(grid_size, grid_size),
        transform=transform,
        fill=0.0,
        all_touched=True,
        dtype="float32",
    )
    occ = np.clip(occ, 0.0, 1.0).astype(np.float32)
    _emit_progress(progress_callback, "channel.distance_transform", detail=f"occ_fraction={float(occ.mean()):.6f}")
    if np.any(occ > 0):
        dist_px = distance_transform_edt(1.0 - occ).astype(np.float32)
        dist = np.exp(-dist_px / max(1.0, grid_size / 32.0)).astype(np.float32)
    else:
        dist = np.zeros_like(occ)

    _emit_progress(progress_callback, "channel.collect_segments", detail=f"features={len(canonical):,}")
    starts, stops, density_seed = _collect_orientation_segments_and_density(canonical, grid_size, limit)
    _emit_progress(progress_callback, "channel.collect_segments_done", detail=f"segments={len(starts):,}")
    if parallel_threads is None:
        parallel_threads = int(os.environ.get("RB_AFL_CHANNEL_THREADS_PER_ATTACK", "1"))
    cos2, sin2, weight = _orientation_accumulators(
        starts, stops, grid_size, limit, max(1, int(parallel_threads)), progress_callback=progress_callback
    )
    _emit_progress(progress_callback, "channel.orientation_propagate", detail=f"segments={len(starts):,}")
    orient_mask = weight > 0
    orientation = np.zeros_like(occ)
    if np.any(orient_mask):
        local_angle = (0.5 * np.arctan2(sin2, cos2)) % math.pi
        nearest = distance_transform_edt(~orient_mask, return_distances=False, return_indices=True)
        orientation = (local_angle[nearest[0], nearest[1]] / math.pi).astype(np.float32)

    _emit_progress(progress_callback, "channel.density_filter", detail=f"sigma={float(density_sigma):.3f}")
    density = gaussian_filter(density_seed, sigma=max(0.0, float(density_sigma))).astype(np.float32)
    density = _normalise01(density)
    _emit_progress(progress_callback, "channel.stack", detail="occupancy+distance+orientation+density")
    tensor = np.stack((occ, dist, orientation, density), axis=0).astype(np.float32)
    _emit_progress(progress_callback, "channel.done", detail=f"tensor_shape={tuple(tensor.shape)}")
    meta: Dict[str, float] = {
        **canonical_meta,
        "feature_count": float(len(canonical)),
        "grid_size": float(grid_size),
        "occ_fraction": float(occ.mean()),
        "dist_mean": float(dist.mean()),
        "orient_mean": float(orientation.mean()),
        "density_mean": float(density.mean()),
        "segment_count": float(len(starts)),
        "channel_parallel_threads": float(max(1, int(parallel_threads))),
        "numba_fastpath": float(bool(_NUMBA_AVAILABLE)),
    }
    return tensor, meta

def _transform_all(gdf: gpd.GeoDataFrame, fn) -> gpd.GeoDataFrame:
    clean = sanitize_gdf(gdf)
    return gpd.GeoDataFrame(geometry=[fn(g) for g in clean.geometry], crs=clean.crs)


def _bbox_origin(gdf: gpd.GeoDataFrame) -> Tuple[float, float]:
    minx, miny, maxx, maxy, _ = dataset_extent(gdf)
    return (0.5 * (minx + maxx), 0.5 * (miny + maxy))


def rotate_attack(gdf: gpd.GeoDataFrame, degrees_clockwise: float) -> gpd.GeoDataFrame:
    origin = _bbox_origin(gdf)
    return _transform_all(
        gdf,
        lambda g: affinity.rotate(g, -float(degrees_clockwise), origin=origin),
    )


def scale_attack(gdf: gpd.GeoDataFrame, factor: float) -> gpd.GeoDataFrame:
    if factor <= 0:
        raise ValueError("Scale factor must be > 0")
    origin = _bbox_origin(gdf)
    return _transform_all(
        gdf,
        lambda g: affinity.scale(g, xfact=float(factor), yfact=float(factor), origin=origin),
    )


def translate_attack(gdf: gpd.GeoDataFrame, factor: float) -> gpd.GeoDataFrame:
    *_, span = dataset_extent(gdf)
    offset = float(factor) * span
    return _transform_all(gdf, lambda g: affinity.translate(g, xoff=offset, yoff=offset))


def delete_attack(gdf: gpd.GeoDataFrame, ratio: float, seed: int) -> gpd.GeoDataFrame:
    clean = sanitize_gdf(gdf)
    if ratio <= 0:
        return clean.copy()
    delete_n = min(len(clean) - 1, max(1, int(round(len(clean) * float(ratio)))))
    rng = np.random.default_rng(seed)
    drop = rng.choice(len(clean), size=delete_n, replace=False)
    return clean.drop(index=drop).reset_index(drop=True)


def _append_geometries(base: gpd.GeoDataFrame, additions: Sequence) -> gpd.GeoDataFrame:
    if not additions:
        return sanitize_gdf(base)
    geoms = list(sanitize_gdf(base).geometry) + [g for g in additions if g is not None and not g.is_empty]
    return sanitize_gdf(gpd.GeoDataFrame(geometry=geoms, crs=base.crs))


def _external_in_crs(external: gpd.GeoDataFrame, base: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    ext = sanitize_gdf(external)
    if base.crs is not None and ext.crs is not None and base.crs != ext.crs:
        ext = ext.to_crs(base.crs)
    return ext


def add_attack(
    gdf: gpd.GeoDataFrame,
    ratio: float,
    seed: int,
    external: Optional[gpd.GeoDataFrame] = None,
) -> Tuple[gpd.GeoDataFrame, str, int]:
    base = sanitize_gdf(gdf)
    if ratio <= 0:
        return base.copy(), "baseline", 0
    add_n = max(1, int(math.ceil(len(base) * float(ratio))))
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy, span = dataset_extent(base)
    center = _bbox_origin(base)
    additions: List = []

    if external is not None and len(external):
        ext = _external_in_crs(external, base)
        indices = rng.choice(len(ext), size=add_n, replace=len(ext) < add_n)
        selected = [ext.geometry.iloc[int(i)] for i in indices]
        selected_gdf = gpd.GeoDataFrame(geometry=selected, crs=base.crs)
        ex_minx, ex_miny, ex_maxx, ex_maxy, ex_span = dataset_extent(selected_gdf)
        scale = min(1.0, 0.5 * span / max(ex_span, 1e-12))
        ex_center = (0.5 * (ex_minx + ex_maxx), 0.5 * (ex_miny + ex_maxy))
        for geom in selected:
            g = affinity.scale(geom, xfact=scale, yfact=scale, origin=ex_center)
            g = affinity.translate(g, xoff=center[0] - ex_center[0], yoff=center[1] - ex_center[1])
            jitter_x = rng.uniform(-0.35, 0.35) * (maxx - minx)
            jitter_y = rng.uniform(-0.35, 0.35) * (maxy - miny)
            additions.append(affinity.translate(g, xoff=jitter_x, yoff=jitter_y))
        mode = "external_objects_fitted_to_base_extent"
    else:
        indices = rng.choice(len(base), size=add_n, replace=len(base) < add_n)
        for i in indices:
            g = base.geometry.iloc[int(i)]
            additions.append(
                affinity.translate(
                    g,
                    xoff=rng.uniform(-0.12, 0.12) * span,
                    yoff=rng.uniform(-0.12, 0.12) * span,
                )
            )
        mode = "deterministic_clone_fallback"
    return _append_geometries(base, additions), mode, len(additions)


def clip_attack(gdf: gpd.GeoDataFrame, ratio: float) -> gpd.GeoDataFrame:
    clean = sanitize_gdf(gdf)
    if ratio <= 0:
        return clean.copy()
    if not 0.0 < ratio < 1.0:
        raise ValueError("Clip ratio must be in [0, 1)")
    minx, miny, maxx, maxy, span = dataset_extent(clean)
    window = box(minx, miny, minx + (1.0 - float(ratio)) * (maxx - minx), maxy)
    grid_size = max(span * 1e-10, 1e-12)
    clipped: List = []
    for geom in clean.geometry:
        repaired = _valid_geometry(geom)
        if repaired is None:
            continue
        try:
            out = shapely.intersection(repaired, window, grid_size=grid_size)
        except Exception:
            repaired = fast_make_valid(np.asarray([repaired], dtype=object))[0]
            out = repaired.intersection(window)
        out = _valid_geometry(out)
        if out is not None and not out.is_empty:
            clipped.append(out)
    if not clipped:
        raise RuntimeError(f"Clip ratio {ratio} removed all geometry")
    return sanitize_gdf(gpd.GeoDataFrame(geometry=clipped, crs=clean.crs))


def merge_attack(
    gdf: gpd.GeoDataFrame,
    ratio: float,
    seed: int,
    external: Optional[gpd.GeoDataFrame],
) -> Tuple[gpd.GeoDataFrame, int]:
    base = sanitize_gdf(gdf)
    if ratio <= 0:
        return base.copy(), 0
    if external is None or len(external) == 0:
        raise ValueError("Merge attack requires --external-vector; geometry dissolve is not a merge attack")
    ext = _external_in_crs(external, base)
    merge_n = max(1, int(math.ceil(len(base) * float(ratio))))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ext), size=merge_n, replace=len(ext) < merge_n)
    additions = [ext.geometry.iloc[int(i)] for i in indices]
    return _append_geometries(base, additions), len(additions)




def non_uniform_scale_attack(gdf: gpd.GeoDataFrame, x_factor: float) -> gpd.GeoDataFrame:
    if x_factor <= 0:
        raise ValueError("Non-uniform scale factor must be > 0")
    origin = _bbox_origin(gdf)
    return _transform_all(
        gdf,
        lambda g: affinity.scale(g, xfact=float(x_factor), yfact=1.0, origin=origin),
    )


def _geometry_vertex_count(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if isinstance(geom, Point):
        return 1
    if isinstance(geom, (LineString, LinearRing)):
        coords = list(geom.coords)
        if isinstance(geom, LinearRing) and len(coords) > 1 and coords[0][:2] == coords[-1][:2]:
            return len(coords) - 1
        return len(coords)
    if isinstance(geom, Polygon):
        total = max(0, len(geom.exterior.coords) - 1)
        total += sum(max(0, len(ring.coords) - 1) for ring in geom.interiors)
        return total
    if isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon, GeometryCollection)):
        return sum(_geometry_vertex_count(part) for part in geom.geoms)
    return 0


def vertex_count(gdf: gpd.GeoDataFrame) -> int:
    return sum(_geometry_vertex_count(geom) for geom in sanitize_gdf(gdf).geometry)


def _allocate_segment_additions(coords: Sequence[Tuple[float, float]], add_count: int) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y, *_ in coords]
    if add_count <= 0 or len(pts) < 2:
        return pts
    lengths = np.asarray(
        [math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:])],
        dtype=np.float64,
    )
    if not np.any(lengths > 0):
        return pts
    raw = add_count * lengths / float(lengths.sum())
    counts = np.floor(raw).astype(np.int64)
    remaining = int(add_count - int(counts.sum()))
    if remaining > 0:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remaining]] += 1
    out: List[Tuple[float, float]] = [pts[0]]
    for index, ((x0, y0), (x1, y1)) in enumerate(zip(pts[:-1], pts[1:])):
        n = int(counts[index])
        for step in range(1, n + 1):
            alpha = step / float(n + 1)
            out.append((x0 + alpha * (x1 - x0), y0 + alpha * (y1 - y0)))
        out.append((x1, y1))
    return out


def _interpolate_geometry(geom, ratio: float):
    if geom is None or geom.is_empty or isinstance(geom, Point):
        return geom
    if isinstance(geom, LinearRing):
        coords = list(geom.coords)
        base_count = max(0, len(coords) - 1)
        return LinearRing(_allocate_segment_additions(coords, int(round(base_count * ratio))))
    if isinstance(geom, LineString):
        coords = list(geom.coords)
        return LineString(_allocate_segment_additions(coords, int(round(len(coords) * ratio))))
    if isinstance(geom, Polygon):
        exterior = list(geom.exterior.coords)
        ext_count = max(0, len(exterior) - 1)
        new_exterior = _allocate_segment_additions(exterior, int(round(ext_count * ratio)))
        holes = []
        for ring in geom.interiors:
            coords = list(ring.coords)
            base_count = max(0, len(coords) - 1)
            holes.append(_allocate_segment_additions(coords, int(round(base_count * ratio))))
        return Polygon(new_exterior, holes)
    if isinstance(geom, MultiPoint):
        return geom
    if isinstance(geom, MultiLineString):
        return MultiLineString([_interpolate_geometry(part, ratio) for part in geom.geoms])
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([_interpolate_geometry(part, ratio) for part in geom.geoms])
    if isinstance(geom, GeometryCollection):
        return GeometryCollection([_interpolate_geometry(part, ratio) for part in geom.geoms])
    return geom


def interpolation_attack(gdf: gpd.GeoDataFrame, ratio: float) -> gpd.GeoDataFrame:
    if not 0.0 <= ratio <= 2.0:
        raise ValueError("Interpolation ratio must be in [0, 2]")
    clean = sanitize_gdf(gdf)
    out = gpd.GeoDataFrame(
        geometry=[_interpolate_geometry(geom, float(ratio)) for geom in clean.geometry],
        crs=clean.crs,
    )
    return sanitize_gdf(out)


def simplification_attack(gdf: gpd.GeoDataFrame, target_delete_ratio: float) -> Tuple[gpd.GeoDataFrame, float]:
    if not 0.0 <= target_delete_ratio < 1.0:
        raise ValueError("Simplification delete ratio must be in [0, 1)")
    clean = sanitize_gdf(gdf)
    original_count = vertex_count(clean)
    if target_delete_ratio <= 0 or original_count <= 1:
        return clean.copy(), 0.0
    target_count = max(1, int(round(original_count * (1.0 - float(target_delete_ratio)))))
    *_, span = dataset_extent(clean)
    low, high = 0.0, span
    best = clean.copy()
    best_count = original_count
    best_error = abs(best_count - target_count)
    for _ in range(28):
        tolerance = 0.5 * (low + high)
        candidate = gpd.GeoDataFrame(
            geometry=[geom.simplify(tolerance, preserve_topology=True) for geom in clean.geometry],
            crs=clean.crs,
        )
        candidate = sanitize_gdf(candidate)
        count = vertex_count(candidate)
        error = abs(count - target_count)
        if error < best_error or (error == best_error and count < best_count):
            best, best_count, best_error = candidate, count, error
        if count > target_count:
            low = tolerance
        else:
            high = tolerance
    realized = max(0.0, (original_count - best_count) / max(1, original_count))
    return best, realized


def _delete_vertices_from_coords(
    coords: Sequence[Tuple[float, float]], ratio: float, rng: np.random.Generator, closed: bool
) -> List[Tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y, *_ in coords]
    if closed:
        unique = pts[:-1] if len(pts) > 1 and pts[0] == pts[-1] else pts
        max_delete = max(0, len(unique) - 3)
        delete_n = min(max_delete, int(round(len(unique) * ratio)))
        if delete_n <= 0:
            return unique + [unique[0]] if unique else pts
        drop = set(int(i) for i in rng.choice(len(unique), size=delete_n, replace=False))
        kept = [pt for i, pt in enumerate(unique) if i not in drop]
        return kept + [kept[0]]
    max_delete = max(0, len(pts) - 2)
    delete_n = min(max_delete, int(round(len(pts) * ratio)))
    if delete_n <= 0:
        return pts
    eligible = np.arange(1, len(pts) - 1, dtype=np.int64)
    drop = set(int(i) for i in rng.choice(eligible, size=delete_n, replace=False))
    return [pt for i, pt in enumerate(pts) if i not in drop]


def _delete_vertices_geometry(geom, ratio: float, rng: np.random.Generator):
    if geom is None or geom.is_empty or isinstance(geom, Point):
        return geom
    if isinstance(geom, LinearRing):
        return LinearRing(_delete_vertices_from_coords(list(geom.coords), ratio, rng, True))
    if isinstance(geom, LineString):
        return LineString(_delete_vertices_from_coords(list(geom.coords), ratio, rng, False))
    if isinstance(geom, Polygon):
        exterior = _delete_vertices_from_coords(list(geom.exterior.coords), ratio, rng, True)
        holes = [
            _delete_vertices_from_coords(list(ring.coords), ratio, rng, True)
            for ring in geom.interiors
        ]
        return Polygon(exterior, holes)
    if isinstance(geom, MultiPoint):
        return geom
    if isinstance(geom, MultiLineString):
        return MultiLineString([_delete_vertices_geometry(part, ratio, rng) for part in geom.geoms])
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([_delete_vertices_geometry(part, ratio, rng) for part in geom.geoms])
    if isinstance(geom, GeometryCollection):
        return GeometryCollection([_delete_vertices_geometry(part, ratio, rng) for part in geom.geoms])
    return geom


def vertex_delete_attack(gdf: gpd.GeoDataFrame, ratio: float, seed: int) -> gpd.GeoDataFrame:
    if not 0.0 <= ratio < 1.0:
        raise ValueError("Vertex delete ratio must be in [0, 1)")
    clean = sanitize_gdf(gdf)
    rng = np.random.default_rng(seed)
    attacked = gpd.GeoDataFrame(
        geometry=[_delete_vertices_geometry(geom, float(ratio), rng) for geom in clean.geometry],
        crs=clean.crs,
    )
    return sanitize_gdf(attacked)


def _coordinate_noise_attack_legacy(
    gdf: gpd.GeoDataFrame, amplitude_ratio: float, seed: int
) -> gpd.GeoDataFrame:
    """Original Python-dictionary implementation kept as an exact fallback."""
    if amplitude_ratio < 0:
        raise ValueError("Coordinate-noise amplitude ratio must be >= 0")
    clean = sanitize_gdf(gdf)
    *_, span = dataset_extent(clean)
    amplitude = float(amplitude_ratio) * span
    if amplitude <= 0:
        return clean.copy()
    rng = np.random.default_rng(seed)
    offsets: Dict[Tuple[float, float], Tuple[float, float]] = {}

    def perturb(coords: np.ndarray) -> np.ndarray:
        result = np.asarray(coords, dtype=np.float64).copy()
        for i in range(result.shape[0]):
            key = (float(result[i, 0]), float(result[i, 1]))
            offset = offsets.get(key)
            if offset is None:
                offset = (
                    float(rng.uniform(-amplitude, amplitude)),
                    float(rng.uniform(-amplitude, amplitude)),
                )
                offsets[key] = offset
            result[i, 0] += offset[0]
            result[i, 1] += offset[1]
        return result

    attacked = gpd.GeoDataFrame(
        geometry=[shapely.transform(geom, perturb) for geom in clean.geometry],
        crs=clean.crs,
    )
    return sanitize_gdf(attacked)


def prepare_coordinate_noise_index(gdf: gpd.GeoDataFrame) -> Tuple[np.ndarray, int]:
    """Precompute the expensive duplicate-coordinate map once per source layer.

    ``coordinate_noise_attack`` needs the first-occurrence identity of every
    coordinate so duplicate vertices receive the same random offset.  Computing
    that map with ``np.unique(..., axis=0)`` dominates runtime on multi-million
    vertex water-system maps.  The map is independent of noise strength and RNG
    seed, so V13.0.2 persists it once and reuses it for all six noise levels.
    """
    clean = sanitize_gdf(gdf)
    geoms = np.asarray(clean.geometry.array, dtype=object)
    coords = np.asarray(shapely.get_coordinates(geoms), dtype=np.float64)
    if coords.size == 0:
        return np.empty((0,), dtype=np.int32), 0
    _, first_idx, inverse_sorted = np.unique(
        coords, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first_idx, kind="stable")
    sorted_to_first = np.empty(len(order), dtype=np.int64)
    sorted_to_first[order] = np.arange(len(order), dtype=np.int64)
    inverse_first = sorted_to_first[inverse_sorted]
    # int32 cuts persistent/mmap memory in half and is ample for realistic GIS
    # layers (<2.1 billion unique coordinate positions).
    if len(order) < np.iinfo(np.int32).max:
        inverse_first = inverse_first.astype(np.int32, copy=False)
    return inverse_first, int(len(order))


def coordinate_noise_attack(
    gdf: gpd.GeoDataFrame,
    amplitude_ratio: float,
    seed: int,
    *,
    inverse_first: Optional[np.ndarray] = None,
    unique_count: Optional[int] = None,
    assume_clean: bool = False,
    parallel_threads: int = 1,
    progress_callback=None,
    checkpoint_dir: Optional[str | Path] = None,
    checkpoint_signature: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Exact-RNG vectorized coordinate jitter with optional reusable index.

    When ``inverse_first``/``unique_count`` are supplied, the O(N log N)
    duplicate-coordinate discovery is skipped entirely.  This preserves the
    exact legacy RNG assignment while making repeated noise strengths roughly
    O(N) rather than repeating ``np.unique`` six times per identity.
    """
    if amplitude_ratio < 0:
        raise ValueError("Coordinate-noise amplitude ratio must be >= 0")
    _emit_progress(progress_callback, "attack.prepare", detail=f"features={len(gdf):,} threads={max(1,int(parallel_threads))}")
    clean = gdf if assume_clean else sanitize_gdf(gdf)
    *_, span = dataset_extent(clean)
    amplitude = float(amplitude_ratio) * span
    if amplitude <= 0:
        return clean.copy()

    cp_dir = Path(checkpoint_dir) if checkpoint_dir else None
    full_cp = cp_dir / "attacked_sanitized_wkb.npy" if cp_dir is not None else None
    full_meta = cp_dir / "attacked_sanitized_complete.json" if cp_dir is not None else None
    if cp_dir is not None and checkpoint_signature and full_cp is not None and full_meta is not None:
        try:
            info = json.loads(full_meta.read_text(encoding="utf-8")) if full_meta.is_file() else {}
            if info.get("signature") == str(checkpoint_signature) and full_cp.is_file():
                wkb = np.load(full_cp, allow_pickle=True)
                restored = np.asarray(shapely.from_wkb(wkb), dtype=object)
                _emit_progress(progress_callback, "attack.resume_checkpoint", done=1, total=1, detail=f"restored attacked geometry features={len(restored):,}")
                return gpd.GeoDataFrame(geometry=restored, crs=clean.crs).reset_index(drop=True)
        except Exception:
            pass
    try:
        _emit_progress(progress_callback, "attack.extract_coordinates", detail=f"features={len(clean):,}")
        geoms = np.asarray(clean.geometry.array, dtype=object).copy()
        coords = np.asarray(shapely.get_coordinates(geoms), dtype=np.float64)
        _emit_progress(progress_callback, "attack.extract_coordinates_done", detail=f"coordinates={len(coords):,}")
        if coords.size == 0:
            return clean.copy()
        if inverse_first is None or unique_count is None or len(inverse_first) != len(coords):
            inverse_first, unique_count = prepare_coordinate_noise_index(clean)
        inv = np.asarray(inverse_first)
        n_unique = int(unique_count)
        if n_unique <= 0:
            return clean.copy()
        _emit_progress(progress_callback, "attack.generate_offsets", detail=f"unique_coordinates={n_unique:,} amplitude={amplitude:.9g}")
        rng = np.random.default_rng(seed)
        offsets = rng.uniform(
            -amplitude, amplitude, size=(n_unique, 2)
        ).astype(np.float64, copy=False)
        new_coords = coords + offsets[inv]
        _emit_progress(progress_callback, "attack.offsets_ready", detail=f"coordinates={len(new_coords):,}")

        # V13.0.5: the final few million-vertex water-system noise attacks used
        # to collapse to a single CPU core inside ``shapely.set_coordinates``.
        # Split by whole geometries (never inside a feature), while slicing the
        # coordinate array at the exact corresponding offsets.  Each thread
        # therefore receives exactly the same coordinates as the serial call,
        # preserving deterministic RNG assignment and geometry ordering.
        n_threads = max(1, int(parallel_threads))
        if n_threads > 1 and len(geoms) > 1 and len(coords) >= 100_000:
            counts = np.asarray(shapely.get_num_coordinates(geoms), dtype=np.int64)
            ends = np.cumsum(counts, dtype=np.int64)
            total_coords = int(ends[-1]) if len(ends) else 0
            if total_coords != len(new_coords):
                # Defensive fallback: Shapely's global coordinate order must
                # match the per-geometry counts for chunking to be exact.
                _emit_progress(progress_callback, "attack.set_coordinates", done=0, total=1, threads=1, detail=f"coordinates={len(new_coords):,} fallback=count-mismatch")
                attacked_geoms = shapely.set_coordinates(geoms, new_coords)
                _emit_progress(progress_callback, "attack.set_coordinates", done=1, total=1, threads=1, detail=f"coordinates={len(new_coords):,} fallback=count-mismatch")
            else:
                # Coordinate-balanced feature boundaries.  This matters for
                # hydrography where a few LineStrings can be far larger than
                # the median feature.
                target = np.linspace(0, total_coords, min(n_threads, len(geoms)) + 1)
                boundaries = [0]
                for t in target[1:-1]:
                    j = int(np.searchsorted(ends, t, side="left") + 1)
                    j = max(boundaries[-1] + 1, min(j, len(geoms) - 1))
                    boundaries.append(j)
                boundaries.append(len(geoms))
                # Remove any duplicate/non-increasing boundaries introduced by
                # extreme feature-size imbalance.
                b2 = [boundaries[0]]
                for x in boundaries[1:]:
                    if x > b2[-1]:
                        b2.append(x)
                boundaries = b2

                pairs = [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]
                _emit_progress(progress_callback, "attack.set_coordinates", done=0, total=len(pairs), threads=len(pairs), detail=f"coordinates={len(new_coords):,}")
                counter = {"done": 0}
                lock = threading.Lock()

                def _set_chunk(pair):
                    a, b = pair
                    c0 = 0 if a == 0 else int(ends[a - 1])
                    c1 = int(ends[b - 1])
                    gg = np.asarray(geoms[a:b], dtype=object).copy()
                    cc = np.asarray(new_coords[c0:c1], dtype=np.float64)
                    result = (a, b, shapely.set_coordinates(gg, cc))
                    with lock:
                        counter["done"] += 1
                        done = counter["done"]
                    _emit_progress(progress_callback, "attack.set_coordinates", done=done, total=len(pairs), threads=len(pairs), detail=f"coordinates={len(new_coords):,}")
                    return result

                with ThreadPoolExecutor(max_workers=len(pairs)) as ex:
                    parts = list(ex.map(_set_chunk, pairs))
                attacked_geoms = np.empty_like(geoms, dtype=object)
                for a, b, part in parts:
                    attacked_geoms[a:b] = part
        else:
            _emit_progress(progress_callback, "attack.set_coordinates", done=0, total=1, threads=1, detail=f"coordinates={len(new_coords):,}")
            attacked_geoms = shapely.set_coordinates(geoms, new_coords)
            _emit_progress(progress_callback, "attack.set_coordinates", done=1, total=1, threads=1, detail=f"coordinates={len(new_coords):,}")

        attacked = gpd.GeoDataFrame(geometry=attacked_geoms, crs=clean.crs)
        # Coordinate jitter cannot make point/line-only water networks invalid in
        # the polygon-topology sense.  Re-validating every LineString after each
        # noise strength is pure overhead on million-vertex hydrography layers.
        # Keep the conservative sanitize path whenever polygons/collections exist.
        try:
            tids = np.asarray(shapely.get_type_id(attacked_geoms), dtype=np.int8)
            line_safe = np.all(np.isin(tids, np.asarray([0, 1, 2, 4, 5], dtype=np.int8)))
        except Exception:
            line_safe = False
        if line_safe:
            _emit_progress(progress_callback, "attack.sanitize_skipped", done=1, total=1, detail="point/line-only geometry")
            result = attacked.reset_index(drop=True)
        else:
            sanitize_cp = (cp_dir / "sanitize") if cp_dir is not None else None
            result = _sanitize_gdf_parallel(
                attacked, n_threads, progress_callback=progress_callback,
                checkpoint_dir=sanitize_cp, checkpoint_signature=checkpoint_signature,
            )

        if cp_dir is not None and checkpoint_signature:
            try:
                cp_dir.mkdir(parents=True, exist_ok=True)
                wkb = np.asarray(shapely.to_wkb(np.asarray(result.geometry.array, dtype=object)), dtype=object)
                tmp_wkb = full_cp.with_name(full_cp.name + f".tmp.{os.getpid()}")
                with tmp_wkb.open("wb") as fh:
                    np.save(fh, wkb, allow_pickle=True)
                os.replace(tmp_wkb, full_cp)
                tmp_meta = full_meta.with_name(full_meta.name + f".tmp.{os.getpid()}")
                tmp_meta.write_text(json.dumps({
                    "version": "v13.0.8-attacked-checkpoint-v1",
                    "signature": str(checkpoint_signature),
                    "features": int(len(result)),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp_meta, full_meta)
            except Exception:
                pass
        _emit_progress(progress_callback, "attack.done", detail=f"features={len(result):,}")
        return result
    except Exception as exc:
        _emit_progress(progress_callback, "attack.fallback_legacy", detail=f"{type(exc).__name__}: {exc}")
        return _coordinate_noise_attack_legacy(gdf, amplitude_ratio, seed)


def _reverse_storage_geometry(geom):
    if geom is None or geom.is_empty or isinstance(geom, Point):
        return geom
    if isinstance(geom, LinearRing):
        return LinearRing(list(geom.coords)[::-1])
    if isinstance(geom, LineString):
        return LineString(list(geom.coords)[::-1])
    if isinstance(geom, Polygon):
        exterior = list(geom.exterior.coords)[::-1]
        holes = [list(ring.coords)[::-1] for ring in reversed(geom.interiors)]
        return Polygon(exterior, holes)
    if isinstance(geom, MultiPoint):
        return MultiPoint(list(geom.geoms)[::-1])
    if isinstance(geom, MultiLineString):
        return MultiLineString([_reverse_storage_geometry(part) for part in reversed(geom.geoms)])
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([_reverse_storage_geometry(part) for part in reversed(geom.geoms)])
    if isinstance(geom, GeometryCollection):
        return GeometryCollection([_reverse_storage_geometry(part) for part in reversed(geom.geoms)])
    return geom


def reorder_attack(gdf: gpd.GeoDataFrame, ratio: float, seed: int) -> gpd.GeoDataFrame:
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("Reorder ratio must be in [0, 1]")
    clean = sanitize_gdf(gdf)
    if ratio <= 0 or len(clean) == 0:
        return clean.copy()
    rng = np.random.default_rng(seed)
    count = min(len(clean), max(1, int(round(len(clean) * ratio))))
    selected = np.asarray(rng.choice(len(clean), size=count, replace=False), dtype=np.int64)
    geoms = list(clean.geometry)
    reversed_geoms = [_reverse_storage_geometry(geoms[int(i)]) for i in selected]
    permutation = rng.permutation(count)
    for destination_pos, selected_index in enumerate(selected):
        geoms[int(selected_index)] = reversed_geoms[int(permutation[destination_pos])]
    return sanitize_gdf(gpd.GeoDataFrame(geometry=geoms, crs=clean.crs))


def apply_attack(
    gdf: gpd.GeoDataFrame,
    attack: str,
    strength: float,
    seed: int,
    external: Optional[gpd.GeoDataFrame] = None,
    *,
    coordinate_noise_inverse: Optional[np.ndarray] = None,
    coordinate_noise_unique_count: Optional[int] = None,
    source_is_sanitized: bool = False,
    coordinate_noise_threads: int = 1,
    progress_callback=None,
    coordinate_noise_checkpoint_dir: Optional[str | Path] = None,
    coordinate_noise_checkpoint_signature: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, Dict[str, object]]:
    name = attack.lower()
    source_count = len(gdf)
    meta: Dict[str, object] = {
        "attack": name,
        "strength": float(strength),
        "seed": int(seed),
        "source_feature_count": int(source_count),
    }
    if name == "rotation":
        out = rotate_attack(gdf, strength)
        meta["direction"] = "clockwise"
    elif name == "scale":
        out = scale_attack(gdf, strength)
    elif name == "translation":
        out = translate_attack(gdf, strength)
        meta["definition"] = "xoff=yoff=factor*max(width,height)"
    elif name == "object_delete":
        out = delete_attack(gdf, strength, seed)
        meta["deleted_features"] = len(gdf) - len(out)
        meta["realized_delete_ratio"] = (len(gdf) - len(out)) / max(1, len(gdf))
    elif name == "object_add":
        out, mode, count = add_attack(gdf, strength, seed, external)
        meta.update(
            {
                "addition_source": mode,
                "added_features": count,
                "realized_add_ratio": count / max(1, source_count),
            }
        )
    elif name == "clip":
        out = clip_attack(gdf, strength)
        meta["definition"] = "rightmost bbox-width fraction removed"
    elif name == "merge":
        out, count = merge_attack(gdf, strength, seed, external)
        meta.update(
            {
                "merged_features": count,
                "realized_merge_ratio": count / max(1, source_count),
                "definition": "append external map features",
            }
        )
    elif name == "non_uniform_scale":
        out = non_uniform_scale_attack(gdf, strength)
        meta.update({"xfact": float(strength), "yfact": 1.0})
    elif name == "interpolation":
        before = vertex_count(gdf)
        out = interpolation_attack(gdf, strength)
        after = vertex_count(out)
        meta.update(
            {
                "source_vertex_count": before,
                "result_vertex_count": after,
                "added_vertices": max(0, after - before),
                "realized_vertex_add_ratio": max(0, after - before) / max(1, before),
                "definition": "linear interpolation along existing segments",
            }
        )
    elif name == "simplification":
        before = vertex_count(gdf)
        out, realized = simplification_attack(gdf, strength)
        after = vertex_count(out)
        meta.update(
            {
                "source_vertex_count": before,
                "result_vertex_count": after,
                "deleted_vertices": max(0, before - after),
                "realized_vertex_delete_ratio": realized,
                "definition": "Douglas-Peucker with target deletion ratio",
            }
        )
    elif name == "vertex_delete":
        before = vertex_count(gdf)
        out = vertex_delete_attack(gdf, strength, seed)
        after = vertex_count(out)
        meta.update(
            {
                "source_vertex_count": before,
                "result_vertex_count": after,
                "deleted_vertices": max(0, before - after),
                "realized_vertex_delete_ratio": max(0, before - after) / max(1, before),
                "definition": "random vertex deletion with endpoints/rings preserved",
            }
        )
    elif name == "coordinate_noise":
        *_, span = dataset_extent(gdf)
        out = coordinate_noise_attack(
            gdf, strength, seed,
            inverse_first=coordinate_noise_inverse,
            unique_count=coordinate_noise_unique_count,
            assume_clean=bool(source_is_sanitized),
            parallel_threads=max(1, int(coordinate_noise_threads)),
            progress_callback=progress_callback,
            checkpoint_dir=coordinate_noise_checkpoint_dir,
            checkpoint_signature=coordinate_noise_checkpoint_signature,
        )
        meta.update(
            {
                "amplitude_to_span_ratio": float(strength),
                "uniform_noise_amplitude": float(strength) * span,
                "definition": "uniform independent coordinate jitter",
            }
        )
    elif name == "reorder":
        out = reorder_attack(gdf, strength, seed)
        meta.update(
            {
                "reordered_feature_fraction": float(strength),
                "definition": "feature-order permutation plus coordinate-direction reversal",
            }
        )
    else:
        raise KeyError(f"Unsupported attack: {attack}")
    meta["result_feature_count"] = len(out)
    return out, meta


def random_training_attack(
    gdf: gpd.GeoDataFrame,
    seed: int,
) -> Tuple[gpd.GeoDataFrame, Dict[str, float]]:
    """Mild vector-domain augmentation used only during encoder training."""
    rng = np.random.default_rng(seed)
    rotation = float(rng.uniform(-15.0, 15.0))
    scale = float(rng.uniform(0.90, 1.10))
    translation = float(rng.uniform(-0.08, 0.08))
    out = rotate_attack(gdf, rotation)
    out = scale_attack(out, scale)
    out = translate_attack(out, translation)
    delete_ratio = float(rng.choice([0.0, 0.0, 0.05, 0.10]))
    if delete_ratio > 0 and len(out) > 1:
        out = delete_attack(out, delete_ratio, seed + 17)
    return out, {
        "rotation_clockwise_deg": rotation,
        "scale": scale,
        "translation_factor": translation,
        "object_delete_ratio": delete_ratio,
    }


def build_four_channels(
    gdf: gpd.GeoDataFrame,
    grid_size: int = 256,
    density_sigma: float = 3.0,
    parallel_threads: Optional[int] = None,
    assume_clean: bool = False,
    progress_callback=None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Manuscript equations 1-6; previous raster code is explicitly legacy-only.

    PCA alignment is inherited from the historical code, not specified by the
    manuscript. The grid expands symmetrically if necessary to cover the complete
    canonical extent. See docs/MANUSCRIPT_ALIGNMENT.md before interpreting results.
    """
    from .fields import fields_from_geometry
    _emit_progress(progress_callback, "channel.canonicalize", detail=f"features={len(gdf)}")
    canonical, meta = canonicalize_geometry(gdf, assume_clean=assume_clean)
    limit = max(1.05, float(np.max(np.abs(canonical.total_bounds))) * 1.05)
    _emit_progress(progress_callback, "channel.equations", detail="exact distances and representative-point KDE")
    tensor, field_meta = fields_from_geometry(canonical.geometry, grid_size, density_sigma, limit=limit)
    meta.update(field_meta)
    meta.update(feature_count=float(len(canonical)), grid_size=float(grid_size),
                occ_fraction=float(tensor[0].mean()), dist_mean=float(tensor[1].mean()),
                orient_mean=float(tensor[2].mean()), density_mean=float(tensor[3].mean()))
    _emit_progress(progress_callback, "channel.done", detail=f"tensor_shape={tensor.shape}")
    return tensor, meta
