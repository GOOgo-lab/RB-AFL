"""Equation-based fields for manuscript Section 3.1 (not the historical raster approximation)."""
from __future__ import annotations
import math
import numpy as np
import shapely
from shapely.geometry import Point

CHANNEL_SCHEMA = "manuscript_equations_1_6_v1.1"


def fields_from_geometry(geometries, grid_size=256, density_sigma=3.0, *,
                         limit=1.05, distance_sigma_px=None, occupancy_tau_px=0.5):
    """Compute exact geometric distances and untruncated Gaussian KDE on cell centers.

    sigma defaults to grid_size/32 pixels, tau to half a pixel, and h to
    density_sigma pixels. These bandwidth choices are implementation defaults;
    the manuscript does not specify numerical sigma/tau/h values.
    Polygon interiors have zero distance to the polygon; rings supply directed
    segments. Segment ties select the earliest segment in input order. Point-only
    layers use orientation 0.5 (theta=0, an explicit missing-direction convention).
    """
    from .vector import _iter_parts, _iter_coordinate_sequences
    if grid_size < 1 or not np.isfinite(limit) or limit <= 0:
        raise ValueError("Invalid grid")
    pixel = 2 * limit / grid_size
    sigma = (grid_size / 32 if distance_sigma_px is None else distance_sigma_px) * pixel
    tau = occupancy_tau_px * pixel
    h = density_sigma * pixel
    if not np.isfinite([sigma, tau, h]).all() or sigma <= 0 or h <= 0 or tau < 0:
        raise ValueError("sigma and h must be positive; tau must be nonnegative")
    geoms = np.asarray([g for g in geometries if g is not None and not g.is_empty], dtype=object)
    if not len(geoms):
        raise ValueError("Cannot encode empty geometry")
    x = -limit + (np.arange(grid_size) + 0.5) * pixel
    y = limit - (np.arange(grid_size) + 0.5) * pixel
    xx, yy = np.meshgrid(x, y)
    queries = shapely.points(xx.ravel(), yy.ravel())
    pairs, distances = shapely.STRtree(geoms).query_nearest(queries, return_distance=True, all_matches=False)
    d = np.empty(len(queries), dtype=np.float64)
    d[pairs[0]] = distances
    occ = (d <= tau).astype(np.float64)
    dist = np.exp(-(d * d) / (2 * sigma * sigma))
    starts, stops, representatives = [], [], []
    for geom in geoms:
        for part in _iter_parts(geom):
            if isinstance(part, Point):
                representatives.append((part.x, part.y))
            else:
                for seq in _iter_coordinate_sequences(part):
                    coords = np.asarray(seq, dtype=np.float64)[:, :2]
                    for a, b in zip(coords[:-1], coords[1:]):
                        representatives.append((a + b) / 2)
                        if np.linalg.norm(b - a) > 0:
                            starts.append(a)
                            stops.append(b)
    orientation = np.full(len(queries), 0.5)
    if starts:
        aa, bb = np.asarray(starts), np.asarray(stops)
        segments = shapely.linestrings(np.stack([aa, bb], axis=1))
        pairs = shapely.STRtree(segments).query_nearest(queries, all_matches=True)
        # STRtree traversal order is unspecified. Break exact-distance ties by index.
        idx = np.full(len(queries), len(segments), dtype=np.int64)
        np.minimum.at(idx, pairs[0], pairs[1])
        theta = np.arctan2((bb - aa)[:, 1], (bb - aa)[:, 0])
        orientation = (theta[idx] + math.pi) / (2 * math.pi)
    raw = np.zeros((grid_size, grid_size), dtype=np.float64)
    points = np.asarray(representatives, dtype=np.float64).reshape(-1, 2)
    # Gaussian separability gives exactly the sum in Eq. 4 without rounding
    # representative points to pixels or truncating the kernel at a radius.
    for first in range(0, len(points), 4096):
        chunk = points[first:first + 4096]
        gx = np.exp(-((x[:, None] - chunk[:, 0]) ** 2) / (2 * h * h))
        gy = np.exp(-((y[:, None] - chunk[:, 1]) ** 2) / (2 * h * h))
        raw += gy @ gx.T
    density = raw / (float(raw.max()) + 1e-12)
    tensor = np.stack([occ.reshape(raw.shape), dist.reshape(raw.shape),
                       orientation.reshape(raw.shape), density]).astype(np.float32)
    return tensor, {"distance_sigma": float(sigma), "occupancy_tau": float(tau),
                    "density_bandwidth": float(h), "grid_limit": float(limit),
                    "representative_point_count": float(len(points)), "segment_count": float(len(starts))}
