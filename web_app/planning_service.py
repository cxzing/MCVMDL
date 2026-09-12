from __future__ import annotations

import json
import math
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy import ndimage


class PlanningError(Exception):
    def __init__(self, message: str, detail: str = "", status: int = 400):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.status = int(status)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise PlanningError("Direction cannot be a zero vector")
    return (vector / norm).astype(np.float32)


def build_ellipsoid_mask(
    shape: tuple[int, int, int], center: list[float] | np.ndarray, radii: list[float] | np.ndarray
) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    if c.shape != (3,) or r.shape != (3,):
        raise PlanningError("ROI center and radii must each contain three values")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)) or np.any(r < 1):
        raise PlanningError("ROI radii must be finite and at least one voxel")
    if np.any(c < 0) or np.any(c > np.asarray(shape, dtype=np.float64) - 1):
        raise PlanningError("ROI center is outside the tissue volume")
    grid = np.ogrid[tuple(slice(0, dim) for dim in shape)]
    distance = sum(((axis - c[index]) / r[index]) ** 2 for index, axis in enumerate(grid))
    return np.asarray(distance <= 1.0, dtype=bool)


def compute_surface_points(tissue: np.ndarray, maximum: int = 50000) -> np.ndarray:
    foreground = np.asarray(tissue) > 0
    if not np.any(foreground):
        raise PlanningError("Tissue contains no foreground voxels")
    interior = np.zeros_like(foreground, dtype=bool)
    interior[1:-1, 1:-1, 1:-1] = (
        foreground[1:-1, 1:-1, 1:-1]
        & foreground[:-2, 1:-1, 1:-1]
        & foreground[2:, 1:-1, 1:-1]
        & foreground[1:-1, :-2, 1:-1]
        & foreground[1:-1, 2:, 1:-1]
        & foreground[1:-1, 1:-1, :-2]
        & foreground[1:-1, 1:-1, 2:]
    )
    points = np.argwhere(foreground & ~interior).astype(np.float32)
    if len(points) > maximum:
        indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
        points = points[indices]
    return points


def _outside_background(tissue: np.ndarray) -> np.ndarray:
    foreground = np.asarray(tissue) > 0
    if not np.any(foreground):
        raise PlanningError("Tissue contains no foreground voxels")
    background = ~foreground
    outside = np.zeros_like(background, dtype=bool)
    outside[0, :, :] = background[0, :, :]
    outside[-1, :, :] = background[-1, :, :]
    outside[:, 0, :] |= background[:, 0, :]
    outside[:, -1, :] |= background[:, -1, :]
    outside[:, :, 0] |= background[:, :, 0]
    outside[:, :, -1] |= background[:, :, -1]
    structure = ndimage.generate_binary_structure(3, 1)
    return np.asarray(ndimage.binary_propagation(outside, structure=structure, mask=background), dtype=bool)


def detect_outer_shell_label(tissue: np.ndarray, requested: int | str | None = "auto") -> int:
    values = np.asarray(tissue)
    outside = _outside_background(values)
    structure = ndimage.generate_binary_structure(3, 1)
    exposed_foreground = ndimage.binary_dilation(outside, structure=structure) & (values > 0)
    labels, counts = np.unique(values[exposed_foreground], return_counts=True)
    valid = labels > 0
    labels, counts = labels[valid], counts[valid]
    if not len(labels):
        raise PlanningError("No tissue label touches the external boundary")
    if requested not in (None, "", "auto"):
        try:
            chosen = int(requested)
        except Exception as exc:
            raise PlanningError("surfaceLabel must be 'auto' or an integer", str(exc)) from exc
        if chosen <= 0 or chosen not in labels:
            raise PlanningError("Requested surface label is not exposed to outside air", str(chosen))
        return chosen
    return int(labels[int(np.argmax(counts))])


def compute_external_source_points(
    tissue: np.ndarray,
    maximum: int = 50000,
    surface_label: int | None = None,
) -> np.ndarray:
    values = np.asarray(tissue)
    foreground = values > 0
    outside = _outside_background(values)
    structure = ndimage.generate_binary_structure(3, 1)
    source_surface = foreground if surface_label is None else values == int(surface_label)
    if not np.any(source_surface):
        raise PlanningError("Requested surface label is absent from the tissue", str(surface_label))
    external = ndimage.binary_dilation(source_surface, structure=structure) & outside
    points = np.argwhere(external).astype(np.float32)
    if not len(points):
        raise PlanningError("No external source points were found next to the tissue surface")
    if len(points) > maximum:
        indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
        points = points[indices]
    return points


def _local_inward_normal(tissue: np.ndarray, surface_label: int, source: np.ndarray, radius: int = 2) -> np.ndarray:
    values = np.asarray(tissue)
    point = np.rint(source).astype(np.int64)
    lower = np.maximum(point - radius, 0)
    upper = np.minimum(point + radius + 1, np.asarray(values.shape))
    region = values[tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))]
    local = np.argwhere(region == int(surface_label))
    if not len(local):
        raise PlanningError("Could not estimate the local surface normal")
    coordinates = local.astype(np.float64) + lower.astype(np.float64)
    vector = np.mean(coordinates - np.asarray(source, dtype=np.float64), axis=0)
    return normalize_vector(vector)


def roi_surface_metadata(
    tissue: np.ndarray,
    center: list[float] | np.ndarray,
    radii: list[float] | np.ndarray,
    target_labels: list[int] | np.ndarray | None = None,
    surface_label: int | str | None = "auto",
) -> dict[str, Any]:
    values = np.asarray(tissue)
    c = np.asarray(center, dtype=np.float64)
    r = np.asarray(radii, dtype=np.float64)
    roi = build_ellipsoid_mask(tuple(values.shape), c, r)
    shell = detect_outer_shell_label(values, surface_label)
    sources = compute_external_source_points(values, maximum=500000, surface_label=shell)
    squared = np.sum((sources.astype(np.float64) - c) ** 2, axis=1)
    anchor = sources[int(np.argmin(squared))]
    depth = float(math.sqrt(float(np.min(squared))))
    target = values > 0 if not target_labels else np.isin(values, np.asarray(target_labels, dtype=np.int64))
    count = max(1, int(np.count_nonzero(roi)))
    return {
        "center": [float(value) for value in c],
        "surfaceAnchor": [float(value) for value in anchor],
        "centerDepthVox": depth,
        "coverDepthVox": depth - float(np.max(r)),
        "tissueFraction": float(np.count_nonzero(roi & (values > 0)) / count),
        "targetFraction": float(np.count_nonzero(roi & target) / count),
        "surfaceLabel": shell,
    }


def local_surface_samples(
    tissue: np.ndarray,
    count: int,
    roi_center: np.ndarray,
    roi_radii: np.ndarray,
    patch_radius: float | None = None,
    surface_label: int | str | None = "auto",
    max_incidence_deg: float = 60.0,
    direction_count: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    if count <= 0:
        raise PlanningError("Candidate position count must be positive")
    if not math.isfinite(float(max_incidence_deg)) or not 0 < float(max_incidence_deg) <= 90:
        raise PlanningError("maxIncidenceDeg must be in (0, 90]")
    center = np.asarray(roi_center, dtype=np.float64)
    radii = np.asarray(roi_radii, dtype=np.float64)
    targets = direction_targets(center, radii, int(direction_count))
    shell = detect_outer_shell_label(tissue, surface_label)
    points = compute_external_source_points(tissue, maximum=500000, surface_label=shell).astype(np.float64)
    anchor = points[int(np.argmin(np.sum((points - center) ** 2, axis=1)))]
    automatic = patch_radius is None
    radius = float(max(10.0, 1.5 * float(np.max(radii)))) if automatic else float(patch_radius)
    if not math.isfinite(radius) or not 4.0 <= radius <= 40.0:
        raise PlanningError("patchRadiusVox must be between 4 and 40")
    maximum_radius = 24.0 if automatic else radius
    required = max(count, count * 4)
    accepted_points: np.ndarray | None = None
    accepted_angles: np.ndarray | None = None
    while radius <= maximum_radius + 1e-9:
        patch = points[np.linalg.norm(points - anchor, axis=1) <= radius + 1e-6]
        angles: list[list[float]] = []
        valid: list[bool] = []
        for source in patch:
            inward = _local_inward_normal(tissue, shell, source)
            source_angles: list[float] = []
            for target in targets:
                direction = normalize_vector(target - source)
                cosine = float(np.clip(np.dot(inward, direction), -1.0, 1.0))
                source_angles.append(float(np.degrees(np.arccos(cosine))))
            angles.append(source_angles)
            valid.append(max(source_angles) <= float(max_incidence_deg) + 1e-6)
        mask = np.asarray(valid, dtype=bool)
        accepted_points = patch[mask]
        accepted_angles = np.asarray(angles, dtype=np.float64)[mask]
        if len(accepted_points) >= required or (radius >= maximum_radius and len(accepted_points) >= count):
            break
        radius += 2.0
    if accepted_points is None or accepted_angles is None or len(accepted_points) < count:
        raise PlanningError(
            "The local surface patch has too few admissible source positions",
            f"needed {count}, found {0 if accepted_points is None else len(accepted_points)}",
        )
    selected = farthest_surface_samples(accepted_points.astype(np.float32), count, center).astype(np.float32)
    angle_by_point = {
        tuple(np.rint(point).astype(np.int64).tolist()): [float(value) for value in angle]
        for point, angle in zip(accepted_points, accepted_angles, strict=True)
    }
    selected_angles_by_direction = [
        angle_by_point[tuple(np.rint(point).astype(np.int64).tolist())] for point in selected
    ]
    metadata = {
        "mode": "localPatch",
        "algorithmVersion": "local-surface-v2",
        "surfaceLabel": shell,
        "surfaceAnchor": [float(value) for value in anchor],
        "patchRadiusVox": float(radius),
        "automaticPatchRadius": bool(automatic),
        "maxIncidenceDeg": float(max_incidence_deg),
        "availablePointCount": int(len(accepted_points)),
        "selectedIncidenceDeg": [float(max(values)) for values in selected_angles_by_direction],
        "selectedIncidenceDegByDirection": selected_angles_by_direction,
        "maxSourceDistanceVox": float(np.max(np.linalg.norm(selected.astype(np.float64) - anchor, axis=1))),
    }
    return selected, metadata


def suggest_shallow_rois(
    tissue: np.ndarray,
    current_center: list[float] | np.ndarray,
    radii: list[float] | np.ndarray,
    target_labels: list[int] | np.ndarray | None = None,
    depth_range: tuple[float, float] = (9.0, 14.0),
    count: int = 1,
    min_tissue_fraction: float = 0.95,
    min_target_fraction: float = 0.90,
    min_separation: float = 18.0,
    surface_label: int | str | None = "auto",
    reference_center: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    values = np.asarray(tissue)
    center = np.asarray(current_center, dtype=np.float64)
    roi_radii = np.asarray(radii, dtype=np.float64)
    if count < 1 or count > 32:
        raise PlanningError("Suggestion count must be between 1 and 32")
    if len(depth_range) != 2 or not 0 <= float(depth_range[0]) < float(depth_range[1]):
        raise PlanningError("depthRangeVox must contain increasing non-negative values")
    shell = detect_outer_shell_label(values, surface_label)
    sources = compute_external_source_points(values, maximum=500000, surface_label=shell).astype(np.int64)
    source_mask = np.zeros(values.shape, dtype=bool)
    source_mask[tuple(sources.T)] = True
    distance, nearest = ndimage.distance_transform_edt(~source_mask, return_indices=True)
    labels = [int(value) for value in (target_labels or []) if int(value) > 0]
    if not labels:
        rounded = np.clip(np.rint(center).astype(np.int64), 0, np.asarray(values.shape) - 1)
        current_label = int(values[tuple(rounded)])
        labels = [current_label] if current_label > 0 else [int(value) for value in np.unique(values) if value > 0]
    missing = sorted(set(labels) - set(int(value) for value in np.unique(values)))
    if missing:
        raise PlanningError("Target labels are absent from the tissue", ", ".join(map(str, missing)))
    target_mask = np.isin(values, np.asarray(labels, dtype=np.int64))
    extents = np.ceil(roi_radii).astype(np.int64)
    kernel_grid = np.ogrid[tuple(slice(-extent, extent + 1) for extent in extents)]
    kernel = np.asarray(
        sum((axis / roi_radii[index]) ** 2 for index, axis in enumerate(kernel_grid)) <= 1.0,
        dtype=np.float32,
    )
    roi_voxels = max(1.0, float(np.sum(kernel)))
    tissue_fraction = ndimage.convolve((values > 0).astype(np.float32), kernel, mode="constant") / roi_voxels
    target_fraction = ndimage.convolve(target_mask.astype(np.float32), kernel, mode="constant") / roi_voxels
    low, high = (float(depth_range[0]), float(depth_range[1]))
    eligible = (
        target_mask
        & (distance >= low)
        & (distance <= high)
        & (tissue_fraction >= float(min_tissue_fraction))
        & (target_fraction >= float(min_target_fraction))
    )
    for axis, extent in enumerate(extents):
        indices = np.arange(values.shape[axis])
        valid_axis = (indices >= extent) & (indices < values.shape[axis] - extent)
        reshape = [1, 1, 1]
        reshape[axis] = values.shape[axis]
        eligible &= valid_axis.reshape(reshape)
    coordinates = np.argwhere(eligible).astype(np.float64)
    if not len(coordinates):
        raise PlanningError(
            "No shallow ROI satisfies the requested depth and occupancy constraints",
            f"depth {low:g}-{high:g}, tissue >= {min_tissue_fraction:.2f}, target >= {min_target_fraction:.2f}",
        )
    preferred_depth = 0.5 * (low + high)
    coord_index = tuple(coordinates.astype(np.int64).T)
    quality = (
        0.45 * target_fraction[coord_index]
        + 0.30 * tissue_fraction[coord_index]
        + 0.25 * (1.0 - np.minimum(1.0, np.abs(distance[coord_index] - preferred_depth) / max(1e-6, 0.5 * (high - low))))
    )
    selected: list[int] = []
    if reference_center is not None:
        reference = np.asarray(reference_center, dtype=np.float64)
        selected.append(int(np.argmin(np.sum((coordinates - reference) ** 2, axis=1))))
    elif count == 1:
        proximity = np.sum((coordinates - center) ** 2, axis=1)
        selected.append(int(np.lexsort((-quality, proximity))[0]))
    else:
        selected.append(int(np.argmax(quality)))
    minimum_distance = np.linalg.norm(coordinates - coordinates[selected[0]], axis=1)
    while len(selected) < count:
        valid = minimum_distance >= float(min_separation)
        valid[np.asarray(selected, dtype=np.int64)] = False
        if not np.any(valid):
            break
        objective = minimum_distance + quality
        objective[~valid] = -np.inf
        next_index = int(np.argmax(objective))
        selected.append(next_index)
        minimum_distance = np.minimum(minimum_distance, np.linalg.norm(coordinates - coordinates[next_index], axis=1))
    if len(selected) < count:
        raise PlanningError(
            "Too few separated shallow ROI suggestions were found",
            f"requested {count}, found {len(selected)} at separation {min_separation:g}",
        )
    suggestions: list[dict[str, Any]] = []
    for item_index in selected:
        coordinate = coordinates[item_index].astype(np.int64)
        key = tuple(coordinate.tolist())
        anchor = nearest[:, key[0], key[1], key[2]]
        suggestions.append(
            {
                "center": [int(value) for value in coordinate],
                "surfaceAnchor": [int(value) for value in anchor],
                "centerDepthVox": float(distance[key]),
                "coverDepthVox": float(distance[key] - np.max(roi_radii)),
                "tissueFraction": float(tissue_fraction[key]),
                "targetFraction": float(target_fraction[key]),
                "surfaceLabel": shell,
                "targetLabels": labels,
                "quality": float(quality[item_index]),
            }
        )
    return {
        "surfaceLabel": shell,
        "targetLabels": labels,
        "depthRangeVox": [low, high],
        "suggestions": suggestions,
        "current": roi_surface_metadata(values, center, roi_radii, labels, shell),
    }


def farthest_surface_samples(points: np.ndarray, count: int, roi_center: np.ndarray) -> np.ndarray:
    if count <= 0:
        raise PlanningError("Candidate position count must be positive")
    if len(points) < count:
        raise PlanningError("Tissue surface has fewer points than requested candidates")
    center = np.asarray(roi_center, dtype=np.float64)
    first = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
    selected = [first]
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    while len(selected) < count:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        min_distance = np.minimum(min_distance, np.sum((points - points[next_index]) ** 2, axis=1))
    return points[np.asarray(selected, dtype=np.int64)]


def direction_targets(center: np.ndarray, radii: np.ndarray, direction_count: int) -> list[np.ndarray]:
    center = np.asarray(center, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    targets = [center.copy()]
    if direction_count <= 1:
        return targets
    ordered_axes = np.argsort(-radii)
    offsets: list[np.ndarray] = []
    for axis in ordered_axes:
        delta = np.zeros(3, dtype=np.float64)
        delta[int(axis)] = max(0.5, float(radii[int(axis)]) * 0.5)
        offsets.extend([delta, -delta])
    for offset in offsets:
        if len(targets) >= direction_count:
            break
        targets.append(center + offset)
    return targets


def prediction_to_physical(prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float32)
    clipped = np.clip(values, 0.0, 1.0)
    physical = np.power(10.0, clipped * 10.0 - 10.0, dtype=np.float32)
    physical[values <= 0] = 0.0
    return physical


def dose_metrics(physical: np.ndarray, tissue: np.ndarray, roi_mask: np.ndarray) -> dict[str, float]:
    foreground = np.asarray(tissue) > 0
    target = foreground & roi_mask
    off_target = foreground & ~roi_mask
    if not np.any(target):
        raise PlanningError("ROI does not overlap foreground tissue")
    if not np.any(off_target):
        raise PlanningError("ROI leaves no off-target tissue for safety metrics")
    target_values = np.asarray(physical[target], dtype=np.float64)
    off_values = np.asarray(physical[off_target], dtype=np.float64)
    return {
        "targetAbsorption": float(np.mean(target_values)),
        "offTargetExposure": float(np.mean(off_values)),
        "hotspotRisk": float(np.percentile(off_values, 99.9)),
    }


def _scaled(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum <= 1e-20:
        return np.full(values.shape, 0.5, dtype=np.float64)
    return (values - minimum) / (maximum - minimum)


def rank_candidates(candidates: list[dict[str, Any]], weights: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidates:
        raise PlanningError("No candidate metrics are available")
    weight_values = np.asarray(
        [weights.get("target", 0.5), weights.get("offTarget", 0.3), weights.get("hotspot", 0.2)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weight_values)) or np.any(weight_values < 0) or float(weight_values.sum()) <= 0:
        raise PlanningError("Ranking weights must be non-negative and have a positive sum")
    weight_values /= weight_values.sum()
    normalized_weights = {
        "target": float(weight_values[0]),
        "offTarget": float(weight_values[1]),
        "hotspot": float(weight_values[2]),
    }

    target = np.asarray([item["targetAbsorption"] for item in candidates], dtype=np.float64)
    off_target = np.asarray([item["offTargetExposure"] for item in candidates], dtype=np.float64)
    hotspot = np.asarray([item["hotspotRisk"] for item in candidates], dtype=np.float64)
    target_n, off_n, hotspot_n = _scaled(target), _scaled(off_target), _scaled(hotspot)

    pareto = np.ones(len(candidates), dtype=bool)
    for index in range(len(candidates)):
        dominated = (
            (target >= target[index])
            & (off_target <= off_target[index])
            & (hotspot <= hotspot[index])
            & ((target > target[index]) | (off_target < off_target[index]) | (hotspot < hotspot[index]))
        )
        dominated[index] = False
        if np.any(dominated):
            pareto[index] = False

    ranked: list[dict[str, Any]] = []
    for index, original in enumerate(candidates):
        item = dict(original)
        item["normalized"] = {
            "target": float(target_n[index]),
            "offTarget": float(off_n[index]),
            "hotspot": float(hotspot_n[index]),
        }
        item["score"] = float(
            normalized_weights["target"] * target_n[index]
            + normalized_weights["offTarget"] * (1.0 - off_n[index])
            + normalized_weights["hotspot"] * (1.0 - hotspot_n[index])
        )
        item["pareto"] = bool(pareto[index])
        ranked.append(item)

    eligible = [item for item in ranked if item["pareto"]] or ranked
    eligible.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["targetAbsorption"]),
            float(item["offTargetExposure"]),
            float(item["hotspotRisk"]),
            str(item["id"]),
        )
    )
    best = dict(eligible[0])
    ranked.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    best = next(item for item in ranked if item["id"] == best["id"])
    return ranked, {"candidate": best, "weights": normalized_weights}


class PlanningManager:
    def __init__(
        self,
        output_root: Path,
        load_npz: Callable[[Path, str], np.ndarray],
        validate_tissue: Callable[[np.ndarray, int], np.ndarray],
        load_model: Callable[[Path, int, torch.device], torch.nn.Module],
        prepare_batch: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        get_device: Callable[[], torch.device],
        gpu_lock: threading.Lock,
    ) -> None:
        self.root = Path(output_root) / "plans"
        self.root.mkdir(parents=True, exist_ok=True)
        self.load_npz = load_npz
        self.validate_tissue = validate_tissue
        self.load_model = load_model
        self.prepare_batch = prepare_batch
        self.get_device = get_device
        self.gpu_lock = gpu_lock
        self.record_lock = threading.RLock()

    def _dir(self, plan_id: str) -> Path:
        return self.root / plan_id

    def _record_path(self, plan_id: str) -> Path:
        return self._dir(plan_id) / "plan.json"

    def _write(self, record: dict[str, Any]) -> None:
        with self.record_lock:
            directory = self._dir(record["planId"])
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "plan.json"
            temporary = directory / f"plan.{threading.get_ident()}.tmp"
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            for attempt in range(8):
                try:
                    temporary.replace(target)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.02 * (attempt + 1))

    def get(self, plan_id: str) -> dict[str, Any]:
        path = self._record_path(plan_id)
        if not path.exists():
            raise PlanningError("Planning job was not found", plan_id, 404)
        with self.record_lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self.root.glob("*/plan.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                records.append(
                    {
                        "planId": record.get("planId"),
                        "status": record.get("status"),
                        "createdAt": record.get("createdAt"),
                        "updatedAt": record.get("updatedAt"),
                        "bestCandidate": record.get("bestCandidate"),
                        "roi": record.get("config", {}).get("roi"),
                        "tissuePath": record.get("config", {}).get("tissuePath"),
                        "sampling": record.get("sampling") or {"mode": "legacyGlobal"},
                    }
                )
            except Exception:
                continue
        records.sort(key=lambda item: float(item.get("createdAt") or 0), reverse=True)
        return records

    def create(self, config: dict[str, Any]) -> dict[str, Any]:
        plan_id = uuid.uuid4().hex[:12]
        now = time.time()
        record = {
            "planId": plan_id,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "config": config,
            "progress": {"completed": 0, "total": int(config["positionCount"]) * int(config["directionCount"])},
            "candidates": [],
            "bestCandidate": None,
            "error": None,
        }
        with self.record_lock:
            self._write(record)
        thread = threading.Thread(target=self._run, args=(plan_id,), daemon=True, name=f"plan-{plan_id}")
        thread.start()
        return record

    def _prepare_context(self, config: dict[str, Any]) -> dict[str, Any]:
        channels = int(config["numChannels"])
        tissue_path = Path(config["tissuePath"])
        model_path = Path(config["modelPath"])
        tissue_np = self.validate_tissue(self.load_npz(tissue_path, "Tissue"), channels)
        tissue_tensor = torch.from_numpy(tissue_np).long()
        shape = tuple(int(value) for value in tissue_np.shape)
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(shape[0], dtype=torch.float32),
                torch.arange(shape[1], dtype=torch.float32),
                torch.arange(shape[2], dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        )
        device = self.get_device()
        with self.gpu_lock:
            model = self.load_model(model_path, channels, device)
        return {
            "channels": channels,
            "tissue": tissue_np,
            "tissueTensor": tissue_tensor,
            "tissueBatch": tissue_tensor.unsqueeze(0).to(device),
            "grid": grid,
            "device": device,
            "model": model,
        }

    @staticmethod
    def _internal_source(tissue: torch.Tensor, source: np.ndarray, direction: np.ndarray) -> torch.Tensor:
        position = torch.from_numpy(source.reshape(1, 3).astype(np.float32)).clone()
        vector = torch.from_numpy(direction.reshape(1, 3).astype(np.float32))
        nx, ny, nz = tissue.shape
        for _ in range(1000):
            x, y, z = (int(position[0, 0]), int(position[0, 1]), int(position[0, 2]))
            if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
                return torch.from_numpy(source.reshape(1, 3).astype(np.float32))
            if int(tissue[x, y, z]) != 0:
                return position
            position += vector * 0.5
        raise PlanningError("Could not move candidate source into foreground tissue")

    def _predict(self, context: dict[str, Any], source: np.ndarray, direction: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, float]:
        direction = normalize_vector(direction)
        internal = self._internal_source(context["tissueTensor"], source, direction)
        relative = context["grid"] - internal.view(1, 1, 1, 3)
        distance_squared = torch.sum(relative**2, dim=-1)
        mask = torch.exp(-distance_squared / (2.0 * sigma**2))
        relative_direction = relative / (torch.norm(relative, dim=-1, keepdim=True) + 1e-8)
        cos_theta = torch.sum(relative_direction * torch.from_numpy(direction).view(1, 1, 1, 3), dim=-1)
        mask[cos_theta < 0] = 0
        device: torch.device = context["device"]
        mask_batch = mask.unsqueeze(0).unsqueeze(0).to(device)
        start = time.perf_counter()
        with self.gpu_lock:
            if device.type == "cuda":
                torch.cuda.synchronize()
            with torch.inference_mode():
                model_input = self.prepare_batch(context["tissueBatch"], mask_batch, context["channels"])
                prediction = context["model"](model_input).squeeze(0).squeeze(0)
            if device.type == "cuda":
                torch.cuda.synchronize()
        inference = time.perf_counter() - start
        output = prediction.detach().float().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(output)):
            raise PlanningError("Model produced NaN or infinite values")
        return output, internal.numpy().reshape(3).astype(np.float32), float(inference)

    def _save_best(self, plan_id: str, prediction: np.ndarray, roi_mask: np.ndarray) -> None:
        directory = self._dir(plan_id)
        np.savez_compressed(directory / "best_prediction.npz", prediction.astype(np.float32))
        np.savez_compressed(directory / "roi_mask.npz", roi_mask.astype(np.uint8))

    def _run(self, plan_id: str) -> None:
        start = time.perf_counter()
        try:
            record = self.get(plan_id)
            config = record["config"]
            context = self._prepare_context(config)
            tissue = context["tissue"]
            center = np.asarray(config["roi"]["center"], dtype=np.float64)
            radii = np.asarray(config["roi"]["radii"], dtype=np.float64)
            roi_mask = build_ellipsoid_mask(tuple(tissue.shape), center, radii)
            _ = dose_metrics(np.zeros(tissue.shape, dtype=np.float32), tissue, roi_mask)
            sampling_config = config.get("sampling") or {"mode": "localPatch"}
            sampling_mode = str(sampling_config.get("mode") or "localPatch")
            if sampling_mode == "legacyGlobal":
                positions = farthest_surface_samples(
                    compute_external_source_points(tissue), int(config["positionCount"]), center
                )
                sampling_metadata = {
                    "mode": "legacyGlobal",
                    "algorithmVersion": "global-farthest-v1",
                }
            elif sampling_mode == "localPatch":
                positions, sampling_metadata = local_surface_samples(
                    tissue,
                    int(config["positionCount"]),
                    center,
                    radii,
                    patch_radius=sampling_config.get("patchRadiusVox"),
                    surface_label=sampling_config.get("surfaceLabel", "auto"),
                    max_incidence_deg=float(sampling_config.get("maxIncidenceDeg", 60.0)),
                    direction_count=int(config["directionCount"]),
                )
            else:
                raise PlanningError("Unsupported surface sampling mode", sampling_mode)
            targets = direction_targets(center, radii, int(config["directionCount"]))
            total = len(positions) * len(targets)
            record.update(
                {
                    "status": "running",
                    "updatedAt": time.time(),
                    "shape": [int(value) for value in tissue.shape],
                    "tissueLabels": [int(value) for value in np.unique(tissue).tolist()],
                    "roiVoxelCount": int(np.count_nonzero(roi_mask)),
                    "roiTissueVoxelCount": int(np.count_nonzero(roi_mask & (tissue > 0))),
                    "sampling": sampling_metadata,
                    "progress": {"completed": 0, "total": total},
                }
            )
            self._write(record)

            candidates: list[dict[str, Any]] = []
            sigma = float(config["sigma"])
            for position_index, source in enumerate(positions, start=1):
                for direction_index, target in enumerate(targets, start=1):
                    direction = normalize_vector(target - source)
                    prediction, internal, inference = self._predict(context, source, direction, sigma)
                    metric_values = dose_metrics(prediction_to_physical(prediction), tissue, roi_mask)
                    candidate = {
                        "id": f"P{position_index:02d}-D{direction_index}",
                        "positionIndex": position_index,
                        "directionIndex": direction_index,
                        "source": [float(value) for value in source],
                        "direction": [float(value) for value in direction],
                        "internalSource": [float(value) for value in internal],
                        "aimTarget": [float(value) for value in target],
                        "incidenceAngleDeg": (
                            sampling_metadata.get("selectedIncidenceDegByDirection", [])[position_index - 1][direction_index - 1]
                            if sampling_metadata.get("mode") == "localPatch"
                            and sampling_metadata.get("selectedIncidenceDegByDirection")
                            else (
                                sampling_metadata.get("selectedIncidenceDeg", [None] * len(positions))[position_index - 1]
                                if sampling_metadata.get("mode") == "localPatch"
                                else None
                            )
                        ),
                        "inferenceSec": inference,
                        **metric_values,
                    }
                    candidates.append(candidate)
                    record["candidates"] = candidates
                    record["progress"] = {"completed": len(candidates), "total": total}
                    record["updatedAt"] = time.time()
                    self._write(record)

            ranked, ranking = rank_candidates(candidates, config["weights"])
            best = ranking["candidate"]
            best_prediction, internal, best_inference = self._predict(
                context,
                np.asarray(best["source"], dtype=np.float32),
                np.asarray(best["direction"], dtype=np.float32),
                sigma,
            )
            self._save_best(plan_id, best_prediction, roi_mask)
            best = dict(best)
            best["internalSource"] = [float(value) for value in internal]
            best["resultInferenceSec"] = best_inference
            record.update(
                {
                    "status": "completed",
                    "updatedAt": time.time(),
                    "completedAt": time.time(),
                    "durationSec": float(time.perf_counter() - start),
                    "weights": ranking["weights"],
                    "candidates": ranked,
                    "bestCandidate": best,
                    "prediction": {
                        "min": float(np.min(best_prediction)),
                        "max": float(np.max(best_prediction)),
                    },
                    "progress": {"completed": total, "total": total},
                    "error": None,
                }
            )
            self._write(record)
        except Exception as exc:
            try:
                record = self.get(plan_id)
                record.update(
                    {
                        "status": "failed",
                        "updatedAt": time.time(),
                        "durationSec": float(time.perf_counter() - start),
                        "error": {
                            "message": getattr(exc, "message", str(exc)),
                            "detail": getattr(exc, "detail", traceback.format_exc(limit=6)),
                        },
                    }
                )
                self._write(record)
            except Exception:
                traceback.print_exc()

    def rerank(self, plan_id: str, weights: dict[str, float]) -> dict[str, Any]:
        with self.record_lock:
            record = self.get(plan_id)
            if record.get("status") != "completed":
                raise PlanningError("Only completed plans can be reranked", plan_id, 409)
            ranked, ranking = rank_candidates(record["candidates"], weights)
            previous_id = record.get("bestCandidate", {}).get("id")
            best = dict(ranking["candidate"])
            if best["id"] != previous_id:
                context = self._prepare_context(record["config"])
                prediction, internal, inference = self._predict(
                    context,
                    np.asarray(best["source"], dtype=np.float32),
                    np.asarray(best["direction"], dtype=np.float32),
                    float(record["config"]["sigma"]),
                )
                roi_mask = build_ellipsoid_mask(
                    tuple(context["tissue"].shape),
                    record["config"]["roi"]["center"],
                    record["config"]["roi"]["radii"],
                )
                self._save_best(plan_id, prediction, roi_mask)
                record["prediction"] = {"min": float(np.min(prediction)), "max": float(np.max(prediction))}
                best["internalSource"] = [float(value) for value in internal]
                best["resultInferenceSec"] = inference
            else:
                best["internalSource"] = record["bestCandidate"].get("internalSource", best.get("internalSource"))
                best["resultInferenceSec"] = record["bestCandidate"].get("resultInferenceSec")
            record["weights"] = ranking["weights"]
            record["config"]["weights"] = ranking["weights"]
            record["candidates"] = ranked
            record["bestCandidate"] = best
            record["updatedAt"] = time.time()
            self._write(record)
            return record

    def load_best(self, plan_id: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        record = self.get(plan_id)
        if record.get("status") != "completed":
            raise PlanningError("Planning result is not complete", plan_id, 409)
        prediction_path = self._dir(plan_id) / "best_prediction.npz"
        roi_path = self._dir(plan_id) / "roi_mask.npz"
        if not prediction_path.exists() or not roi_path.exists():
            raise PlanningError("Persisted planning volume is missing", plan_id, 500)
        tissue = self.validate_tissue(
            self.load_npz(Path(record["config"]["tissuePath"]), "Tissue"),
            int(record["config"]["numChannels"]),
        )
        with np.load(prediction_path) as data:
            prediction = np.asarray(data["arr_0"], dtype=np.float32)
        with np.load(roi_path) as data:
            roi = np.asarray(data["arr_0"], dtype=np.uint8).astype(bool)
        return record, tissue, prediction, roi

    def predict_source(
        self,
        plan_id: str,
        source: np.ndarray,
        direction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run one persisted plan model for a manually supplied source."""

        record = self.get(plan_id)
        if record.get("status") != "completed":
            raise PlanningError("Planning result is not complete", plan_id, 409)
        context = self._prepare_context(record["config"])
        return self._predict(
            context,
            np.asarray(source, dtype=np.float32),
            np.asarray(direction, dtype=np.float32),
            float(record["config"]["sigma"]),
        )
