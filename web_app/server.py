from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import torch


WEB_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mcvmdl import UNet3D_SourceMod  # noqa: E402
from utils.data_loader import prepare_batch_on_gpu  # noqa: E402
from web_app.mcvm_service import MCVMError, MCVMManager  # noqa: E402
from web_app.planning_service import PlanningError, PlanningManager, roi_surface_metadata, suggest_shallow_rois  # noqa: E402
from web_app.visual_style import (  # noqa: E402
    compose_difference_rgba,
    compose_scalar_rgba,
    display_floor,
    tissue_grayscale_rgba,
)


DEFAULT_MODEL_PATH = PROJECT_ROOT / "weight" / "best_model.pth"
DEFAULT_TISSUE_PATH = (
    PROJECT_ROOT
    / "data"
    / "ScatterBrains-Subject01-Full"
    / "Test"
    / "Tissue_Compressed.npz"
)
DEFAULT_SOURCE = [55.358497619628906, 25.445829391479492, 88.87934875488281]
DEFAULT_DIRECTION = [0.3178660571575165, 0.8198574185371399, -0.4762299954891205]
DEFAULT_SIGMA = 10.0
DEFAULT_NUM_CHANNELS = 17
DEFAULT_MCVM_EXE = WEB_ROOT / "bin" / "MCVM.exe"
PERSISTENT_RUN_ROOT = PROJECT_ROOT / "outputs" / "web_app"
MAX_CACHED_RUNS = 4
MAX_CACHED_PREVIEWS = 3
MAX_CACHED_MESHES = 3
UPLOAD_ROOT = PERSISTENT_RUN_ROOT / "uploads"
MODEL_EXTENSIONS = {".pth", ".pt"}
TISSUE_EXTENSIONS = {".npz"}


class ApiError(Exception):
    def __init__(self, message: str, detail: str = "", status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.status = int(status)


@dataclass
class RunResult:
    run_id: str
    tissue: np.ndarray
    prediction: np.ndarray
    mask: np.ndarray
    source: np.ndarray
    direction: np.ndarray
    internal_source: np.ndarray
    pred_min: float
    pred_max: float
    mask_min: float
    mask_max: float
    inference_sec: float
    total_sec: float
    device: str
    created_at: float


RUN_CACHE: OrderedDict[str, RunResult] = OrderedDict()
MODEL_CACHE: dict[tuple[str, int, str], torch.nn.Module] = {}
PREVIEW_CACHE: OrderedDict[tuple[str, int, int, float], dict[str, Any]] = OrderedDict()
MESH_CACHE: OrderedDict[tuple[str, int, int, int, float], dict[str, Any]] = OrderedDict()
CACHE_LOCK = Lock()
GPU_INFERENCE_LOCK = Lock()
SERVER_DEVICE: torch.device | None = None


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def normalize_path(value: str | None, default_path: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default_path.resolve()

    raw = Path(str(value).strip().strip('"'))
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        (PROJECT_ROOT / raw).resolve(),
        (Path.cwd() / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extensions_for_kind(kind: str) -> set[str]:
    if kind == "model":
        return MODEL_EXTENSIONS
    if kind == "tissue":
        return TISSUE_EXTENSIONS
    return MODEL_EXTENSIONS | TISSUE_EXTENSIONS


def default_browse_path(kind: str) -> Path:
    if kind == "model":
        return DEFAULT_MODEL_PATH.parent
    if kind == "tissue":
        return DEFAULT_TISSUE_PATH.parent
    return PROJECT_ROOT


def list_local_path(path_value: str | None, kind: str) -> dict[str, Any]:
    base = normalize_path(path_value, default_browse_path(kind))
    if base.is_file():
        base = base.parent
    if not base.exists():
        base = default_browse_path(kind)
    if not base.is_dir():
        raise ApiError("Browse path is not a directory", str(base))

    allowed = extensions_for_kind(kind)
    entries: list[dict[str, Any]] = []
    try:
        children = sorted(base.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except PermissionError as exc:
        raise ApiError("Permission denied while reading this directory", str(base), HTTPStatus.FORBIDDEN) from exc

    for item in children[:500]:
        is_dir = item.is_dir()
        suffix = item.suffix.lower()
        selectable = (not is_dir) and suffix in allowed
        if not is_dir and not selectable:
            continue
        try:
            stat = item.stat()
            size = int(stat.st_size) if not is_dir else None
            modified = float(stat.st_mtime)
        except OSError:
            size = None
            modified = None
        entries.append(
            {
                "name": item.name,
                "path": str(item),
                "isDir": is_dir,
                "selectable": selectable,
                "size": size,
                "modified": modified,
            }
        )

    parent = base.parent if base.parent != base else None
    return {
        "path": str(base),
        "parent": str(parent) if parent else None,
        "kind": kind,
        "entries": entries,
    }


def safe_upload_filename(filename: str, kind: str) -> str:
    name = Path(filename or f"{kind}_upload").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not cleaned:
        cleaned = f"{kind}_upload"
    suffix = Path(cleaned).suffix.lower()
    if suffix not in extensions_for_kind(kind):
        expected = ", ".join(sorted(extensions_for_kind(kind)))
        raise ApiError("Unsupported upload file type", f"{cleaned}; expected {expected}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}_{cleaned}"


def load_npz_array(path: Path, label: str) -> np.ndarray:
    if not path.exists():
        raise ApiError(f"{label} file not found", str(path), HTTPStatus.NOT_FOUND)
    if not path.is_file():
        raise ApiError(f"{label} is not a file", str(path))
    try:
        with np.load(path) as data:
            if "arr_0" not in data:
                raise ApiError(f"{label} is missing arr_0", f"{path} does not contain key arr_0")
            return np.asarray(data["arr_0"])
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(f"Failed to read {label}", f"{path}: {exc}") from exc


def inspect_tissue(tissue_path: Path, num_channels: int) -> dict[str, Any]:
    tissue_exists = tissue_path.exists() and tissue_path.is_file()
    info: dict[str, Any] = {
        "exists": tissue_exists,
        "path": str(tissue_path),
    }
    if not tissue_exists:
        return info

    tissue = load_npz_array(tissue_path, "Tissue")
    labels = np.unique(tissue)
    info.update(
        {
            "shape": [int(x) for x in tissue.shape],
            "dtype": str(tissue.dtype),
            "min": float(np.nanmin(tissue)),
            "max": float(np.nanmax(tissue)),
            "labels": [int(x) for x in labels.tolist()],
            "labelCount": int(labels.size),
            "labelsWithinChannels": bool(labels.size == 0 or (int(labels.min()) >= 0 and int(labels.max()) < num_channels)),
        }
    )
    if tissue.ndim == 3:
        info["sliderRanges"] = {
            "x": [0, int(tissue.shape[0] - 1)],
            "y": [0, int(tissue.shape[1] - 1)],
            "z": [0, int(tissue.shape[2] - 1)],
        }
        info["middleSlices"] = {
            "x": int(tissue.shape[0] // 2),
            "y": int(tissue.shape[1] // 2),
            "z": int(tissue.shape[2] // 2),
        }
    return info


def validate_tissue(tissue: np.ndarray, num_channels: int) -> np.ndarray:
    if tissue.ndim != 3:
        raise ApiError("Tissue array has invalid dimensions", f"Expected 3D array, got shape {tissue.shape}")
    if any(dim <= 0 for dim in tissue.shape):
        raise ApiError("Tissue array is empty", f"Invalid shape {tissue.shape}")
    if any(dim % 16 != 0 for dim in tissue.shape):
        raise ApiError("Tissue voxel size is incompatible", f"Each dimension must be divisible by 16 for the current U-Net, got {tissue.shape}")
    if not np.issubdtype(tissue.dtype, np.integer):
        if not np.all(np.equal(tissue, np.round(tissue))):
            raise ApiError("Tissue labels must be integers", f"Got dtype {tissue.dtype}")
    tissue_int = tissue.astype(np.int64, copy=False)
    label_min = int(np.min(tissue_int))
    label_max = int(np.max(tissue_int))
    if label_min < 0:
        raise ApiError("Tissue labels cannot be negative", f"Minimum label is {label_min}")
    if label_max >= num_channels:
        raise ApiError("Tissue labels exceed numChannels", f"Maximum label {label_max}, numChannels {num_channels}")
    return tissue_int


def parse_vector(payload: dict[str, Any], key: str) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ApiError(f"{key} must be an array of length 3", f"Got {value!r}")
    try:
        vector = np.asarray([float(v) for v in value], dtype=np.float32)
    except Exception as exc:
        raise ApiError(f"{key} contains a non-numeric value", str(exc)) from exc
    if not np.all(np.isfinite(vector)):
        raise ApiError(f"{key} must contain finite numbers", f"Got {value!r}")
    return vector


def parse_positive_float(payload: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(payload.get(key, default))
    except Exception as exc:
        raise ApiError(f"{key} must be numeric", str(exc)) from exc
    if not math.isfinite(value) or value <= 0:
        raise ApiError(f"{key} must be greater than 0", f"Got {value}")
    return value


def parse_num_channels(payload: dict[str, Any]) -> int:
    try:
        value = int(payload.get("numChannels", DEFAULT_NUM_CHANNELS))
    except Exception as exc:
        raise ApiError("numChannels must be an integer", str(exc)) from exc
    if value <= 0:
        raise ApiError("numChannels must be greater than 0", f"Got {value}")
    return value


def parse_max_points(payload: dict[str, Any]) -> int:
    try:
        value = int(payload.get("maxPoints", 10000))
    except Exception as exc:
        raise ApiError("maxPoints must be an integer", str(exc)) from exc
    return min(max(value, 1000), 24000)


def parse_mesh_stride(payload: dict[str, Any]) -> int:
    try:
        value = int(payload.get("stride", 2))
    except Exception as exc:
        raise ApiError("stride must be an integer", str(exc)) from exc
    return min(max(value, 1), 8)


def parse_max_faces(payload: dict[str, Any]) -> int:
    try:
        value = int(payload.get("maxFaces", 30000))
    except Exception as exc:
        raise ApiError("maxFaces must be an integer", str(exc)) from exc
    return min(max(value, 4000), 90000)


def build_tissue_preview(payload: dict[str, Any]) -> dict[str, Any]:
    num_channels = parse_num_channels(payload)
    max_points = parse_max_points(payload)
    tissue_path = normalize_path(payload.get("tissuePath"), DEFAULT_TISSUE_PATH)
    if not tissue_path.exists():
        raise ApiError("Tissue file not found", str(tissue_path), HTTPStatus.NOT_FOUND)
    cache_key = (str(tissue_path), num_channels, max_points, float(tissue_path.stat().st_mtime))
    with CACHE_LOCK:
        cached = PREVIEW_CACHE.get(cache_key)
        if cached is not None:
            PREVIEW_CACHE.move_to_end(cache_key)
            return cached

    tissue = validate_tissue(load_npz_array(tissue_path, "Tissue"), num_channels)
    foreground = tissue > 0
    if not np.any(foreground):
        raise ApiError("Tissue contains no foreground labels", str(tissue_path))

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
    surface = foreground & ~interior
    coords = np.argwhere(surface)
    if coords.size == 0:
        coords = np.argwhere(foreground)

    total_surface_points = int(coords.shape[0])
    if total_surface_points > max_points:
        rng = np.random.default_rng(20260313)
        chosen = np.sort(rng.choice(total_surface_points, size=max_points, replace=False))
        coords = coords[chosen]

    labels = tissue[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.int16, copy=False)
    points = np.column_stack((coords.astype(np.int16, copy=False), labels)).tolist()
    result = {
        "path": str(tissue_path),
        "shape": [int(x) for x in tissue.shape],
        "points": points,
        "pointCount": int(len(points)),
        "totalSurfacePoints": total_surface_points,
        "labels": [int(x) for x in np.unique(labels).tolist()],
    }
    with CACHE_LOCK:
        PREVIEW_CACHE[cache_key] = result
        PREVIEW_CACHE.move_to_end(cache_key)
        while len(PREVIEW_CACHE) > MAX_CACHED_PREVIEWS:
            PREVIEW_CACHE.popitem(last=False)
    return result


def downsample_tissue_for_mesh(tissue: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    shape = tissue.shape
    block_shape = tuple(int(math.ceil(dim / stride)) for dim in shape)
    padded_shape = tuple(dim * stride for dim in block_shape)
    pad_width = [(0, padded_shape[idx] - shape[idx]) for idx in range(3)]
    padded = np.pad(tissue, pad_width, mode="constant", constant_values=0)
    blocks = padded.reshape(block_shape[0], stride, block_shape[1], stride, block_shape[2], stride)
    occupancy = np.any(blocks > 0, axis=(1, 3, 5))
    labels = np.max(blocks, axis=(1, 3, 5)).astype(np.int16, copy=False)
    return occupancy, labels


def exposed_face_count(occupancy: np.ndarray) -> int:
    count = 0
    for axis in range(3):
        before = np.zeros_like(occupancy, dtype=bool)
        after = np.zeros_like(occupancy, dtype=bool)
        if axis == 0:
            before[1:, :, :] = occupancy[:-1, :, :]
            after[:-1, :, :] = occupancy[1:, :, :]
        elif axis == 1:
            before[:, 1:, :] = occupancy[:, :-1, :]
            after[:, :-1, :] = occupancy[:, 1:, :]
        else:
            before[:, :, 1:] = occupancy[:, :, :-1]
            after[:, :, :-1] = occupancy[:, :, 1:]
        count += int(np.sum(occupancy & ~before))
        count += int(np.sum(occupancy & ~after))
    return count


def voxel_bounds(i: int, j: int, k: int, stride: int, shape: tuple[int, int, int]) -> tuple[float, float, float, float, float, float]:
    x0 = max(i * stride - 0.5, 0.0)
    y0 = max(j * stride - 0.5, 0.0)
    z0 = max(k * stride - 0.5, 0.0)
    x1 = min((i + 1) * stride - 0.5, float(shape[0] - 1))
    y1 = min((j + 1) * stride - 0.5, float(shape[1] - 1))
    z1 = min((k + 1) * stride - 0.5, float(shape[2] - 1))
    return x0, x1, y0, y1, z0, z1


def append_mesh_faces(
    faces: list[dict[str, Any]],
    coords: np.ndarray,
    labels: np.ndarray,
    stride: int,
    shape: tuple[int, int, int],
    direction: str,
) -> None:
    normal_map = {
        "x-": [-1, 0, 0],
        "x+": [1, 0, 0],
        "y-": [0, -1, 0],
        "y+": [0, 1, 0],
        "z-": [0, 0, -1],
        "z+": [0, 0, 1],
    }
    for i, j, k in coords.tolist():
        x0, x1, y0, y1, z0, z1 = voxel_bounds(i, j, k, stride, shape)
        if direction == "x-":
            vertices = [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]
        elif direction == "x+":
            vertices = [[x1, y0, z0], [x1, y0, z1], [x1, y1, z1], [x1, y1, z0]]
        elif direction == "y-":
            vertices = [[x0, y0, z0], [x0, y0, z1], [x1, y0, z1], [x1, y0, z0]]
        elif direction == "y+":
            vertices = [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]
        elif direction == "z-":
            vertices = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]
        else:
            vertices = [[x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]]
        faces.append(
            {
                "v": vertices,
                "n": normal_map[direction],
                "label": int(labels[i, j, k]),
            }
        )


def build_faces_from_occupancy(occupancy: np.ndarray, labels: np.ndarray, stride: int, shape: tuple[int, int, int]) -> list[dict[str, Any]]:
    faces: list[dict[str, Any]] = []
    for axis, negative_name, positive_name in ((0, "x-", "x+"), (1, "y-", "y+"), (2, "z-", "z+")):
        before = np.zeros_like(occupancy, dtype=bool)
        after = np.zeros_like(occupancy, dtype=bool)
        if axis == 0:
            before[1:, :, :] = occupancy[:-1, :, :]
            after[:-1, :, :] = occupancy[1:, :, :]
        elif axis == 1:
            before[:, 1:, :] = occupancy[:, :-1, :]
            after[:, :-1, :] = occupancy[:, 1:, :]
        else:
            before[:, :, 1:] = occupancy[:, :, :-1]
            after[:, :, :-1] = occupancy[:, :, 1:]
        append_mesh_faces(faces, np.argwhere(occupancy & ~before), labels, stride, shape, negative_name)
        append_mesh_faces(faces, np.argwhere(occupancy & ~after), labels, stride, shape, positive_name)
    return faces


def build_tissue_mesh(payload: dict[str, Any]) -> dict[str, Any]:
    num_channels = parse_num_channels(payload)
    stride = parse_mesh_stride(payload)
    max_faces = parse_max_faces(payload)
    tissue_path = normalize_path(payload.get("tissuePath"), DEFAULT_TISSUE_PATH)
    if not tissue_path.exists():
        raise ApiError("Tissue file not found", str(tissue_path), HTTPStatus.NOT_FOUND)
    cache_key = (str(tissue_path), num_channels, stride, max_faces, float(tissue_path.stat().st_mtime))
    with CACHE_LOCK:
        cached = MESH_CACHE.get(cache_key)
        if cached is not None:
            MESH_CACHE.move_to_end(cache_key)
            return cached

    tissue = validate_tissue(load_npz_array(tissue_path, "Tissue"), num_channels)
    chosen_stride = stride
    occupancy, labels = downsample_tissue_for_mesh(tissue, chosen_stride)
    face_count = exposed_face_count(occupancy)
    while face_count > max_faces and chosen_stride < 8:
        chosen_stride += 1
        occupancy, labels = downsample_tissue_for_mesh(tissue, chosen_stride)
        face_count = exposed_face_count(occupancy)
    if not np.any(occupancy):
        raise ApiError("Tissue contains no foreground labels", str(tissue_path))

    faces = build_faces_from_occupancy(occupancy, labels, chosen_stride, tuple(int(x) for x in tissue.shape))
    result = {
        "path": str(tissue_path),
        "shape": [int(x) for x in tissue.shape],
        "stride": int(chosen_stride),
        "faces": faces,
        "faceCount": int(len(faces)),
        "labels": [int(x) for x in np.unique(labels[labels > 0]).tolist()],
    }
    with CACHE_LOCK:
        MESH_CACHE[cache_key] = result
        MESH_CACHE.move_to_end(cache_key)
        while len(MESH_CACHE) > MAX_CACHED_MESHES:
            MESH_CACHE.popitem(last=False)
    return result


def find_internal_light_source(tissue: torch.Tensor, initial_light_pos: torch.Tensor, light_dir: torch.Tensor) -> torch.Tensor:
    light_pos = initial_light_pos.clone()
    nx, ny, nz = tissue.shape

    def is_out_of_bounds(pos: torch.Tensor) -> bool:
        x, y, z = int(pos[0, 0]), int(pos[0, 1]), int(pos[0, 2])
        return not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz)

    if is_out_of_bounds(light_pos):
        return initial_light_pos

    if tissue[int(light_pos[0, 0]), int(light_pos[0, 1]), int(light_pos[0, 2])] != 0:
        return light_pos

    step_size = 0.5
    max_steps = 1000
    for _ in range(max_steps):
        light_pos += light_dir * step_size
        if is_out_of_bounds(light_pos):
            return initial_light_pos
        if tissue[int(light_pos[0, 0]), int(light_pos[0, 1]), int(light_pos[0, 2])] != 0:
            return light_pos
    raise ApiError("Could not find an internal light source position", "Check source and direction")


def create_gaussian_mask(tissue: torch.Tensor, light_pos: torch.Tensor, light_dir: torch.Tensor, sigma: float) -> torch.Tensor:
    direction_norm = torch.norm(light_dir)
    if float(direction_norm.item()) <= 0:
        raise ApiError("Light direction cannot be a zero vector", "direction norm is zero")
    light_dir = light_dir / direction_norm
    nx, ny, nz = tissue.shape
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(nx, dtype=torch.float32),
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nz, dtype=torch.float32),
            indexing="ij",
        ),
        dim=-1,
    )
    relative_vec = grid - light_pos.view(1, 1, 1, 3)
    distance_squared = torch.sum(relative_vec**2, dim=-1)
    mask = torch.exp(-distance_squared / (2 * sigma**2))

    relative_dir = relative_vec / (torch.norm(relative_vec, dim=-1, keepdim=True) + 1e-8)
    cos_theta = torch.sum(relative_dir * light_dir.view(1, 1, 1, 3), dim=-1)
    mask[cos_theta < 0] = 0
    return mask


def load_model(model_path: Path, num_channels: int, device: torch.device) -> torch.nn.Module:
    key = (str(model_path), num_channels, str(device))
    with CACHE_LOCK:
        cached = MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    if not model_path.exists():
        raise ApiError("Weight file not found", str(model_path), HTTPStatus.NOT_FOUND)
    model = UNet3D_SourceMod(in_channels=num_channels + 1, out_channels=1).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            for state_key in ("state_dict", "model_state_dict", "model"):
                if state_key in checkpoint and isinstance(checkpoint[state_key], dict):
                    checkpoint = checkpoint[state_key]
                    break
        if isinstance(checkpoint, dict):
            checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError("Failed to load model weights", f"{model_path}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR) from exc
    model.eval()

    with CACHE_LOCK:
        MODEL_CACHE[key] = model
    return model


def run_simulation(payload: dict[str, Any]) -> dict[str, Any]:
    total_start = time.perf_counter()
    num_channels = parse_num_channels(payload)
    model_path = normalize_path(payload.get("modelPath"), DEFAULT_MODEL_PATH)
    tissue_path = normalize_path(payload.get("tissuePath"), DEFAULT_TISSUE_PATH)
    source = parse_vector(payload, "source")
    direction = parse_vector(payload, "direction")
    sigma = parse_positive_float(payload, "sigma", DEFAULT_SIGMA)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 0:
        raise ApiError("Light direction cannot be a zero vector", "direction norm is zero")
    direction = direction / direction_norm

    tissue_np = validate_tissue(load_npz_array(tissue_path, "Tissue"), num_channels)
    tissue_tensor = torch.from_numpy(tissue_np).long()
    source_tensor = torch.from_numpy(source.reshape(1, 3)).float()
    direction_tensor = torch.from_numpy(direction.reshape(1, 3)).float()
    internal_source = torch.round(find_internal_light_source(tissue_tensor, source_tensor, direction_tensor)).float()
    mask_tensor = create_gaussian_mask(tissue_tensor, internal_source, direction_tensor, sigma=sigma)

    device = SERVER_DEVICE or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tissue_batch = tissue_tensor.unsqueeze(0).to(device)
    mask_batch = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)
    with GPU_INFERENCE_LOCK:
        model = load_model(model_path, num_channels, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_start = time.perf_counter()
        with torch.inference_mode():
            model_input = prepare_batch_on_gpu(tissue_batch, mask_batch, num_channels)
            prediction = model(model_input).squeeze(0).squeeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
    inference_sec = time.perf_counter() - infer_start

    pred_np = prediction.detach().float().cpu().numpy().astype(np.float32, copy=False)
    if not np.all(np.isfinite(pred_np)):
        raise ApiError("Model output contains NaN/Inf", "Prediction is not finite", HTTPStatus.INTERNAL_SERVER_ERROR)
    mask_np = mask_tensor.cpu().numpy().astype(np.float32, copy=False)

    run_id = uuid.uuid4().hex[:12]
    result = RunResult(
        run_id=run_id,
        tissue=tissue_np.astype(np.int16, copy=False),
        prediction=pred_np,
        mask=mask_np,
        source=source.astype(np.float32, copy=False),
        direction=direction.astype(np.float32, copy=False),
        internal_source=internal_source.cpu().numpy().reshape(3).astype(np.float32),
        pred_min=float(np.min(pred_np)),
        pred_max=float(np.max(pred_np)),
        mask_min=float(np.min(mask_np)),
        mask_max=float(np.max(mask_np)),
        inference_sec=float(inference_sec),
        total_sec=float(time.perf_counter() - total_start),
        device=str(device),
        created_at=time.time(),
    )
    with CACHE_LOCK:
        RUN_CACHE[run_id] = result
        RUN_CACHE.move_to_end(run_id)
        while len(RUN_CACHE) > MAX_CACHED_RUNS:
            RUN_CACHE.popitem(last=False)

    return {
        "runId": run_id,
        "shape": [int(x) for x in pred_np.shape],
        "prediction": {"min": result.pred_min, "max": result.pred_max},
        "mask": {"min": result.mask_min, "max": result.mask_max},
        "source": source.tolist(),
        "direction": direction.tolist(),
        "internalSource": result.internal_source.tolist(),
        "inferenceSec": result.inference_sec,
        "totalSec": result.total_sec,
        "device": result.device,
    }


def extract_slice(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "x":
        return volume[index, :, :]
    if axis == "y":
        return volume[:, index, :]
    if axis == "z":
        return volume[:, :, index]
    raise ApiError("axis must be x, y, or z", f"Got {axis!r}")


def source_projection(axis: str, index: int, source: np.ndarray) -> tuple[int, int] | None:
    sx, sy, sz = [int(round(float(v))) for v in source.tolist()]
    if axis == "x" and sx == index:
        return sz, sy
    if axis == "y" and sy == index:
        return sz, sx
    if axis == "z" and sz == index:
        return sy, sx
    return None


def draw_source_marker(rgba: np.ndarray, axis: str, index: int, source: np.ndarray) -> tuple[int, int] | None:
    projection = source_projection(axis, index, source)
    if projection is None:
        return None
    col, row = projection
    h, w, _ = rgba.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    for radius, color in ((5, (255, 255, 255, 255)), (3, (224, 42, 42, 255)), (1, (255, 244, 168, 255))):
        for dy in range(-radius, radius + 1):
            rr = row + dy
            if 0 <= rr < h:
                rgba[rr, col] = color
        for dx in range(-radius, radius + 1):
            cc = col + dx
            if 0 <= cc < w:
                rgba[row, cc] = color
    return col, row


def render_slice(result: RunResult, axis: str, index: int) -> tuple[np.ndarray, dict[str, str]]:
    shape = result.prediction.shape
    axis_to_dim = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_dim:
        raise ApiError("axis must be x, y, or z", f"Got {axis!r}")
    dim = axis_to_dim[axis]
    if index < 0 or index >= shape[dim]:
        raise ApiError("Slice index is out of range", f"{axis}={index}, valid range 0..{shape[dim] - 1}")

    pred = extract_slice(result.prediction, axis, index)
    tissue = extract_slice(result.tissue, axis, index)
    maximum = max(float(result.pred_max), 1e-6)
    rgba = compose_scalar_rgba(tissue, pred, maximum)
    source_pixel = draw_source_marker(rgba, axis, index, result.internal_source)

    headers = {
        "X-Width": str(int(rgba.shape[1])),
        "X-Height": str(int(rgba.shape[0])),
        "X-Source-On-Slice": "1" if source_pixel else "0",
        "X-Display-Max": f"{maximum:.9g}",
        "X-Display-Floor": f"{display_floor(maximum):.9g}",
    }
    if source_pixel:
        headers["X-Source-Col"] = str(source_pixel[0])
        headers["X-Source-Row"] = str(source_pixel[1])
    return rgba, headers


def _slice_headers(rgba: np.ndarray) -> dict[str, str]:
    return {"X-Width": str(int(rgba.shape[1])), "X-Height": str(int(rgba.shape[0]))}


def render_tissue_slice(tissue: np.ndarray, axis: str, index: int) -> tuple[np.ndarray, dict[str, str]]:
    if axis not in {"x", "y", "z"}:
        raise ApiError("axis must be x, y, or z", axis)
    dim = {"x": 0, "y": 1, "z": 2}[axis]
    if index < 0 or index >= tissue.shape[dim]:
        raise ApiError("Slice index is out of range", f"{axis}={index}")
    labels = extract_slice(tissue, axis, index).astype(np.int32, copy=False)
    rgba = tissue_grayscale_rgba(labels)
    return rgba, _slice_headers(rgba)


def _roi_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = mask.copy()
    if mask.shape[0] > 2 and mask.shape[1] > 2:
        interior = np.zeros_like(mask, dtype=bool)
        interior[1:-1, 1:-1] = mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
        boundary &= ~interior
    return boundary


def render_scalar_slice(
    tissue: np.ndarray,
    volume: np.ndarray,
    axis: str,
    index: int,
    value_min: float,
    value_max: float,
    roi_mask: np.ndarray | None = None,
    source: np.ndarray | None = None,
    difference: bool = False,
) -> tuple[np.ndarray, dict[str, str]]:
    values = extract_slice(volume, axis, index).astype(np.float32, copy=False)
    tissue_slice = extract_slice(tissue, axis, index)
    if difference:
        rgba = compose_difference_rgba(tissue_slice, values, value_max)
    else:
        rgba = compose_scalar_rgba(tissue_slice, values, value_max)
    if roi_mask is not None:
        boundary = _roi_boundary(extract_slice(roi_mask, axis, index).astype(bool))
        rgba[boundary] = np.asarray([42, 238, 209, 255], dtype=np.uint8)
    if source is not None:
        draw_source_marker(rgba, axis, index, source)
    headers = _slice_headers(rgba)
    headers["X-Display-Max"] = f"{float(value_max):.9g}"
    headers["X-Display-Floor"] = f"{display_floor(abs(float(value_max))):.9g}"
    headers["X-Display-Kind"] = "difference" if difference else "absorption"
    return rgba, headers


def current_device() -> torch.device:
    return SERVER_DEVICE or torch.device("cuda" if torch.cuda.is_available() else "cpu")


PLAN_MANAGER = PlanningManager(
    output_root=PERSISTENT_RUN_ROOT,
    load_npz=load_npz_array,
    validate_tissue=validate_tissue,
    load_model=load_model,
    prepare_batch=prepare_batch_on_gpu,
    get_device=current_device,
    gpu_lock=GPU_INFERENCE_LOCK,
)
MCVM_MANAGER = MCVMManager(
    output_root=PERSISTENT_RUN_ROOT,
    mcvm_exe=DEFAULT_MCVM_EXE,
    planning=PLAN_MANAGER,
)


class MCVMRequestHandler(BaseHTTPRequestHandler):
    server_version = "MCVMWeb/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ApiError("Failed to parse request JSON", str(exc)) from exc
        if not isinstance(payload, dict):
            raise ApiError("Request body must be a JSON object", f"Got {type(payload).__name__}")
        return payload

    def handle_upload(self, kind: str) -> dict[str, Any]:
        if kind not in {"model", "tissue"}:
            raise ApiError("Invalid upload target", f"Got {kind!r}")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ApiError("Invalid upload length", str(exc)) from exc
        if length <= 0:
            raise ApiError("Uploaded file is empty", "Content-Length is 0")
        filename = unquote(self.headers.get("X-Filename", f"{kind}_upload"))
        safe_name = safe_upload_filename(filename, kind)
        target_dir = UPLOAD_ROOT / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        remaining = length
        with target_path.open("wb") as output:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                output.write(chunk)
                remaining -= len(chunk)
        if remaining != 0:
            try:
                target_path.unlink()
            except OSError:
                pass
            raise ApiError("Upload did not complete", f"{remaining} bytes remaining")
        return {
            "path": str(target_path),
            "name": target_path.name,
            "size": int(target_path.stat().st_size),
            "kind": kind,
        }

    def send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_api_error(self, exc: Exception) -> None:
        if isinstance(exc, ApiError):
            self.send_json({"error": exc.message, "detail": exc.detail}, exc.status)
            return
        if isinstance(exc, (PlanningError, MCVMError)):
            self.send_json({"error": exc.message, "detail": exc.detail}, exc.status)
            return
        traceback.print_exc()
        self.send_json({"error": "Internal server error", "detail": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_rgba(self, rgba: np.ndarray, headers: dict[str, str]) -> None:
        body = rgba.tobytes(order="C")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/inspect":
                payload = self.read_json_body()
                num_channels = parse_num_channels(payload)
                tissue_path = normalize_path(payload.get("tissuePath"), DEFAULT_TISSUE_PATH)
                model_path = normalize_path(payload.get("modelPath"), DEFAULT_MODEL_PATH)
                self.send_json(
                    {
                        "tissue": inspect_tissue(tissue_path, num_channels),
                        "model": {"exists": model_path.exists() and model_path.is_file(), "path": str(model_path)},
                        "numChannels": num_channels,
                    }
                )
                return
            if parsed.path == "/api/simulate":
                payload = self.read_json_body()
                self.send_json(run_simulation(payload))
                return
            if parsed.path == "/api/tissue-preview":
                payload = self.read_json_body()
                self.send_json(build_tissue_preview(payload))
                return
            if parsed.path == "/api/tissue-mesh":
                payload = self.read_json_body()
                self.send_json(build_tissue_mesh(payload))
                return
            if parsed.path == "/api/tissue-slice":
                payload = self.read_json_body()
                num_channels = parse_num_channels(payload)
                tissue_path = normalize_path(payload.get("tissuePath"), DEFAULT_TISSUE_PATH)
                tissue = validate_tissue(load_npz_array(tissue_path, "Tissue"), num_channels)
                axis = str(payload.get("axis") or "").lower()
                try:
                    index = int(payload.get("index"))
                except Exception as exc:
                    raise ApiError("index must be an integer", str(exc)) from exc
                rgba, headers = render_tissue_slice(tissue, axis, index)
                self.send_rgba(rgba, headers)
                return
            if parsed.path == "/api/roi-suggestions":
                payload = self.read_json_body()
                num_channels = parse_num_channels(payload)
                tissue_path = normalize_path(payload.get("tissuePath"), DEFAULT_TISSUE_PATH)
                tissue = validate_tissue(load_npz_array(tissue_path, "Tissue"), num_channels)
                roi = payload.get("roi")
                if not isinstance(roi, dict):
                    raise ApiError("roi must be an object")
                center = parse_vector(roi, "center")
                radii = parse_vector(roi, "radii")
                raw_labels = payload.get("targetLabels") or []
                if not isinstance(raw_labels, list):
                    raise ApiError("targetLabels must be an array")
                try:
                    target_labels = [int(value) for value in raw_labels]
                    depth_values = payload.get("depthRangeVox") or [9.0, 14.0]
                    if not isinstance(depth_values, list) or len(depth_values) != 2:
                        raise ValueError("depthRangeVox must contain two values")
                    depth_range = (float(depth_values[0]), float(depth_values[1]))
                    suggestion_count = int(payload.get("count", 1))
                    min_separation = float(payload.get("minSeparationVox", 18.0))
                    reference_center = payload.get("referenceCenter")
                    if reference_center is not None:
                        reference_center = np.asarray(reference_center, dtype=np.float64)
                        if reference_center.shape != (3,) or not np.all(np.isfinite(reference_center)):
                            raise ValueError("referenceCenter must contain three finite values")
                except Exception as exc:
                    raise ApiError("Invalid ROI suggestion parameters", str(exc)) from exc
                if bool(payload.get("inspectOnly")):
                    current = roi_surface_metadata(
                        tissue,
                        center,
                        radii,
                        target_labels,
                        payload.get("surfaceLabel", "auto"),
                    )
                    self.send_json(
                        {
                            "surfaceLabel": current["surfaceLabel"],
                            "targetLabels": target_labels,
                            "suggestions": [],
                            "current": current,
                        }
                    )
                else:
                    self.send_json(
                        suggest_shallow_rois(
                            tissue,
                            center,
                            radii,
                            target_labels=target_labels,
                            depth_range=depth_range,
                            count=suggestion_count,
                            min_tissue_fraction=float(payload.get("minTissueFraction", 0.95)),
                            min_target_fraction=float(payload.get("minTargetFraction", 0.90)),
                            min_separation=min_separation,
                            surface_label=payload.get("surfaceLabel", "auto"),
                            reference_center=reference_center,
                        )
                    )
                return
            if parsed.path == "/api/plans":
                payload = self.read_json_body()
                num_channels = parse_num_channels(payload)
                model_path = normalize_path(payload.get("modelPath"), DEFAULT_MODEL_PATH)
                tissue_path = normalize_path(payload.get("tissuePath"), DEFAULT_TISSUE_PATH)
                if not model_path.exists() or not model_path.is_file():
                    raise ApiError("Weight file not found", str(model_path), HTTPStatus.NOT_FOUND)
                tissue = validate_tissue(load_npz_array(tissue_path, "Tissue"), num_channels)
                roi = payload.get("roi")
                if not isinstance(roi, dict):
                    raise ApiError("roi must be an object")
                center = parse_vector(roi, "center")
                radii = parse_vector(roi, "radii")
                try:
                    position_count = int(payload.get("positionCount", 16))
                    direction_count = int(payload.get("directionCount", 3))
                except Exception as exc:
                    raise ApiError("Candidate counts must be integers", str(exc)) from exc
                if not 4 <= position_count <= 64:
                    raise ApiError("positionCount must be between 4 and 64")
                if not 1 <= direction_count <= 5:
                    raise ApiError("directionCount must be between 1 and 5")
                weights = payload.get("weights") or {"target": 0.5, "offTarget": 0.3, "hotspot": 0.2}
                if not isinstance(weights, dict):
                    raise ApiError("weights must be an object")
                sampling_value = payload.get("sampling") or {}
                if not isinstance(sampling_value, dict):
                    raise ApiError("sampling must be an object")
                sampling_mode = str(sampling_value.get("mode") or "localPatch")
                if sampling_mode not in {"localPatch", "legacyGlobal"}:
                    raise ApiError("sampling.mode must be localPatch or legacyGlobal")
                radius_value = sampling_value.get("patchRadiusVox")
                if radius_value in (None, "", "auto"):
                    patch_radius = None
                else:
                    try:
                        patch_radius = float(radius_value)
                    except Exception as exc:
                        raise ApiError("sampling.patchRadiusVox must be numeric or null", str(exc)) from exc
                    if not 4 <= patch_radius <= 40:
                        raise ApiError("sampling.patchRadiusVox must be between 4 and 40")
                surface_value = sampling_value.get("surfaceLabel", "auto")
                if surface_value not in (None, "", "auto"):
                    try:
                        surface_value = int(surface_value)
                    except Exception as exc:
                        raise ApiError("sampling.surfaceLabel must be auto or an integer", str(exc)) from exc
                try:
                    max_incidence = float(sampling_value.get("maxIncidenceDeg", 60.0))
                except Exception as exc:
                    raise ApiError("sampling.maxIncidenceDeg must be numeric", str(exc)) from exc
                if not 0 < max_incidence <= 90:
                    raise ApiError("sampling.maxIncidenceDeg must be in (0, 90]")
                config = {
                    "modelPath": str(model_path),
                    "tissuePath": str(tissue_path),
                    "numChannels": num_channels,
                    "sigma": parse_positive_float(payload, "sigma", DEFAULT_SIGMA),
                    "roi": {"center": center.tolist(), "radii": radii.tolist()},
                    "positionCount": position_count,
                    "directionCount": direction_count,
                    "weights": {
                        "target": float(weights.get("target", 0.5)),
                        "offTarget": float(weights.get("offTarget", 0.3)),
                        "hotspot": float(weights.get("hotspot", 0.2)),
                    },
                    "sampling": {
                        "mode": sampling_mode,
                        "patchRadiusVox": patch_radius,
                        "surfaceLabel": surface_value or "auto",
                        "maxIncidenceDeg": max_incidence,
                    },
                    "shape": [int(value) for value in tissue.shape],
                }
                self.send_json(PLAN_MANAGER.create(config), HTTPStatus.ACCEPTED)
                return
            plan_rerank = re.fullmatch(r"/api/plans/([a-f0-9]{12})/rerank", parsed.path)
            if plan_rerank:
                payload = self.read_json_body()
                weights = payload.get("weights") if isinstance(payload.get("weights"), dict) else payload
                self.send_json(PLAN_MANAGER.rerank(plan_rerank.group(1), weights))
                return
            if parsed.path == "/api/mcvm/jobs":
                self.send_json(MCVM_MANAGER.create(self.read_json_body()), HTTPStatus.ACCEPTED)
                return
            mcvm_cancel = re.fullmatch(r"/api/mcvm/jobs/([a-f0-9]{12})/cancel", parsed.path)
            if mcvm_cancel:
                self.send_json(MCVM_MANAGER.cancel(mcvm_cancel.group(1)), HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/upload":
                params = parse_qs(parsed.query)
                kind = params.get("kind", [""])[0]
                self.send_json(self.handle_upload(kind))
                return
            raise ApiError("API endpoint not found", parsed.path, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_api_error(exc)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/defaults":
                device = SERVER_DEVICE or torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.send_json(
                    {
                        "modelPath": str(DEFAULT_MODEL_PATH),
                        "tissuePath": str(DEFAULT_TISSUE_PATH),
                        "numChannels": DEFAULT_NUM_CHANNELS,
                        "source": DEFAULT_SOURCE,
                        "direction": DEFAULT_DIRECTION,
                        "sigma": DEFAULT_SIGMA,
                        "device": str(device),
                        "mcvmExePath": str(DEFAULT_MCVM_EXE),
                        "verificationMode": "sameGrid",
                        "defaultsRevision": "mcvmdl-demo-v2",
                        "persistentRunRoot": str(PERSISTENT_RUN_ROOT),
                    }
                )
                return
            if parsed.path == "/api/list-path":
                params = parse_qs(parsed.query)
                kind = params.get("kind", [""])[0]
                path_value = params.get("path", [""])[0]
                self.send_json(list_local_path(path_value, kind))
                return
            if parsed.path == "/api/plans":
                self.send_json({"plans": PLAN_MANAGER.list()})
                return
            plan_slice = re.fullmatch(r"/api/plans/([a-f0-9]{12})/slice", parsed.path)
            if plan_slice:
                params = parse_qs(parsed.query)
                axis = params.get("axis", [""])[0].lower()
                try:
                    index = int(params.get("index", [""])[0])
                except Exception as exc:
                    raise ApiError("index must be an integer", str(exc)) from exc
                record, tissue, prediction, roi = PLAN_MANAGER.load_best(plan_slice.group(1))
                source = np.asarray(record["bestCandidate"].get("internalSource") or record["bestCandidate"]["source"], dtype=np.float32)
                rgba, headers = render_scalar_slice(
                    tissue,
                    prediction,
                    axis,
                    index,
                    0.0,
                    max(float(record.get("prediction", {}).get("max", np.max(prediction))), 1e-8),
                    roi_mask=roi,
                    source=source,
                )
                self.send_rgba(rgba, headers)
                return
            plan_match = re.fullmatch(r"/api/plans/([a-f0-9]{12})", parsed.path)
            if plan_match:
                self.send_json(PLAN_MANAGER.get(plan_match.group(1)))
                return
            if parsed.path == "/api/mcvm/profile":
                params = parse_qs(parsed.query)
                plan_id = params.get("planId", [""])[0]
                self.send_json(MCVM_MANAGER.profile_for_plan(plan_id))
                return
            if parsed.path == "/api/mcvm/jobs":
                self.send_json({"jobs": MCVM_MANAGER.list()})
                return
            mcvm_comparison = re.fullmatch(r"/api/mcvm/jobs/([a-f0-9]{12})/comparison-meta", parsed.path)
            if mcvm_comparison:
                self.send_json(MCVM_MANAGER.comparison_meta(mcvm_comparison.group(1)))
                return
            mcvm_slice = re.fullmatch(r"/api/mcvm/jobs/([a-f0-9]{12})/slice", parsed.path)
            if mcvm_slice:
                params = parse_qs(parsed.query)
                axis = params.get("axis", [""])[0].lower()
                kind = params.get("kind", ["model"])[0].lower()
                overlays_value = params.get("overlays", ["1"])[0]
                if overlays_value not in {"0", "1"}:
                    raise ApiError("overlays must be 0 or 1", overlays_value)
                show_overlays = overlays_value == "1"
                try:
                    index = int(params.get("index", [""])[0])
                except Exception as exc:
                    raise ApiError("index must be an integer", str(exc)) from exc
                record, tissue, model_volume, mcvm_volume, difference_volume = MCVM_MANAGER.load_result(mcvm_slice.group(1))
                if kind == "model":
                    volume, is_difference = model_volume, False
                elif kind == "mcvm":
                    volume, is_difference = mcvm_volume, False
                elif kind == "difference":
                    volume, is_difference = difference_volume, True
                else:
                    raise ApiError("kind must be model, mcvm, or difference", kind)
                plan, _, _, roi = PLAN_MANAGER.load_best(record["planId"])
                display = record.get("displayRange") or {}
                maximum = float(display.get("differenceMax" if is_difference else "commonMax", 1.0))
                source = np.asarray(
                    record.get("internalSource")
                    or record.get("source")
                    or plan["bestCandidate"].get("internalSource")
                    or plan["bestCandidate"]["source"],
                    dtype=np.float32,
                )
                rgba, headers = render_scalar_slice(
                    tissue,
                    volume,
                    axis,
                    index,
                    0.0,
                    maximum,
                    roi_mask=roi if show_overlays else None,
                    source=source if show_overlays else None,
                    difference=is_difference,
                )
                self.send_rgba(rgba, headers)
                return
            mcvm_match = re.fullmatch(r"/api/mcvm/jobs/([a-f0-9]{12})", parsed.path)
            if mcvm_match:
                self.send_json(MCVM_MANAGER.get(mcvm_match.group(1)))
                return
            if parsed.path == "/api/slice":
                params = parse_qs(parsed.query)
                run_id = params.get("runId", [""])[0]
                axis = params.get("axis", [""])[0].lower()
                try:
                    index = int(params.get("index", [""])[0])
                except Exception as exc:
                    raise ApiError("index must be an integer", str(exc)) from exc
                with CACHE_LOCK:
                    result = RUN_CACHE.get(run_id)
                    if result is not None:
                        RUN_CACHE.move_to_end(run_id)
                if result is None:
                    raise ApiError("runId does not exist or has expired", run_id, HTTPStatus.NOT_FOUND)
                rgba, headers = render_slice(result, axis, index)
                self.send_rgba(rgba, headers)
                return
            self.serve_static(parsed.path)
        except Exception as exc:
            self.send_api_error(exc)

    def serve_static(self, request_path: str) -> None:
        rel = unquote(request_path).lstrip("/") or "index.html"
        if rel.endswith("/"):
            rel += "index.html"
        parts = Path(rel).parts
        if ".." in parts:
            raise ApiError("Invalid path", rel, HTTPStatus.BAD_REQUEST)
        target = (WEB_ROOT / rel).resolve()
        if WEB_ROOT not in target.parents and target != WEB_ROOT:
            raise ApiError("Invalid path", rel, HTTPStatus.BAD_REQUEST)
        if not target.exists() or not target.is_file():
            raise ApiError("Page asset not found", rel, HTTPStatus.NOT_FOUND)
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif target.suffix in {".html", ".css"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCVMDL web dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mcvm-exe", default=str(DEFAULT_MCVM_EXE))
    return parser.parse_args()


def main() -> None:
    global SERVER_DEVICE
    args = parse_args()
    MCVM_MANAGER.set_executable(Path(args.mcvm_exe))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        SERVER_DEVICE = torch.device("cuda")
    elif args.device == "cpu":
        SERVER_DEVICE = torch.device("cpu")
    else:
        SERVER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    httpd = ThreadingHTTPServer((args.host, args.port), MCVMRequestHandler)
    print(f"MCVM web app serving at http://{args.host}:{args.port}")
    print(f"Device: {SERVER_DEVICE}")
    print(f"Web root: {WEB_ROOT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
