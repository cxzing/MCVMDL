from __future__ import annotations

import json
import hashlib
import math
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from web_app.planning_service import PlanningManager, dose_metrics, prediction_to_physical
from web_app.visual_style import display_floor


class MCVMError(Exception):
    def __init__(self, message: str, detail: str = "", status: int = 400):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.status = int(status)


DEFAULT_OPTICAL_PROPERTIES = [
    (1, "Scalp", 1.37, 0.155, 310.0, 0.900),
    (2, "Skull", 1.43, 0.100, 210.0, 0.910),
    (6, "CSF", 1.33, 0.004, 0.1, 0.900),
    (7, "Gray matter", 1.37, 0.200, 90.0, 0.850),
    (8, "White matter", 1.37, 0.800, 410.0, 0.835),
]
DEFAULT_PROFILE_VERSION = "same-grid-v1"
SAME_GRID_VERIFICATION_MODE = "sameGrid"
MCVM_MCI_VERSION = 2
MCVM_BEAM_MODE = "sameGridConfigurable"
MCVM_STARTUP_TIMEOUT_SEC = 30.0
MCVM_ACTUAL_STARTUP_TIMEOUT_SEC = 120.0
MCVM_JOB_TIMEOUT_SEC = 2.0 * 60.0 * 60.0
MCVM_PREFLIGHT_TIMEOUT_SEC = 3.0 * 60.0
DEFAULT_VOXEL_MM = (2.0, 2.0, 2.0)
TRAINING_LOG_FLOOR = 1e-10
TRAINING_NOISE_THRESHOLD = 0.0
DISPLAY_PHYSICAL_FLOOR = 5e-9


def default_optical_profile(num_channels: int, labels: list[int] | None = None) -> dict[str, Any]:
    defaults = {row[0]: row[1:] for row in DEFAULT_OPTICAL_PROPERTIES}
    requested_labels = sorted(set(labels if labels is not None else range(1, max(1, int(num_channels)))))
    tissues: list[dict[str, Any]] = []
    for label in requested_labels:
        if label <= 0 or label >= int(num_channels):
            continue
        name, refractive, mua, mus, anisotropy = defaults.get(
            label,
            (f"Tissue {label}", 1.37, 0.1, 100.0, 0.9),
        )
        tissues.append(
            {
                "label": label,
                "name": name,
                "n": refractive,
                "mua": mua,
                "mus": mus,
                "g": anisotropy,
                "present": bool(labels is None or label in requested_labels),
            }
        )
    return {
        "profileVersion": DEFAULT_PROFILE_VERSION,
        "verificationMode": SAME_GRID_VERIFICATION_MODE,
        "voxelSizeMm": list(DEFAULT_VOXEL_MM),
        "modelVoxelSizeMm": list(DEFAULT_VOXEL_MM),
        "referenceVoxelSizeMm": list(DEFAULT_VOXEL_MM),
        "outsideRefractiveIndex": 1.0,
        "beam": {"radiusCm": 0.1, "profile": "top-hat", "fiberNA": 0.0},
        "noiseThreshold": TRAINING_NOISE_THRESHOLD,
        "displayPhysicalFloor": DISPLAY_PHYSICAL_FLOOR,
        "tissues": tissues,
    }


def same_grid_profile(num_channels: int, labels: list[int]) -> dict[str, Any]:
    return default_optical_profile(num_channels, labels=labels)


def xyz_to_xzy(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[..., [0, 2, 1]]


def xzy_to_xyz(values: np.ndarray) -> np.ndarray:
    return np.asarray(values)[..., [0, 2, 1]]


def export_tissue_xzy(tissue_xyz: np.ndarray, path: Path) -> None:
    flat = np.asarray(tissue_xyz).transpose(0, 2, 1).reshape(-1)
    with Path(path).open("w", encoding="ascii", newline="\n") as output:
        output.writelines(f"{int(value)}\n" for value in flat)


def load_mcvm_absorption(path: Path, shape_xyz: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape_xyz
    # Read the MCVM X-Z-Y output ordering and return the model's X-Y-Z ordering.
    values = np.fromfile(path, sep=" ", dtype=np.float64)
    expected = nx * ny * nz
    if values.size != expected:
        raise MCVMError(
            "MCVM absorption output has an unexpected size",
            f"{path} contains {values.size} values; expected {expected}",
            500,
        )
    return values.reshape((nx, nz, ny)).transpose(0, 2, 1).astype(np.float32, copy=False)


def physical_to_normalized(physical: np.ndarray, noise_threshold: float) -> np.ndarray:
    # Match the model's log transformation before float32 loading.
    values = np.asarray(physical, dtype=np.float64).copy()
    if noise_threshold > 0:
        values[values < noise_threshold] = 0.0
    normalized = (np.log10(np.maximum(values, TRAINING_LOG_FLOOR)) + 10.0) / 10.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def comparison_metadata(
    model_normalized: np.ndarray,
    mcvm_normalized: np.ndarray,
    roi_center: list[float] | np.ndarray,
) -> dict[str, Any]:
    model = np.asarray(model_normalized, dtype=np.float32)
    mcvm = np.asarray(mcvm_normalized, dtype=np.float32)
    if model.shape != mcvm.shape or model.ndim != 3:
        raise MCVMError("Model and MCVM comparison volumes must be aligned 3D arrays")
    finite_mcvm = np.where(np.isfinite(mcvm), mcvm, 0.0)
    finite_model = np.where(np.isfinite(model), model, 0.0)
    if np.any(finite_mcvm > 0):
        point = np.unravel_index(int(np.argmax(finite_mcvm)), finite_mcvm.shape)
        mode = "mcvmPeak"
    elif np.any(finite_model > 0):
        point = np.unravel_index(int(np.argmax(finite_model)), finite_model.shape)
        mode = "modelPeak"
    else:
        center = np.rint(np.asarray(roi_center, dtype=np.float64)).astype(np.int64)
        center = np.clip(center, 0, np.asarray(model.shape, dtype=np.int64) - 1)
        point = tuple(int(value) for value in center.tolist())
        mode = "roiCenter"
    foreground_max = max(float(np.max(finite_model)), float(np.max(finite_mcvm)), 1e-8)
    difference_max = max(float(np.max(np.abs(finite_model - finite_mcvm))), 1e-8)
    return {
        "referencePoint": [int(value) for value in point],
        "referenceMode": mode,
        "commonMax": foreground_max,
        "displayFloor": display_floor(foreground_max),
        "differenceMax": difference_max,
        "shape": [int(value) for value in model.shape],
    }


def resolve_source_selection(
    payload: dict[str, Any],
    plan: dict[str, Any],
    shape_xyz: tuple[int, int, int],
) -> dict[str, Any]:
    requested = payload.get("sourceSelection") or {}
    mode = str(requested.get("mode") or "planned").lower()
    best = plan["bestCandidate"]
    if mode == "planned":
        surface_source = [float(value) for value in best["source"]]
        simulation_source = [float(value) for value in (best.get("internalSource") or best["source"])]
        return {
            "mode": "planned",
            "source": simulation_source,
            "surfaceSource": surface_source,
            "direction": [float(value) for value in best["direction"]],
            "aimTarget": [
                float(value)
                for value in (best.get("aimTarget") or plan["config"]["roi"]["center"])
            ],
            "candidateId": best.get("id"),
        }
    if mode != "manual":
        raise MCVMError("Source mode must be planned or manual", mode)

    try:
        source = np.asarray(requested.get("source"), dtype=np.float64)
        aim_target = np.asarray(
            requested.get("aimTarget", plan["config"]["roi"]["center"]),
            dtype=np.float64,
        )
    except Exception as exc:
        raise MCVMError("Manual source and aim target must contain numeric X, Y, and Z values", str(exc)) from exc
    if source.shape != (3,) or aim_target.shape != (3,) or not np.all(np.isfinite(source)) or not np.all(np.isfinite(aim_target)):
        raise MCVMError("Manual source and aim target must contain three finite values")
    upper = np.asarray(shape_xyz, dtype=np.float64) - 1.0
    if np.any(source < 0) or np.any(source > upper):
        raise MCVMError("Manual source is outside the tissue grid", f"source={source.tolist()}, shape={list(shape_xyz)}")
    if np.any(aim_target < 0) or np.any(aim_target > upper):
        raise MCVMError("Manual aim target is outside the tissue grid", f"aimTarget={aim_target.tolist()}, shape={list(shape_xyz)}")
    direction = aim_target - source
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise MCVMError("Manual source and aim target cannot be the same point")
    direction /= norm
    return {
        "mode": "manual",
        "source": [float(value) for value in source],
        "direction": [float(value) for value in direction],
        "aimTarget": [float(value) for value in aim_target],
        "candidateId": None,
    }


def remap_tissue_for_profile(tissue: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    """Map arbitrary web labels to the contiguous 1..N labels expected by MCVM."""

    source = np.asarray(tissue)
    remapped = np.zeros(source.shape, dtype=np.int16)
    for mcvm_label, row in enumerate(profile["tissues"], start=1):
        remapped[source == int(row["label"])] = mcvm_label
    missing = np.unique(source[(source > 0) & (remapped == 0)])
    if missing.size:
        raise MCVMError("Optical parameters are missing for tissue labels", ", ".join(str(int(value)) for value in missing))
    return remapped


def validate_profile(profile: dict[str, Any], tissue: np.ndarray, num_channels: int) -> dict[str, Any]:
    try:
        voxel = np.asarray(profile.get("voxelSizeMm"), dtype=np.float64)
    except Exception as exc:
        raise MCVMError("Voxel size must contain numeric X, Y, and Z values", str(exc)) from exc
    if voxel.shape != (3,) or not np.all(np.isfinite(voxel)) or np.any(voxel <= 0):
        raise MCVMError("Voxel size must contain three positive values in millimetres")
    try:
        reference_voxel = np.asarray(profile.get("referenceVoxelSizeMm", voxel), dtype=np.float64)
    except Exception as exc:
        raise MCVMError("Reference voxel size must contain numeric X, Y, and Z values", str(exc)) from exc
    if reference_voxel.shape != (3,) or not np.all(np.isfinite(reference_voxel)) or np.any(reference_voxel <= 0):
        raise MCVMError("Reference voxel size must contain three positive values in millimetres")
    try:
        outside_n = float(profile.get("outsideRefractiveIndex", 1.0))
        noise_threshold = float(profile.get("noiseThreshold", 5e-9))
    except Exception as exc:
        raise MCVMError("Outside refractive index and noise threshold must be numeric", str(exc)) from exc
    if not math.isfinite(outside_n) or outside_n <= 0 or not math.isfinite(noise_threshold) or noise_threshold < 0:
        raise MCVMError("Outside refractive index must be positive and noise threshold cannot be negative")

    rows = profile.get("tissues")
    if not isinstance(rows, list) or not rows:
        raise MCVMError("Optical tissue table is empty")
    parsed_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            label = int(row["label"])
            parsed = {
                "label": label,
                "name": str(row.get("name") or f"Tissue {label}"),
                "n": float(row["n"]),
                "mua": float(row["mua"]),
                "mus": float(row["mus"]),
                "g": float(row["g"]),
            }
        except Exception as exc:
            raise MCVMError("Optical tissue table contains an invalid row", str(row)) from exc
        values = [parsed["n"], parsed["mua"], parsed["mus"], parsed["g"]]
        if label <= 0 or label >= num_channels or not all(math.isfinite(value) for value in values):
            raise MCVMError("Optical tissue label or numeric value is out of range", str(row))
        if parsed["n"] <= 0 or parsed["mua"] < 0 or parsed["mus"] < 0 or not (0 <= parsed["g"] <= 1):
            raise MCVMError("Optical parameters require n>0, mua/mus>=0, and 0<=g<=1", str(row))
        if label in parsed_rows:
            raise MCVMError("Optical tissue labels must be unique", str(label))
        parsed_rows[label] = parsed

    required_labels = [int(value) for value in np.unique(tissue) if int(value) > 0]
    missing = [label for label in required_labels if label not in parsed_rows]
    if missing:
        raise MCVMError("Optical parameters are missing for tissue labels", ", ".join(map(str, missing)))

    beam = profile.get("beam") or {}
    try:
        radius_cm = float(beam.get("radiusCm", 0.1))
        fiber_na = float(beam.get("fiberNA", 0.0))
    except Exception as exc:
        raise MCVMError("Beam radius and fiber NA must be numeric", str(exc)) from exc
    beam_profile = str(beam.get("profile", "top-hat"))
    if radius_cm < 0 or not math.isfinite(radius_cm) or fiber_na < 0 or not math.isfinite(fiber_na):
        raise MCVMError("Beam radius and fiber NA cannot be negative")
    if beam_profile not in {"top-hat", "gaussian"}:
        raise MCVMError("Beam profile must be top-hat or gaussian")

    ordered_rows = [parsed_rows[label] for label in sorted(parsed_rows)]
    return {
        "profileVersion": str(profile.get("profileVersion") or "custom"),
        "voxelSizeMm": [float(value) for value in voxel],
        "referenceVoxelSizeMm": [float(value) for value in reference_voxel],
        "absorptionScale": float(np.prod(reference_voxel / voxel)),
        "outsideRefractiveIndex": outside_n,
        "beam": {"radiusCm": radius_cm, "profile": beam_profile, "fiberNA": fiber_na},
        "noiseThreshold": noise_threshold,
        "tissues": ordered_rows,
    }


def validate_same_grid_profile(
    profile: dict[str, Any],
    tissue: np.ndarray,
    num_channels: int,
) -> dict[str, Any]:
    validated = validate_profile(profile, tissue, num_channels)
    if tissue.ndim != 3 or any(int(value) <= 0 for value in tissue.shape):
        raise MCVMError("Same-grid verification requires a non-empty three-dimensional tissue volume")
    voxel = [float(value) for value in validated["voxelSizeMm"]]
    validated.update(
        {
            "verificationMode": SAME_GRID_VERIFICATION_MODE,
            "modelVoxelSizeMm": voxel,
            "referenceVoxelSizeMm": voxel,
            "absorptionScale": 1.0,
            "mciVersion": MCVM_MCI_VERSION,
            "displayPhysicalFloor": DISPLAY_PHYSICAL_FLOOR,
        }
    )
    return validated


def map_source_to_same_grid(
    source_model_xyz: np.ndarray,
    shape_xyz: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(source_model_xyz, dtype=np.float64)
    upper = np.asarray(shape_xyz, dtype=np.float64)
    if source.shape != (3,) or not np.all(np.isfinite(source)):
        raise MCVMError("Source mapping requires finite X, Y, and Z coordinates")
    if np.any(source < 0.0) or np.any(source >= upper):
        raise MCVMError(
            "The source is outside the model grid",
            f"source={source.tolist()}; shape={upper.astype(int).tolist()}",
        )
    return source, {
        "mode": SAME_GRID_VERIFICATION_MODE,
        "modelSource": [float(value) for value in source],
        "mcvmSource": [float(value) for value in source],
        "modelToMcvmScale": [1.0, 1.0, 1.0],
    }


def build_mci_text(
    tissue_path: Path,
    shape_xyz: tuple[int, int, int],
    source_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    photons: int,
    profile: dict[str, Any],
) -> str:
    voxel_mm_xyz = np.asarray(profile["voxelSizeMm"], dtype=np.float64)
    source_cm_xzy = xyz_to_xzy(np.asarray(source_xyz, dtype=np.float64) * voxel_mm_xyz / 10.0)
    direction_xzy = xyz_to_xzy(direction_xyz)
    direction_xzy = direction_xzy / np.linalg.norm(direction_xzy)
    voxel_cm_xzy = xyz_to_xzy(voxel_mm_xyz / 10.0)
    nx, ny, nz = shape_xyz
    beam = profile["beam"]
    gaussian_flag = 1 if beam["profile"] == "gaussian" else 0
    rows = profile["tissues"]
    lines = [
        "2\t\t\t\t\t\t# file version",
        "1\t\t\t\t\t\t# number of runs",
        "",
        "### Specify data for run 1",
        "result.mco A\t\t# output filename, ASCII/Binary",
        "result_abs 3 A\t\t# absorption distribution filename, ASCII/Binary",
        "result_outphoton A\t# information of out photons",
        f"{tissue_path}\t# tissue file name",
        "1 10\t\t\t\t# sample times and interval/ns",
        f"{int(photons)}\t\t\t\t# No. of photons",
        f"{source_cm_xzy[0]:.8f} {source_cm_xzy[1]:.8f} {source_cm_xzy[2]:.8f}\t# x,z,y source position (cm)",
        f"{direction_xzy[0]:.8f} {direction_xzy[1]:.8f} {direction_xzy[2]:.8f}\t# ux,uz,uy direction cosine",
        f"{beam['radiusCm']:.8f} {gaussian_flag} {beam['fiberNA']:.8f}\t# radius, Gaussian/top-hat, fiber NA",
        f"{voxel_cm_xzy[0]:.8f} {voxel_cm_xzy[1]:.8f} {voxel_cm_xzy[2]:.8f}\t# dx,dz,dy (cm)",
        f"{nx} {nz} {ny} 1\t\t# Nx,Nz,Ny,Na",
        "",
        f"{len(rows)}\t\t\t\t\t# Number of tissue types",
        "# n mua mus g",
        f"{profile['outsideRefractiveIndex']:.8f}\t# n for outside medium",
    ]
    for row in rows:
        lines.append(
            f"{row['n']:.8f} {row['mua']:.8f} {row['mus']:.8f} {row['g']:.8f}\t# {row['label']} {row['name']}"
        )
    return "\n".join(lines) + "\n"


def build_same_grid_mci_text(
    tissue_path: Path,
    shape_xyz: tuple[int, int, int],
    source_xyz: np.ndarray,
    direction_xyz: np.ndarray,
    photons: int,
    profile: dict[str, Any],
) -> str:
    text = build_mci_text(
        tissue_path=tissue_path,
        shape_xyz=shape_xyz,
        source_xyz=source_xyz,
        direction_xyz=direction_xyz,
        photons=photons,
        profile=profile,
    )
    return text.replace("\n", "\r\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def run_mcvm_process(
    executable: Path,
    directory: Path,
    log_path: Path,
    photons: int,
    cancel_event: threading.Event,
    startup_timeout: float = MCVM_STARTUP_TIMEOUT_SEC,
    overall_timeout: float = MCVM_JOB_TIMEOUT_SEC,
    preflight_validated: bool = False,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    on_process: Callable[[subprocess.Popen[str]], None] | None = None,
) -> dict[str, Any]:
    """Run MCVM v2 with a startup watchdog and structured progress events.

    MCVM v2 uses fully buffered stdout when it is launched without a console.  A
    long run can therefore be computing normally for many minutes before the
    banner and ``Checking input`` line become visible to this process.  The
    short compatibility run still validates that marker.  Once that exact
    executable/profile/source combination has passed the compatibility run,
    ``preflight_validated`` prevents the formal run from being killed merely
    because its buffered stdout has not yet flushed.
    """

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        [str(executable)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(directory),
        creationflags=creation_flags,
    )
    if on_process is not None:
        on_process(process)
    assert process.stdin is not None
    process.stdin.write("input.mci\n.\n")
    process.stdin.flush()
    process.stdin.close()
    assert process.stdout is not None

    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            while True:
                output_character = process.stdout.read(1)
                if output_character == "":
                    break
                output_queue.put(output_character)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True, name="mcvm-stdout")
    reader.start()
    started = time.monotonic()
    startup_validated = bool(preflight_validated)
    last_synthetic_progress = -1.0
    expected_duration = max(5.0, float(photons) / 10_000_000.0 * 3900.0)
    progress_pattern = re.compile(r"(\d+)\s+photons.*?End\s+(.+)$")
    pending_text = ""

    def parse_output_line(output_line: str) -> None:
        stripped = output_line.strip()
        match = progress_pattern.search(stripped)
        if match:
            remaining = int(match.group(1))
            fraction = max(0.0, min(1.0, 1.0 - remaining / max(int(photons), 1)))
            if on_event is not None:
                on_event(
                    "simulation",
                    {
                        "fraction": fraction,
                        "remainingPhotons": remaining,
                        "estimatedEnd": match.group(2).strip(),
                    },
                )
        if stripped.startswith("User time:") and on_event is not None:
            on_event("writingOutputs", {"fraction": 1.0, "remainingPhotons": 0, "estimatedEnd": None})

    with Path(log_path).open("w", encoding="utf-8", newline="\n") as log_file:
        while True:
            if cancel_event.is_set():
                terminate_process(process)
                break
            elapsed = time.monotonic() - started
            if not startup_validated and elapsed > float(startup_timeout):
                terminate_process(process)
                raise MCVMError(
                    f"MCVM did not produce an input/progress marker within {float(startup_timeout):g} seconds",
                    "No 'Checking input data for run 1' marker was produced; verify the MCI version and beam line.",
                    500,
                )
            if elapsed > float(overall_timeout):
                terminate_process(process)
                raise MCVMError(
                    "MCVM exceeded the two-hour execution limit",
                    f"Elapsed {elapsed:.1f} seconds",
                    504,
                )
            if preflight_validated and elapsed - last_synthetic_progress >= 2.0:
                # The legacy executable does not flush redirected stdout while
                # propagating photons.  Keep the UI honest and responsive with
                # an elapsed-time estimate until real progress is available.
                last_synthetic_progress = elapsed
                estimated_fraction = max(0.0, min(0.99, elapsed / expected_duration))
                if on_event is not None:
                    on_event(
                        "simulation",
                        {
                            "fraction": estimated_fraction,
                            "remainingPhotons": int(round(photons * (1.0 - estimated_fraction))),
                            "estimatedEnd": time.strftime(
                                "%Y-%m-%d %H:%M:%S",
                                time.localtime(time.time() + max(expected_duration - elapsed, 0.0)),
                            ),
                        },
                    )
            try:
                chunk = output_queue.get(timeout=0.25)
            except queue.Empty:
                if process.poll() is not None and not reader.is_alive():
                    break
                continue
            if chunk is None:
                break
            log_file.write(chunk)
            pending_text += chunk
            if "Checking input data for run 1" in pending_text and not startup_validated:
                startup_validated = True
                log_file.flush()
                if on_event is not None:
                    on_event("inputValidated", {})
            while "\n" in pending_text:
                complete_line, pending_text = pending_text.split("\n", 1)
                parse_output_line(complete_line)
                log_file.flush()
        if pending_text:
            parse_output_line(pending_text)
            log_file.flush()
    process.wait()
    return {
        "returnCode": int(process.returncode),
        "startupValidated": bool(startup_validated),
        "startupValidationMode": "cached20PhotonPreflight" if preflight_validated else "stdoutMarker",
        "durationSec": float(time.monotonic() - started),
    }


def comparison_metrics(
    prediction_normalized: np.ndarray,
    mcvm_physical: np.ndarray,
    tissue: np.ndarray,
    roi: np.ndarray,
    noise_threshold: float,
) -> dict[str, Any]:
    prediction_physical = prediction_to_physical(prediction_normalized)
    mcvm_values = np.asarray(mcvm_physical, dtype=np.float32).copy()
    mcvm_values[mcvm_values < noise_threshold] = 0.0
    model_dose = dose_metrics(prediction_physical, tissue, roi)
    mcvm_dose = dose_metrics(mcvm_values, tissue, roi)
    foreground = tissue > 0
    model_fg = prediction_physical[foreground].astype(np.float64)
    mcvm_fg = mcvm_values[foreground].astype(np.float64)
    physical_difference = model_fg - mcvm_fg
    physical_denominator = max(float(np.linalg.norm(mcvm_fg)), 1e-20)
    if model_fg.size > 1 and float(np.std(model_fg)) > 1e-20 and float(np.std(mcvm_fg)) > 1e-20:
        physical_correlation: float | None = float(np.corrcoef(model_fg, mcvm_fg)[0, 1])
    else:
        physical_correlation = None

    model_normalized_fg = np.clip(np.asarray(prediction_normalized, dtype=np.float32), 0.0, 1.0)[foreground].astype(np.float64)
    mcvm_normalized_fg = physical_to_normalized(mcvm_values, noise_threshold)[foreground].astype(np.float64)
    normalized_difference = model_normalized_fg - mcvm_normalized_fg
    normalized_denominator = max(float(np.linalg.norm(mcvm_normalized_fg)), 1e-20)
    if (
        model_normalized_fg.size > 1
        and float(np.std(model_normalized_fg)) > 1e-20
        and float(np.std(mcvm_normalized_fg)) > 1e-20
    ):
        normalized_correlation: float | None = float(np.corrcoef(model_normalized_fg, mcvm_normalized_fg)[0, 1])
    else:
        normalized_correlation = None

    relative: dict[str, float] = {}
    for key in ("targetAbsorption", "offTargetExposure", "hotspotRisk"):
        relative[key] = float(abs(model_dose[key] - mcvm_dose[key]) / max(abs(mcvm_dose[key]), 1e-20))
    return {
        "model": model_dose,
        "mcvm": mcvm_dose,
        "relativeError": relative,
        "spatial": {
            "domain": "logNormalized",
            "relativeL2": float(np.linalg.norm(normalized_difference) / normalized_denominator),
            "nrmse": float(
                np.sqrt(np.mean(normalized_difference**2))
                / max(float(np.sqrt(np.mean(mcvm_normalized_fg**2))), 1e-20)
            ),
            "mae": float(np.mean(np.abs(normalized_difference))),
            "pearson": normalized_correlation,
        },
        "physicalSpatial": {
            "domain": "physicalAbsorption",
            "relativeL2": float(np.linalg.norm(physical_difference) / physical_denominator),
            "nrmse": float(
                np.sqrt(np.mean(physical_difference**2)) / max(float(np.sqrt(np.mean(mcvm_fg**2))), 1e-20)
            ),
            "mae": float(np.mean(np.abs(physical_difference))),
            "pearson": physical_correlation,
        },
    }


class MCVMManager:
    def __init__(
        self,
        output_root: Path,
        mcvm_exe: Path,
        planning: PlanningManager,
    ) -> None:
        self.root = Path(output_root) / "mcvm"
        self.root.mkdir(parents=True, exist_ok=True)
        self.mcvm_exe = Path(mcvm_exe)
        self.planning = planning
        self.lock = threading.RLock()
        self.preflight_lock = threading.Lock()
        self.active_job_id: str | None = None
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        self.exe_hash = sha256_file(self.mcvm_exe) if self.mcvm_exe.exists() else None
        self.compatibility_directory = self.root / "_compatibility_same_grid"
        self.compatibility_cache_path = self.compatibility_directory / "compatibility.json"
        self._mark_interrupted_jobs()

    def set_executable(self, executable: Path) -> None:
        self.mcvm_exe = Path(executable).resolve()
        self.exe_hash = sha256_file(self.mcvm_exe) if self.mcvm_exe.exists() else None

    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def _record_path(self, job_id: str) -> Path:
        return self._dir(job_id) / "job.json"

    def _write(self, record: dict[str, Any]) -> None:
        with self.lock:
            directory = self._dir(record["jobId"])
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "job.json"
            temporary = directory / f"job.{threading.get_ident()}.tmp"
            temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            for attempt in range(8):
                try:
                    temporary.replace(target)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.02 * (attempt + 1))

    def _mark_interrupted_jobs(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") in {"queued", "running", "processing"}:
                    record["status"] = "interrupted"
                    record["updatedAt"] = time.time()
                    record["error"] = {"message": "Server restarted before this MCVM job completed", "detail": ""}
                    self._write(record)
            except Exception:
                continue

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._record_path(job_id)
        if not path.exists():
            raise MCVMError("MCVM job was not found", job_id, 404)
        with self.lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self.root.glob("*/job.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                records.append(
                    {
                        "jobId": record.get("jobId"),
                        "planId": record.get("planId"),
                        "status": record.get("status"),
                        "photons": record.get("photons"),
                        "profileVersion": record.get("profileVersion") or record.get("profile", {}).get("profileVersion"),
                        "mciVersion": record.get("mciVersion"),
                        "createdAt": record.get("createdAt"),
                        "updatedAt": record.get("updatedAt"),
                    }
                )
            except Exception:
                continue
        records.sort(key=lambda item: float(item.get("createdAt") or 0), reverse=True)
        return records

    def profile_for_plan(self, plan_id: str) -> dict[str, Any]:
        record, tissue, _, _ = self.planning.load_best(plan_id)
        labels = [int(value) for value in np.unique(tissue) if int(value) > 0]
        profile = same_grid_profile(int(record["config"]["numChannels"]), labels)
        profile["compatibleTissue"] = tissue.ndim == 3 and all(int(value) > 0 for value in tissue.shape)
        profile["tissueShape"] = [int(value) for value in tissue.shape]
        return profile

    def _compatibility_signature(self, profile: dict[str, Any], tissue: np.ndarray) -> dict[str, Any]:
        profile_hash = hashlib.sha256(
            json.dumps(profile, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "profileVersion": profile["profileVersion"],
            "profileHash": profile_hash,
            "mciVersion": MCVM_MCI_VERSION,
            "beamMode": MCVM_BEAM_MODE,
            "beam": profile["beam"],
            "exeHash": self.exe_hash,
            "tissueHash": hashlib.sha256(np.asarray(tissue, dtype=np.int16).tobytes()).hexdigest(),
        }

    def _read_compatibility_cache(self, signature: dict[str, Any]) -> dict[str, Any] | None:
        try:
            cached = json.loads(self.compatibility_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if cached.get("signature") != signature or cached.get("status") != "passed":
            return None
        return cached

    def _ensure_compatibility(
        self,
        profile: dict[str, Any],
        tissue: np.ndarray,
        source_xyz: np.ndarray,
        direction_xyz: np.ndarray,
        cancel_event: threading.Event,
        on_event: Callable[[str, dict[str, Any]], None],
    ) -> dict[str, Any]:
        signature = self._compatibility_signature(profile, tissue)
        cached = self._read_compatibility_cache(signature)
        if cached is not None:
            return cached
        with self.preflight_lock:
            cached = self._read_compatibility_cache(signature)
            if cached is not None:
                return cached
            self.compatibility_directory.mkdir(parents=True, exist_ok=True)
            self._cleanup_raw(self.compatibility_directory, include_tissue=True)
            for name in ("result.mco", "input.mci", "mcvm.log"):
                path = self.compatibility_directory / name
                try:
                    if path.exists():
                        path.unlink()
                except OSError:
                    pass
            tissue_path = self.compatibility_directory / "tissue_xzy.txt"
            export_tissue_xzy(remap_tissue_for_profile(tissue, profile), tissue_path)
            shape_xyz = tuple(int(value) for value in tissue.shape)
            mci_text = build_same_grid_mci_text(
                tissue_path=tissue_path,
                shape_xyz=shape_xyz,
                source_xyz=source_xyz,
                direction_xyz=direction_xyz,
                photons=20,
                profile=profile,
            )
            (self.compatibility_directory / "input.mci").write_bytes(mci_text.encode("ascii"))
            on_event("compatibilityCheck", {"fraction": 0.0, "remainingPhotons": 20, "estimatedEnd": None})
            result = run_mcvm_process(
                executable=self.mcvm_exe,
                directory=self.compatibility_directory,
                log_path=self.compatibility_directory / "mcvm.log",
                photons=20,
                cancel_event=cancel_event,
                startup_timeout=MCVM_STARTUP_TIMEOUT_SEC,
                overall_timeout=MCVM_PREFLIGHT_TIMEOUT_SEC,
                on_event=lambda phase, data: on_event(
                    "compatibilitySimulation" if phase in {"inputValidated", "simulation"} else phase,
                    data,
                ),
            )
            if cancel_event.is_set():
                self._cleanup_raw(self.compatibility_directory, include_tissue=True)
                return {
                    "status": "cancelled",
                    "signature": signature,
                    "startupValidated": bool(result["startupValidated"]),
                    "durationSec": result["durationSec"],
                }
            if result["returnCode"] != 0 or not result["startupValidated"]:
                raise MCVMError("MCVM v2 compatibility check failed", json.dumps(result), 500)
            absorption_path = self.compatibility_directory / "result_absxzy"
            if not absorption_path.exists():
                raise MCVMError(
                    "MCVM v2 compatibility check produced no absorption field",
                    str(absorption_path),
                    500,
                )
            same_grid = load_mcvm_absorption(absorption_path, shape_xyz)
            if same_grid.shape != shape_xyz or not np.all(np.isfinite(same_grid)):
                raise MCVMError(
                    "MCVM v2 compatibility output does not match the model grid",
                    str(same_grid.shape),
                    500,
                )
            cached = {
                "status": "passed",
                "checkedAt": time.time(),
                "signature": signature,
                "startupValidated": True,
                "outputVoxelCount": int(np.prod(shape_xyz)),
                "outputShape": list(same_grid.shape),
                "durationSec": result["durationSec"],
            }
            self.compatibility_cache_path.write_text(
                json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._cleanup_raw(self.compatibility_directory, include_tissue=True)
            return cached

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.active_job_id is not None:
                try:
                    active = self.get(self.active_job_id)
                    if active.get("status") in {"queued", "running", "processing"}:
                        raise MCVMError("Another MCVM verification is already running", self.active_job_id, 409)
                except MCVMError as exc:
                    if exc.status != 404:
                        raise
                self.active_job_id = None
            if not self.mcvm_exe.exists():
                raise MCVMError("MCVM executable was not found", str(self.mcvm_exe), 404)
            plan_id = str(payload.get("planId") or "")
            plan, tissue, _, _ = self.planning.load_best(plan_id)
            try:
                photons = int(payload.get("photons", 100000))
            except Exception as exc:
                raise MCVMError("Photon count must be an integer", str(exc)) from exc
            if photons < 10 or photons > 100000000:
                raise MCVMError("Photon count must be between 10 and 100,000,000")
            profile = validate_same_grid_profile(
                payload.get("profile") or {}, tissue, int(plan["config"]["numChannels"])
            )
            source_selection = resolve_source_selection(payload, plan, tuple(int(value) for value in tissue.shape))
            job_id = uuid.uuid4().hex[:12]
            directory = self._dir(job_id)
            directory.mkdir(parents=True, exist_ok=True)
            shape_xyz = tuple(int(value) for value in tissue.shape)
            output_voxels = int(np.prod(shape_xyz))
            estimated_bytes = int(output_voxels * 20 + photons * 240 + 512 * 1024 * 1024)
            free_bytes = shutil.disk_usage(directory).free
            if free_bytes < estimated_bytes:
                raise MCVMError(
                    "Insufficient free disk space for MCVM output",
                    f"Estimated {estimated_bytes / 1024**3:.1f} GB, available {free_bytes / 1024**3:.1f} GB",
                    507,
                )
            now = time.time()
            record = {
                "jobId": job_id,
                "planId": plan_id,
                "status": "queued",
                "photons": photons,
                "retainRaw": bool(payload.get("retainRaw", False)),
                "verificationMode": SAME_GRID_VERIFICATION_MODE,
                "profileVersion": profile["profileVersion"],
                "mciVersion": MCVM_MCI_VERSION,
                "beamMode": MCVM_BEAM_MODE,
                "exeHash": self.exe_hash,
                "startupValidated": False,
                "tissuePath": str(Path(plan["config"]["tissuePath"]).resolve()),
                "profile": profile,
                "sourceSelection": source_selection,
                "source": source_selection["source"],
                "direction": source_selection["direction"],
                "aimTarget": source_selection["aimTarget"],
                "shape": [int(value) for value in tissue.shape],
                "mcvmShapeXzy": [shape_xyz[0], shape_xyz[2], shape_xyz[1]],
                "compression": {
                    "method": "sameGrid",
                    "blockShape": [1, 1, 1],
                    "outputShapeXyz": list(shape_xyz),
                    "threshold": float(profile["noiseThreshold"]),
                    "trainingLogFloor": TRAINING_LOG_FLOOR,
                    "displayPhysicalFloor": float(profile.get("displayPhysicalFloor", DISPLAY_PHYSICAL_FLOOR)),
                    "logTransform": "(log10(max(A, 1e-10)) + 10) / 10",
                },
                "createdAt": now,
                "updatedAt": now,
                "progress": {
                    "phase": "queued",
                    "fraction": 0.0,
                    "remainingPhotons": photons,
                    "estimatedEnd": None,
                },
                "metrics": None,
                "error": None,
            }
            self._write(record)
            self.active_job_id = job_id
            cancel_event = threading.Event()
            self.cancel_events[job_id] = cancel_event
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True, name=f"mcvm-{job_id}")
            thread.start()
            return record

    @staticmethod
    def _cleanup_raw(directory: Path, include_tissue: bool = True) -> None:
        names = ["result_absxzy", "result_outphoton"]
        if include_tissue:
            names.append("tissue_xzy.txt")
        for name in names:
            path = directory / name
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass

    def _run(self, job_id: str) -> None:
        start = time.perf_counter()
        directory = self._dir(job_id)
        log_path = directory / "mcvm.log"
        try:
            record = self.get(job_id)
            plan, tissue, prediction, roi = self.planning.load_best(record["planId"])
            best = plan["bestCandidate"]
            profile = record["profile"]
            source_selection = record.get("sourceSelection") or resolve_source_selection({}, plan, tuple(int(value) for value in tissue.shape))
            source = np.asarray(source_selection["source"], dtype=np.float64)
            direction = np.asarray(source_selection["direction"], dtype=np.float64)
            if source_selection["mode"] == "manual":
                prediction, internal_source, inference_seconds = self.planning.predict_source(record["planId"], source, direction)
                np.savez_compressed(directory / "model_prediction.npz", prediction.astype(np.float32))
                record["modelInferenceSec"] = inference_seconds
                record["internalSource"] = [float(value) for value in internal_source]
            else:
                record["internalSource"] = best.get("internalSource", best["source"])
            surface_source = np.asarray(
                source_selection.get("surfaceSource") or source_selection["source"],
                dtype=np.float64,
            )
            mcvm_source, mapping = map_source_to_same_grid(
                source_model_xyz=surface_source,
                shape_xyz=tuple(int(value) for value in tissue.shape),
            )
            remapped_tissue = remap_tissue_for_profile(tissue, profile)
            tissue_path = directory / "tissue_xzy.txt"
            export_tissue_xzy(remapped_tissue, tissue_path)
            mci_text = build_same_grid_mci_text(
                tissue_path=tissue_path,
                shape_xyz=tuple(int(value) for value in tissue.shape),
                source_xyz=mcvm_source,
                direction_xyz=direction,
                photons=int(record["photons"]),
                profile=profile,
            )
            (directory / "input.mci").write_bytes(mci_text.encode("ascii"))
            record["status"] = "running"
            record["updatedAt"] = time.time()
            record["source"] = source_selection["source"]
            record["direction"] = source_selection["direction"]
            record["aimTarget"] = source_selection["aimTarget"]
            record["surfaceSource"] = [float(value) for value in surface_source]
            record["mcvmSource"] = [float(value) for value in mcvm_source]
            record["sourceMapping"] = mapping
            record["progress"] = {
                "phase": "compatibilityCheck",
                "fraction": 0.0,
                "remainingPhotons": 20,
                "estimatedEnd": None,
            }
            self._write(record)

            def update_progress(phase: str, data: dict[str, Any]) -> None:
                record["progress"] = {
                    "phase": phase,
                    "fraction": float(data.get("fraction", 0.0)),
                    "remainingPhotons": data.get("remainingPhotons"),
                    "estimatedEnd": data.get("estimatedEnd"),
                }
                record["updatedAt"] = time.time()
                self._write(record)

            compatibility = self._ensure_compatibility(
                profile=profile,
                tissue=tissue,
                source_xyz=mcvm_source,
                direction_xyz=direction,
                cancel_event=self.cancel_events[job_id],
                on_event=update_progress,
            )
            record["compatibility"] = compatibility
            record["startupValidated"] = True
            if self.cancel_events[job_id].is_set():
                record.update(
                    {
                        "status": "cancelled",
                        "updatedAt": time.time(),
                        "durationSec": float(time.perf_counter() - start),
                    }
                )
                self._cleanup_raw(directory)
                self._write(record)
                return
            update_progress(
                "startingSimulation",
                {"fraction": 0.0, "remainingPhotons": int(record["photons"]), "estimatedEnd": None},
            )
            result = run_mcvm_process(
                executable=self.mcvm_exe,
                directory=directory,
                log_path=log_path,
                photons=int(record["photons"]),
                cancel_event=self.cancel_events[job_id],
                startup_timeout=MCVM_ACTUAL_STARTUP_TIMEOUT_SEC,
                overall_timeout=MCVM_JOB_TIMEOUT_SEC,
                preflight_validated=True,
                on_event=update_progress,
                on_process=lambda running: self.processes.__setitem__(job_id, running),
            )

            if self.cancel_events[job_id].is_set():
                record.update(
                    {
                        "status": "cancelled",
                        "updatedAt": time.time(),
                        "durationSec": float(time.perf_counter() - start),
                    }
                )
                self._cleanup_raw(directory)
                self._write(record)
                return
            if result["returnCode"] != 0:
                unsigned_code = int(result["returnCode"]) & 0xFFFFFFFF
                detail = f"Exit code {result['returnCode']} (0x{unsigned_code:08X})"
                if unsigned_code == 0xC0000005:
                    detail += "; the legacy MCVM executable reported a memory access violation"
                raise MCVMError("MCVM process failed", detail, 500)
            if not result["startupValidated"]:
                raise MCVMError("MCVM completed without validating the v2 input", str(log_path), 500)
            record["startupValidationMode"] = result.get("startupValidationMode", "stdoutMarker")
            absorption_path = directory / "result_absxzy"
            if not absorption_path.exists():
                raise MCVMError("MCVM completed without an absorption output", str(absorption_path), 500)

            record["status"] = "processing"
            record["progress"] = {
                "phase": "normalization",
                "fraction": 1.0,
                "remainingPhotons": 0,
                "estimatedEnd": None,
            }
            record["updatedAt"] = time.time()
            self._write(record)
            mcvm_physical = load_mcvm_absorption(absorption_path, tuple(int(value) for value in tissue.shape))
            if mcvm_physical.shape != tissue.shape:
                raise MCVMError(
                    "MCVM output did not match the model grid",
                    f"mcvm={mcvm_physical.shape}; model={tissue.shape}",
                    500,
                )
            threshold = float(profile["noiseThreshold"])
            mcvm_physical[mcvm_physical < threshold] = 0.0
            np.savez_compressed(directory / "mcvm_absorption.npz", mcvm_physical.astype(np.float32))
            metrics = comparison_metrics(prediction, mcvm_physical, tissue, roi, threshold)
            model_normalized = np.clip(prediction, 0.0, 1.0)
            mcvm_normalized = physical_to_normalized(mcvm_physical, threshold)
            foreground = tissue > 0
            common_max = float(
                max(
                    np.max(model_normalized[foreground]) if np.any(foreground) else 1.0,
                    np.max(mcvm_normalized[foreground]) if np.any(foreground) else 1.0,
                    1e-8,
                )
            )
            difference_max = float(max(np.max(np.abs(model_normalized - mcvm_normalized)), 1e-8))
            comparison = comparison_metadata(
                model_normalized,
                mcvm_normalized,
                plan["config"]["roi"]["center"],
            )
            comparison.update(
                {
                    "commonMax": common_max,
                    "displayFloor": display_floor(common_max),
                    "differenceMax": difference_max,
                }
            )
            record.update(
                {
                    "status": "completed",
                    "updatedAt": time.time(),
                    "completedAt": time.time(),
                    "durationSec": float(time.perf_counter() - start),
                    "progress": {
                        "phase": "completed",
                        "fraction": 1.0,
                        "remainingPhotons": 0,
                        "estimatedEnd": None,
                    },
                    "metrics": metrics,
                    "displayRange": {"commonMax": common_max, "differenceMax": difference_max},
                    "comparison": comparison,
                    "error": None,
                }
            )
            if not record["retainRaw"]:
                self._cleanup_raw(directory)
            self._write(record)
        except Exception as exc:
            try:
                record = self.get(job_id)
                if record.get("status") != "cancelled":
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
                    self._cleanup_raw(directory)
                    self._write(record)
            except Exception:
                traceback.print_exc()
        finally:
            with self.lock:
                self.processes.pop(job_id, None)
                self.cancel_events.pop(job_id, None)
                if self.active_job_id == job_id:
                    self.active_job_id = None

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            record = self.get(job_id)
            if record.get("status") not in {"queued", "running", "processing"}:
                raise MCVMError("MCVM job is not running", job_id, 409)
            event = self.cancel_events.get(job_id)
            if event is not None:
                event.set()
            process = self.processes.get(job_id)
            if process is not None and process.poll() is None:
                process.terminate()
            record["status"] = "cancelling"
            record["updatedAt"] = time.time()
            self._write(record)
            return record

    def load_result(self, job_id: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        record = self.get(job_id)
        if record.get("status") != "completed":
            raise MCVMError("MCVM result is not complete", job_id, 409)
        plan, tissue, prediction, _ = self.planning.load_best(record["planId"])
        custom_prediction_path = self._dir(job_id) / "model_prediction.npz"
        if custom_prediction_path.exists():
            with np.load(custom_prediction_path) as data:
                prediction = np.asarray(data["arr_0"], dtype=np.float32)
        path = self._dir(job_id) / "mcvm_absorption.npz"
        if not path.exists():
            raise MCVMError("MCVM result is missing", str(path), 500)
        with np.load(path) as data:
            mcvm_physical = np.asarray(data["arr_0"], dtype=np.float32)
        mcvm_normalized = physical_to_normalized(mcvm_physical, float(record["profile"]["noiseThreshold"]))
        model_normalized = np.clip(prediction, 0.0, 1.0).astype(np.float32)
        difference = model_normalized - mcvm_normalized
        return record, tissue, model_normalized, mcvm_normalized, difference

    def comparison_meta(self, job_id: str) -> dict[str, Any]:
        record = self.get(job_id)
        stored = record.get("comparison")
        required = {"referencePoint", "referenceMode", "commonMax", "displayFloor", "differenceMax", "shape"}
        if isinstance(stored, dict) and required.issubset(stored):
            computed = dict(stored)
        else:
            record, _, model, mcvm, _ = self.load_result(job_id)
            plan = self.planning.get(record["planId"])
            computed = comparison_metadata(model, mcvm, plan["config"]["roi"]["center"])
        display = record.get("displayRange") or {}
        common_max = float(display.get("commonMax", computed["commonMax"]))
        computed.update(
            {
                "commonMax": common_max,
                "displayFloor": display_floor(common_max),
                "differenceMax": float(display.get("differenceMax", computed["differenceMax"])),
            }
        )
        return {"jobId": job_id, **computed}
