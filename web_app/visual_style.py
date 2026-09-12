from __future__ import annotations

import numpy as np


TISSUE_GRAY_MIN = 0.10
TISSUE_GRAY_MAX = 0.90
REFERENCE_LABEL_MAX = 17
WEIGHT_TISSUE = 0.50
WEIGHT_ABSORPTION = 0.50
DISPLAY_FLOOR_FRACTION = 0.01
DISPLAY_FLOOR_MINIMUM = 1e-5


def label_to_grayscale_float(labels: np.ndarray) -> np.ndarray:
    """Reference Figure 5 label-to-gray mapping in the [0, 1] range."""

    integer = np.clip(np.asarray(labels).astype(np.int16, copy=False), 0, REFERENCE_LABEL_MAX)
    lookup = np.linspace(
        TISSUE_GRAY_MIN,
        TISSUE_GRAY_MAX,
        REFERENCE_LABEL_MAX + 1,
        dtype=np.float32,
    )
    grayscale = lookup[integer]
    return np.where(np.asarray(labels) > 0, grayscale, 0.0).astype(np.float32, copy=False)


def label_to_grayscale(labels: np.ndarray) -> np.ndarray:
    return np.rint(label_to_grayscale_float(labels) * 255.0).astype(np.uint8)


def tissue_rgb(labels: np.ndarray) -> np.ndarray:
    grayscale = label_to_grayscale_float(labels)
    return np.repeat(grayscale[..., None], 3, axis=-1)


def jet_rgb(normalized: np.ndarray) -> np.ndarray:
    """Project-specific Jet mapping, kept identical to the reference script."""

    values = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    red = np.where(
        values < 0.125,
        0.0,
        np.where(
            values < 0.375,
            0.0,
            np.where(
                values < 0.625,
                4.0 * (values - 0.375),
                np.where(values < 0.875, 1.0, 1.0 - 4.0 * (values - 0.875)),
            ),
        ),
    )
    green = np.where(
        values < 0.125,
        0.0,
        np.where(
            values < 0.375,
            4.0 * (values - 0.125),
            np.where(
                values < 0.625,
                1.0,
                np.where(values < 0.875, 1.0 - 4.0 * (values - 0.625), 0.0),
            ),
        ),
    )
    blue = np.where(
        values < 0.125,
        0.5 + 4.0 * values,
        np.where(values < 0.375, 1.0, np.where(values < 0.625, 1.0 - 4.0 * (values - 0.375), 0.0)),
    )
    return np.clip(np.stack((red, green, blue), axis=-1), 0.0, 1.0).astype(np.float32, copy=False)


def diverging_rgb(normalized_signed: np.ndarray) -> np.ndarray:
    """Blue-white-red map for values normalized to [-1, 1]."""

    values = np.clip(np.asarray(normalized_signed, dtype=np.float32), -1.0, 1.0)
    position = (values + 1.0) * 0.5
    red = np.where(position < 0.5, 2.0 * position, 1.0)
    green = np.where(position < 0.5, 2.0 * position, 2.0 * (1.0 - position))
    blue = np.where(position < 0.5, 1.0, 2.0 * (1.0 - position))
    return np.clip(np.stack((red, green, blue), axis=-1), 0.0, 1.0).astype(np.float32, copy=False)


def display_floor(value_max: float) -> float:
    return float(max(float(value_max) * DISPLAY_FLOOR_FRACTION, DISPLAY_FLOOR_MINIMUM))


def compose_scalar_rgb(labels: np.ndarray, values: np.ndarray, value_max: float) -> np.ndarray:
    """Blend anatomy and a positive absorption field using the Figure 5 style."""

    tissue = tissue_rgb(labels)
    scalar = np.asarray(values, dtype=np.float32)
    maximum = max(float(value_max), 1e-6)
    heat = jet_rgb(scalar / maximum)
    combined = tissue.copy()
    signal = (np.asarray(labels) > 0) & (scalar >= display_floor(maximum))
    combined[signal] = WEIGHT_TISSUE * tissue[signal] + WEIGHT_ABSORPTION * heat[signal]
    combined[np.asarray(labels) <= 0] = 0.0
    return np.clip(combined, 0.0, 1.0).astype(np.float32, copy=False)


def compose_difference_rgb(labels: np.ndarray, values: np.ndarray, difference_max: float) -> np.ndarray:
    """Blend anatomy with a signed difference field on an independent scale."""

    tissue = tissue_rgb(labels)
    difference = np.asarray(values, dtype=np.float32)
    maximum = max(abs(float(difference_max)), 1e-6)
    colors = diverging_rgb(difference / maximum)
    combined = tissue.copy()
    signal = (np.asarray(labels) > 0) & (np.abs(difference) >= display_floor(maximum))
    combined[signal] = WEIGHT_TISSUE * tissue[signal] + WEIGHT_ABSORPTION * colors[signal]
    combined[np.asarray(labels) <= 0] = 0.0
    return np.clip(combined, 0.0, 1.0).astype(np.float32, copy=False)


def rgb_to_rgba(rgb: np.ndarray) -> np.ndarray:
    color = np.rint(np.clip(np.asarray(rgb), 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha = np.full((*color.shape[:-1], 1), 255, dtype=np.uint8)
    return np.concatenate([color, alpha], axis=-1)


def tissue_grayscale_rgba(labels: np.ndarray) -> np.ndarray:
    return rgb_to_rgba(tissue_rgb(labels))


def compose_scalar_rgba(labels: np.ndarray, values: np.ndarray, value_max: float) -> np.ndarray:
    return rgb_to_rgba(compose_scalar_rgb(labels, values, value_max))


def compose_difference_rgba(labels: np.ndarray, values: np.ndarray, difference_max: float) -> np.ndarray:
    return rgb_to_rgba(compose_difference_rgb(labels, values, difference_max))
