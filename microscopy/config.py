"""Fail early on contradictory or nonfinite numerical configuration."""

import math


def validate(config):
    def positive(value, name, zero=False, integer=False):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (not zero and value == 0)
            or (integer and int(value) != value)
        ):
            raise ValueError(
                f'{name} must be a finite {"nonnegative" if zero else "positive"} {"integer" if integer else "number"}'
            )

    if config.get("frame_interval_s") is not None:
        positive(config["frame_interval_s"], "frame_interval_s")
    positive(config.get("expected_frame_step", 1), "expected_frame_step", integer=True)
    if config.get("pixel_size_um") is not None:
        positive(config["pixel_size_um"], "pixel_size_um")
    for channel, v in config.get("exposure_ms", {}).items():
        if v is not None:
            positive(v, f"exposure_ms.{channel}")
    segmentation = config.get("segmentation", {})
    if segmentation.get("polarity", "dark") not in {"dark", "bright"}:
        raise ValueError("polarity must be dark or bright")
    for key, default in [
        ("background_sigma_px", 8),
        ("min_area_px", 12),
        ("max_area_px", 5000),
    ]:
        positive(segmentation.get(key, default), key)
    if segmentation.get("min_area_px", 12) > segmentation.get("max_area_px", 5000):
        raise ValueError("min_area_px exceeds max_area_px")
    tracking = config.get("tracking", {})
    positive(
        tracking.get("confirmation_frames", 1),
        "confirmation_frames",
        zero=True,
        integer=True,
    )
    positive(tracking.get("max_displacement_px", 15), "max_displacement_px")
    ratio = tracking.get("division_area_ratio", [0.65, 1.5])
    if len(ratio) != 2 or not 0 < ratio[0] < ratio[1]:
        raise ValueError("Invalid division_area_ratio")
    if not 0 < tracking.get("min_child_fraction", 0.2) < 0.5:
        raise ValueError("min_child_fraction must be between 0 and .5")
    quality = config.get("quality", {})
    if quality.get("min_focus_score") is not None:
        positive(quality["min_focus_score"], "min_focus_score", zero=True)
    for key, default in [("edge_margin_px", 3), ("isolation_px", 0)]:
        positive(
            config.get("founders", {}).get(key, default), key, zero=True, integer=True
        )
    names = set()
    for recipe in config.get("measurements", []):
        name = recipe.get("name")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or name == "first_observed_division_h"
        ):
            raise ValueError(
                "Measurement names must be unique, nonempty and not reserved"
            )
        names.add(name)
        if recipe.get("channel") not in config.get("fluorescence_channels", []):
            raise ValueError(
                "Measurement channel must be declared in fluorescence_channels"
            )
        selection = recipe.get("selection", "baseline")
        if selection not in {"baseline", "all", "frame", "event_relative"}:
            raise ValueError("Unknown measurement selection")
        if selection == "frame":
            positive(recipe.get("frame"), "measurement frame", zero=True, integer=True)
        if selection == "event_relative":
            positive(recipe.get("tolerance_s"), "event tolerance_s", zero=True)
        inner = recipe.get("background_inner_px", 3)
        outer = recipe.get("background_outer_px", 10)
        positive(inner, "background_inner_px", integer=True)
        positive(outer, "background_outer_px", integer=True)
        if inner >= outer:
            raise ValueError("Background inner radius must be less than outer radius")
        positive(
            recipe.get("min_background_pixels", 30),
            "min_background_pixels",
            integer=True,
        )
        if (
            recipe.get("normalize_exposure")
            and recipe.get("linear_response_verified") is not True
        ):
            raise ValueError("Exposure normalization requires verified linear response")
        positive(recipe.get("reference_exposure_ms", 100), "reference_exposure_ms")
    pairs = config.get("analysis", {}).get("comparisons", [])
    seen = set()
    for pair in pairs:
        if (
            len(pair) != 2
            or not all(isinstance(c, str) and c for c in pair)
            or pair[0] == pair[1]
            or tuple(sorted(pair)) in seen
        ):
            raise ValueError(
                "Comparisons must be distinct pairs of different conditions"
            )
        seen.add(tuple(sorted(pair)))
    return config
