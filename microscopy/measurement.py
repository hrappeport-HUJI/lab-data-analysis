"""Fluorophore measurements use raw pixels, never rendered overlays."""

from functools import lru_cache
from collections import defaultdict
import numpy as np
from scipy import ndimage
from .images import read_frame
from .review import observation_mask
from .events import quality_map


def measure(project, run_id, data, founders):
    cfg = project.config
    nodes = {r["id"]: r for r in data["observations"]}
    usable = quality_map(data)
    frames = defaultdict(list)
    for n in data["observations"]:
        frames[(n["field"], n["frame"])].append(n)

    @lru_cache(maxsize=8)
    def cached_mask(oid):
        return observation_mask(project, run_id, nodes[oid])

    def mask(node):
        return cached_mask(node["id"])

    @lru_cache(maxsize=4)
    def occupied_frame(field, frame):
        shape = project.frames[(field, frame, cfg["structure_channel"])]["shape"]
        occupied = np.zeros(shape, bool)
        for other in frames[(field, frame)]:
            occupied |= mask(other)
        return occupied

    output = []
    recipes = cfg.get("measurements", [])
    for recipe in recipes:
        channel = recipe["channel"]
        selection = recipe.get("selection", "baseline")
        name = recipe["name"]
        if selection not in {"baseline", "all", "frame", "event_relative"}:
            raise ValueError("Unknown measurement selection")
        normalize = recipe.get("normalize_exposure", False)
        if normalize and recipe.get("linear_response_verified") is not True:
            raise ValueError(
                "Exposure normalization requires an explicitly verified linear acquisition response"
            )
        for f in founders:
            if f["decision"] == "exclude":
                continue
            reachable = [nodes[n] for n in f["observation_ids"]]
            candidates = reachable
            chosen = data["measurement_frames"].get(f"{f['id']}|{channel}")
            if chosen is not None:
                candidates = [n for n in reachable if n["frame"] == chosen]
            elif selection == "baseline":
                candidates = [nodes[f["observation_id"]]]
            elif selection == "frame":
                candidates = [n for n in reachable if n["frame"] == recipe["frame"]]
            elif selection == "event_relative":
                if f["decision"] != "division" or f["division_midpoint_s"] is None:
                    candidates = []
                else:
                    if "tolerance_s" not in recipe:
                        raise ValueError(
                            "Event-relative selection requires an explicit tolerance_s"
                        )
                    target = f["division_midpoint_s"] + recipe.get("offset_s", 0)
                    possible = [
                        (
                            abs(
                                project.frames[(n["field"], n["frame"], channel)][
                                    "time_s"
                                ]
                                - target
                            ),
                            n,
                        )
                        for n in reachable
                        if (n["field"], n["frame"], channel) in project.frames
                        and project.frames[(n["field"], n["frame"], channel)]["time_s"]
                        is not None
                    ]
                    nearest = min((d for d, n in possible), default=float("inf"))
                    candidates = [
                        n
                        for d, n in possible
                        if d == nearest and d <= recipe["tolerance_s"]
                    ]
            if not candidates:
                output.append(
                    {
                        "founder_id": f["id"],
                        "condition": f["condition"],
                        "field": f["field"],
                        "measurement": name,
                        "channel": channel,
                        "status": "no_tracked_mask_at_requested_time",
                    }
                )
            for node in candidates:
                row = {
                    "founder_id": f["id"],
                    "observation_id": node["id"],
                    "condition": f["condition"],
                    "field": node["field"],
                    "measurement": name,
                    "channel": channel,
                    "frame": node["frame"],
                    "status": "ok",
                    "selection": selection,
                    "manual_frame_override": chosen is not None,
                }
                key = (node["field"], node["frame"], channel)
                if key not in project.frames:
                    row["status"] = "channel_not_acquired"
                    output.append(row)
                    continue
                rec = project.frames[key]
                row.update(time_s=rec["time_s"], exposure_ms=rec["exposure_ms"])
                if not usable.get(key, True) or not usable.get(
                    (node["field"], node["frame"], cfg["structure_channel"]), True
                ):
                    row["status"] = "unusable_image_or_mask"
                    output.append(row)
                    continue
                image = read_frame(project, *key).astype(float)
                cell = mask(node)
                occupied = occupied_frame(node["field"], node["frame"])
                shift = recipe.get("channel_shift_yx_px", [0, 0])
                if len(shift) != 2 or any(not float(v).is_integer() for v in shift):
                    raise ValueError(
                        "Initial implementation supports integer channel-mask shifts only"
                    )
                if any(shift):
                    shifted = (
                        ndimage.shift(
                            cell.astype("uint8"),
                            shift,
                            order=0,
                            mode="constant",
                            cval=0,
                        )
                        > 0
                    )
                    if shifted.sum() != cell.sum():
                        row["status"] = "mask_outside_channel"
                        output.append(row)
                        continue
                    cell = shifted
                    occupied = (
                        ndimage.shift(
                            occupied.astype("uint8"),
                            shift,
                            order=0,
                            mode="constant",
                            cval=0,
                        )
                        > 0
                    )
                inner = int(recipe.get("background_inner_px", 3))
                outer = int(recipe.get("background_outer_px", 10))
                if not 0 < inner < outer:
                    raise ValueError("Background annulus requires 0 < inner < outer")
                ring = (
                    ndimage.binary_dilation(cell, iterations=outer)
                    & ~ndimage.binary_dilation(cell, iterations=inner)
                    & ~occupied
                )
                if ring.sum() < recipe.get("min_background_pixels", 30):
                    row["status"] = "insufficient_background"
                    output.append(row)
                    continue
                values = image[cell]
                background = float(np.median(image[ring]))
                net = float(values.mean() - background)
                row.update(
                    area_px=int(cell.sum()),
                    raw_mean=float(values.mean()),
                    raw_median=float(np.median(values)),
                    background_median=background,
                    background_pixels=int(ring.sum()),
                    net_mean=net,
                    net_median=float(np.median(values) - background),
                    net_integrated=float(np.sum(values - background)),
                    value=net,
                    units="ADU/pixel",
                    normalization="none",
                )
                if normalize:
                    if rec["exposure_ms"] is None:
                        row.update(status="missing_exposure", value=None)
                    else:
                        reference = float(recipe.get("reference_exposure_ms", 100))
                        if reference <= 0:
                            raise ValueError("Reference exposure must be positive")
                        row.update(
                            value=net * reference / rec["exposure_ms"],
                            units=f"ADU/pixel at {reference:g} ms",
                            normalization="exposure",
                            reference_exposure_ms=reference,
                        )
                if np.issubdtype(np.dtype(rec["dtype"]), np.integer):
                    row["saturated_cell_fraction"] = float(
                        np.mean(values >= np.iinfo(np.dtype(rec["dtype"])).max)
                    )
                output.append(row)
    return output
