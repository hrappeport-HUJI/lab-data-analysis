"""Segment and track every plate of one scanner, straight from the raw scans.

Per frame, the scan is read once (memory-mapped). Every plate is cropped at its aligned
position, segmented by the chosen detector, and its colonies are tracked. The default
footprint tracker (``scanlag.track.tracker``) keeps each colony's own pixels and splits merged
blobs between their colonies; ``tracker="seed"`` is MATLAB's seed-pixel tracker
(``scanlag.track.seed_tracker``), where merged colonies all report the blob. Outputs per plate,
under ``<results>/<scanner>/plate_<k>/``: plate.json, mask.png, labels.zarr (detected objects),
colonies.zarr (colony IDs, footprint tracker only), tracks.parquet, colonies.parquet (see
``write_plate``). ``retrack_plate`` tracks an already segmented plate again from its labels.zarr.

Detectors:
* ``classical``: MATLAB's threshold (im2L.m), with contrast limits from the plate's last frame
* a trained model directory (U-Net or Cellpose-SAM, ``scanlag.detect.unet.load_detector``)
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from scanlag.detect.channels import PREVIOUS_GAP
from scanlag.detect.classical import DEFAULT_THRESHOLD, binarize, label_column_major, stretch_limits, subtract_background
from scanlag.io.matlab_import import ColonyStatus
from scanlag.io.raw import read_scan
from scanlag.pipeline.flags import colony_flags
from scanlag.pipeline.plates import Aligner, Plate, circle_mask, crop, find_plates, is_empty_slot
from scanlag.pipeline.scans import Scanner
from scanlag.store import PROJECT_ROOT, ResultsStore
from scanlag.track.seed_tracker import MIN_AREA, SeedTracker, border_status
from scanlag.track.tracker import FootprintTracker

MODELS = {
    "unet": PROJECT_ROOT / "models" / "unet_scanlag_v1",
    "cellpose": PROJECT_ROOT / "models" / "cpsam_scanlag_v1",
}


@dataclass
class RunConfig:
    detector: str = "unet"  # "classical", "unet", "cellpose" or a model directory
    n_plates: int = 6
    mask_fraction: float = 0.8
    threshold: float = DEFAULT_THRESHOLD  # classical only
    min_upper_limit: float = 0.06  # classical only: floor on the contrast stretch's upper limit
    min_area: int = MIN_AREA  # px; smaller unmatched blobs don't start a colony
    tracker: str = "footprint"  # or "seed" (MATLAB: merged colonies report the merged blob)
    workers: int = 6  # plates processed in parallel threads per frame (1 = one after another)


class ClassicalSegmenter:
    """MATLAB's threshold detector.

    Images2Colonies.m stretches contrast to the cleaned last frame's 1-99 % range inside the mask,
    so the effective threshold is TH x upper limit. On an empty or sparse plate the upper limit
    is only a few gray levels and noise floods the plate (tens of thousands of blobs), and a few
    pixels' difference in the mask moves it by one gray level. ``min_upper_limit`` puts a floor
    under it (0 = exact MATLAB behaviour).
    """

    name = "classical"

    def __init__(self, threshold: float, min_upper_limit: float = 0.0, log=print):
        self.threshold = threshold
        self.min_upper_limit = min_upper_limit
        self.limits: dict[int, np.ndarray] = {}
        self.log = log

    def set_last_frame(self, plate: int, last: np.ndarray, background: np.ndarray, mask: np.ndarray) -> None:
        lo, hi = stretch_limits(subtract_background(last, background)[mask])
        if hi < self.min_upper_limit:
            self.log(f"  plate {plate}: contrast upper limit {hi:.3f} raised to {self.min_upper_limit:.3f}")
            hi = self.min_upper_limit
        self.limits[plate] = np.array([lo, hi])

    def segment(self, plate: int, image, background, previous, mask) -> np.ndarray:
        labels, _ = label_column_major(binarize(image, background, self.limits[plate], self.threshold, mask))
        return labels


class ModelSegmenter:
    def __init__(self, model_dir: Path):
        from scanlag.detect.unet import load_detector

        self.det = load_detector(model_dir)
        self.name = Path(model_dir).name

    def segment(self, plate: int, image, background, previous, mask) -> np.ndarray:
        labels = self.det.segment([self.det.prepare(image, background, previous)])[0].astype(np.int32)
        return keep_centered(labels, mask)


def keep_centered(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The objects whose (rounded) centroid is in the working area, renumbered 1..n."""
    rows, cols = np.nonzero(labels)
    if not len(rows):
        return labels
    ids_px = labels[rows, cols]
    area = np.bincount(ids_px)
    ids = np.flatnonzero(area[1:]) + 1
    # integer coordinate sums are exact in float64, so this equals ndimage.center_of_mass
    cy = (np.bincount(ids_px, weights=rows)[ids] / area[ids]).round().astype(int)
    cx = (np.bincount(ids_px, weights=cols)[ids] / area[ids]).round().astype(int)
    keep = ids[mask[np.clip(cy, 0, mask.shape[0] - 1), np.clip(cx, 0, mask.shape[1] - 1)]]
    lut = np.zeros(len(area), np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[labels]


def model_dir(detector: str) -> Path | None:
    """The model folder of a detector: an alias (unet, cellpose) or a folder; None for classical.
    Raises ValueError if it is not a trained model folder (one with a config.json)."""
    if detector == "classical":
        return None
    folder = MODELS.get(detector, Path(detector)).resolve()
    if not (folder / "config.json").is_file():
        raise ValueError(f"unknown detector {detector!r}: use unet, cellpose, classical or a trained model folder "
                         f"(one with a config.json)")
    return folder


def make_segmenter(cfg: RunConfig, log=print):
    """The segmenter for ``cfg.detector``: ClassicalSegmenter or ModelSegmenter."""
    if cfg.detector == "classical":
        return ClassicalSegmenter(cfg.threshold, cfg.min_upper_limit, log)
    return ModelSegmenter(model_dir(cfg.detector))


def make_tracker(cfg: RunConfig):
    """A new tracker for ``cfg.tracker``: FootprintTracker or SeedTracker."""
    if cfg.tracker == "footprint":
        return FootprintTracker(cfg.min_area)
    if cfg.tracker == "seed":
        return SeedTracker(cfg.min_area)
    raise ValueError(f"unknown tracker {cfg.tracker!r} (footprint or seed)")


class ScanFeed:
    """Scans read ahead in one IO thread and shared by the plate workers.

    Frame ``k`` is taken once per plate (``n_takers`` times) and then dropped, and no frame more than
    ``ahead`` past the oldest one still in use is handed out, so a fast worker waits for the slow
    ones and at most ``ahead + 2`` scans are in memory.
    """

    def __init__(self, paths: list[Path], first: np.ndarray, n_takers: int, ahead: int = 3, on_scan_done=None):
        self.paths, self.n_takers, self.ahead = paths, n_takers, ahead
        self.on_scan_done = on_scan_done or (lambda: None)
        self.io = ThreadPoolExecutor(1)
        self.cond = threading.Condition()
        self.failed = False
        self.futures: dict[int, Future] = {0: Future()}
        self.futures[0].set_result(first)
        self.taken = collections.Counter()
        self.oldest = 0  # the first frame some plate hasn't taken yet
        self.next = 1  # the next frame to read
        self._prefetch()

    def _prefetch(self) -> None:
        while self.next < len(self.paths) and self.next <= self.oldest + self.ahead + 1:
            # np.array, not np.asarray: a memory-mapped scan is only read from disk when copied,
            # and that must happen here in the IO thread
            self.futures[self.next] = self.io.submit(lambda j=self.next: np.array(read_scan(self.paths[j])))
            self.next += 1

    def get(self, k: int) -> np.ndarray:
        with self.cond:
            self.cond.wait_for(lambda: self.failed or k <= self.oldest + self.ahead)
            if self.failed:
                raise RuntimeError("stopped: another plate worker failed")
            future = self.futures[k]
        image = future.result()
        with self.cond:
            self.taken[k] += 1
            while self.taken[self.oldest] == self.n_takers:
                del self.futures[self.oldest], self.taken[self.oldest]
                self.oldest += 1
                self.on_scan_done()
            self._prefetch()
            self.cond.notify_all()
        return image

    def fail(self) -> None:
        with self.cond:
            self.failed = True
            self.cond.notify_all()

    def close(self) -> None:
        self.io.shutdown(cancel_futures=True)


@contextlib.contextmanager
def opencv_threads(n: int):
    """OpenCV's own thread count for the block: with several plate workers, every OpenCV call
    starting a thread per core oversubscribes the CPU."""
    import cv2

    before = cv2.getNumThreads()
    cv2.setNumThreads(n)
    try:
        yield
    finally:
        cv2.setNumThreads(before)


def find_dishes(background_scan: np.ndarray, cfg: RunConfig) -> tuple[list[Plate], list[int]]:
    """(plates holding a dish, numbers of the empty slots)."""
    plates = find_plates(background_scan, cfg.n_plates)
    empty = [p.number for p in plates if is_empty_slot(background_scan, p)]
    return [p for p in plates if p.number not in empty], empty


def plate_crop(image: np.ndarray, k: int, plate: Plate, aligner: Aligner) -> tuple[np.ndarray, tuple[int, int]]:
    """Frame ``k``'s crop of ``plate``, shifted to follow the plate (frame 0 is the reference)."""
    dx, dy = (0, 0) if k == 0 else aligner.shift(image)
    x0, y0, s = plate.crop
    return crop(image, x0 + dx, y0 + dy, s), (dx, dy)


def last_frame_counts(scanner: Scanner, cfg: RunConfig, seg=None, log=print) -> dict[int, int]:
    """Objects of at least ``cfg.min_area`` px on each plate's last frame: a quick first estimate of
    how dense the plates are, before the whole time-lapse is processed."""
    seg = seg or make_segmenter(cfg, log)
    n = len(scanner.scans)
    background_scan = read_scan(scanner.scans[0])
    last = read_scan(scanner.scans[-1])
    k_prev = max(n - 1 - PREVIOUS_GAP, 0)
    previous = read_scan(scanner.scans[k_prev])
    counts = {}
    for p in find_dishes(background_scan, cfg)[0]:
        aligner = Aligner(background_scan, p, mask_fraction=cfg.mask_fraction)
        mask = circle_mask(p.crop[2], p, cfg.mask_fraction)
        background = crop(background_scan, *p.crop)
        image = plate_crop(last, n - 1, p, aligner)[0]
        if isinstance(seg, ClassicalSegmenter):
            seg.set_last_frame(p.number, image, background, mask)
        labels = seg.segment(p.number, image, background, plate_crop(previous, k_prev, p, aligner)[0], mask)
        counts[p.number] = int((np.bincount(labels.ravel())[1:] >= cfg.min_area).sum())
    return counts


def process_scanner(scanner: Scanner, store: ResultsStore, cfg: RunConfig, log=print, seg=None,
                    exclude=()) -> list[Path]:
    """Segment and track every plate of ``scanner`` and write its tables (see ``write_plate``).

    ``seg``: a segmenter to reuse (default: a new one from ``cfg``); ``exclude``: plate numbers to
    leave out. Returns the plate folders written (empty if no plate is left).
    """
    t0 = time.time()
    n = len(scanner.scans)
    background_scan = read_scan(scanner.scans[0])
    plates, empty = find_dishes(background_scan, cfg)
    if empty:
        log(f"{scanner.name}: plates {empty} are empty slots (no dish), skipped")
    exclude = set(exclude)
    if excluded := sorted(p.number for p in plates if p.number in exclude):
        log(f"{scanner.name}: plates {excluded} excluded")
    if unknown := sorted(exclude - {p.number for p in plates} - set(empty)):
        log(f"{scanner.name}: no plates {unknown} to exclude")
    plates = [p for p in plates if p.number not in exclude]
    if not plates:
        log(f"{scanner.name}: no plates left to process")
        return []
    aligners = {p.number: Aligner(background_scan, p, mask_fraction=cfg.mask_fraction) for p in plates}
    size = {p.number: p.crop[2] for p in plates}
    masks = {p.number: circle_mask(size[p.number], p, cfg.mask_fraction) for p in plates}
    backgrounds = {p.number: crop(background_scan, *p.crop) for p in plates}
    seg = seg or make_segmenter(cfg, log)
    log(f"{scanner.name}: {n} scans, plates at " + ", ".join(f"{p.number}:({p.cx:.0f},{p.cy:.0f})" for p in plates))

    if isinstance(seg, ClassicalSegmenter):
        last = read_scan(scanner.scans[-1])
        for p in plates:
            seg.set_last_frame(p.number, plate_crop(last, n - 1, p, aligners[p.number])[0], backgrounds[p.number], masks[p.number])

    out = {p.number: (scanner.name, f"plate_{p.number}") for p in plates}
    labels_out = {p.number: store.create_labels((n, size[p.number], size[p.number]), *out[p.number], "labels.zarr") for p in plates}
    trackers = {p.number: make_tracker(cfg) for p in plates}
    colonies_out = {}
    if cfg.tracker == "footprint":
        colonies_out = {p.number: store.create_labels((n, size[p.number], size[p.number]), *out[p.number], "colonies.zarr")
                        for p in plates}
    shifts = {p.number: [] for p in plates}
    history = {p.number: collections.deque(maxlen=PREVIOUS_GAP + 1) for p in plates}

    bar = tqdm(total=n, desc=f"{scanner.name} ({len(plates)} plates)", unit="scan", dynamic_ncols=True)
    feed = ScanFeed(scanner.scans, np.array(background_scan), len(plates), on_scan_done=bar.update)
    writer = ThreadPoolExecutor(2)  # zarr compression and writes, off the workers' path
    writes = []

    def process_plate(k, p):
        im, shift = plate_crop(feed.get(k), k, p, aligners[p.number])
        shifts[p.number].append(shift)
        hist = history[p.number]
        hist.append(im)
        labels = seg.segment(p.number, im, backgrounds[p.number], hist[0], masks[p.number])
        writes.append(writer.submit(labels_out[p.number].__setitem__, k, labels.astype(np.uint16)))
        own = trackers[p.number].update(labels, int(labels.max()))
        if p.number in colonies_out:
            writes.append(writer.submit(colonies_out[p.number].__setitem__, k, own.astype(np.uint16)))

    def run_worker(my_plates):
        try:
            for k in range(n):
                for p in my_plates:
                    process_plate(k, p)
        except BaseException:
            feed.fail()  # don't leave the other workers waiting for this one
            raise

    # Plates are independent, so they run in parallel threads (numpy, OpenCV, scipy and torch
    # release the GIL; the detectors serialize their GPU calls). Each worker moves through the
    # frames on its own, so one plate's CPU work overlaps another's GPU pass. A plate's frames
    # always run in order, so the results don't depend on the number of workers.
    groups = [plates[i :: max(cfg.workers, 1)] for i in range(min(max(cfg.workers, 1), len(plates)))]
    with opencv_threads(max(1, (os.cpu_count() or 1) // len(groups))), ThreadPoolExecutor(len(groups)) as pool:
        for job in [pool.submit(run_worker, g) for g in groups]:
            job.result()
    for w in writes:
        w.result()
    writer.shutdown()
    feed.close()
    bar.close()
    log(f"{scanner.name}: segmented in {(time.time() - t0) / 60:.1f} min")

    with ThreadPoolExecutor(max(1, len(groups))) as pool:  # each plate's tables and flags on its own
        return list(pool.map(lambda p: write_plate(store, out[p.number], scanner, p, cfg, seg, masks[p.number],
                                                   shifts[p.number], trackers[p.number], labels_out[p.number],
                                                   colonies_out.get(p.number)), plates))


def write_plate(store, out, scanner: Scanner, plate: Plate, cfg: RunConfig, seg, mask, shifts,
                tracker, labels, own=None) -> Path:
    """Write one plate's plate.json, mask.png, tracks.parquet and colonies.parquet; returns its folder."""
    detector = seg.name
    times = np.asarray(scanner.times_min)
    tracks, colonies = track_tables(tracker, times, mask, labels, own)
    x0, y0, s = plate.crop
    store.write_json({
        "scanner": scanner.name, "scanner_label": scanner.label, "plate": plate.number,
        "raw_dir": scanner.directory, "raw_scans": [p.name for p in scanner.scans],
        "times_min": times, "start_time": scanner.start.isoformat(sep=" "),
        "circle": plate.to_dict(), "crop_xy_size": [x0, y0, s],
        "shift_xy": shifts, "mask_fraction": cfg.mask_fraction,
        "detector": detector, "threshold": cfg.threshold if detector == "classical" else None,
        "contrast_limits": getattr(seg, "limits", {}).get(plate.number),
        "min_area": cfg.min_area, "tracker": cfg.tracker,
    }, *out, "plate.json")
    store.write_mask(mask, *out, "mask.png")
    store.write_table(tracks, *out, "tracks.parquet")
    store.write_table(colonies, *out, "colonies.parquet")
    return store.path(*out)


def track_tables(tracker, times: np.ndarray, mask: np.ndarray, labels, own=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tracks, colonies) tables of a plate from its tracker, with automatic statuses and flags."""
    area, centroid, blob = tracker.tracks.as_arrays()
    n_frames, n_col = area.shape
    # merged: the colony shares its blob with another colony. With the seed tracker its area is
    # then the blob's; the footprint tracker gives it its share of the blob.
    merged = np.zeros_like(area, bool)
    for k in range(n_frames):
        b = blob[k]
        counts = np.bincount(b[b > 0], minlength=1)
        merged[k] = (b > 0) & (counts[b] > 1)

    t_idx, c_idx = np.nonzero(area)
    cx, cy = centroid[t_idx, c_idx, 0] - 1, centroid[t_idx, c_idx, 1] - 1  # tracker centroids: 1-based (x, y), as MATLAB
    tracks = pd.DataFrame({
        "frame": t_idx, "time_min": times[t_idx], "colony_id": c_idx + 1, "area_px": area[t_idx, c_idx],
        "x": cx, "y": cy, "blob": blob[t_idx, c_idx], "merged": merged[t_idx, c_idx],
    })

    seeds = tracker.tracks.seeds
    status = np.full(n_col, int(ColonyStatus.VALID))
    if n_col:
        status[~border_status(mask, seeds)] = ColonyStatus.NEAR_BORDER
        status[area[-1] == 0] = ColonyStatus.ZERO_FINAL_AREA
    appear = np.argmax(area > 0, axis=0) if n_col else np.zeros(0, int)
    ever_merged = merged.any(axis=0)
    colonies = pd.DataFrame({
        "colony_id": np.arange(1, n_col + 1),
        "status": status,
        "status_name": [ColonyStatus(s).name for s in status],
        "auto_status": status,
        "appearance_frame": appear,
        "appearance_min": times[appear] if n_col else [],
        "first_merged_frame": np.where(ever_merged, np.argmax(merged, axis=0), -1) if n_col else [],
        "final_area_px": area[-1] if n_col else [],
        "seed_x": [c for _, c in seeds],  # seeds are 0-based (row, col)
        "seed_y": [r for r, _ in seeds],
    })
    colonies = colonies.join(colony_flags(area, merged, blob, labels, own).set_index("colony_id"), on="colony_id")
    return tracks, colonies


def retrack_plate(store: ResultsStore, plate_dir: Path, tracker: str = "footprint") -> dict:
    """Track an already segmented plate again from its labels.zarr, without reading the scans.

    Rewrites tracks.parquet, colonies.parquet, colonies.zarr (footprint tracker) and the tracker in
    plate.json. Colony IDs change, so review decisions can't carry over: the old colonies.parquet and
    review.json are kept as colonies.<old tracker>.bak.parquet and review.<old tracker>.bak.json, and
    the plate is reviewed afresh. Returns the old and new colony counts.
    """
    import zarr
    from PIL import Image

    plate_dir = Path(plate_dir)
    rel = plate_dir.resolve().relative_to(store.root).parts
    info = plate_info(plate_dir)
    cfg = RunConfig(tracker=tracker, min_area=info.get("min_area", MIN_AREA))
    labels = zarr.open_array(str(plate_dir / "labels.zarr"), mode="r")
    mask = np.array(Image.open(plate_dir / "mask.png")).astype(bool)
    times = np.asarray(info["times_min"], float)

    old = info.get("tracker", "seed")
    n_old = len(pd.read_parquet(plate_dir / "colonies.parquet"))
    for name, suffix in [("colonies", ".parquet"), ("review", ".json")]:
        if (plate_dir / f"{name}{suffix}").exists():
            (plate_dir / f"{name}{suffix}").replace(store.path(*rel, f"{name}.{old}.bak{suffix}"))
    if (plate_dir / "colonies.zarr").exists():
        shutil.rmtree(store.path(*rel, "colonies.zarr"))

    tr = make_tracker(cfg)
    own_out = store.create_labels(labels.shape, *rel, "colonies.zarr") if tracker == "footprint" else None
    for k in range(labels.shape[0]):
        lab = np.asarray(labels[k]).astype(np.int32)
        own = tr.update(lab, int(lab.max()))
        if own_out is not None:
            own_out[k] = own.astype(np.uint16)
    tracks, colonies = track_tables(tr, times, mask, labels, own_out)
    store.write_json(info | {"tracker": tracker, "min_area": cfg.min_area}, *rel, "plate.json")
    store.write_table(tracks, *rel, "tracks.parquet")
    store.write_table(colonies, *rel, "colonies.parquet")
    return {"old_tracker": old, "old_colonies": n_old, "colonies": len(colonies),
            "flagged": int((colonies.n_flags > 0).sum())}


def refresh_flags(plate_dir: Path) -> int:
    """Recompute the review flags of a processed plate (e.g. after the flag rules changed).

    Statuses and review decisions are kept. Returns the number of flagged colonies.
    """
    import zarr

    colonies = pd.read_parquet(plate_dir / "colonies.parquet")
    tracks = pd.read_parquet(plate_dir / "tracks.parquet")
    labels = zarr.open_array(str(plate_dir / "labels.zarr"), mode="r")
    own = zarr.open_array(str(plate_dir / "colonies.zarr"), mode="r") if merged_area_is_own(plate_info(plate_dir)) else None
    shape = (labels.shape[0], len(colonies))
    area, merged, blob = np.zeros(shape, np.int64), np.zeros(shape, bool), np.zeros(shape, np.int64)
    t, c = tracks.frame.to_numpy(), tracks.colony_id.to_numpy() - 1
    area[t, c], merged[t, c], blob[t, c] = tracks.area_px, tracks.merged, tracks.blob
    flags = colony_flags(area, merged, blob, labels, own).set_index("colony_id")
    colonies = colonies.drop(columns=["flags", "n_flags"]).join(flags, on="colony_id")
    colonies.to_parquet(plate_dir / "colonies.parquet", index=False)
    return int((colonies.n_flags > 0).sum())


def load_plate_dirs(results: Path) -> list[Path]:
    """The plate folders under ``results`` that have a colonies.parquet, sorted."""
    return sorted(p.parent for p in Path(results).glob("*/plate_*/colonies.parquet"))


def plate_info(plate_dir: Path) -> dict:
    """A plate folder's plate.json."""
    return json.loads((Path(plate_dir) / "plate.json").read_text())


def merged_area_is_own(info: dict) -> bool:
    """True if a merged colony's area is its own share (footprint tracker), not the merged blob's.
    Plates processed before the footprint tracker existed have no "tracker" key."""
    return info.get("tracker", "seed") == "footprint"
