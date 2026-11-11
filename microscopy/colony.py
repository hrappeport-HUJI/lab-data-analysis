"""Founder events from microcolony following, not from frame-to-frame single-cell linking.

Each founder's colony is the set of segmented objects that overlap the colony's footprint in the
previous frame (after correcting the field-wide stage shift). Splits and touching daughters
therefore never end a colony. Events:

* division: the first frame at which the colony holds two or more cells, persisting, once the
  colony has grown. A dented (non-convex) outline that has grown counts as two cells, because
  daughters bend at the septum before a bright septum line is visible.
* lost ("fused"): the colony shares an object with another colony before dividing. Touching
  colonies that have not grown are split by their previous footprints instead.
* lost ("vanished"): no object for more than ``gap_frames`` consecutive frames.
* no division: followed to the last frame of the recording as one cell.

Parameters were chosen on Pilot_3 Set A human tags (2026-11-09); see
docs/microscopy/founder_events.md for the validation numbers.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

DEFAULTS = {
    "gap_frames": 2,
    "persist": [2, 3],  # >=2 cells in at least 2 of the 3 frames starting at the candidate
    "min_colony_growth": 1.5,  # colony area relative to the founder before a division counts
    "dent_solidity": 0.8,  # an object less convex than this counts as two cells ...
    "dent_min_growth": 1.7,  # ... once it is this much larger than the founder
    "static_split_growth": 1.4,  # touching colonies both below this growth are split, not fused
    "doublet_solidity": 0.89,  # founders less convex than this at baseline are flagged
    "drift_radius_px": 30,
}


def _solidity(mask):
    from skimage.morphology import convex_hull_image

    area = int(mask.sum())
    return area / max(1, int(convex_hull_image(mask).sum()))


def _centroids(labels):
    index = np.arange(1, int(labels.max()) + 1)
    if not len(index):
        return np.zeros((0, 2))
    present = np.bincount(labels.ravel(), minlength=len(index) + 1)[1:] > 0
    index = index[present]
    if not len(index):
        return np.zeros((0, 2))
    return np.array(ndimage.center_of_mass(labels > 0, labels, index))


def field_shift(previous, current, radius=30):
    """Stage shift between frames: median displacement of mutual-nearest object centroids."""
    a, b = _centroids(previous), _centroids(current)
    if len(a) < 3 or len(b) < 3:
        return 0, 0
    d1, j = cKDTree(b).query(a, distance_upper_bound=radius)
    _, i = cKDTree(a).query(b, distance_upper_bound=radius)
    pairs = [(k, j[k]) for k in range(len(a)) if np.isfinite(d1[k]) and i[j[k]] == k]
    if len(pairs) < 3:
        return 0, 0
    dy, dx = np.median(np.array([b[q] - a[p] for p, q in pairs]), axis=0)
    return int(round(dy)), int(round(dx))


def _shift(array, dy, dx):
    if not dy and not dx:
        return array
    out = np.zeros_like(array)
    h, w = array.shape
    if abs(dy) >= h or abs(dx) >= w:
        return out
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = array[ys, xs]
    return out


class ColonyEvents:
    """Feed one field's usable frames in order: start(), then update()/skip(), then finish()."""

    def __init__(self, parameters=None):
        self.p = {**DEFAULTS, **(parameters or {})}
        self.founders = {}
        self.frames = []

    def start(self, frame, labels, nodes, founders):
        bylabel = {n["label"]: n for n in nodes}
        byobs = {f["observation_id"]: f for f in founders}
        self.K = np.asarray(labels, np.int64).copy()  # pixel -> colony id; founders keep their label
        self.previous = np.asarray(labels).copy()
        self.next_id = int(labels.max()) + 1
        self.frames = [frame]
        self.skipped_after_last_use = False
        boxes = ndimage.find_objects(np.asarray(labels))
        for label, node in bylabel.items():
            founder = byobs.get(node["id"])
            if founder is None:
                continue
            box = boxes[label - 1]
            solidity = _solidity(labels[box] == label)
            self.founders[label] = {
                "id": founder["id"],
                "area0": node["area_px"],
                "solidity0": solidity,
                "counts": [1],
                "areas": [node["area_px"]],
                "shapes": [[(node["area_px"], solidity)]],
                "observations": [[node["id"]]],
                "stop": None,  # (frame, reason)
                "missing": 0,
                "last_seen": frame,
                "last_area": node["area_px"],  # last observed area (dropouts and skipped frames keep it)
                "gap": False,  # an unusable frame fell inside the observed window
                "contact": None,
            }

    def skip(self, frame):
        """An unusable frame: nothing observed, colonies keep their footprints."""
        self.skipped_after_last_use = True
        self.frames.append(frame)
        for f in self.founders.values():
            if f["stop"] is None:
                f["gap"] = True
                f["counts"].append(0)
                f["areas"].append(0)
                f["shapes"].append([])
                f["observations"].append([])

    def update(self, frame, labels, nodes):
        p = self.p
        labels = np.asarray(labels)
        self.frames.append(frame)
        self.skipped_after_last_use = False
        dy, dx = field_shift(self.previous, labels, p["drift_radius_px"])
        K = _shift(self.K, dy, dx)
        self.previous = labels.copy()
        L = labels.astype(np.int64).copy()
        n = int(L.max())
        both = (L > 0) & (K > 0)
        owners = {}
        if both.any():
            for o, c in np.unique(np.stack([L[both], K[both]]), axis=1).T:
                owners.setdefault(int(o), set()).add(int(c))
        piece_of = {}
        original_boxes = ndimage.find_objects(labels)
        for o, ow in list(owners.items()):
            if len(ow) < 2 or not all(self._static(c) for c in ow):
                continue
            box = original_boxes[o - 1]
            box = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in box)
            obj = L[box] == o
            prior = np.where(np.isin(K[box], list(ow)), K[box], 0)
            _, (iy, ix) = ndimage.distance_transform_edt(prior == 0, return_indices=True)
            nearest = prior[iy, ix]
            sub = L[box]
            for c in ow:
                piece = obj & (nearest == c)
                if piece.any():
                    n += 1
                    sub[piece] = n
                    owners[n] = {c}
                    piece_of[n] = o
                    self.founders[c]["contact"] = self.founders[c]["contact"] or frame
            owners.pop(o)
        lut = np.zeros(n + 1, np.int64)
        members = {}
        for o in range(1, n + 1):
            ow = owners.get(o)
            if not ow:
                lut[o] = self.next_id
                self.next_id += 1
            elif len(ow) == 1:
                c = next(iter(ow))
                lut[o] = c
                members.setdefault(c, []).append(o)
            else:
                lut[o] = self.next_id
                self.next_id += 1
                for c in ow:
                    f = self.founders.get(c)
                    if f and f["stop"] is None:
                        f["stop"] = (frame, "fused")
        newK = lut[L]
        boxes = ndimage.find_objects(L)
        node_of = {nd["label"]: nd["id"] for nd in nodes}
        carry = []
        for c, f in self.founders.items():
            if f["stop"] is not None:
                continue
            if c in members:
                shapes, ids = [], []
                for o in members[c]:
                    m = L[boxes[o - 1]] == o
                    shapes.append((int(m.sum()), _solidity(m)))
                    ids.append(node_of.get(piece_of.get(o, o)))
                f["counts"].append(len(members[c]))
                f["areas"].append(sum(a for a, _ in shapes))
                f["shapes"].append(shapes)
                f["observations"].append(sorted({i for i in ids if i}))
                f["missing"] = 0
                f["last_seen"] = frame
                f["last_area"] = f["areas"][-1]
            else:
                f["counts"].append(0)
                f["areas"].append(0)
                f["shapes"].append([])
                f["observations"].append([])
                f["missing"] += 1
                if f["missing"] == 1:
                    f["first_missing"] = frame
                if f["missing"] > p["gap_frames"]:
                    f["stop"] = (f["first_missing"], "vanished")  # loss interval ends where it first went missing
                else:
                    carry.append(c)
        if carry:  # carry missing colonies' footprints through a short dropout
            keep = np.isin(K, carry) & (newK == 0)
            newK[keep] = K[keep]
        self.K = newK

    def _static(self, c):
        f = self.founders.get(c)
        if f is None or f["stop"] is not None:
            return False
        return 0 < f["last_area"] < self.p["static_split_growth"] * f["area0"]

    def _cells(self, f):
        p = self.p
        out = []
        for shapes, count in zip(f["shapes"], f["counts"]):
            if not shapes:
                out.append(count)
                continue
            out.append(sum(
                2 if solidity < p["dent_solidity"] and area >= p["dent_min_growth"] * f["area0"] else 1
                for area, solidity in shapes
            ))
        return out

    def finish(self, recording_last_frame):
        p = self.p
        need, window = p["persist"]
        frames = self.frames
        events = {}
        for f in self.founders.values():
            cells = self._cells(f)
            division = unconfirmed = None
            for t in range(1, len(cells)):
                if cells[t] >= 2 and f["areas"][t] >= p["min_colony_growth"] * f["area0"]:
                    ahead = cells[t:t + window]
                    positive = sum(1 for x in ahead if x >= 2)
                    if positive >= need:
                        division = t
                        break
                    if unconfirmed is None and len(ahead) < window and positive + window - len(ahead) >= need:
                        unconfirmed = t  # the recording or the colony ended before it could be confirmed
            def before(t):  # last observed frame before t (an unobserved frame is no evidence of one cell)
                k = t - 1
                while k > 0 and not f["shapes"][k]:
                    k -= 1
                return frames[k]
            stop_frame, stop_reason = f["stop"] or (None, None)
            flags = []
            if f["solidity0"] < p["doublet_solidity"]:
                flags.append("possible_doublet")
            if f["contact"] is not None:
                flags.append("touched_neighbour")
            if division is not None and (stop_frame is None or frames[division] < stop_frame):
                decision, reason = "division", "division"
                lower, upper = before(division), frames[division]
                end = frames[division]
            elif unconfirmed is not None:
                decision, reason = "lost", "unconfirmed_division_candidate"
                lower, upper = before(unconfirmed), frames[unconfirmed]
                end = lower
            elif stop_frame is not None:
                decision, reason = "lost", stop_reason
                lower, upper = f["last_seen"], stop_frame
                end = f["last_seen"]
            elif f["last_seen"] >= recording_last_frame:
                decision, reason = "no_division", "recording_end"
                lower = upper = end = recording_last_frame
            else:
                decision = "lost"
                reason = "unusable_frames" if self.skipped_after_last_use or f["last_seen"] < frames[-1] else "ended"
                end = f["last_seen"]
                later = [fr for fr in frames if fr > end]
                lower, upper = end, (later[0] if later else end)
            shown = [i for frame_ids in f["observations"] for i in frame_ids]
            events[f["id"]] = {
                "decision": decision,
                "lower_frame": int(lower),
                "upper_frame": int(upper),
                "last_identity_frame": int(end),
                "tracking_end_reason": reason,
                "flags": flags,
                "contains_gap": bool(f["gap"]),
                "observation_ids": shown,
            }
        return events
