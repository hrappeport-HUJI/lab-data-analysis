"""Quality report of processed plates: what to check before the colony review.

``write_qc`` writes ``<results>/qc/``:

* ``qc.csv``: one row per plate (``plate_qc``): the density check's estimate, the tracked and
  whole-plate counts and how far they are from the estimate, merges, flags by kind, possible
  doublets and bad frames
* ``notes.txt``: plain-text warnings worth reading before the review (``qc_notes``), and the
  appearance-vs-growth control per scanner (``control_notes``)
* ``index.html``: both, plus per plate the last frame with the tracker's colony outlines (flagged
  included colonies marked) and galleries of flagged colonies: each row is one colony over time,
  with its outline in red and its neighbours' in cyan, and its area track (red dots: merged frames,
  grey bands: bad frames)

Control, on every run: the Spearman rank correlation of appearance time with radial growth rate
(``lag_growth_rho``, ``_p``, ``_n`` per plate; pooled per scanner in the notes). Appearance is the
first detection (about 10 px), so a fast grower is detected sooner even with the same lag. A clearly
negative rho means appearance times partly reflect growth rate (the pilot 2026_10_15: rho -0.29,
p 1e-36 on the dense scanner, -0.13, p 0.02 on the sparse one).

Everything is read from the results and the raw scans (read only). The thresholds of the notes are
module constants.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scanlag.io.matlab_import import ColonyStatus

EST_OFF = 0.5  # note when the tracked count is this far (relative) from the density estimate
FLAGGED_SHARE = 0.15  # note when more than this share of included colonies is flagged
MERGE_EXCESS = 0.10  # note when n_final_full_est exceeds n_final_full by more than this
DOUBLET_SHARE = 0.02  # note when more than this share of included colonies are possible doublets
LAG_GROWTH_RHO = -0.3  # note a plate whose appearance-vs-growth rho is below this (and p < LAG_GROWTH_P)
LAG_GROWTH_P = 0.01
GALLERY_PER_KIND = 3
WINDOW = 70  # px: half size of a gallery crop
KIND = re.compile(r"\s*\(.*$|\s*x[\d.]+$|\s*to [\d.]+%$")


def flag_kinds(flags: pd.Series) -> pd.Series:
    """'shrinks to 46%; jump x2.1' -> ['shrinks', 'jump'] per colony (exploded, index kept)."""
    k = flags.fillna("").str.split("; ").explode().str.replace(KIND, "", regex=True)
    return k[k != ""]


def appearance_growth(plate_dir: Path) -> pd.DataFrame:
    """Included colonies' appearance time (h) and radial growth rate (sqrt(px)/h, the analysis' default
    fit, bad frames skipped), for the control."""
    from scanlag.analysis.conditions import FitConfig, radial_growth
    from scanlag.pipeline.export import plate_tables

    col, tr, _ = plate_tables(Path(plate_dir))
    col, tr = col[col.included], tr[tr.included & ~tr.bad_frame].drop(columns="included")
    growth = radial_growth(tr, FitConfig()).set_index("colony_id").slope if len(tr) else pd.Series(dtype=float)
    return pd.DataFrame({"scanner": col.scanner.to_numpy(), "plate": col.plate.to_numpy(),
                         "appearance_h": col.appearance_h.to_numpy(),
                         "growth": col.colony_id.map(growth).to_numpy(float)})


def plate_qc(plate_dir: Path, density: pd.DataFrame | None = None) -> dict:
    """One plate's QC numbers (see the module docstring)."""
    plate_dir = Path(plate_dir)
    info = json.loads((plate_dir / "plate.json").read_text())
    col = pd.read_parquet(plate_dir / "colonies.parquet")
    inc = col[col.status == ColonyStatus.VALID]
    final = info.get("final_count") or {}
    est = None
    if density is not None:
        row = density[(density.scanner == info["scanner"]) & (density.plate == info["plate"])]
        est = float(row.est_colonies.iloc[0]) if len(row) else None
    n_col = len(col)  # tracked in the working area, near the border included, as the estimate
    kinds = flag_kinds(inc["flags"]).value_counts()
    from scanlag.pipeline.flags import unreviewed_mask

    open_flags = int(unreviewed_mask(inc).sum()) if len(inc) else 0
    review = plate_dir / "review.json"
    full, full_est = final.get("n_full"), final.get("n_full_est")
    from scanlag.analysis.conditions import appearance_growth_correlation

    ag = appearance_growth(plate_dir)
    rho, p, n_ag = appearance_growth_correlation(ag.appearance_h, ag.growth)
    return {
        "scanner": info["scanner"], "plate": info["plate"],
        "reviewed": bool(json.loads(review.read_text()).get("visited")) if review.exists() else False,
        "est_colonies": est, "n_colonies": n_col, "tracked_vs_est": n_col / est - 1 if est else None,
        "n_included": len(inc),
        "n_whole_plate": len(inc) + final["n_rim"] if final else None,
        "n_final_full": full, "n_final_full_est": full_est,
        "merge_excess": full_est / full - 1 if full else None,
        "merged_share": float((inc.first_merged_frame >= 0).mean()) if len(inc) else None,
        "n_flagged_included": int((inc.n_flags > 0).sum()),
        "flagged_share": float((inc.n_flags > 0).mean()) if len(inc) else None,
        "n_flags_unreviewed": open_flags,
        "n_possible_doublets": int(kinds.get("possible doublet", 0)),
        "bad_frames": " ".join(map(str, info.get("bad_frames") or [])),
        "lag_growth_rho": rho, "lag_growth_p": p, "lag_growth_n": n_ag,
        "flags": "; ".join(f"{k} {v}" for k, v in kinds.items()),
    }


def qc_notes(table: pd.DataFrame) -> list[str]:
    """Warnings worth reading before the review, one line each."""
    notes = []
    for r in table.itertuples():
        name = f"{r.scanner} plate {r.plate}"
        if r.tracked_vs_est is not None and np.isfinite(r.tracked_vs_est) and abs(r.tracked_vs_est) > EST_OFF:
            notes.append(f"{name}: {r.n_colonies} colonies tracked against {r.est_colonies:.0f} estimated by the "
                         f"density check ({r.tracked_vs_est:+.0%}): look at the outline picture")
        open_flags = getattr(r, "n_flags_unreviewed", r.n_flagged_included)
        if r.n_included and open_flags / r.n_included > FLAGGED_SHARE:
            notes.append(f"{name}: {open_flags} of {r.n_included} included flagged colonies not reviewed yet "
                         f"({open_flags / r.n_included:.0%}): {r.flags}")
        if r.merge_excess is not None and r.merge_excess > MERGE_EXCESS:
            notes.append(f"{name}: merged clusters at the end hold about {r.n_final_full_est - r.n_final_full} "
                         f"more colonies than objects ({r.merge_excess:+.0%}); many growth fits stop at a merge")
        if r.n_included and r.n_possible_doublets / r.n_included > DOUBLET_SHARE:
            notes.append(f"{name}: {r.n_possible_doublets} possible doublets (two colonies tracked as one): "
                         f"the plate likely holds that many more colonies")
        if r.bad_frames:
            notes.append(f"{name}: bad frame(s) {r.bad_frames} (many colonies' areas off for one scan); "
                         f"ignored by the flags and the growth metrics")
        rho, p = getattr(r, "lag_growth_rho", np.nan), getattr(r, "lag_growth_p", np.nan)
        if np.isfinite(rho) and rho < LAG_GROWTH_RHO and p < LAG_GROWTH_P:
            notes.append(f"{name}: appearance time and growth rate correlate (Spearman rho {rho:+.2f}, p {p:.1g}): "
                         f"fast growers are detected earlier, so appearance is partly growth rate here")
    return notes


def control_notes(pairs: pd.DataFrame) -> list[str]:
    """The appearance-vs-growth control pooled per scanner (always reported)."""
    from scanlag.analysis.conditions import appearance_growth_correlation

    out = []
    for scanner, d in pairs.groupby("scanner", sort=False):
        rho, p, n = appearance_growth_correlation(d.appearance_h, d.growth)
        if np.isfinite(rho):
            out.append(f"control, {scanner}: appearance vs radial growth rate, Spearman rho {rho:+.2f} "
                       f"(p {p:.1g}, n {n}); negative = fast growers detected earlier")
    return out


class PlateImages:
    """Crops of one plate's aligned frames (memory-mapped scans) and its colony ID stack."""

    def __init__(self, plate_dir: Path):
        import zarr

        self.dir = Path(plate_dir)
        self.info = json.loads((self.dir / "plate.json").read_text())
        self.raw = Path(self.info["raw_dir"])
        self.x0, self.y0, self.size = self.info["crop_xy_size"]
        own = self.dir / "colonies.zarr"
        self.own = zarr.open_array(str(own if own.exists() else self.dir / "labels.zarr"), mode="r")
        self.colony_ids = own.exists()

    def frame(self, k: int, rows: slice | None = None, cols: slice | None = None) -> np.ndarray:
        from scanlag.io.raw import read_window

        dx, dy = self.info["shift_xy"][k]
        rows = rows or slice(0, self.size)
        cols = cols or slice(0, self.size)
        y, x = self.y0 + dy, self.x0 + dx
        return read_window(self.raw / self.info["raw_scans"][k], slice(y + rows.start, y + rows.stop),
                           slice(x + cols.start, x + cols.stop))


def _normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(float)
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1), 0, 1)


def outline_picture(images: PlateImages, flagged_ids, width: int = 900) -> np.ndarray:
    """The last frame (RGB uint8, ``width`` px) with colony outlines in cyan and flagged colonies'
    in orange."""
    import cv2
    from skimage.segmentation import find_boundaries

    k = images.own.shape[0] - 1
    rgb = _normalize(images.frame(k))
    lab = np.asarray(images.own[k])
    rgb[find_boundaries(lab, mode="inner") & (lab > 0)] = [0, 0.9, 1]
    flagged = np.isin(lab, list(flagged_ids)) & (lab > 0)
    rgb[find_boundaries(np.where(flagged, lab, 0), mode="inner") & flagged] = [1, 0.55, 0]
    out = (rgb * 255).astype(np.uint8)
    return cv2.resize(out, (width, width), interpolation=cv2.INTER_AREA)


def gallery_frames(appear: int, first_merge: int, n_frames: int, track: pd.DataFrame) -> list[int]:
    """Six frames telling a colony's story: appearance, a little later, just before and at its first
    merge (or later growth), the first frame it is separate again (or later), the last."""
    last = n_frames - 1
    if first_merge >= 0:
        after = track[(track.frame > first_merge) & ~track.merged].frame
        mid = [max(first_merge - 1, appear), first_merge,
               int(after.iloc[0]) if len(after) else min(first_merge + 20, last)]
    else:
        mid = [min(appear + 12, last), min(appear + 24, last), min(appear + 48, last)]
    return [min(f, last) for f in [appear, min(appear + 4, last), *mid, last]]


def gallery(images: PlateImages, colonies: pd.DataFrame, tracks: pd.DataFrame, title: str, path: Path) -> None:
    """One row per colony: six crops over time with its outline and its area track."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries

    t_h = np.asarray(images.info["times_min"], float) / 60
    bad = images.info.get("bad_frames") or []
    n_frames = images.own.shape[0]
    fig, axes = plt.subplots(len(colonies), 7, figsize=(17, 2.6 * len(colonies)), squeeze=False)
    for row, c in zip(axes, colonies.itertuples()):
        cid = int(c.colony_id)
        tr = tracks[tracks.colony_id == cid].sort_values("frame")
        last = np.argwhere(np.asarray(images.own[n_frames - 1]) == cid) if images.colony_ids else []
        cy, cx = last.mean(0) if len(last) else (c.seed_y, c.seed_x)
        r0, c0 = int(np.clip(cy - WINDOW, 0, images.size - 2 * WINDOW)), int(np.clip(cx - WINDOW, 0, images.size - 2 * WINDOW))
        rows, cols = slice(r0, r0 + 2 * WINDOW), slice(c0, c0 + 2 * WINDOW)
        for ax, k in zip(row[:6], gallery_frames(int(c.appearance_frame), int(c.first_merged_frame), n_frames, tr)):
            rgb = _normalize(images.frame(k, rows, cols))
            lab = np.asarray(images.own[k, rows, cols])
            rgb[find_boundaries(lab, mode="inner") & (lab > 0)] = [0, 0.9, 1]
            if images.colony_ids:
                rgb[find_boundaries(lab == cid, mode="inner") & (lab == cid)] = [1, 0.1, 0.1]
            ax.imshow(rgb)
            ax.set_title(f"frame {k}, {t_h[k]:.1f} h" + (" (bad)" if k in bad else ""), fontsize=7)
            ax.axis("off")
        ax = row[6]
        for k in bad:
            ax.axvspan(t_h[k] - 0.1, t_h[k] + 0.1, color="0.85")
        ax.plot(t_h[tr.frame], tr.area_px, "k-", lw=0.8)
        m = tr[tr.merged]
        ax.plot(t_h[m.frame], m.area_px, "r.", ms=2)
        ax.set_title(f"colony {cid}: {c.flags}"[:80], fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("h", fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def write_qc(results: Path, plate_dirs: list[Path], store, galleries: int = GALLERY_PER_KIND,
             log=print) -> Path:
    """Write ``<results>/qc/`` (see the module docstring) for ``plate_dirs``; returns index.html's path.
    ``galleries``: colonies per flag kind and plate in the galleries (0: tables and outlines only)."""
    results = Path(results)
    dens_path = results / "density" / "density.csv"
    density = pd.read_csv(dens_path) if dens_path.exists() else None
    table = pd.DataFrame([plate_qc(d, density) for d in plate_dirs])
    notes = qc_notes(table) + control_notes(pd.concat([appearance_growth(d) for d in plate_dirs]))
    store.write_table(table, "qc", "qc.csv")
    (store.path("qc", "notes.txt")).write_text("\n".join(notes) + ("\n" if notes else ""))

    sections = []
    for d in plate_dirs:
        images = PlateImages(d)
        name = f"{images.info['scanner']}_p{images.info['plate']}"
        col = pd.read_parquet(d / "colonies.parquet")
        inc = col[(col.status == ColonyStatus.VALID) & (col.n_flags > 0)]
        try:
            store.write_image(outline_picture(images, set(inc.colony_id)), "qc", f"{name}_outlines.png")
        except (OSError, ValueError) as e:  # raw scans moved or unreadable: tables only
            log(f"qc: {name}: no pictures ({e})")
            sections.append((name, None, []))
            continue
        pics = []
        if galleries and len(inc):
            tracks = pd.read_parquet(d / "tracks.parquet")
            kinds = flag_kinds(inc["flags"])
            for kind in kinds.value_counts().index:
                ids = kinds[kinds == kind].index.unique()
                chosen = inc.loc[ids].sample(min(galleries, len(ids)), random_state=0).sort_values("colony_id")
                fname = f"{name}_{re.sub(r'[^a-z]+', '_', kind)}.png"
                gallery(images, chosen, tracks, f"{images.info['scanner']} plate {images.info['plate']}: {kind} "
                        f"({len(ids)} included colonies; red: this colony, cyan: others)", store.path("qc", fname))
                pics.append((kind, len(ids), fname))
        sections.append((name, f"{name}_outlines.png", pics))
        log(f"qc: {name} done")
    path = store.path("qc", "index.html")
    path.write_text(_html(results, table, notes, sections))
    log(f"qc report: {path}")
    return path


def _html(results: Path, table: pd.DataFrame, notes: list[str], sections) -> str:
    fmt = table.copy()
    fmt["lag_growth_rho"] = fmt.lag_growth_rho.map(lambda v: "" if not np.isfinite(v) else f"{v:+.2f}")
    fmt["lag_growth_p"] = fmt.lag_growth_p.map(lambda v: "" if not np.isfinite(v) else f"{v:.1g}")
    for c in ("tracked_vs_est", "merge_excess", "merged_share", "flagged_share"):
        fmt[c] = fmt[c].map(lambda v: "" if v is None or not np.isfinite(v) else f"{v:+.0%}" if c in
                            ("tracked_vs_est", "merge_excess") else f"{v:.0%}")
    fmt["est_colonies"] = fmt.est_colonies.map(lambda v: "" if v is None or not np.isfinite(v) else f"{v:.0f}")
    parts = [f"<h1>ScanLag QC: {html.escape(results.name)}</h1>",
             f"<p class=muted>{html.escape(str(results))}</p>",
             "<h2>Notes</h2>" + ("<ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>"
                                 if notes else "<p>None.</p>"),
             "<h2>Plates</h2>" + fmt.to_html(index=False, na_rep="", escape=True, border=0)]
    for name, outline, pics in sections:
        parts.append(f"<h2 id='{name}'>{html.escape(name)}</h2>")
        if outline:
            parts.append(f"<p><img class=plate src='{outline}' alt='{name} outlines'></p>")
        for kind, n, fname in pics:
            parts.append(f"<h3>{html.escape(kind)} ({n})</h3><img class=gallery src='{fname}' alt='{kind}'>")
    style = ("body{font:14px system-ui,sans-serif;margin:24px;max-width:1300px;color:#222;background:#fff}"
             "table{border-collapse:collapse;font-size:12px}td,th{padding:3px 8px;border-bottom:1px solid #ddd;"
             "text-align:right}th{background:#f3f3f3}.muted{color:#777}img.plate{width:min(900px,100%)}"
             "img.gallery{width:100%}li{margin:4px 0}")
    return (f"<!doctype html><html><head><meta charset=utf-8><title>ScanLag QC</title><style>{style}</style>"
            f"</head><body>{''.join(parts)}</body></html>")
