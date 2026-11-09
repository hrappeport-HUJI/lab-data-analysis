"""Colony review GUI, modelled on MATLAB's PlateAnalyzer.

Layout as in PlateAnalyzer: the plate on the left (time slider below, last frame by default)
and the area of each colony over time on the right. Click a colony on the plate or its line in
the graph to select it in both. Ctrl+E excludes it, Ctrl+I includes it. "Find colony no." jumps to a colony.
Colony numbers use PlateAnalyzer's colours: white, red when the colony appears in this frame, magenta
near the border, yellow excluded.

Flagged colonies (suspicious track or shape, ``scanlag.pipeline.flags``) are listed on the right in
two lists. "To review" starts with every flagged colony; marked on the plate with an orange "!". Include
(Ctrl+I) or exclude (Ctrl+E) a colony and it moves to "Reviewed". In both lists green means included and
red excluded. "Unreview" (Ctrl+U) moves the selected reviewed colonies back to "To review", keeping their
status, to think about them again. F / Shift+F step through "To review". The box above the lists shows
one kind of flag only (e.g. "large at appearance", pieces split off a neighbour); "Include" /
"Exclude" under a list decide the colonies selected in it (Ctrl/Shift+click for several), "Exclude
listed" every colony "To review" shows. Reviewed flag kinds are saved in colonies.parquet's
``flags_reviewed`` (scanlag.pipeline.flags, Reviewing) and logged in review.json.

Clicking a colony in a list or its curve in the graph centres the plate image on it, zoomed in with its
neighbourhood still visible; Esc (or "Full plate") shows the whole plate again. Clicking on the plate
selects without moving the view. The selected colony is marked by a wide yellow ring, to find it quickly
on a crowded plate.

With the footprint tracker (the default), merged colonies keep their own colour and the line
where their blob was split is drawn in white ("Labels only") or as outlines. With the seed tracker,
a blob holding several colonies is white.

Decisions are saved as they are made (colonies.parquet + an append-only review.json log), so
the window can be closed and reopened. "Finish && export" writes the CSV tables.

Below the plate: the time slider (redraws while dragging; numbers come back on release), the
image layer (raw scan or gray(frame - background), the difference MATLAB thresholds) and its
black/white levels, with "Auto contrast" to reset them from the current frame.

Keys: Ctrl+E exclude, Ctrl+I include, Ctrl+U unreview, Esc full plate, Left/Right one frame, Home/End first/last frame,
PageUp/PageDown previous/next plate, F/Shift+F next/previous flagged colony.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path

import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets  # before matplotlib's Qt backend, which otherwise may pick another
                                            # installed binding (conda-forge's matplotlib brings PySide6)

matplotlib.use("QtAgg")
import numpy as np
import pandas as pd
import zarr
from matplotlib import patheffects
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PIL import Image
from skimage.segmentation import find_boundaries

from scanlag.detect.classical import subtract_background
from scanlag.io.matlab_import import ColonyStatus
from scanlag.io.raw import read_scan
from scanlag.pipeline.flags import is_unreviewed, kinds_of, reviewed_column
from scanlag.pipeline.plates import crop
from scanlag.pipeline.run import merged_area_is_own

VALID, BORDER, EXCLUDED, ZERO = (int(s) for s in ColonyStatus)
STATUS_RGB = {BORDER: (255, 0, 255), EXCLUDED: (255, 255, 0), ZERO: (128, 128, 128)}
STATUS_TEXT = {VALID: "white", BORDER: "magenta", EXCLUDED: "yellow", ZERO: "0.6"}
UNASSIGNED_RGB = (90, 90, 90)  # blobs that are no colony (too small, never tracked)
MERGED_RGB = (255, 255, 255)  # seed tracker: a blob holding several colonies; footprint: their split line
FLAG_COLOR = "#ff8c00"


def colony_rgb(cid: int) -> np.ndarray:
    """A stable, bright colour per colony."""
    rng = np.random.default_rng(cid * 7919)
    hsv = np.array([[[rng.random(), 0.55 + 0.45 * rng.random(), 1.0]]])
    return (matplotlib.colors.hsv_to_rgb(hsv)[0, 0] * 255).astype(np.uint8)


class PlateData:
    def __init__(self, plate_dir: Path):
        self.dir = Path(plate_dir)
        self.info = json.loads((self.dir / "plate.json").read_text())
        self.colonies = pd.read_parquet(self.dir / "colonies.parquet")
        self.colonies["flags_reviewed"] = reviewed_column(self.colonies)  # derived for older plates
        self.colonies = self.colonies.drop(columns=["flags_resolved"], errors="ignore")  # its old name
        self.tracks = pd.read_parquet(self.dir / "tracks.parquet")
        self.labels = zarr.open_array(str(self.dir / "labels.zarr"), mode="r")
        # colony IDs per pixel (footprint tracker); None for the seed tracker
        self.own = zarr.open_array(str(self.dir / "colonies.zarr"), mode="r") if merged_area_is_own(self.info) else None
        self.mask = np.array(Image.open(self.dir / "mask.png")).astype(bool)
        self.times = np.asarray(self.info["times_min"], float)
        self.n_frames, self.n = len(self.times), len(self.colonies)
        shape = (self.n_frames, self.n + 1)  # column 0 unused, colony IDs are 1-based
        self.area = np.zeros(shape)
        self.merged = np.zeros(shape, bool)
        self.cx = np.zeros(shape)
        self.cy = np.zeros(shape)
        self.blob = np.zeros(shape, int)
        t, c = self.tracks.frame.to_numpy(), self.tracks.colony_id.to_numpy()
        self.area[t, c] = self.tracks.area_px
        self.merged[t, c] = self.tracks.merged
        self.cx[t, c], self.cy[t, c] = self.tracks.x, self.tracks.y
        self.blob[t, c] = self.tracks.blob
        if self.own is None:  # merged colonies report the blob's centroid; show them at their own seed instead
            seed_x = np.r_[0, self.colonies.seed_x.to_numpy()]
            seed_y = np.r_[0, self.colonies.seed_y.to_numpy()]
            self.cx = np.where(self.merged, seed_x[None, :], self.cx)
            self.cy = np.where(self.merged, seed_y[None, :], self.cy)
        self.colors = np.array([colony_rgb(i) for i in range(self.n + 1)])
        self._images: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._reader = ThreadPoolExecutor(1)
        review = self.dir / "review.json"
        self.review = json.loads(review.read_text()) if review.exists() else {"visited": False, "log": []}

    @property
    def title(self) -> str:
        return f"{self.info['scanner']} / plate {self.info['plate']}"

    def status(self, cid: int) -> int:
        return int(self.colonies.status.iat[cid - 1])

    def flags(self, cid: int) -> str:
        return self.colonies["flags"].iat[cid - 1] or ""

    def reviewed(self, cid: int) -> bool:
        """The colony has flags and all of them are reviewed."""
        return bool(self.flags(cid)) and not is_unreviewed(self.flags(cid), self.colonies["flags_reviewed"].iat[cid - 1])

    def to_review(self, cid: int) -> bool:
        """The colony has flags that are not reviewed."""
        return is_unreviewed(self.flags(cid), self.colonies["flags_reviewed"].iat[cid - 1])

    def _mark(self, ids, on: bool) -> list[int]:
        changed = []
        for cid in ids:
            if self.flags(cid) and self.reviewed(cid) != on:
                self.colonies.loc[self.colonies.colony_id == cid, "flags_reviewed"] = (
                    "; ".join(kinds_of(self.flags(cid))) if on else "")
                changed.append(int(cid))
        return changed

    def decide(self, ids, status: int, action: str) -> int:
        """Set the status of the given colonies and mark the flagged ones reviewed; returns how many
        changed (status or review mark)."""
        ids = [int(c) for c in ids]
        old = {c: self.status(c) for c in ids}
        sel = self.colonies.colony_id.isin(ids)
        self.colonies.loc[sel, ["status", "status_name"]] = [status, ColonyStatus(status).name]
        marked = self._mark(ids, True)
        changed = sorted({c for c in ids if old[c] != status} | set(marked))
        if changed:
            self.review["log"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action,
                                       "to": status, "colony_ids": changed})
            self.save()
        return len(changed)

    def unreview(self, ids) -> int:
        """Move reviewed colonies back to "to review" (their status stays)."""
        changed = self._mark(ids, False)
        if changed:
            self.review["log"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "action": "unreview",
                                       "colony_ids": changed})
            self.save()
        return len(changed)

    @property
    def background(self) -> np.ndarray:
        """Frame 0, kept out of the frame cache so the difference layer never re-reads it."""
        if not hasattr(self, "_background"):
            self._background = self.image(0)
        return self._background

    def _read(self, k: int) -> np.ndarray:
        x0, y0, size = self.info["crop_xy_size"]
        dx, dy = self.info["shift_xy"][k]
        return crop(read_scan(Path(self.info["raw_dir"]) / self.info["raw_scans"][k]), x0 + dx, y0 + dy, size)

    def image(self, k: int) -> np.ndarray:
        with self._lock:
            if k in self._images:
                self._images.move_to_end(k)
                return self._images[k]
        img = self._read(k)
        self._store(k, img)
        return img

    def _store(self, k: int, img: np.ndarray) -> None:
        with self._lock:
            self._images[k] = img
            while len(self._images) > 24:
                self._images.popitem(last=False)

    def prefetch(self, k: int, ahead: int = 3) -> None:
        """Read the neighbouring frames in the background (reading a scan window takes ~0.15 s)."""
        for j in [k + d for d in range(1, ahead + 1)] + [k - d for d in range(1, ahead + 1)]:
            if 0 <= j < self.n_frames:
                with self._lock:
                    if j in self._images:
                        continue
                self._reader.submit(lambda j=j: self._store(j, self._read(j)))

    def set_status(self, cid: int, status: int) -> bool:
        old = self.status(cid)
        if old == status:
            return False
        self.colonies.loc[self.colonies.colony_id == cid, ["status", "status_name"]] = [status, ColonyStatus(status).name]
        self.review["log"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "colony_id": cid, "from": old, "to": status})
        self.save()
        return True

    def set_statuses(self, new: pd.Series, action: str) -> int:
        """Bulk change: ``new`` is the status of every colony, in colony order. Returns the number changed."""
        changed = self.colonies.status.to_numpy() != new.to_numpy()
        if not changed.any():
            return 0
        self.colonies["status"] = new.to_numpy()
        self.colonies["status_name"] = [ColonyStatus(int(s)).name for s in self.colonies.status]
        self.review["log"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "action": action,
                                   "colony_ids": self.colonies.colony_id[changed].astype(int).tolist()})
        self.save()
        return int(changed.sum())

    def exclude(self, ids, action: str) -> int:
        """Exclude the given colonies (those included)."""
        s = self.colonies.status
        return self.set_statuses(s.where(~(self.colonies.colony_id.isin(ids) & (s == VALID)), EXCLUDED), action)

    def reset(self) -> int:
        """Undo all review changes: back to the statuses the detection gave, every flag to review."""
        unmarked = int((self.colonies["flags_reviewed"] != "").sum())
        self.colonies["flags_reviewed"] = ""
        n = self.set_statuses(self.colonies.auto_status, "reset")
        if unmarked and not n:
            self.save()
        return max(n, unmarked)

    def n_changed(self) -> int:
        return int((self.colonies.status != self.colonies.auto_status).sum())

    def save(self) -> None:
        self.colonies.to_parquet(self.dir / "colonies.parquet", index=False)
        (self.dir / "review.json").write_text(json.dumps(self.review, indent=1))

    def colony_at(self, k: int, x: float, y: float) -> int | None:
        """Colony under (x, y) in frame k, else the nearest one within 10 px."""
        xi, yi = int(round(x)), int(round(y))
        if self.own is not None and 0 <= yi < self.mask.shape[0] and 0 <= xi < self.mask.shape[1]:
            c = int(self.own[k, yi, xi])
            if c:
                return c
        elif 0 <= yi < self.mask.shape[0] and 0 <= xi < self.mask.shape[1]:
            lab = int(self.labels[k, yi, xi])
            if lab:
                hits = np.nonzero(self.blob[k] == lab)[0]
                hits = hits[hits > 0]
                if len(hits):
                    return int(min(hits, key=lambda c: np.hypot(self.cx[k, c] - x, self.cy[k, c] - y)))
        present = np.nonzero(self.area[k] > 0)[0]
        if not len(present):
            return None
        d = np.hypot(self.cx[k, present] - x, self.cy[k, present] - y)
        return int(present[d.argmin()]) if d.min() < 10 else None


class ReviewWindow(QtWidgets.QMainWindow):
    def __init__(self, plate_dirs: list[Path], on_finish=None):
        super().__init__()
        self.plate_dirs = plate_dirs
        self.on_finish = on_finish
        self.plate: PlateData | None = None
        self.k = 0
        self.selected: int | None = None
        self.lines: dict[int, matplotlib.lines.Line2D] = {}
        self.texts = []
        self.ring = []
        self.setWindowTitle("ScanLag colony review")
        self._build()
        self.fit_to_screen()
        self.load_plate(0)

    # ---------- layout ----------
    def fit_to_screen(self):
        """Open at most 1500 x 950 px, and never larger than the screen's free area (menu bar and Dock
        excluded)."""
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else QtCore.QRect(0, 0, 1500, 950)
        w, h = min(1500, int(avail.width() * 0.96)), min(950, int(avail.height() * 0.94))
        self.resize(w, h)
        self.move(avail.x() + (avail.width() - w) // 2, avail.y() + (avail.height() - h) // 2)
        self.split_evenly()

    def split_evenly(self):
        """Image : graph : lists about 40 : 36 : 24 of the window's width (the lists at least 300 px)."""
        w = self.width()
        side = max(300, int(w * 0.24))
        rest = max(w - side - 30, 450)
        self.splitter.setSizes([int(rest * 0.53), int(rest * 0.47), side])

    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        # Two rows of controls, so the window fits a laptop screen (about 1,300 px wide)
        bar = QtWidgets.QHBoxLayout()
        self.plate_box = QtWidgets.QComboBox()
        for d in self.plate_dirs:
            self.plate_box.addItem(f"{d.parent.name} / {d.name}")
        self.plate_box.currentIndexChanged.connect(self.load_plate)
        bar.addWidget(QtWidgets.QLabel("Plate:"))
        bar.addWidget(self.plate_box)
        for text, slot in (("◀", lambda: self.step_plate(-1)), ("▶", lambda: self.step_plate(1))):
            b = QtWidgets.QPushButton(text)
            b.setFixedWidth(32)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Find colony"))
        self.find_edit = QtWidgets.QLineEdit()
        self.find_edit.setFixedWidth(55)
        self.find_edit.returnPressed.connect(self.find_colony)
        bar.addWidget(self.find_edit)
        bar.addSpacing(12)
        for text, slot, tip in (("Exclude", lambda: self.set_selected(EXCLUDED), "Exclude the selected colony (Ctrl+E)"),
                                ("Include", lambda: self.set_selected(VALID), "Include the selected colony (Ctrl+I)"),
                                ("Unreview", self.unreview_selected, "Back to To review (Ctrl+U)"),
                                ("◀ flagged", lambda: self.step_flag(-1), "Previous colony to review (Shift+F)"),
                                ("flagged ▶", lambda: self.step_flag(1), "Next colony to review (F)")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            bar.addWidget(b)
        bar.addStretch(1)
        finish = QtWidgets.QPushButton("Finish && export")
        finish.setStyleSheet("font-weight: bold")
        finish.clicked.connect(self.finish)
        bar.addWidget(finish)
        root.addLayout(bar)

        checks_row = QtWidgets.QHBoxLayout()
        self.checks = {}
        for name, default in (("Outlines", True), ("Numbers", True), ("Labels only", False),
                              ("Show excluded", True), ("Flagged only", False), ("Log scale", True)):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(default)
            cb.toggled.connect(self.redraw_all)
            self.checks[name] = cb
            checks_row.addWidget(cb)
        checks_row.addStretch(1)
        for text, slot, tip in (("Full plate", self.full_view, "Show the whole plate (Esc)"),
                                ("Reset plate", self.reset_plate, "Undo all changes on this plate")):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            checks_row.addWidget(b)
        root.addLayout(checks_row)

        # Separate canvases: moving in time redraws only the plate; the graph (up to ~1,000 curves,
        # 0.25 s to draw) only gets its time marker blitted. Image, graph and lists sit in a splitter:
        # drag the dividers to give one more room.
        self.fig_img = Figure(figsize=(5, 5), layout="constrained")
        self.ax_img = self.fig_img.add_subplot()
        self.canvas = FigureCanvasQTAgg(self.fig_img)
        self.fig_graph = Figure(figsize=(5, 5), layout="constrained")
        self.ax_graph = self.fig_graph.add_subplot()
        self.canvas_graph = FigureCanvasQTAgg(self.fig_graph)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar_graph = NavigationToolbar2QT(self.canvas_graph, self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        for toolbar, canvas in ((self.toolbar, self.canvas), (self.toolbar_graph, self.canvas_graph)):
            canvas.setMinimumSize(220, 220)
            canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            toolbar.setMinimumWidth(0)
            panel = QtWidgets.QWidget()
            col = QtWidgets.QVBoxLayout(panel)
            col.setContentsMargins(0, 0, 0, 0)
            col.addWidget(toolbar)
            col.addWidget(canvas, 1)
            splitter.addWidget(panel)
        self._graph_bg = None
        self.canvas_graph.mpl_connect("draw_event", self.on_graph_draw)

        side_panel = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(side_panel)
        side.setContentsMargins(4, 0, 0, 0)
        self.flag_kind = QtWidgets.QComboBox()
        self.flag_kind.currentIndexChanged.connect(self.change_flag_kind)
        legend = QtWidgets.QLabel("<b>Flagged colonies</b> &nbsp; <span style='color:#2e7d32'>green: included</span>, "
                                  "<span style='color:#c62828'>red: excluded</span>")
        legend.setWordWrap(True)
        side.addWidget(legend)
        side.addWidget(self.flag_kind)
        self.lists, self.list_labels, self._ids = {}, {}, {"todo": [], "done": []}
        for key in ("todo", "done"):
            self.list_labels[key] = QtWidgets.QLabel()
            side.addWidget(self.list_labels[key])
            lw = QtWidgets.QListWidget()
            lw.setMinimumWidth(200)
            lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            lw.itemSelectionChanged.connect(lambda *_, key=key: self.list_selection_changed(key))
            self.lists[key] = lw
            side.addWidget(lw, 3 if key == "todo" else 2)
            row = QtWidgets.QHBoxLayout()
            # *_ swallows the "checked" argument of clicked(), which would otherwise land in key
            buttons = (("Include", lambda *_, key=key: self.decide_selected(key, VALID),
                        "Include the colonies selected in this list"),
                       ("Exclude", lambda *_, key=key: self.decide_selected(key, EXCLUDED),
                        "Exclude the colonies selected in this list"))
            buttons += ((("Exclude listed", self.exclude_all_to_review, "Exclude every colony this list shows (asks first)"),)
                        if key == "todo" else
                        (("Unreview", self.unreview_selected, "Move the selected colonies back to To review"),))
            for text, slot, tip in buttons:
                b = QtWidgets.QPushButton(text)
                b.setObjectName(f"{key} {text.lower()}")  # e.g. "todo include", "done unreview"
                b.setToolTip(tip)
                b.clicked.connect(slot)
                row.addWidget(b)
            side.addLayout(row)
        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(QtCore.Qt.RichText)
        self.info_label.setMinimumHeight(120)
        self.info_label.setAlignment(QtCore.Qt.AlignTop)
        side.addWidget(self.info_label)
        side_panel.setMinimumWidth(270)
        splitter.addWidget(side_panel)
        self.splitter = splitter
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        root.addWidget(splitter, 1)

        slider_row = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        # Continuous: every move asks for a frame, a zero-delay timer renders only the latest one,
        # and numbers are left out while dragging (they are the slow part on dense plates).
        self.slider.valueChanged.connect(self.request_frame)
        self.slider.sliderReleased.connect(lambda: self.request_frame(self.slider.value()))
        self._frame_timer = QtCore.QTimer(self, singleShot=True, interval=0)
        self._frame_timer.timeout.connect(lambda: self.set_frame(self._pending_frame))
        self.time_label = QtWidgets.QLabel()
        self.time_label.setMinimumWidth(190)
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.time_label)
        root.addLayout(slider_row)

        view_row = QtWidgets.QHBoxLayout()
        view_row.addWidget(QtWidgets.QLabel("Image:"))
        self.layer_box = QtWidgets.QComboBox()
        self.layer_box.addItems(["Raw", "Difference (frame - background)"])
        self.layer_box.currentIndexChanged.connect(self.change_layer)
        view_row.addWidget(self.layer_box)
        view_row.addSpacing(12)
        self.level_sliders = {}
        for name in ("Black", "White"):
            view_row.addWidget(QtWidgets.QLabel(name))
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.setRange(0, 255)
            sl.setMinimumWidth(60)
            sl.setMaximumWidth(180)
            sl.valueChanged.connect(self.change_levels)
            self.level_sliders[name] = sl
            view_row.addWidget(sl, 1)
        self.level_label = QtWidgets.QLabel()
        self.level_label.setMinimumWidth(60)
        view_row.addWidget(self.level_label)
        auto = QtWidgets.QPushButton("Auto contrast")
        auto.clicked.connect(lambda: self.auto_levels(redraw=True))
        view_row.addWidget(auto)
        view_row.addStretch(1)
        root.addLayout(view_row)
        self.status = self.statusBar()

        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas_graph.mpl_connect("pick_event", self.on_pick)
        for keys, slot in (("Ctrl+E", lambda: self.set_selected(EXCLUDED)), ("Ctrl+I", lambda: self.set_selected(VALID)),
                           ("Ctrl+U", self.unreview_selected), ("Esc", self.full_view),
                           ("Left", lambda: self.slider.setValue(self.k - 1)), ("Right", lambda: self.slider.setValue(self.k + 1)),
                           ("Home", lambda: self.slider.setValue(0)), ("End", lambda: self.slider.setValue(self.slider.maximum())),
                           ("PgUp", lambda: self.step_plate(-1)), ("PgDown", lambda: self.step_plate(1)),
                           ("F", lambda: self.step_flag(1)), ("Shift+F", lambda: self.step_flag(-1))):
            QtWidgets.QShortcut(QtGui.QKeySequence(keys), self, activated=slot)

    # ---------- plate / frame ----------
    def step_plate(self, d: int):
        self.plate_box.setCurrentIndex(int(np.clip(self.plate_box.currentIndex() + d, 0, len(self.plate_dirs) - 1)))

    def load_plate(self, index: int):
        self.plate = PlateData(self.plate_dirs[index])
        self.plate.review["visited"] = True
        self.plate.save()
        self.selected = None
        self.ax_img.clear()
        self.toolbar.update()  # forget the previous plate's zoom history
        self.img_artist, self.texts, self.ring = None, [], []
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.plate.n_frames - 1)
        self.slider.setValue(self.plate.n_frames - 1)
        self.slider.blockSignals(False)
        self.k = self.plate.n_frames - 1
        self.levels = {}
        self.auto_levels()
        self.fill_flag_kinds()
        self.fill_flag_list()
        self.redraw_all()

    def request_frame(self, k: int):
        self._pending_frame = int(k)
        self.show_time(int(k))
        self._frame_timer.start()

    def set_frame(self, k: int):
        self.k = int(np.clip(k, 0, self.plate.n_frames - 1))
        self.draw_plate(fast=self.slider.isSliderDown())
        self.canvas.draw_idle()
        self.draw_time_line()
        self.plate.prefetch(self.k)

    # ---------- image layer and contrast ----------
    @property
    def layer(self) -> str:
        return "diff" if self.layer_box.currentIndex() == 1 else "raw"

    def layer_image(self, k: int) -> np.ndarray:
        """The frame as stored (raw RGB) or gray(frame - background), before contrast."""
        p = self.plate
        if self.layer == "diff":
            return subtract_background(p.image(k), p.background).astype(np.float32)
        return p.image(k).astype(np.float32)

    def auto_levels(self, redraw: bool = False):
        """Black/white at the 0.5 / 99.8 % (raw) or 1 / 99.9 % (diff) points inside the working area."""
        values = self.layer_image(self.k)[self.plate.mask]
        lo, hi = (0.5, 99.8) if self.layer == "raw" else (1, 99.9)
        lo, hi = np.percentile(values, [lo, hi])
        self.levels[self.layer] = (int(lo), int(max(hi, lo + 2)))
        self.sync_level_sliders()
        if redraw:
            self.set_frame(self.k)

    def sync_level_sliders(self):
        lo, hi = self.levels[self.layer]
        for name, v in (("Black", lo), ("White", hi)):
            sl = self.level_sliders[name]
            sl.blockSignals(True)
            sl.setValue(v)
            sl.blockSignals(False)
        self.level_label.setText(f"{lo} - {hi}")

    def change_levels(self, _=None):
        lo, hi = self.level_sliders["Black"].value(), self.level_sliders["White"].value()
        if hi <= lo:
            hi = lo + 1
        self.levels[self.layer] = (lo, hi)
        self.level_label.setText(f"{lo} - {hi}")
        self.request_frame(self.k)

    def change_layer(self, _=None):
        if self.layer not in self.levels:
            self.auto_levels()
        self.sync_level_sliders()
        self.set_frame(self.k)

    def display_rgb(self, k: int) -> np.ndarray:
        img = self.layer_image(k)
        lo, hi = self.levels[self.layer]
        out = (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        return np.repeat(out[..., None], 3, axis=2) if out.ndim == 2 else out

    def show_time(self, k: int):
        minutes = self.plate.times[k]
        self.time_label.setText(f"frame {k + 1}/{self.plate.n_frames}   {minutes:.0f} min ({minutes / 60:.1f} h)")

    def redraw_all(self, *_):
        self.draw_plate()
        self.draw_graph()
        self.show_info()
        self.canvas.draw_idle()
        self.canvas_graph.draw_idle()

    # ---------- plate image ----------
    def colony_status_rgb(self, cid: int) -> np.ndarray:
        s = self.plate.status(cid)
        return self.plate.colors[cid] if s == VALID else np.array(STATUS_RGB[s], np.uint8)

    def draw_plate(self, fast: bool = False):
        """``fast`` (while dragging the slider) skips the colony numbers."""
        p, k = self.plate, self.k
        labels = np.asarray(p.labels[k])
        present = np.nonzero(p.area[k] > 0)[0]
        present = present[present > 0]
        if p.own is None:
            lut = np.tile(np.array(UNASSIGNED_RGB, np.uint8), (int(labels.max()) + 1, 1))
            for c in present:
                b = p.blob[k, c]
                lut[b] = MERGED_RGB if p.merged[k, c] else self.colony_status_rgb(c)
        else:  # colony IDs, and n + 1 for detected objects that are no colony
            own = np.asarray(p.own[k]).astype(np.int64)
            labels = np.where(own > 0, own, np.where(labels > 0, p.n + 1, 0))
            lut = np.tile(np.array(UNASSIGNED_RGB, np.uint8), (p.n + 2, 1))
            for c in present:
                lut[c] = self.colony_status_rgb(c)
        if self.checks["Labels only"].isChecked():
            rgb = lut[labels]
            rgb[labels == 0] = 0
            if p.own is not None:  # the split lines between merged colonies
                merged = np.zeros(p.n + 2, bool)
                merged[: p.n + 1] = p.merged[k]
                split = find_boundaries(labels, mode="inner") & ~find_boundaries(labels > 0, mode="inner")
                split &= merged[labels]
                rgb[split] = MERGED_RGB
        else:
            rgb = self.display_rgb(k)
            if self.checks["Outlines"].isChecked():
                edge = find_boundaries(labels, mode="inner")
                rgb[edge] = lut[labels[edge]]
        rgb[~p.mask] = (rgb[~p.mask] * 0.45).astype(np.uint8)
        if self.img_artist is None:
            self.img_artist = self.ax_img.imshow(rgb, interpolation="nearest")
            self.ax_img.set_axis_off()
        else:
            self.img_artist.set_data(rgb)

        for t in self.texts:
            t.remove()
        self.texts = []
        n_included = int(sum(p.status(c) == VALID for c in present))
        if self.checks["Numbers"].isChecked() and not fast:
            for c in present:
                s = p.status(c)
                flagged = p.to_review(c)  # reviewed colonies lose their "!"
                # PlateAnalyzer order: status colour (excluded yellow, near border magenta) wins over
                # "new in this frame" red; flagged included colonies are orange.
                if s != VALID:
                    color = STATUS_TEXT[s]
                elif flagged:
                    color = FLAG_COLOR
                elif k > 0 and p.area[k - 1, c] == 0:
                    color = "red"
                else:
                    color = STATUS_TEXT[VALID]
                label = f"{c}!" if flagged else str(c)
                t = self.ax_img.text(p.cx[k, c], p.cy[k, c], label, fontsize=7, ha="center", va="center",
                                     color=color, fontweight="bold" if flagged or s == EXCLUDED else "normal",
                                     clip_on=True)  # no numbers in the margins when zoomed in
                t.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground="black")])
                self.texts.append(t)
        self.draw_ring()
        minutes = p.times[k]
        self.ax_img.set_title(f"{p.title}, {minutes:.0f} min ({minutes / 60:.1f} h), colonies: {n_included}", fontsize=10)
        self.show_time(k)

    def draw_ring(self):
        """A wide yellow ring (with a dark edge, visible on any background) around the selected colony,
        leaving the colony's own outline visible, so the eye finds it on a crowded plate."""
        for ring in self.ring:
            ring.remove()
        self.ring = []
        c = self.selected
        if c is None:
            return
        present = np.nonzero(self.plate.area[:, c])[0]
        if not len(present):
            return
        k = self.k if self.plate.area[self.k, c] > 0 else int(present[-1])
        r = max(np.sqrt(self.plate.area[k, c] / np.pi) + 6, 10)
        xy = (self.plate.cx[k, c], self.plate.cy[k, c])
        ring = matplotlib.patches.Circle(xy, 2.2 * r + 25, fill=False, ec="yellow", lw=2.5)
        ring.set_path_effects([patheffects.withStroke(linewidth=4.5, foreground="black")])
        self.ring = [ring]
        for ring in self.ring:
            self.ax_img.add_patch(ring)

    # ---------- graph ----------
    def draw_graph(self):
        p = self.plate
        ax = self.ax_graph
        xlim = ax.get_xlim() if self.lines else None
        ax.clear()
        self.lines = {}
        show_excl = self.checks["Show excluded"].isChecked()
        flagged_only = self.checks["Flagged only"].isChecked()
        listed = set(self._flag_ids)
        hours = p.times / 60
        for c in range(1, p.n + 1):
            s = p.status(c)
            if (s != VALID and not show_excl) or (flagged_only and c not in listed):
                continue
            a = np.where(p.area[:, c] > 0, p.area[:, c], np.nan)
            color = p.colors[c] / 255 if s == VALID else np.array(STATUS_RGB[s]) / 255
            (line,) = ax.plot(hours, a, color=color, lw=0.9, picker=4, ls="-" if s == VALID else ":")
            line.colony_id = c
            self.lines[c] = line
        ax.set_yscale("log" if self.checks["Log scale"].isChecked() else "linear")
        if not self.lines:  # nothing to show (e.g. all excluded): keep the full time range
            ax.set_xlim(hours[0], hours[-1])
        ax.set_xlabel("time (h)")
        ax.set_ylabel("area (pixels)")
        ax.set_title("Area of each colony over time" + (" (flagged only)" if flagged_only else ""), fontsize=10)
        ax.grid(alpha=0.3, which="both", lw=0.4)
        self.vline = ax.axvline(hours[self.k], color="k", lw=0.8, ls="--", animated=True)
        self.highlight_line()

    def on_graph_draw(self, _event=None):
        """After a full graph draw: keep the curves as a bitmap and draw the (animated) time marker."""
        self._graph_bg = self.canvas_graph.copy_from_bbox(self.fig_graph.bbox)
        self.ax_graph.draw_artist(self.vline)

    def draw_time_line(self):
        self.vline.set_xdata([self.plate.times[self.k] / 60] * 2)
        if self._graph_bg is None:
            self.canvas_graph.draw_idle()
            return
        self.canvas_graph.restore_region(self._graph_bg)
        self.ax_graph.draw_artist(self.vline)
        self.canvas_graph.blit(self.fig_graph.bbox)

    def highlight_line(self):
        for c, line in self.lines.items():
            sel = c == self.selected
            line.set_linewidth(3 if sel else 0.9)
            line.set_zorder(10 if sel else 2)
            line.set_path_effects([patheffects.withStroke(linewidth=5, foreground="black")] if sel else [])

    # ---------- selection ----------
    def select(self, cid: int | None, focus: bool = False):
        """Select a colony; ``focus`` also centres the plate view on it (from the lists and the graph)."""
        self.selected = cid
        if focus and cid is not None:
            self.focus_on(cid)
        self.draw_ring()
        self.highlight_line()
        self.show_info()
        self.canvas.draw_idle()
        self.canvas_graph.draw_idle()

    FOCUS_MIN = 110  # px: half width of the focused view at least ...
    FOCUS_RADII = 7  # ... or this many colony radii

    def focus_on(self, cid: int):
        """Centre the plate image on the colony, zoomed so that it and its neighbours are visible
        (half width max(FOCUS_MIN px, FOCUS_RADII x its radius in the last frame it was seen))."""
        p = self.plate
        present = np.nonzero(p.area[:, cid])[0]
        if not len(present):
            return
        k = self.k if p.area[self.k, cid] > 0 else int(present[-1])
        r = np.sqrt(p.area[int(present[-1]), cid] / np.pi)
        half = max(self.FOCUS_MIN, self.FOCUS_RADII * r)
        size = p.mask.shape[0]
        half = min(half, size / 2)
        cx = float(np.clip(p.cx[k, cid], half - 0.5, size - half - 0.5))
        cy = float(np.clip(p.cy[k, cid], half - 0.5, size - half - 0.5))
        if not self.toolbar._nav_stack():  # so the toolbar's Home button returns to the full plate
            self.toolbar.push_current()
        self.ax_img.set_xlim(cx - half, cx + half)
        self.ax_img.set_ylim(cy + half, cy - half)  # image rows grow downwards

    def full_view(self):
        size = self.plate.mask.shape
        self.ax_img.set_xlim(-0.5, size[1] - 0.5)
        self.ax_img.set_ylim(size[0] - 0.5, -0.5)
        self.canvas.draw_idle()

    def on_click(self, event):
        if self.toolbar.mode or event.inaxes is not self.ax_img or event.xdata is None:
            return
        self.select(self.plate.colony_at(self.k, event.xdata, event.ydata))

    def on_pick(self, event):
        cid = getattr(event.artist, "colony_id", None)
        if cid is not None and event.mouseevent.button == 1:
            self.select(cid, focus=True)

    def find_colony(self):
        try:
            cid = int(self.find_edit.text())
        except ValueError:
            return
        if 1 <= cid <= self.plate.n:
            self.select(cid, focus=True)
        else:
            self.status.showMessage(f"no colony {cid} on this plate", 4000)

    def show_info(self):
        c, p = self.selected, self.plate
        if c is None:
            self.info_label.setText("Click a colony on the plate or its line in the graph.")
            return
        s = p.status(c)
        a = p.area[:, c]
        first = np.nonzero(a)[0]
        appear = f"{p.times[first[0]] / 60:.1f} h" if len(first) else "never"
        merged = np.nonzero(p.merged[:, c])[0]
        merged_txt = f"merged from {p.times[merged[0]] / 60:.1f} h" if len(merged) else "never merged"
        flags = p.flags(c)
        if not flags:
            flags_html = "<br>No flags."
        elif p.reviewed(c):
            flags_html = f"<br><b>Flags (reviewed):</b> {flags}"
        else:
            flags_html = f"<br><span style='color:{FLAG_COLOR}'><b>Flags:</b> {flags}</span>"
        self.info_label.setText(
            f"<b>Colony {c}</b> &mdash; {ColonyStatus(s).name.replace('_', ' ').lower()}"
            f" (detected as {ColonyStatus(int(p.colonies.auto_status.iat[c - 1])).name.replace('_', ' ').lower()})<br>"
            f"Appears at {appear}, final area {int(a[-1])} px, {merged_txt}.{flags_html}<br><br>"
            "<i>Ctrl+E exclude, Ctrl+I include, Ctrl+U unreview, Esc full plate</i>")

    def set_selected(self, status: int):
        c = self.selected
        if c is None:
            self.status.showMessage("select a colony first", 3000)
            return
        if self.plate.decide([c], status, "include" if status == VALID else "exclude"):
            self.status.showMessage(f"colony {c}: {ColonyStatus(status).name.lower()} (saved)", 4000)
            self._after_decision()

    def _confirm(self, title: str, text: str) -> bool:
        ans = QtWidgets.QMessageBox.question(self, title, text, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                             QtWidgets.QMessageBox.No)
        return ans == QtWidgets.QMessageBox.Yes

    def _after_bulk(self, message: str):
        self.status.showMessage(message, 5000)
        self.fill_flag_list()
        self.show_info()
        self.redraw_all()

    def reset_plate(self):
        n = self.plate.n_changed()
        if not n:
            self.status.showMessage("no changes to undo on this plate", 3000)
            return
        if self._confirm("Reset plate", f"Undo your changes to {n} colonies on {self.plate.title} "
                                        "and return to the statuses from detection?"):
            self._after_bulk(f"{self.plate.reset()} colonies reset to their detected status (saved)")

    # ---------- flagged lists ----------
    ALL_FLAGS = "All flags"
    COLORS = {VALID: "#2e7d32", EXCLUDED: "#c62828"}

    @staticmethod
    def flag_kinds(flags: str) -> list[str]:
        """'gap (3 frames); jump x2.1' -> ['gap', 'jump']: the flag names without their numbers."""
        return kinds_of(flags)

    def flagged_ids(self, reviewed: bool) -> list[int]:
        """Flagged colonies (included or excluded by the reviewer; not the automatic border / zero-area
        statuses) still to review, or reviewed; filtered by the flag-kind box."""
        col = self.plate.colonies
        f = col[(col.n_flags > 0) & ((col.status == VALID) | (col.status == EXCLUDED))]
        kind = self.flag_kind.currentText()
        if kind and kind != self.ALL_FLAGS:
            f = f[np.array([kind in self.flag_kinds(x) for x in f["flags"]], bool)]
        # a numpy mask, not a list: an empty list would select no *columns* (KeyError: 'status' below)
        f = f[np.array([self.plate.reviewed(int(c)) == reviewed for c in f.colony_id], bool)]
        return f.sort_values(["status", "n_flags", "colony_id"], ascending=[True, False, True]).colony_id.tolist()

    def fill_flag_kinds(self):
        """The flag kinds on this plate, with their counts; keeps the chosen kind if it is still there."""
        col = self.plate.colonies
        col = col[(col.n_flags > 0) & ((col.status == VALID) | (col.status == EXCLUDED))]
        counts = pd.Series([k for x in col["flags"] for k in set(self.flag_kinds(x))], dtype=str).value_counts()
        current = self.flag_kind.currentText()
        self.flag_kind.blockSignals(True)
        self.flag_kind.clear()
        self.flag_kind.addItem(self.ALL_FLAGS)
        for kind, n in counts.items():
            self.flag_kind.addItem(kind)
            self.flag_kind.setItemData(self.flag_kind.count() - 1, f"{n} colonies", QtCore.Qt.ToolTipRole)
        self.flag_kind.setCurrentIndex(max(self.flag_kind.findText(current), 0))
        self.flag_kind.blockSignals(False)

    def fill_flag_list(self, *_):
        """Fill both lists, keeping the selected colony selected in the list it is now in."""
        for key, reviewed in (("todo", False), ("done", True)):
            lw = self.lists[key]
            lw.blockSignals(True)
            lw.clear()
            self._ids[key] = self.flagged_ids(reviewed)
            for c in self._ids[key]:
                s = self.plate.status(c)
                item = QtWidgets.QListWidgetItem(f"#{c}  {'included' if s == VALID else 'excluded'}  {self.plate.flags(c)}")
                item.setForeground(QtGui.QColor(self.COLORS.get(s, "#888888")))
                item.setToolTip(item.text())  # long flag lists don't fit the panel
                lw.addItem(item)
            if self.selected in self._ids[key]:
                lw.setCurrentRow(self._ids[key].index(self.selected))
            lw.blockSignals(False)
        self.list_labels["todo"].setText(f"<b>To review</b> ({len(self._ids['todo'])})")
        self.list_labels["done"].setText(f"<b>Reviewed</b> ({len(self._ids['done'])})")

    @property
    def _flag_ids(self) -> list[int]:
        """Both lists' colonies (the graph's "Flagged only")."""
        return self._ids["todo"] + self._ids["done"]

    def selected_in(self, key: str) -> list[int]:
        return [self._ids[key][i.row()] for i in self.lists[key].selectedIndexes() if i.row() < len(self._ids[key])]

    def list_selection_changed(self, key: str):
        other = self.lists["done" if key == "todo" else "todo"]
        if self.lists[key].selectedItems() and other.selectedItems():
            other.blockSignals(True)
            other.clearSelection()
            other.blockSignals(False)
        row = self.lists[key].currentRow()
        if self.lists[key].selectedItems() and 0 <= row < len(self._ids[key]):
            self.select(self._ids[key][row], focus=True)

    def _after_decision(self):
        """Lists, plate and graph again after a decision; done on the next event-loop turn, because it
        rebuilds the list whose signal may have started it."""
        QtCore.QTimer.singleShot(0, lambda: (self.fill_flag_list(), self.redraw_all()))

    def decide_selected(self, key: str, status: int):
        ids = self.selected_in(key)
        if not ids:
            self.status.showMessage("select colonies in the list first", 3000)
            return
        word = "include" if status == VALID else "exclude"
        n = self.plate.decide(ids, status, f"{word} selected")
        self.status.showMessage(f"{n} colonies {word}d and reviewed (saved)", 4000)
        self._after_decision()

    def exclude_all_to_review(self):
        ids = list(self._ids["todo"])
        if not ids:
            self.status.showMessage("nothing left to review in this list", 3000)
            return
        kind = self.flag_kind.currentText()
        if self._confirm("Exclude flagged", f"Exclude all {len(ids)} colonies still to review ({kind.lower()})?"):
            n = self.plate.decide(ids, EXCLUDED, f"exclude flagged: {kind}")
            self._after_bulk(f"{n} flagged colonies excluded and reviewed (saved)")

    def unreview_selected(self):
        """Ctrl+U: the colonies selected in "Reviewed" (else the selected colony) back to "To review"."""
        ids = self.selected_in("done") or ([self.selected] if self.selected is not None else [])
        ids = [c for c in ids if self.plate.reviewed(c)]
        if not ids:
            self.status.showMessage("select a reviewed colony first", 3000)
            return
        n = self.plate.unreview(ids)
        self.status.showMessage(f"{n} colonies back to review (saved)", 4000)
        self._after_decision()

    def change_flag_kind(self, _=None):
        self.fill_flag_list()
        if self.checks["Flagged only"].isChecked():
            self.draw_graph()

    def step_flag(self, d: int):
        """F / Shift+F: the next / previous colony in "To review"."""
        ids = self._ids["todo"]
        if not ids:
            return
        if self.selected in ids:
            i = (ids.index(self.selected) + d) % len(ids)
        else:
            i = 0 if d > 0 else len(ids) - 1
        lw = self.lists["todo"]
        lw.clearSelection()
        lw.setCurrentRow(i)

    # ---------- finish ----------
    def finish(self):
        unvisited = [d for d in self.plate_dirs if not _visited(d)]
        if unvisited:
            names = ", ".join(f"{d.parent.name}/{d.name}" for d in unvisited[:6])
            ans = QtWidgets.QMessageBox.question(self, "Export", f"{len(unvisited)} plates not opened yet ({names}...). Export anyway?")
            if ans != QtWidgets.QMessageBox.Yes:
                return
        if self.on_finish:
            self.on_finish()
        QtWidgets.QMessageBox.information(self, "Export", "Results exported.")
        self.close()


def _visited(plate_dir: Path) -> bool:
    r = Path(plate_dir) / "review.json"
    return r.exists() and json.loads(r.read_text()).get("visited", False)


def run_review(plate_dirs: list[Path], on_finish=None) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = ReviewWindow(plate_dirs, on_finish)

    def report(kind, value, tb):
        """An error in a button or key handler: show it and keep the window open (PyQt5 would abort the
        whole program). Decisions are saved as they are made, so nothing is lost either way."""
        import traceback

        text = "".join(traceback.format_exception(kind, value, tb))
        print(text, file=sys.stderr, flush=True)
        QtWidgets.QMessageBox.warning(win, "Error", f"{kind.__name__}: {value}\n\nThe window stays open; the full "
                                      "error is in the terminal. Please report it.")

    sys.excepthook = report
    win.show()
    try:
        app.exec_()
    finally:
        sys.excepthook = sys.__excepthook__
