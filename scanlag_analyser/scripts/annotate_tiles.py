"""Correct the MATLAB segmentation of the selected tiles in napari.

    python scripts/annotate_tiles.py [--testset testset_matlab]

Layers (top to bottom): ``colonies`` (editable labels), ``image`` (tile frame),
``difference`` (image minus background, contrast-stretched), ``previous`` and ``background``
(hidden; toggle with the eye icon).

Goal: every colony is one label, and nothing else is labelled.
  * split merged colonies: pick a new label (press M for a fresh ID) and paint over one part
  * delete non-colonies (dust, writing, rim reflections): fill bucket with label 0
  * add missed colonies, including tiny ones that are visible in ``difference``
  * a colony cut by the tile edge still counts

Keys: S = save, N or PageDown = next tile, Shift+N or PageUp = previous tile.
(B is taken by napari's labels layer: it toggles "preserve labels".)
Saved as ``labels_corrected.png`` next to ``labels_init.png``; tiles already saved open with
the corrected version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import napari
import numpy as np
from PIL import Image

from scanlag.store import PROJECT_ROOT, REGISTRY_FILE, ResultsStore
from scanlag.testset.gold import difference_image


def load(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


class Annotator:
    def __init__(self, testset: Path):
        raw_dirs = json.loads((testset / REGISTRY_FILE).read_text())["raw_dirs"]
        self.store = ResultsStore(testset, raw_dirs)
        self.tiles = sorted((testset / "manual").glob("tile_*"))
        if not self.tiles:
            raise SystemExit("no tiles found; run scripts/select_tiles.py first")
        self.index = 0
        self.viewer = napari.Viewer(title="ScanLag tile correction")
        self.viewer.bind_key("s", lambda _: self.save(), overwrite=True)
        # layer keymaps shadow viewer ones, so avoid keys the labels layer uses (B, E, P, F, L, Z, M, X, 1-7)
        for key, step in [("n", 1), ("PageDown", 1), ("Shift-N", -1), ("PageUp", -1)]:
            self.viewer.bind_key(key, lambda _, step=step: self.show(self.index + step), overwrite=True)
        self.show(0)

    @property
    def tile(self) -> Path:
        return self.tiles[self.index]

    def show(self, index: int) -> None:
        self.index = int(np.clip(index, 0, len(self.tiles) - 1))
        tile = self.tile
        image, background = load(tile / "image.png"), load(tile / "background.png")
        corrected = tile / "labels_corrected.png"
        labels = load(corrected if corrected.exists() else tile / "labels_init.png").astype(np.uint16)

        v = self.viewer
        v.layers.clear()
        v.add_image(background, name="background", visible=False)
        v.add_image(load(tile / "previous.png"), name="previous", visible=False)
        v.add_image(difference_image(image, background), name="difference", colormap="magma", opacity=0.6)
        v.add_image(image, name="image", opacity=0.6)
        layer = v.add_labels(labels, name="colonies", opacity=0.45)
        layer.contour = 1
        v.layers.selection.active = layer
        meta = json.loads((tile / "tile.json").read_text())
        state = "corrected" if corrected.exists() else "not yet saved"
        v.title = f"[{self.index + 1}/{len(self.tiles)}] {tile.name} - {meta['plate_id']} t={meta['time_min']:.0f} min ({state})"
        v.reset_view()

    def save(self) -> None:
        labels = self.viewer.layers["colonies"].data.astype(np.uint16)
        self.store.write_image(labels, "manual", self.tile.name, "labels_corrected.png")
        self.viewer.status = f"saved {self.tile.name}"
        self.show(self.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, default=PROJECT_ROOT / "testset_matlab",
                        help="test set folder (default: testset_matlab in the project)")
    args = parser.parse_args()
    Annotator(args.testset)
    napari.run()


if __name__ == "__main__":
    main()
