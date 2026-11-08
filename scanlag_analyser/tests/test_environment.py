"""The conda environment files agree with each other and with pyproject.toml."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# conda-forge name -> PyPI name, where they differ
PYPI_NAME = {"pytorch": "torch", "matplotlib-base": "matplotlib"}


def conda_packages(name: str) -> dict[str, str]:
    """{package: spec} of an environment file's conda dependencies (the pip section is skipped)."""
    out = {}
    for line in (ROOT / name).read_text().splitlines():
        if m := re.fullmatch(r"  - ([A-Za-z0-9_.-]+)(.*)", line):
            out[m[1]] = m[2]
    return out


def pip_packages(name: str) -> set[str]:
    """Package names in an environment file's pip section (the editable project install skipped)."""
    lines = (ROOT / name).read_text().splitlines()
    return {m[1] for line in lines if (m := re.fullmatch(r"      - ([A-Za-z0-9_.-]+).*", line))}


def pyproject_names(*extras: str) -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    specs = project["dependencies"] + [d for e in extras for d in project["optional-dependencies"][e]]
    return {re.match(r"[A-Za-z0-9_.-]+", s)[0] for s in specs}


def test_cpu_file_differs_only_in_the_torch_build():
    gpu, cpu = conda_packages("environment.yml"), conda_packages("environment-cpu.yml")
    assert cpu.pop("pytorch-cpu") == gpu.pop("pytorch-gpu") == "" and gpu == cpu


def test_review_window_uses_pyqt5():
    """matplotlib's Qt backend must use the review window's binding, also when PySide6 is installed
    (e.g. in an environment made from an older version of the files)."""
    import scanlag.gui.review  # noqa: F401
    from matplotlib.backends import qt_compat

    assert qt_compat.QT_API == "PyQt5"


def test_user_environment_covers_pyproject():
    # opencv-python-headless comes from PyPI with the package; napari[pyqt5] is napari + pyqt here
    conda = {PYPI_NAME.get(p, p) for p in conda_packages("environment.yml")} | {"opencv-python-headless", "napari[pyqt5]"}
    assert pyproject_names("gui", "dl") <= conda
    training = set(conda_packages("environment-training.yml")) | pip_packages("environment-training.yml")
    assert pyproject_names("cellpose", "dev") <= conda | training


def test_no_conda_package_that_brings_a_second_qt_or_opencv():
    """conda-forge's matplotlib installs PySide6, its cellpose and segment-anything also a Qt 6 OpenCV
    next to opencv-python-headless: use matplotlib-base, and cellpose from pip."""
    for name in ("environment.yml", "environment-cpu.yml", "environment-training.yml"):
        assert not {"matplotlib", "cellpose", "segment-anything", "opencv", "py-opencv", "pyside6"} & set(
            conda_packages(name)), name
