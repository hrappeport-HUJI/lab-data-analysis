import pytest

from scanlag.store import RawDataProtectionError, ResultsStore, default_results_dir


def test_default_results_dir_named_after_data_folder(tmp_path):
    assert default_results_dir(tmp_path / "raw" / "EPEC260323", root=tmp_path) == tmp_path / "EPEC260323"


def test_results_inside_raw_rejected(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(RawDataProtectionError):
        ResultsStore(raw / "results", [raw])


def test_raw_inside_results_rejected(tmp_path):
    with pytest.raises(RawDataProtectionError):
        ResultsStore(tmp_path, [tmp_path / "raw"])


def test_paths_cannot_escape_results(tmp_path):
    raw = tmp_path / "raw"
    store = ResultsStore(tmp_path / "results", [raw])
    with pytest.raises(RawDataProtectionError):
        store.path("..", "raw", "scan.tiff")
    with pytest.raises(RawDataProtectionError):
        store.path("..", "elsewhere.txt")


def test_results_dir_bound_to_its_raw_dirs(tmp_path):
    ResultsStore(tmp_path / "results_AMP7", [tmp_path / "expA" / "AMP7"])
    with pytest.raises(FileExistsError):
        ResultsStore(tmp_path / "results_AMP7", [tmp_path / "expB" / "AMP7"])
    ResultsStore(tmp_path / "results_AMP7", [tmp_path / "expA" / "AMP7"])  # same owner: fine
