"""The command-line scripts' prompts and checks, with the terminal and the heavy steps replaced."""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from scanlag.pipeline.scans import Scanner
from scanlag.store import ResultsStore

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    return load_script("scanlag_pipeline")


@pytest.fixture(scope="module")
def analysis():
    return load_script("scanlag_analysis")


def answers(monkeypatch, *replies):
    """Make input() return ``replies`` in turn (EOFError after the last), and record the prompts."""
    prompts, replies = [], list(replies)

    def fake_input(prompt=""):
        prompts.append(prompt)
        if not replies:
            raise EOFError
        return replies.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def scanner(name: str) -> Scanner:
    t = datetime(2026, 10, 26, 12, 0)
    return Scanner(name, name, Path("."), [Path(f"x_{name}_20261026_1200.tif")], [t], t)


def test_ask_repeats_until_yes_or_no(pipeline, monkeypatch, capsys):
    prompts = answers(monkeypatch, "maybe", "Y")
    assert pipeline.ask("proceed?") is True
    assert prompts == ["proceed? y/n "] * 2
    answers(monkeypatch, "no")
    assert pipeline.ask("proceed?") is False


def test_ask_without_terminal_answers_no(pipeline, monkeypatch, capsys):
    answers(monkeypatch)
    assert pipeline.ask("proceed?") is False
    assert "no terminal: answering n" in capsys.readouterr().out


def test_conflicting_flags_are_refused(pipeline, capsys):
    for argv in (["raw", "--review-only", "--no-review"], ["raw", "--redo", "--review-only"],
                 ["raw", "--reflag", "--retrack"], ["raw", "--export-only", "--redo"]):
        with pytest.raises(SystemExit):
            pipeline.parse_args(argv)
    assert "--export-only" in capsys.readouterr().err
    assert pipeline.parse_args(["raw", "--export-only"]).export_only


def test_to_segment_and_redo_question(pipeline, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    store = ResultsStore(tmp_path / "res", [raw])
    store.write_json({"detector": "unet"}, "A", "scanner.json")  # complete, with review decisions on one plate
    store.write_json({"visited": True}, "A", "plate_1", "review.json")
    store.write_json({}, "B", "plate_1", "plate.json")  # interrupted run: no scanner.json
    a, b, c = scanner("A"), scanner("B"), scanner("C")

    todo, replace, incomplete = pipeline.to_segment([a, b, c], store, redo=False)
    assert [s.name for s in todo] == ["B", "C"] and replace == [] and [s.name for s in incomplete] == ["B"]
    assert pipeline.confirm_redo(replace, store)  # nothing complete is deleted: no question

    todo, replace, _ = pipeline.to_segment([a, b, c], store, redo=True)
    assert [s.name for s in todo] == ["A", "B", "C"] and [s.name for s in replace] == ["A"]
    prompts = answers(monkeypatch, "n")
    assert not pipeline.confirm_redo(replace, store)
    assert prompts == ["warning: --redo will delete the results of A, including review decisions on 1 plate(s). "
                       "proceed? y/n "]


def fake_steps(pipeline, monkeypatch, counts):
    monkeypatch.setattr(pipeline, "make_segmenter", lambda cfg, log: "segmenter")
    monkeypatch.setattr(pipeline, "last_frame_counts", lambda s, cfg, seg, log: counts[s.name])
    processed = {}

    def fake_process(s, store, cfg, log, seg, exclude):
        processed[s.name] = sorted(exclude)
        return [store.path(s.name, "plate_1", "colonies.parquet").parent]

    monkeypatch.setattr(pipeline, "process_scanner", fake_process)
    return processed


@pytest.mark.parametrize("reply, excluded", [("y", {"A": [2, 3], "B": [1]}), ("n", {"A": [], "B": []})])
def test_dense_plates_question(pipeline, tmp_path, monkeypatch, capsys, reply, excluded):
    raw = tmp_path / "raw"
    raw.mkdir()
    store = ResultsStore(tmp_path / "res", [raw])
    processed = fake_steps(pipeline, monkeypatch, {"A": {1: 40, 2: 900, 3: 501}, "B": {1: 700}})
    prompts = answers(monkeypatch, reply)
    args = pipeline.parse_args(["raw", "--detector", "classical"])
    cfg = pipeline.RunConfig(detector="classical")
    todo = [scanner("A"), scanner("B")]

    pipeline.segment(todo, [], [], store, cfg, args)
    assert prompts == ["warning: A, plates 2, 3 and B, plates 1 are too dense (n>500). expect long runtimes and "
                       "low quality tracks. exclude these plates? y/n "]
    assert processed == excluded
    saved = json.loads(store.path("A", "scanner.json").read_text())
    assert saved["excluded_plates"] == excluded["A"] and saved["model"] is None and "workers" not in saved["config"]


def test_dense_plates_question_is_skipped_below_the_limit(pipeline, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    store = ResultsStore(tmp_path / "res", [raw])
    fake_steps(pipeline, monkeypatch, {"A": {1: 40}})
    prompts = answers(monkeypatch)
    pipeline.segment([scanner("A")], [], [], store, pipeline.RunConfig(detector="classical"),
                     pipeline.parse_args(["raw", "--detector", "classical"]))
    assert prompts == []


def test_unknown_detector(tmp_path):
    from scanlag.pipeline.run import model_dir

    assert model_dir("classical") is None
    assert (model_dir("unet") / "config.json").is_file()
    with pytest.raises(ValueError, match="unknown detector"):
        model_dir(str(tmp_path / "no_model"))


def test_condition_folder_clash(analysis):
    analysis.check_folder_names(["CTRL", "AMP 5", "CTRL"])
    with pytest.raises(ValueError, match="'A B' / 'A/B'"):
        analysis.check_folder_names(["A B", "A/B", "C"])


def test_condition_options_are_checked_for_duplicates(analysis):
    from scanlag.analysis.conditions import expand_layout

    layout = expand_layout(analysis.layout_from_args(["CTRL=1-4.1u:1-3", "AMP=1-4.1u:4-6"]))
    assert len(layout) == 6
    with pytest.raises(ValueError, match="1-4.1u plate 3"):
        expand_layout(analysis.layout_from_args(["CTRL=1-4.1u:1-3", "AMP=1-4.1u:3-6"]))


def test_results_found_by_experiment_name(analysis, tmp_path, monkeypatch):
    (tmp_path / "EPEC260224").mkdir()
    monkeypatch.setattr(analysis, "RESULTS_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    assert analysis.find_results(Path("EPEC260224")) == tmp_path / "EPEC260224"
    assert analysis.find_results(Path("nothing")) == Path("nothing")  # left for the "not a folder" message


def test_export_reports_unreviewed_plates(tmp_path):
    from scanlag.pipeline.export import export

    raw = tmp_path / "raw"
    raw.mkdir()
    store = ResultsStore(tmp_path / "res", [raw])
    for k in (1, 2):
        store.write_json({"scanner": "A", "plate": k, "times_min": [0, 30], "tracker": "footprint"},
                         "A", f"plate_{k}", "plate.json")
        pd.DataFrame({"colony_id": [1], "status": [0], "auto_status": [0], "appearance_frame": [0],
                      "appearance_min": [0.0], "final_area_px": [30], "seed_x": [5], "seed_y": [5], "n_flags": [0],
                      "flags": [""]}).to_parquet(store.path("A", f"plate_{k}", "colonies.parquet"))
        pd.DataFrame({"frame": [0, 1], "time_min": [0.0, 30.0], "colony_id": [1, 1], "area_px": [20, 30],
                      "x": [5.0, 5.0], "y": [5.0, 5.0], "blob": [1, 1], "merged": [False, False]}
                     ).to_parquet(store.path("A", f"plate_{k}", "tracks.parquet"))
    store.write_json({"visited": True}, "A", "plate_1", "review.json")
    lines = []
    export(store.root, store, log=lines.append)
    assert lines[-1] == "1 of them were never opened in the review window (automatic statuses): A/plate_2"

    for k in (1, 2):
        store.path("A", f"plate_{k}", "colonies.csv").unlink()
    export(store.root, store, log=lines.append, plate_dirs=[store.root / "A" / "plate_1"])  # e.g. --scanner
    assert (store.root / "A" / "plate_1" / "colonies.csv").exists()
    assert not (store.root / "A" / "plate_2" / "colonies.csv").exists()
    assert pd.read_csv(store.root / "summary.csv").plate.tolist() == ["plate_1"]
    assert pd.read_csv(store.root / "all_colonies.csv").plate.unique().tolist() == [1]
