#!/usr/bin/env python3
"""Run only the medium-background step of an experiment, before any notebook exists.

Usage:
    python preview_background.py EXPERIMENT_FOLDER/experiment_config.json

Checks the config against its files, loads the plate with the library, and prints which background
each well gets and everything that needs the experimenter's decision. Saves background_table.csv and
figures/<channel>_background_qc.png in the experiment folder. Exits with 0 when nothing is open, 1 on
config problems, and 2 when something needs a decision.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import check_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    folder = args.config.resolve().parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    problems, facts = check_config(config, folder)
    if problems:
        print("Config problems:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    for warning in facts.get("warnings", []):
        print(f"Warning: {warning}")

    os.chdir(folder)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pandas as pd
    import reader_background as rb
    import reader_functions as rf

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 30)
    background = config.get("background", {})
    exp = rf.ReaderExperiment(config["reader_file"], config["plate_map"], config["measurements"],
                              reader=facts["info"]["parser"], sheet_name=config["sheet"])
    rb.trim_trailing_empty_reads(exp)
    loaded_check = rb.verify_loaded(exp).set_index("measurement")
    for channel, (reads, wells) in facts.get("expected", {}).items():
        found = (int(loaded_check.loc[channel, "reads"]), int(loaded_check.loc[channel, "wells_with_data"]))
        if found != (reads, wells):
            print(f"Problem: {channel}: the library loaded {found[0]} reads and {found[1]} wells with data, but the "
                  f"file holds {reads} reads and {wells} wells with data. Don't analyze this until it is explained.")
            sys.exit(1)
    if config.get("exclude_time_windows"):
        rb.blank_time_windows(exp, config["exclude_time_windows"])
    print(rb.verify_loaded(exp).to_string(index=False))
    print()

    result = rb.compute_background(
        exp, control_wells=background.get("control_wells", []), match_traits=background.get("match_traits", []),
        od_measurement=facts["od_channel"], measurements=list(config["measurements"]),
        control_decisions=background.get("control_decisions"), manual_background=background.get("manual"),
        exclude_wells=background.get("exclude_wells"), accept_blank_drift=background.get("accept_blank_drift"),
        accept_own_start=background.get("accept_own_start"),
        raise_if_unresolved=False,
    )
    if not result.unresolved:
        rb.apply_background(exp, result, offset=background.get("offset"))
    print(rb.summarize(result))
    if result.unresolved:
        print("\nThe check for wells falling below their background runs once these are settled.")
    if facts["od_channel"]:
        table = result.wells[result.wells["measurement"] == facts["od_channel"]]
        print("\nOD start by well, reads dropped as the initial jump in brackets:")
        lookup = {row.well: f"{row.start:.3f}[{int(row.jump_reads)}]" for row in table.itertuples(index=False) if row.start == row.start}
        for row in exp.wells_2d:
            cells = [lookup.get(well, ".").rjust(9) for well in row if isinstance(well, str)]
            print(" ".join(cells))

    rb.background_table(result).to_csv("background_table.csv", index=False)
    Path("figures").mkdir(exist_ok=True)
    for channel in config["measurements"]:
        rb.plot_background_qc(exp, result, channel, save_path=f"figures/{channel}_background_qc.png")
    print("\nSaved background_table.csv and figures/*_background_qc.png")
    sys.exit(2 if result.unresolved else 0)


if __name__ == "__main__":
    main()
