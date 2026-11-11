"""Command line entry points; commands never modify raw acquisitions."""

import argparse
import json
from .storage import Project, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lab-microscopy")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser(
        "inspect", help="Inspect TIFF headers and write a configuration template"
    )
    s.add_argument("folder")
    s.add_argument("--config-template")
    s = sub.add_parser(
        "init", help="Hash and index an explicitly configured acquisition"
    )
    s.add_argument("folder")
    s.add_argument("--config", required=True)
    s.add_argument("--project", required=True)
    s = sub.add_parser(
        "demo", help="Generate a synthetic acquisition and a tracked project"
    )
    s.add_argument("destination")
    for command in ["run", "analyze", "serve", "export-reviews", "import-reviews"]:
        s = sub.add_parser(command)
        s.add_argument("project")
        if command != "run":
            s.add_argument("--run-id")
        if command == "run":
            s.add_argument("--fields", nargs="+")
            s.add_argument("--baseline-only", action="store_true")
            s.add_argument("--resume-from", help="Reuse saved label planes of an earlier run with identical segmentation settings")
        if command == "analyze":
            s.add_argument("--reviewed-only", action="store_true")
        if command == "serve":
            s.add_argument("--port", type=int, default=8766)
        if command == "export-reviews":
            s.add_argument("--output", required=True)
        if command == "import-reviews":
            s.add_argument("file")
            s.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            from .importing import inspect, template

            result = inspect(args.folder)
            if args.config_template:
                from pathlib import Path

                if Path(args.config_template).exists():
                    raise ValueError("Configuration already exists")
                write_json(args.config_template, template())
        elif args.command == "init":
            from .importing import initialize

            result = {
                "project": str(initialize(args.folder, args.config, args.project).root)
            }
        elif args.command == "demo":
            from .demo import create

            result = {"project": str(create(args.destination).root)}
        elif args.command == "run":
            from .pipeline import run

            result = run(
                args.project, fields=args.fields, baseline_only=args.baseline_only,
                resume_from=args.resume_from,
            )
            result = {k: result[k] for k in ["run_id", "fields"]}
        elif args.command == "analyze":
            from .analysis import analyze

            result = analyze(args.project, args.run_id, args.reviewed_only)
        elif args.command == "serve":
            from .server import serve

            serve(args.project, args.port, args.run_id)
            return
        else:
            p = Project(args.project)
            run_id = args.run_id or p.latest_run()
            if args.command == "export-reviews":
                from pathlib import Path

                if Path(args.output).exists():
                    raise ValueError(
                        "Export exists; choose a new file to preserve the previous backup"
                    )
                write_json(args.output, p.export_review(run_id))
                result = {"output": args.output}
            else:
                from .review import save

                result = save(
                    p,
                    run_id,
                    {"type": "import", "review": read_json(args.file)},
                    args.expected_revision,
                )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, FileNotFoundError) as e:
        parser.exit(2, f"Error: {e}\n")


if __name__ == "__main__":
    main()
