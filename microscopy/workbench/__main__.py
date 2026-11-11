"""python -m microscopy.workbench PROJECT [--run-id RUN] [--port PORT] [--read-only] [--tagging-set FILE] [--open]"""

import argparse

from .server import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Founder review viewer for a microscopy project")
    parser.add_argument("project", help="Project folder (contains project.json, runs/ and reviews.sqlite)")
    parser.add_argument("--run-id", help="Run to review; defaults to latest_run.json")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--read-only", action="store_true", help="Browse without saving")
    parser.add_argument("--tagging-set", help='JSON file {"sets": {"name": [positions]}} adding a position-set filter')
    parser.add_argument("--open", action="store_true", help="Open the viewer in the browser")
    args = parser.parse_args(argv)
    serve(args.project, args.port, args.run_id, read_only=args.read_only, open_browser=args.open,
          tagging_sets=args.tagging_set)


if __name__ == "__main__":
    main()
