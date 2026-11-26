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
    parser.add_argument("--host", default="127.0.0.1",
                        help="Address to listen on. Default: this computer only. 0.0.0.0 = all networks "
                             "(needs --allow-client; links then carry an access key)")
    parser.add_argument("--allow-client", action="append", default=[], metavar="ADDRESS_OR_NETWORK",
                        help="Computer or network allowed in network mode, e.g. 132.64.81.20 or 100.64.0.0/10 (Tailscale). Repeatable")
    parser.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                        help="Extra name the browser may use for this computer, e.g. its Tailscale name. Repeatable")
    parser.add_argument("--image-format", choices=["png", "jpeg"], default="png",
                        help="Display images: png (lossless) or jpeg (about 4x smaller; for remote review)")
    args = parser.parse_args(argv)
    serve(args.project, args.port, args.run_id, read_only=args.read_only, open_browser=args.open,
          tagging_sets=args.tagging_set, host=args.host, allow_clients=args.allow_client,
          allow_hosts=args.allow_host, image_format=args.image_format)


if __name__ == "__main__":
    main()
