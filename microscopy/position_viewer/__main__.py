"""Command-line entry point for building and serving position caches."""

import argparse
import sys
from pathlib import Path

from .server import FastThreadingHTTPServer


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("--") and argv[0] != "--help":
        argv.insert(0, "serve")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Serve existing position caches")
    serve.add_argument("--cache-dir", type=Path, action="append", required=True,
                       help="Cache root, repeatable in search order")
    serve.add_argument("--positions-file", type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8896)
    serve.add_argument("--allow-client", action="append", default=[], metavar="IP_OR_CIDR")
    serve.add_argument("--no-auth", action="store_true", help="Disable the key on loopback")
    build = commands.add_parser("build", help="Build a cache from a microscopy project run")
    build.add_argument("project_dir", type=Path)
    build.add_argument("output_dir", type=Path, help="New or empty cache directory")
    build.add_argument("--run-id", help="Defaults to the latest run")
    build.add_argument("--field", action="append", dest="fields", help="Field to include, repeatable")
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            from .cache import build_project_viewer_cache

            output = build_project_viewer_cache(args.project_dir, args.output_dir, args.run_id, args.fields)
            print(f"Built position cache at {output}", flush=True)
            return
        if not 0 <= args.port <= 65535:
            raise ValueError("--port must be between 0 and 65535")
        with FastThreadingHTTPServer(
            args.cache_dir, positions_file=args.positions_file, host=args.host, port=args.port,
            allow_clients=args.allow_client, no_auth=args.no_auth,
        ) as server:
            for link in server.links:
                print(f"Position Viewer listening at {link}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    except (ValueError, OSError, MemoryError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
