"""Serve Stopframe, the browser front end for the pipeline (issue #17).

Wraps the tools/ scripts as background jobs with live logs and progress,
shows each match's stage status, and serves its outputs. See app/README.md.

Usage:
    python tools/serve_app.py                       # http://<host>:8780/
    python tools/serve_app.py --port 8781 --host 127.0.0.1
    python tools/serve_app.py --busy-check ~/claude-tg/bin/media-busy

--busy-check names a command that exits 0 while the machine is busy (e.g. the
media server is playing); heavy jobs (tracking, camera motion, rendering)
wait for it to go idle unless started with "Run now". Ports 8765/8766 are
taken by Jellyfin live TV and 8790 by calibrate_pitch.py's web mode.
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--busy-check", default=os.environ.get("FOOTBALL_BUSY_CHECK"),
                        help="Command that exits 0 while heavy jobs should wait "
                             "(env FOOTBALL_BUSY_CHECK)")
    args = parser.parse_args()

    os.chdir(ROOT)  # tools/ assume the repo root as cwd
    sys.path.insert(0, str(ROOT))
    import uvicorn

    from app.server import create_app

    busy = None
    if args.busy_check:
        busy = [os.path.expanduser(a) for a in shlex.split(args.busy_check)]
    app = create_app(ROOT, busy_check=busy)
    print(f"Stopframe on http://{args.host}:{args.port}/  (busy check: {busy or 'off'})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
