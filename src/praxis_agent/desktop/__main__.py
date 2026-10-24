from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Praxis desktop stdio service")
    parser.add_argument("--stdio", action="store_true", required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--bypass", action="store_true", help="Run tools without approval prompts for this desktop process")
    args = parser.parse_args()
    # Reserve the original stdout for protocol messages. Library prints become diagnostics.
    protocol_stdout = sys.stdout
    for stream in (sys.stdin, protocol_stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    from .server import DesktopServer
    try:
        DesktopServer(sys.stdin, protocol_stdout, database_path=args.database,
                      bypass_approvals=args.bypass).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
