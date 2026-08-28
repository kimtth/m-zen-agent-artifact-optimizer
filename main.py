"""Shim so `uv run main.py ...` keeps working. The CLI lives in zen/cli.py."""

from zen.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
