"""Convenience launcher after installing with pip install -e ."""

import sys

from retailiq.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pipeline", *sys.argv[1:]]))
