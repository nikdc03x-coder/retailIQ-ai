"""Start the dashboard with the project's own Python environment and data checks."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start the RetailIQ AI dashboard")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    root = Path(__file__).resolve().parent
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Create the project environment with: python -m venv .venv", file=sys.stderr)
        return 1
    check = subprocess.run([str(python), "-m", "retailiq", "check-data"], cwd=root)
    if check.returncode:
        print(
            "If the project libraries are missing, install them with:\n"
            f'"{python}" -m pip install -e "{root}[dev]"\n'
            "Follow the data instructions above for missing datasets or reports.",
            file=sys.stderr,
        )
        return check.returncode
    return subprocess.run(
        [
            str(python),
            "-m",
            "streamlit",
            "run",
            str(root / "app.py"),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=root,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
