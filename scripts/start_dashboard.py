#!/usr/bin/env python3
"""Dashboard startup script."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Start the Modular RAG Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to run dashboard on")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    args = parser.parse_args()
    
    dashboard_path = Path(__file__).parent.parent / "src" / "observability" / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"[ERROR] Dashboard app not found at {dashboard_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Starting dashboard at http://{args.host}:{args.port}")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Dashboard failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
