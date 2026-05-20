#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saicinpainting.refiner.config import load_config
from saicinpainting.refiner.infer import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LaMa + LightRefine inference.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "refiner" / "lama_refiner.yaml"),
        help="Path to the refiner YAML config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional KEY=VALUE overrides, for example inference.image_dir=/data/images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    run_inference(config)


if __name__ == "__main__":
    main()
