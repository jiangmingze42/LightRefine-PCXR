#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saicinpainting.refiner.config import load_config
from saicinpainting.refiner.train import train_refiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LightRefine-PCXR LaMa refiner.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "refiner" / "lama_refiner.yaml"),
        help="Path to the refiner YAML config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional KEY=VALUE overrides, for example paths.train_images=/data/train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    best_path = train_refiner(config, config_path=args.config)
    print(f"Best validation checkpoint: {best_path}")


if __name__ == "__main__":
    main()
