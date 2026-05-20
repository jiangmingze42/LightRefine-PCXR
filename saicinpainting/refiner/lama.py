from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from saicinpainting.training.modules import make_generator


def _load_train_config(config_path: str | Path) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise ImportError(
            "omegaconf is required to load a LaMa training config. "
            "Install hydra-core/omegaconf from requirements.txt."
        ) from exc

    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return OmegaConf.create(data)


def _extract_generator_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError("Unsupported LaMa checkpoint format: missing state_dict")

    generator_state = {}
    for key, value in state.items():
        if key.startswith("generator."):
            generator_state[key[len("generator.") :]] = value
        elif not key.startswith(("discriminator.", "loss_", "val_", "test_")):
            generator_state[key] = value
    if not generator_state:
        raise ValueError("No generator weights found in LaMa checkpoint")
    return generator_state


def load_frozen_lama_generator(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    strict: bool = True,
) -> torch.nn.Module:
    train_config = _load_train_config(config_path)
    generator = make_generator(train_config, **train_config.generator)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator.load_state_dict(_extract_generator_state(checkpoint), strict=strict)
    generator.eval().to(device)
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    return generator
