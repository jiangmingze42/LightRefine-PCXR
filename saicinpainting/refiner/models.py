from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def build_refiner(config: dict[str, Any]) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required for the LightRefine U-Net. "
            "Install it from requirements.txt."
        ) from exc

    model_cfg = config.get("model", {}).get("refiner", {})
    return smp.Unet(
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_depth=int(model_cfg.get("encoder_depth", 5)),
        encoder_weights=model_cfg.get("encoder_weights", None),
        in_channels=int(model_cfg.get("in_channels", 7)),
        classes=int(model_cfg.get("classes", 3)),
        activation=model_cfg.get("activation", None),
    )


def load_refiner_checkpoint(
    refiner: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cuda",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("refiner", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    refiner.load_state_dict(state, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {"refiner": state}


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)
