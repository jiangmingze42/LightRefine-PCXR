from __future__ import annotations

import torch


def compose_refined_output(
    image: torch.Tensor,
    mask: torch.Tensor,
    base_pred: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    refined = base_pred + mask * delta
    inpainted = mask * refined + (1 - mask) * image
    return refined, inpainted


@torch.no_grad()
def run_lama_backbone(
    lama_generator: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_image = image * (1 - mask)
    lama_input = torch.cat([masked_image, mask], dim=1)
    return lama_generator(lama_input), masked_image


def run_refiner(
    refiner: torch.nn.Module,
    base_pred: torch.Tensor,
    masked_image: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    refiner_input = torch.cat([base_pred, masked_image, mask], dim=1)
    return refiner(refiner_input)
