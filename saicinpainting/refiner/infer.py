from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from saicinpainting.refiner.config import get_by_path, require_path
from saicinpainting.refiner.data import PairedImageMaskDataset, get_first_hw
from saicinpainting.refiner.lama import load_frozen_lama_generator
from saicinpainting.refiner.models import build_refiner, load_refiner_checkpoint
from saicinpainting.refiner.pipeline import compose_refined_output, run_lama_backbone, run_refiner


def tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    array = (array * 255.0).astype(np.uint8)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


@torch.no_grad()
def run_inference(config: dict[str, Any]) -> None:
    device = torch.device(get_by_path(config, "runtime.device", "cuda") if torch.cuda.is_available() else "cpu")
    infer_cfg = config.get("inference", {})
    data_cfg = config.get("data", {})

    lama_generator = load_frozen_lama_generator(
        config_path=require_path(config, "paths.lama_config"),
        checkpoint_path=require_path(config, "paths.lama_checkpoint"),
        device=device,
        strict=bool(get_by_path(config, "lama.strict_load", True)),
    )
    refiner = build_refiner(config).to(device)
    load_refiner_checkpoint(refiner, require_path(config, "paths.refiner_checkpoint"), device=device)
    refiner.eval()

    dataset = PairedImageMaskDataset(
        image_dir=require_path(config, "inference.image_dir"),
        mask_dir=infer_cfg.get("mask_dir"),
        mask_suffix=str(infer_cfg.get("mask_suffix", data_cfg.get("mask_suffix", "_mask000"))),
        pad_out_to_modulo=int(data_cfg.get("pad_out_to_modulo", 32)),
        recursive=bool(data_cfg.get("recursive_images", False)),
        binarize_masks=bool(infer_cfg.get("binarize_masks", data_cfg.get("binarize_masks", True))),
        pad_mode=str(infer_cfg.get("pad_mode", "edge")),
        skip_missing_masks=bool(infer_cfg.get("skip_missing_masks", True)),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(infer_cfg.get("num_workers", 2)),
        pin_memory=bool(infer_cfg.get("pin_memory", True)),
    )

    pred_dir = Path(require_path(config, "inference.output_pred_dir"))
    inpaint_dir = Path(require_path(config, "inference.output_inpaint_dir"))
    pred_dir.mkdir(parents=True, exist_ok=True)
    inpaint_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Inference"):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        h0, w0 = get_first_hw(batch["orig_hw"])
        image_path = batch["path"][0]
        stem = Path(image_path).stem

        base_pred, masked_image = run_lama_backbone(lama_generator, image, mask)
        delta = run_refiner(refiner, base_pred, masked_image, mask)
        refined, inpainted = compose_refined_output(image, mask, base_pred, delta)

        refined = refined[:, :, :h0, :w0]
        inpainted = inpainted[:, :, :h0, :w0]
        cv2.imwrite(str(pred_dir / f"{stem}_refined.png"), tensor_to_bgr(refined))
        cv2.imwrite(str(inpaint_dir / f"{stem}_mask000.png"), tensor_to_bgr(inpainted))
