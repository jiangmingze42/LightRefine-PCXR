from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_gray_numpy_01(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor[0].detach().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    return np.asarray(Image.fromarray(array).convert("L")).astype(np.float32) / 255.0


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0.5)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)


def reconstruction_metrics(
    target: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    filter_identical: bool = True,
) -> dict[str, float | bool | None]:
    target_gray = to_gray_numpy_01(target)
    pred_gray = to_gray_numpy_01(pred)
    mask_np = (mask[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

    identical = bool(np.all((target_gray - pred_gray) == 0))
    diff_full = target_gray - pred_gray
    result: dict[str, float | bool | None] = {
        "identical": identical,
        "full_psnr": float(peak_signal_noise_ratio(target_gray, pred_gray, data_range=1.0)),
        "full_ssim": float(structural_similarity(target_gray, pred_gray, data_range=1.0)),
        "full_mae": float(np.mean(np.abs(diff_full))),
        "full_rmse": float(np.sqrt(np.mean(diff_full**2))),
        "roi_psnr": None,
        "roi_ssim": None,
        "roi_mae": None,
        "roi_rmse": None,
    }
    bbox = bbox_from_mask(mask_np)
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        target_roi = target_gray[y0:y1, x0:x1]
        pred_roi = pred_gray[y0:y1, x0:x1]
        diff_roi = target_roi - pred_roi
        result.update(
            roi_psnr=float(peak_signal_noise_ratio(target_roi, pred_roi, data_range=1.0)),
            roi_ssim=float(structural_similarity(target_roi, pred_roi, data_range=1.0)),
            roi_mae=float(np.mean(np.abs(diff_roi))),
            roi_rmse=float(np.sqrt(np.mean(diff_roi**2))),
        )
    return result


def aggregate_metrics(per_image: list[dict[str, float | bool | None]], filter_identical: bool = True) -> dict[str, float | int]:
    selected = [m for m in per_image if not m["identical"]] if filter_identical else per_image
    if not selected:
        selected = per_image

    def mean(key: str) -> float:
        values = [m[key] for m in selected if m.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    return {
        "full_psnr": mean("full_psnr"),
        "full_ssim": mean("full_ssim"),
        "full_mae": mean("full_mae"),
        "full_rmse": mean("full_rmse"),
        "roi_psnr": mean("roi_psnr"),
        "roi_ssim": mean("roi_ssim"),
        "roi_mae": mean("roi_mae"),
        "roi_rmse": mean("roi_rmse"),
        "num_used": len(selected),
        "num_total": len(per_image),
    }
