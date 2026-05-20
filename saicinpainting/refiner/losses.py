from __future__ import annotations

import torch
import torch.nn.functional as F


def ensure_3ch(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x[:, :3]


def sobel_gradients(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        x = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
    else:
        x = x[:, :1]
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def dilate_mask(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 0:
        return (mask > 0.5).float()
    pad = kernel_size // 2
    return (F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad) > 0).float()


def boundary_band_mask(mask: torch.Tensor, band: int) -> torch.Tensor:
    if band <= 0:
        return torch.zeros_like(mask)
    dilated = dilate_mask(mask, 2 * band + 1)
    eroded = 1.0 - dilate_mask(1.0 - mask, 2 * band + 1)
    return (dilated - eroded).clamp(0.0, 1.0)


def total_variation(image: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    dx = image[:, :, 1:, :] - image[:, :, :-1, :]
    dy = image[:, :, :, 1:] - image[:, :, :, :-1]
    if mask is not None:
        mx = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        my = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        dx = dx * mx
        dy = dy * my
        denom = mx.sum() * image.shape[1] + my.sum() * image.shape[1] + 1e-6
    else:
        denom = torch.tensor(image.numel() / image.shape[0], device=image.device, dtype=image.dtype)
    return (dx.abs().sum() + dy.abs().sum()) / denom


class MaskedLPIPS:
    def __init__(self, backbone: str = "alex", spatial: bool = True, device: str | torch.device = "cuda"):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("lpips is required when loss.lpips_weight > 0") from exc

        self.model = lpips.LPIPS(net=backbone, spatial=spatial).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _to_minus1_1(image: torch.Tensor) -> torch.Tensor:
        return image * 2.0 - 1.0

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pred = self._to_minus1_1(ensure_3ch(pred).clamp(0, 1))
        target = self._to_minus1_1(ensure_3ch(target).clamp(0, 1))
        if getattr(self.model, "spatial", False):
            lp_map = self.model(pred, target, normalize=False)
            mask_small = F.interpolate(mask, size=lp_map.shape[-2:], mode="nearest")
            return (lp_map * mask_small).sum() / mask_small.sum().clamp_min(1e-6)
        pred_roi = mask * ((pred + 1.0) * 0.5) + (1 - mask) * ((target + 1.0) * 0.5)
        return self.model(self._to_minus1_1(pred_roi), target, normalize=False).mean()


class RefinerLoss:
    def __init__(
        self,
        residual_weight: float = 1.0,
        edge_weight: float = 1.0,
        tv_weight: float = 0.1,
        lpips_weight: float = 0.0,
        lpips_backbone: str = "alex",
        boundary_band: int = 4,
        device: str | torch.device = "cuda",
    ):
        self.residual_weight = float(residual_weight)
        self.edge_weight = float(edge_weight)
        self.tv_weight = float(tv_weight)
        self.lpips_weight = float(lpips_weight)
        self.boundary_band = int(boundary_band)
        self.lpips = MaskedLPIPS(lpips_backbone, spatial=True, device=device) if self.lpips_weight > 0 else None

    def __call__(
        self,
        image: torch.Tensor,
        base_pred: torch.Tensor,
        delta: torch.Tensor,
        inpainted: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        target_residual = (image - base_pred) * mask
        pred_residual = delta * mask
        residual = F.l1_loss(pred_residual, target_residual) * self.residual_weight
        loss = residual
        metrics = {"residual_l1": float(residual.detach().item())}

        if self.edge_weight > 0:
            edge = F.l1_loss(sobel_gradients(inpainted) * mask, sobel_gradients(image) * mask) * self.edge_weight
            loss = loss + edge
            metrics["edge"] = float(edge.detach().item())

        if self.tv_weight > 0:
            band = boundary_band_mask(mask, self.boundary_band)
            tv_mask = torch.clamp(mask + band, 0, 1)
            tv = total_variation(inpainted, mask=tv_mask) * self.tv_weight
            loss = loss + tv
            metrics["tv"] = float(tv.detach().item())

        if self.lpips_weight > 0 and self.lpips is not None:
            lp = self.lpips(inpainted, image, mask) * self.lpips_weight
            loss = loss + lp
            metrics["lpips"] = float(lp.detach().item())

        metrics["total"] = float(loss.detach().item())
        return loss, metrics
