from __future__ import annotations

import csv
import json
import os
import random
import shutil
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from saicinpainting.refiner.config import get_by_path, require_path, resolve_path
from saicinpainting.refiner.data import FileMaskPool, ImageFolderDataset, PairedImageMaskDataset, get_first_hw
from saicinpainting.refiner.lama import load_frozen_lama_generator
from saicinpainting.refiner.losses import RefinerLoss
from saicinpainting.refiner.metrics import aggregate_metrics, reconstruction_metrics
from saicinpainting.refiner.models import build_refiner, count_parameters
from saicinpainting.refiner.pipeline import compose_refined_output, run_lama_backbone, run_refiner


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(output_root: str, tag: str, config_path: str | None = None) -> dict[str, str]:
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{tag}"
    run_dir = Path(output_root) / run_name
    paths = {
        "run_dir": str(run_dir),
        "checkpoints": str(run_dir / "checkpoints"),
        "logs": str(run_dir / "logs"),
        "config": str(run_dir / "config"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    if config_path and os.path.isfile(config_path):
        shutil.copy2(config_path, Path(paths["config"]) / "config.yaml")
    return paths


def write_metrics(run_dir: str, metrics: dict[str, Any]) -> None:
    jsonl_path = Path(run_dir) / "metrics.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    csv_path = Path(run_dir) / "metrics.csv"
    fields = [
        "epoch",
        "split",
        "full_psnr",
        "full_ssim",
        "full_mae",
        "full_rmse",
        "roi_psnr",
        "roi_ssim",
        "roi_mae",
        "roi_rmse",
        "num_used",
        "num_total",
    ]
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: metrics.get(key) for key in fields})


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def evaluate(
    refiner: torch.nn.Module,
    lama_generator: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    filter_identical: bool = True,
) -> dict[str, float | int]:
    refiner.eval()
    per_image = []
    for batch in tqdm(loader, desc="Evaluate", leave=False):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        h0, w0 = get_first_hw(batch["orig_hw"])

        base_pred, masked_image = run_lama_backbone(lama_generator, image, mask)
        delta = run_refiner(refiner, base_pred, masked_image, mask)
        _, inpainted = compose_refined_output(image, mask, base_pred, delta)

        target = image[:, :, :h0, :w0]
        pred = inpainted[:, :, :h0, :w0]
        mask0 = mask[:, :, :h0, :w0]
        per_image.append(reconstruction_metrics(target, pred, mask0, filter_identical=filter_identical))
    refiner.train()
    return aggregate_metrics(per_image, filter_identical=filter_identical)


def train_refiner(config: dict[str, Any], config_path: str | None = None) -> str:
    device = torch.device(get_by_path(config, "runtime.device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(get_by_path(config, "runtime.seed"))

    paths_cfg = config.get("paths", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    run_paths = make_run_dir(
        output_root=resolve_path(paths_cfg.get("output_dir"), Path.cwd()) or "outputs/refiner",
        tag=str(get_by_path(config, "run.tag", "lama_refiner")),
        config_path=config_path,
    )

    lama_generator = load_frozen_lama_generator(
        config_path=require_path(config, "paths.lama_config"),
        checkpoint_path=require_path(config, "paths.lama_checkpoint"),
        device=device,
        strict=bool(get_by_path(config, "lama.strict_load", True)),
    )
    refiner = build_refiner(config).to(device)

    train_dataset = ImageFolderDataset(
        image_dir=require_path(config, "paths.train_images"),
        pad_out_to_modulo=int(data_cfg.get("pad_out_to_modulo", 32)),
        recursive=bool(data_cfg.get("recursive_images", False)),
        exclude_mask_suffix=str(data_cfg.get("mask_suffix", "_mask000")),
        pad_mode=str(data_cfg.get("train_pad_mode", "constant")),
    )
    val_dataset = PairedImageMaskDataset(
        image_dir=require_path(config, "paths.val_images"),
        mask_dir=paths_cfg.get("val_masks"),
        mask_suffix=str(data_cfg.get("mask_suffix", "_mask000")),
        pad_out_to_modulo=int(data_cfg.get("pad_out_to_modulo", 32)),
        recursive=bool(data_cfg.get("recursive_images", False)),
        binarize_masks=bool(data_cfg.get("binarize_masks", True)),
        pad_mode=str(data_cfg.get("eval_pad_mode", "constant")),
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(train_cfg.get("eval_num_workers", 2)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
    )

    test_loader = None
    if bool(get_by_path(config, "evaluation.report_test", False)) and paths_cfg.get("test_images"):
        test_dataset = PairedImageMaskDataset(
            image_dir=paths_cfg["test_images"],
            mask_dir=paths_cfg.get("test_masks"),
            mask_suffix=str(data_cfg.get("mask_suffix", "_mask000")),
            pad_out_to_modulo=int(data_cfg.get("pad_out_to_modulo", 32)),
            recursive=bool(data_cfg.get("recursive_images", False)),
            binarize_masks=bool(data_cfg.get("binarize_masks", True)),
            pad_mode=str(data_cfg.get("eval_pad_mode", "constant")),
        )
        test_loader = make_loader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(train_cfg.get("eval_num_workers", 2)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
        )

    mask_pool = FileMaskPool(
        mask_dir=require_path(config, "paths.train_masks"),
        out_size=int(train_cfg.get("mask_out_size", 512)),
        recursive=bool(data_cfg.get("recursive_masks", True)),
        binarize=bool(data_cfg.get("binarize_masks", True)),
    )
    criterion = RefinerLoss(
        residual_weight=float(get_by_path(config, "loss.residual_weight", 1.0)),
        edge_weight=float(get_by_path(config, "loss.edge_weight", 1.0)),
        tv_weight=float(get_by_path(config, "loss.tv_weight", 0.1)),
        lpips_weight=float(get_by_path(config, "loss.lpips_weight", 0.0)),
        lpips_backbone=str(get_by_path(config, "loss.lpips_backbone", "alex")),
        boundary_band=int(get_by_path(config, "loss.boundary_band", 4)),
        device=device,
    )
    optimizer = torch.optim.Adam(refiner.parameters(), lr=float(train_cfg.get("lr", 1e-4)))

    meta = {
        "user": os.environ.get("USERNAME") or os.environ.get("USER"),
        "host": socket.gethostname(),
        "device": str(device),
        "lama_parameters": count_parameters(lama_generator),
        "refiner_parameters": count_parameters(refiner),
        "refiner_trainable_parameters": count_parameters(refiner, trainable_only=True),
    }
    with (Path(run_paths["logs"]) / "run.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    best_metric_name = str(train_cfg.get("checkpoint_metric", "full_psnr"))
    best_score = -float("inf")
    latest_path = Path(run_paths["checkpoints"]) / "latest.pth"
    best_path = Path(run_paths["checkpoints"]) / "refiner_best.pth"
    eval_interval = int(train_cfg.get("eval_interval", 10))
    global_step = 0

    init_metrics = evaluate(
        refiner,
        lama_generator,
        val_loader,
        device,
        filter_identical=bool(get_by_path(config, "evaluation.filter_identical", True)),
    )
    init_metrics.update(epoch=-1, split="val")
    write_metrics(run_paths["run_dir"], init_metrics)
    best_score = float(init_metrics[best_metric_name])
    torch.save({"epoch": -1, "best_metric": best_score, "refiner": refiner.state_dict()}, best_path)

    for epoch in range(int(train_cfg.get("epochs", 400))):
        refiner.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            image = batch["image"].to(device)
            masks = torch.stack([mask_pool.sample(image[i]) for i in range(image.shape[0])], dim=0)

            base_pred, masked_image = run_lama_backbone(lama_generator, image, masks)
            delta = run_refiner(refiner, base_pred, masked_image, masks)
            _, inpainted = compose_refined_output(image, masks, base_pred, delta)
            loss, loss_metrics = criterion(image, base_pred, delta, inpainted, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().item())
            global_step += 1
            if global_step % int(train_cfg.get("log_interval", 50)) == 0:
                pbar.set_postfix({k: f"{v:.4f}" for k, v in loss_metrics.items()})

        train_metrics = {
            "epoch": epoch,
            "split": "train",
            "loss": running_loss / max(1, len(train_loader)),
            "global_step": global_step,
        }
        with (Path(run_paths["run_dir"]) / "train_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(train_metrics) + "\n")

        if epoch % eval_interval == 0:
            val_metrics = evaluate(
                refiner,
                lama_generator,
                val_loader,
                device,
                filter_identical=bool(get_by_path(config, "evaluation.filter_identical", True)),
            )
            val_metrics.update(epoch=epoch, split="val")
            write_metrics(run_paths["run_dir"], val_metrics)

            state = {"epoch": epoch, "global_step": global_step, "refiner": refiner.state_dict()}
            torch.save(state, latest_path)
            score = float(val_metrics[best_metric_name])
            if score > best_score:
                best_score = score
                torch.save({**state, "best_metric": best_score}, best_path)

            if test_loader is not None and bool(get_by_path(config, "evaluation.report_test", False)):
                # Optional reporting only. Checkpoint selection above uses validation metrics.
                report_metrics = evaluate(
                    refiner,
                    lama_generator,
                    test_loader,
                    device,
                    filter_identical=bool(get_by_path(config, "evaluation.filter_identical", True)),
                )
                report_metrics.update(epoch=epoch, split="test")
                write_metrics(run_paths["run_dir"], report_metrics)

    return str(best_path)
