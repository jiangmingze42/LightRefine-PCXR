from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _is_mask_name(path: str, mask_suffix: str) -> bool:
    return Path(path).stem.endswith(mask_suffix)


def list_images(
    root: str,
    suffixes: Iterable[str] = IMAGE_EXTENSIONS,
    recursive: bool = False,
    exclude_mask_suffix: str | None = None,
) -> list[str]:
    pattern = "**/*" if recursive else "*"
    suffixes = tuple(s.lower() for s in suffixes)
    files = []
    for path in glob.glob(os.path.join(root, pattern), recursive=recursive):
        if not os.path.isfile(path):
            continue
        if Path(path).suffix.lower() not in suffixes:
            continue
        if exclude_mask_suffix and _is_mask_name(path, exclude_mask_suffix):
            continue
        files.append(path)
    return sorted(files)


def pad_to_modulo(array: np.ndarray, modulo: int, mode: str = "constant") -> tuple[np.ndarray, tuple[int, int]]:
    h, w = array.shape[:2]
    pad_h = (modulo - h % modulo) % modulo
    pad_w = (modulo - w % modulo) % modulo
    if pad_h == 0 and pad_w == 0:
        return array, (h, w)

    if array.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        pad_width = ((0, pad_h), (0, pad_w))
    return np.pad(array, pad_width, mode=mode), (h, w)


def read_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def read_mask(path: str, binarize: bool = True) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    if binarize:
        return (mask > 127).astype(np.float32)
    return mask.astype(np.float32) / 255.0


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(mask))[None, ...]


def get_first_hw(collated_hw) -> tuple[int, int]:
    if torch.is_tensor(collated_hw):
        if collated_hw.ndim == 1 and collated_hw.numel() == 2:
            return int(collated_hw[0].item()), int(collated_hw[1].item())
        if collated_hw.ndim == 2 and collated_hw.shape[0] >= 1 and collated_hw.shape[1] == 2:
            return int(collated_hw[0, 0].item()), int(collated_hw[0, 1].item())
    if isinstance(collated_hw, (list, tuple)) and len(collated_hw) == 2:
        h, w = collated_hw
        if torch.is_tensor(h):
            h = h.flatten()[0].item()
        if torch.is_tensor(w):
            w = w.flatten()[0].item()
        return int(h), int(w)
    raise ValueError(f"Unexpected collated image size: {collated_hw}")


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        pad_out_to_modulo: int = 32,
        image_suffixes: Iterable[str] = IMAGE_EXTENSIONS,
        recursive: bool = False,
        exclude_mask_suffix: str = "_mask000",
        pad_mode: str = "constant",
    ):
        self.image_dir = image_dir
        self.pad_out_to_modulo = pad_out_to_modulo
        self.pad_mode = pad_mode
        self.image_filenames = list_images(
            image_dir,
            suffixes=image_suffixes,
            recursive=recursive,
            exclude_mask_suffix=exclude_mask_suffix,
        )
        if not self.image_filenames:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        image_path = self.image_filenames[idx]
        image = read_rgb(image_path)
        image, orig_hw = pad_to_modulo(image, self.pad_out_to_modulo, mode=self.pad_mode)
        return {
            "image": image_to_tensor(image),
            "path": image_path,
            "orig_hw": orig_hw,
        }


class PairedImageMaskDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str | None = None,
        image_suffixes: Iterable[str] = IMAGE_EXTENSIONS,
        mask_suffix: str = "_mask000",
        pad_out_to_modulo: int = 32,
        recursive: bool = False,
        binarize_masks: bool = True,
        pad_mode: str = "constant",
        skip_missing_masks: bool = False,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir or image_dir
        self.mask_suffix = mask_suffix
        self.pad_out_to_modulo = pad_out_to_modulo
        self.binarize_masks = binarize_masks
        self.pad_mode = pad_mode

        image_paths = list_images(
            image_dir,
            suffixes=image_suffixes,
            recursive=recursive,
            exclude_mask_suffix=mask_suffix,
        )
        pairs = []
        for image_path in image_paths:
            mask_path = self._find_mask(image_path)
            if mask_path is None:
                if skip_missing_masks:
                    continue
                raise FileNotFoundError(f"Mask not found for image: {image_path}")
            pairs.append((image_path, mask_path))
        if not pairs:
            raise RuntimeError(f"No image-mask pairs found in {image_dir}")
        self.image_filenames = [p[0] for p in pairs]
        self.mask_filenames = [p[1] for p in pairs]

    def _find_mask(self, image_path: str) -> str | None:
        stem = Path(image_path).stem
        candidates = [
            os.path.join(self.mask_dir, stem + self.mask_suffix + ext)
            for ext in IMAGE_EXTENSIONS
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        image_path = self.image_filenames[idx]
        mask_path = self.mask_filenames[idx]
        image = read_rgb(image_path)
        mask = read_mask(mask_path, binarize=self.binarize_masks)
        image, orig_hw = pad_to_modulo(image, self.pad_out_to_modulo, mode=self.pad_mode)
        mask, _ = pad_to_modulo(mask, self.pad_out_to_modulo, mode=self.pad_mode)
        return {
            "image": image_to_tensor(image),
            "mask": mask_to_tensor(mask),
            "path": image_path,
            "mask_path": mask_path,
            "orig_hw": orig_hw,
            "unpad_to_size": orig_hw,
        }


class FileMaskPool:
    def __init__(
        self,
        mask_dir: str,
        out_size: int = 512,
        recursive: bool = True,
        binarize: bool = True,
        extensions: Iterable[str] = IMAGE_EXTENSIONS,
    ):
        self.mask_paths = list_images(mask_dir, suffixes=extensions, recursive=recursive)
        if not self.mask_paths:
            raise RuntimeError(f"No mask files found in {mask_dir}")
        self.out_size = out_size
        self.binarize = binarize
        self.masks = []
        for path in self.mask_paths:
            mask = read_mask(path, binarize=binarize)
            mask = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
            self.masks.append(mask.astype(np.float32))

    def sample(self, image: torch.Tensor) -> torch.Tensor:
        mask = random.choice(self.masks)
        mask_t = torch.from_numpy(mask)[None, ...].to(device=image.device, dtype=image.dtype)
        if mask_t.shape[-2:] != image.shape[-2:]:
            mask_t = F.interpolate(mask_t[None], size=image.shape[-2:], mode="nearest")[0]
        return mask_t
