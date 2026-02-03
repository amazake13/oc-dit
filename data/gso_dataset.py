"""GSO (Google Scanned Objects) dataset loader for OC-DiT training."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.io as io
import torchvision.transforms.v2.functional as TF

from data.dataset import BaseSegmentationDataset


class GSODataset(BaseSegmentationDataset):
    """Dataset loader for GSO (Google Scanned Objects) rendered data.

    Expected directory structure:
        data_root/
        ├── images/
        │   ├── 000000.png
        │   ├── 000001.png
        │   └── ...
        ├── masks/
        │   ├── 000000/
        │   │   ├── obj_0.png
        │   │   ├── obj_1.png
        │   │   └── ...
        │   └── ...
        ├── templates/
        │   ├── class_0/
        │   │   ├── view_0.png
        │   │   ├── view_1.png
        │   │   └── ...
        │   └── ...
        └── annotations.json

    annotations.json format:
        {
            "samples": [
                {
                    "image_id": "000000",
                    "objects": [
                        {"class_id": 0, "template_class": "class_0"},
                        {"class_id": 1, "template_class": "class_1"}
                    ]
                },
                ...
            ],
            "classes": ["class_0", "class_1", ...]
        }
    """

    def __init__(
        self,
        data_root: str,
        image_size: Tuple[int, int] = (480, 640),
        template_size: Tuple[int, int] = (128, 128),
        max_classes: int = 8,
        max_templates: int = 12,
        num_template_views: int = 4,
        split: str = "train",
        transform=None,
    ):
        """Initialize GSO dataset.

        Args:
            data_root: Path to dataset root directory
            image_size: Target image size (H, W)
            template_size: Target template size (h, w)
            max_classes: Maximum number of classes per sample
            max_templates: Maximum number of template views
            num_template_views: Number of template views to load per class
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transforms
        """
        super().__init__(
            image_size=image_size,
            template_size=template_size,
            max_classes=max_classes,
            max_templates=max_templates,
            transform=transform,
        )

        self.data_root = Path(data_root)
        self.num_template_views = min(num_template_views, max_templates)
        self.split = split

        # Load annotations
        annotations_path = self.data_root / "annotations.json"
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)

        self.samples = self.annotations.get("samples", [])
        self.classes = self.annotations.get("classes", [])

        # Filter by split if specified in annotations
        if "splits" in self.annotations:
            split_samples = self.annotations["splits"].get(split, [])
            self.samples = [s for s in self.samples if s["image_id"] in split_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        image_id = sample_info["image_id"]
        objects = sample_info["objects"]

        # Load image
        image_path = self.data_root / "images" / f"{image_id}.png"
        image = self._load_image(image_path)

        # Load masks and templates
        masks_list = []
        templates_list = []

        for obj in objects[: self.max_classes]:
            class_id = obj["class_id"]
            template_class = obj["template_class"]

            # Load mask
            mask_path = self.data_root / "masks" / image_id / f"obj_{class_id}.png"
            mask = self._load_mask(mask_path)
            masks_list.append(mask)

            # Load templates
            template_views = self._load_templates(template_class)
            templates_list.append(template_views)

        # Stack tensors
        num_classes = len(masks_list)

        if num_classes == 0:
            # Return empty sample
            return self._empty_sample()

        masks = torch.stack(masks_list)  # (num_classes, H, W)
        templates = torch.stack(templates_list)  # (num_classes, num_views, 3, h, w)

        # Apply transforms
        if self.transform is not None:
            sample = self.transform({
                "image": image,
                "templates": templates,
                "masks": masks,
            })
            image = sample["image"]
            templates = sample["templates"]
            masks = sample["masks"]

        # Resize to target size
        image = TF.resize(image, self.image_size, antialias=True)
        masks = TF.resize(
            masks.unsqueeze(1),
            self.image_size,
            interpolation=TF.InterpolationMode.NEAREST,
        ).squeeze(1)

        # Pad to max sizes
        templates, masks = self._pad_to_max(templates, masks)

        return {
            "image": image,
            "templates": templates,
            "masks": masks,
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image."""
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = io.read_image(str(path))  # (C, H, W)

        # Convert to float [0, 1]
        image = image.float() / 255.0

        # Handle RGBA
        if image.shape[0] == 4:
            image = image[:3]

        return image

    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load binary mask."""
        if not path.exists():
            # Return empty mask if file doesn't exist
            return torch.zeros(self.image_size)

        mask = io.read_image(str(path))  # (1, H, W) or (C, H, W)

        # Convert to single channel binary mask
        if mask.shape[0] > 1:
            mask = mask[0:1]
        mask = (mask > 127).float().squeeze(0)

        return mask

    def _load_templates(self, template_class: str) -> torch.Tensor:
        """Load template views for a class."""
        template_dir = self.data_root / "templates" / template_class

        if not template_dir.exists():
            # Return zeros if templates don't exist
            return torch.zeros(
                self.num_template_views, 3, self.template_size[0], self.template_size[1]
            )

        # List available template views
        template_files = sorted(template_dir.glob("view_*.png"))

        if len(template_files) == 0:
            template_files = sorted(template_dir.glob("*.png"))

        templates = []
        for i in range(self.num_template_views):
            if i < len(template_files):
                template = self._load_image(template_files[i])
                template = TF.resize(template, self.template_size, antialias=True)
            else:
                # Repeat last template or use zeros
                if len(templates) > 0:
                    template = templates[-1].clone()
                else:
                    template = torch.zeros(3, self.template_size[0], self.template_size[1])
            templates.append(template)

        return torch.stack(templates)

    def _empty_sample(self) -> Dict[str, torch.Tensor]:
        """Return empty sample when no objects found."""
        return {
            "image": torch.zeros(3, self.image_size[0], self.image_size[1]),
            "templates": torch.zeros(
                self.max_classes,
                self.max_templates,
                3,
                self.template_size[0],
                self.template_size[1],
            ),
            "masks": torch.zeros(self.max_classes, self.image_size[0], self.image_size[1]),
        }


class GSOSegmentationDataset(GSODataset):
    """GSO dataset optimized for segmentation with more objects per sample."""

    def __init__(
        self,
        data_root: str,
        image_size: Tuple[int, int] = (480, 640),
        template_size: Tuple[int, int] = (128, 128),
        max_classes: int = 8,
        max_templates: int = 12,
        num_template_views: int = 8,
        split: str = "train",
        transform=None,
    ):
        super().__init__(
            data_root=data_root,
            image_size=image_size,
            template_size=template_size,
            max_classes=max_classes,
            max_templates=max_templates,
            num_template_views=num_template_views,
            split=split,
            transform=transform,
        )
