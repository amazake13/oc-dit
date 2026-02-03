"""Objaverse dataset loader for OC-DiT training."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.io as io
import torchvision.transforms.v2.functional as TF

from data.dataset import BaseSegmentationDataset


class ObjaverseDataset(BaseSegmentationDataset):
    """Dataset loader for Objaverse rendered data.

    Objaverse contains 2,600+ meshes with diverse object categories,
    used for generalization to diverse object classes.

    Expected directory structure:
        data_root/
        ├── images/
        │   ├── scene_000000.png
        │   └── ...
        ├── masks/
        │   ├── scene_000000/
        │   │   ├── obj_0.png
        │   │   └── ...
        │   └── ...
        ├── templates/
        │   ├── object_uid_0/
        │   │   ├── view_0.png
        │   │   └── ...
        │   └── ...
        └── annotations.json
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
        category_filter: Optional[List[str]] = None,
        transform=None,
    ):
        """Initialize Objaverse dataset.

        Args:
            data_root: Path to dataset root directory
            image_size: Target image size (H, W)
            template_size: Target template size (h, w)
            max_classes: Maximum number of classes per sample
            max_templates: Maximum number of template views
            num_template_views: Number of template views to load per class
            split: Dataset split ('train', 'val', 'test')
            category_filter: Optional list of categories to include
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
        self.category_filter = category_filter

        # Load annotations
        annotations_path = self.data_root / "annotations.json"
        if not annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)

        self.samples = self.annotations.get("samples", [])
        self.object_metadata = self.annotations.get("objects", {})

        # Filter by split
        if "splits" in self.annotations:
            split_ids = set(self.annotations["splits"].get(split, []))
            self.samples = [s for s in self.samples if s["scene_id"] in split_ids]

        # Filter by category
        if category_filter is not None:
            category_set = set(category_filter)
            filtered_samples = []
            for sample in self.samples:
                sample_objects = sample.get("objects", [])
                has_category = any(
                    self.object_metadata.get(obj["object_uid"], {}).get("category")
                    in category_set
                    for obj in sample_objects
                )
                if has_category:
                    filtered_samples.append(sample)
            self.samples = filtered_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        scene_id = sample_info["scene_id"]
        objects = sample_info.get("objects", [])

        # Load image
        image_path = self.data_root / "images" / f"{scene_id}.png"
        image = self._load_image(image_path)

        # Load masks and templates
        masks_list = []
        templates_list = []

        for i, obj in enumerate(objects[: self.max_classes]):
            object_uid = obj["object_uid"]

            # Load mask
            mask_path = self.data_root / "masks" / scene_id / f"obj_{i}.png"
            mask = self._load_mask(mask_path)
            masks_list.append(mask)

            # Load templates
            template_views = self._load_templates(object_uid)
            templates_list.append(template_views)

        num_classes = len(masks_list)

        if num_classes == 0:
            return self._empty_sample()

        masks = torch.stack(masks_list)
        templates = torch.stack(templates_list)

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
            return torch.zeros(3, self.image_size[0], self.image_size[1])

        image = io.read_image(str(path))
        image = image.float() / 255.0

        if image.shape[0] == 4:
            image = image[:3]

        return image

    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load binary mask."""
        if not path.exists():
            return torch.zeros(self.image_size)

        mask = io.read_image(str(path))

        if mask.shape[0] > 1:
            mask = mask[0:1]
        mask = (mask > 127).float().squeeze(0)

        return mask

    def _load_templates(self, object_uid: str) -> torch.Tensor:
        """Load template views for an object."""
        template_dir = self.data_root / "templates" / object_uid

        if not template_dir.exists():
            return torch.zeros(
                self.num_template_views, 3, self.template_size[0], self.template_size[1]
            )

        template_files = sorted(template_dir.glob("view_*.png"))
        if len(template_files) == 0:
            template_files = sorted(template_dir.glob("*.png"))

        templates = []
        for i in range(self.num_template_views):
            if i < len(template_files):
                template = self._load_image(template_files[i])
                template = TF.resize(template, self.template_size, antialias=True)
            else:
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

    def get_categories(self) -> List[str]:
        """Get list of all object categories in dataset."""
        categories = set()
        for obj_data in self.object_metadata.values():
            if "category" in obj_data:
                categories.add(obj_data["category"])
        return sorted(categories)
