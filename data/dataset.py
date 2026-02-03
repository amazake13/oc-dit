"""Base dataset class for OC-DiT training."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


class BaseSegmentationDataset(Dataset, ABC):
    """Base dataset class for instance segmentation.

    Each sample should contain:
        - image: (3, H, W) RGB image tensor
        - templates: (num_classes, num_views, 3, h, w) template crops
        - masks: (num_classes, H, W) binary instance masks
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        template_size: Tuple[int, int],
        max_classes: int = 8,
        max_templates: int = 12,
        transform=None,
    ):
        """Initialize dataset.

        Args:
            image_size: Target image size (H, W)
            template_size: Target template size (h, w)
            max_classes: Maximum number of object classes per sample
            max_templates: Maximum number of template views per class
            transform: Optional transforms to apply
        """
        self.image_size = image_size
        self.template_size = template_size
        self.max_classes = max_classes
        self.max_templates = max_templates
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset length."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample.

        Returns:
            Dictionary with keys:
                - image: (3, H, W) RGB image
                - templates: (num_classes, num_views, 3, h, w) templates
                - masks: (num_classes, H, W) binary masks
        """
        pass

    def _pad_to_max(
        self,
        templates: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad templates and masks to max_classes and max_templates.

        Args:
            templates: (num_classes, num_views, 3, h, w)
            masks: (num_classes, H, W)

        Returns:
            Padded templates and masks
        """
        num_classes, num_views = templates.shape[:2]

        # Pad templates
        if num_classes < self.max_classes or num_views < self.max_templates:
            padded_templates = torch.zeros(
                self.max_classes,
                self.max_templates,
                3,
                self.template_size[0],
                self.template_size[1],
            )
            padded_templates[:num_classes, :num_views] = templates
            templates = padded_templates

        # Pad masks
        if num_classes < self.max_classes:
            h, w = masks.shape[1:]
            padded_masks = torch.zeros(self.max_classes, h, w)
            padded_masks[:num_classes] = masks
            masks = padded_masks

        return templates, masks


class SyntheticDataset(BaseSegmentationDataset):
    """Synthetic dataset for testing the training loop.

    Generates random images, templates, and masks for validating
    the training pipeline without real data.
    """

    def __init__(
        self,
        num_samples: int = 100,
        image_size: Tuple[int, int] = (480, 640),
        template_size: Tuple[int, int] = (128, 128),
        max_classes: int = 8,
        max_templates: int = 12,
        num_classes_per_sample: int = 4,
        num_templates_per_class: int = 4,
        transform=None,
    ):
        """Initialize synthetic dataset.

        Args:
            num_samples: Number of synthetic samples to generate
            image_size: Target image size (H, W)
            template_size: Target template size (h, w)
            max_classes: Maximum number of classes (for padding)
            max_templates: Maximum number of templates (for padding)
            num_classes_per_sample: Actual number of classes per sample
            num_templates_per_class: Actual number of templates per class
            transform: Optional transforms
        """
        super().__init__(
            image_size=image_size,
            template_size=template_size,
            max_classes=max_classes,
            max_templates=max_templates,
            transform=transform,
        )
        self.num_samples = num_samples
        self.num_classes_per_sample = min(num_classes_per_sample, max_classes)
        self.num_templates_per_class = min(num_templates_per_class, max_templates)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Set seed for reproducibility per sample
        torch.manual_seed(idx)

        # Generate random RGB image
        image = torch.rand(3, self.image_size[0], self.image_size[1])

        # Generate random templates
        templates = torch.rand(
            self.num_classes_per_sample,
            self.num_templates_per_class,
            3,
            self.template_size[0],
            self.template_size[1],
        )

        # Generate random binary masks (non-overlapping regions)
        masks = torch.zeros(
            self.num_classes_per_sample,
            self.image_size[0],
            self.image_size[1],
        )

        # Create simple non-overlapping rectangular masks
        h, w = self.image_size
        for i in range(self.num_classes_per_sample):
            # Random rectangle
            y1 = torch.randint(0, h // 2, (1,)).item()
            x1 = torch.randint(0, w // 2, (1,)).item()
            y2 = torch.randint(y1 + 20, min(y1 + h // 4, h), (1,)).item()
            x2 = torch.randint(x1 + 20, min(x1 + w // 4, w), (1,)).item()
            masks[i, y1:y2, x1:x2] = 1.0

        # Apply transforms if any
        if self.transform is not None:
            sample = self.transform({
                "image": image,
                "templates": templates,
                "masks": masks,
            })
            image = sample["image"]
            templates = sample["templates"]
            masks = sample["masks"]

        # Pad to max sizes
        templates, masks = self._pad_to_max(templates, masks)

        return {
            "image": image,
            "templates": templates,
            "masks": masks,
        }


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """Custom collate function for segmentation batches.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with stacked tensors
    """
    images = torch.stack([s["image"] for s in batch])
    templates = torch.stack([s["templates"] for s in batch])
    masks = torch.stack([s["masks"] for s in batch])

    return {
        "image": images,
        "templates": templates,
        "masks": masks,
    }
