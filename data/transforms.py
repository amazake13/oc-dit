"""Data augmentation utilities for OC-DiT training."""

from typing import Dict, Optional, Tuple

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF


class SegmentationTransform:
    """Compose multiple transforms for segmentation data.

    Applies consistent transformations to images, templates, and masks.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        template_size: Tuple[int, int],
        horizontal_flip: bool = True,
        color_jitter: bool = True,
        color_jitter_params: Optional[Dict] = None,
    ):
        """Initialize transform pipeline.

        Args:
            image_size: Target image size (H, W)
            template_size: Target template size (h, w)
            horizontal_flip: Enable random horizontal flip
            color_jitter: Enable color jitter
            color_jitter_params: Color jitter parameters
        """
        self.image_size = image_size
        self.template_size = template_size
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter

        if color_jitter_params is None:
            color_jitter_params = {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            }
        self.color_jitter_params = color_jitter_params

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply transforms to sample.

        Args:
            sample: Dictionary with 'image', 'templates', 'masks'

        Returns:
            Transformed sample
        """
        image = sample["image"]
        templates = sample["templates"]
        masks = sample["masks"]

        # Resize image and masks
        image = TF.resize(image, self.image_size, antialias=True)
        masks = TF.resize(
            masks.unsqueeze(1),
            self.image_size,
            interpolation=TF.InterpolationMode.NEAREST,
        ).squeeze(1)

        # Resize templates
        num_classes, num_views = templates.shape[:2]
        templates = templates.view(-1, *templates.shape[2:])
        templates = TF.resize(templates, self.template_size, antialias=True)
        templates = templates.view(num_classes, num_views, *templates.shape[1:])

        # Random horizontal flip
        if self.horizontal_flip and torch.rand(1).item() > 0.5:
            image = TF.hflip(image)
            masks = TF.hflip(masks)

        # Color jitter on image
        if self.color_jitter:
            image = apply_color_jitter(image, **self.color_jitter_params)

        return {
            "image": image,
            "templates": templates,
            "masks": masks,
        }


class TemplateTransform:
    """Transform for template images with rotation augmentation."""

    def __init__(
        self,
        template_size: Tuple[int, int],
        rotation: bool = True,
        rotation_degrees: float = 15.0,
        color_jitter: bool = True,
    ):
        """Initialize template transform.

        Args:
            template_size: Target template size (h, w)
            rotation: Enable random rotation
            rotation_degrees: Maximum rotation angle
            color_jitter: Enable color jitter
        """
        self.template_size = template_size
        self.rotation = rotation
        self.rotation_degrees = rotation_degrees
        self.color_jitter = color_jitter

    def __call__(self, templates: torch.Tensor) -> torch.Tensor:
        """Apply transforms to templates.

        Args:
            templates: (num_classes, num_views, 3, h, w) tensor

        Returns:
            Transformed templates
        """
        num_classes, num_views = templates.shape[:2]

        # Flatten for processing
        templates = templates.view(-1, *templates.shape[2:])

        # Resize
        templates = TF.resize(templates, self.template_size, antialias=True)

        # Random rotation per template
        if self.rotation:
            rotated = []
            for t in templates:
                angle = (torch.rand(1).item() * 2 - 1) * self.rotation_degrees
                t = TF.rotate(t, angle)
                rotated.append(t)
            templates = torch.stack(rotated)

        # Color jitter
        if self.color_jitter:
            jittered = []
            for t in templates:
                t = apply_color_jitter(t, brightness=0.1, contrast=0.1)
                jittered.append(t)
            templates = torch.stack(jittered)

        # Reshape back
        templates = templates.view(num_classes, num_views, *templates.shape[1:])

        return templates


def apply_color_jitter(
    image: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> torch.Tensor:
    """Apply color jitter to an image tensor.

    Args:
        image: (C, H, W) image tensor
        brightness: Brightness jitter factor
        contrast: Contrast jitter factor
        saturation: Saturation jitter factor
        hue: Hue jitter factor

    Returns:
        Jittered image tensor
    """
    # Random brightness
    if brightness > 0:
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * brightness
        image = TF.adjust_brightness(image, factor)

    # Random contrast
    if contrast > 0:
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * contrast
        image = TF.adjust_contrast(image, factor)

    # Random saturation
    if saturation > 0:
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * saturation
        image = TF.adjust_saturation(image, factor)

    # Random hue
    if hue > 0:
        factor = (torch.rand(1).item() * 2 - 1) * hue
        image = TF.adjust_hue(image, factor)

    return image.clamp(0, 1)


class TrainTransform(SegmentationTransform):
    """Training transform with all augmentations enabled."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        template_size: Tuple[int, int],
    ):
        super().__init__(
            image_size=image_size,
            template_size=template_size,
            horizontal_flip=True,
            color_jitter=True,
        )


class EvalTransform(SegmentationTransform):
    """Evaluation transform with augmentations disabled."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        template_size: Tuple[int, int],
    ):
        super().__init__(
            image_size=image_size,
            template_size=template_size,
            horizontal_flip=False,
            color_jitter=False,
        )
