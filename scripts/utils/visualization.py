"""Visualization utilities for OC-DiT."""

from typing import List, Optional, Tuple

import torch


def create_mask_overlay(
    image: torch.Tensor,
    mask: torch.Tensor,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.5,
) -> torch.Tensor:
    """Create image with mask overlay.

    Args:
        image: RGB image tensor (3, H, W)
        mask: Binary mask tensor (H, W)
        color: RGB color for mask overlay
        alpha: Transparency of overlay

    Returns:
        Image with mask overlay (3, H, W)
    """
    overlay = image.clone()

    # Create colored mask
    mask_rgb = torch.zeros_like(image)
    for i, c in enumerate(color):
        mask_rgb[i] = mask * c

    # Blend
    mask_expanded = mask.unsqueeze(0).expand_as(image)
    overlay = torch.where(
        mask_expanded > 0.5,
        alpha * mask_rgb + (1 - alpha) * image,
        image,
    )

    return overlay


def create_comparison_grid(
    images: torch.Tensor,
    pred_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """Create comparison grid of predictions vs ground truth.

    Args:
        images: Batch of images (B, 3, H, W)
        pred_masks: Predicted masks (B, num_classes, H, W)
        gt_masks: Ground truth masks (B, num_classes, H, W)
        num_classes: Number of classes to visualize

    Returns:
        Grid image (3, grid_H, grid_W)
    """
    from torchvision.utils import make_grid

    batch_size = images.shape[0]
    vis_images = []

    # Color palette for different classes
    colors = [
        (1.0, 0.0, 0.0),  # Red
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0),  # Magenta
        (0.0, 1.0, 1.0),  # Cyan
        (1.0, 0.5, 0.0),  # Orange
        (0.5, 0.0, 1.0),  # Purple
    ]

    for b in range(min(batch_size, 4)):
        img = images[b]

        # Original image
        vis_images.append(img)

        # Prediction overlay
        pred_overlay = img.clone()
        for c in range(min(num_classes, pred_masks.shape[1])):
            if pred_masks[b, c].sum() > 0:
                pred_overlay = create_mask_overlay(
                    pred_overlay, pred_masks[b, c], colors[c % len(colors)], alpha=0.4
                )
        vis_images.append(pred_overlay)

        # Ground truth overlay
        gt_overlay = img.clone()
        for c in range(min(num_classes, gt_masks.shape[1])):
            if gt_masks[b, c].sum() > 0:
                gt_overlay = create_mask_overlay(
                    gt_overlay, gt_masks[b, c], colors[c % len(colors)], alpha=0.4
                )
        vis_images.append(gt_overlay)

    grid = make_grid(vis_images, nrow=3, padding=2, normalize=False)
    return grid


def visualize_templates(
    templates: torch.Tensor,
    max_classes: int = 4,
    max_views: int = 4,
) -> torch.Tensor:
    """Visualize template images.

    Args:
        templates: Template tensor (num_classes, num_views, 3, h, w)
        max_classes: Maximum number of classes to show
        max_views: Maximum number of views per class

    Returns:
        Grid image (3, grid_H, grid_W)
    """
    from torchvision.utils import make_grid

    num_classes = min(templates.shape[0], max_classes)
    num_views = min(templates.shape[1], max_views)

    # Collect templates
    template_list = []
    for c in range(num_classes):
        for v in range(num_views):
            template_list.append(templates[c, v])

    grid = make_grid(template_list, nrow=num_views, padding=2, normalize=False)
    return grid


def visualize_denoising_steps(
    steps: List[torch.Tensor],
    num_steps_to_show: int = 5,
) -> torch.Tensor:
    """Visualize intermediate denoising steps.

    Args:
        steps: List of intermediate mask predictions
        num_steps_to_show: Number of steps to visualize

    Returns:
        Grid image showing denoising progression
    """
    from torchvision.utils import make_grid

    total_steps = len(steps)
    if total_steps <= num_steps_to_show:
        indices = list(range(total_steps))
    else:
        indices = [int(i * (total_steps - 1) / (num_steps_to_show - 1)) for i in range(num_steps_to_show)]

    selected_steps = [steps[i] for i in indices]

    # Normalize to [0, 1]
    normalized = []
    for step in selected_steps:
        step_norm = (step - step.min()) / (step.max() - step.min() + 1e-7)
        if step_norm.dim() == 2:
            step_norm = step_norm.unsqueeze(0).repeat(3, 1, 1)
        elif step_norm.dim() == 3 and step_norm.shape[0] == 1:
            step_norm = step_norm.repeat(3, 1, 1)
        normalized.append(step_norm)

    grid = make_grid(normalized, nrow=num_steps_to_show, padding=2)
    return grid
