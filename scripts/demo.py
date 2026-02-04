"""Interactive demo script for OC-DiT inference."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.io as io
import torchvision.transforms.v2.functional as TF
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocdit.models.ocdit import OCDiT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
    use_ema: bool = True,
) -> OCDiT:
    """Load pre-trained OC-DiT model.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to load model on
        use_ema: Whether to use EMA weights

    Returns:
        Loaded model
    """
    model_config = config.get("model", {})
    model = OCDiT(
        image_size=tuple(model_config.get("image_size", [480, 640])),
        template_size=tuple(model_config.get("template_size", [128, 128])),
        embed_dim=model_config.get("embed_dim", 1024),
        depth=model_config.get("depth", 12),
        num_heads=model_config.get("num_heads", 16),
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if use_ema and "ema" in checkpoint:
        logger.info("Loading EMA weights")
        ema_state = checkpoint["ema"]["shadow"]
        model_state = model.state_dict()
        for name, param in ema_state.items():
            if name in model_state:
                model_state[name] = param
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from {checkpoint_path}")
    return model


def load_image(
    image_path: str,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Load and preprocess image.

    Args:
        image_path: Path to image file
        target_size: Target size (H, W)

    Returns:
        Preprocessed image tensor (1, 3, H, W)
    """
    image = io.read_image(image_path)
    image = image.float() / 255.0

    # Handle RGBA
    if image.shape[0] == 4:
        image = image[:3]

    # Resize
    image = TF.resize(image, target_size, antialias=True)

    return image.unsqueeze(0)


def load_templates(
    template_paths: List[str],
    template_size: Tuple[int, int],
    max_classes: int = 8,
    max_templates: int = 12,
) -> torch.Tensor:
    """Load template images.

    Args:
        template_paths: List of paths to template directories or images
        template_size: Target template size (h, w)
        max_classes: Maximum number of classes
        max_templates: Maximum number of templates per class

    Returns:
        Template tensor (1, num_classes, num_templates, 3, h, w)
    """
    templates_list = []

    for class_path in template_paths[:max_classes]:
        class_path = Path(class_path)

        if class_path.is_dir():
            # Load multiple views from directory
            view_files = sorted(class_path.glob("*.png")) + sorted(class_path.glob("*.jpg"))
            views = []
            for view_file in view_files[:max_templates]:
                view = io.read_image(str(view_file)).float() / 255.0
                if view.shape[0] == 4:
                    view = view[:3]
                view = TF.resize(view, template_size, antialias=True)
                views.append(view)

            # Pad to max_templates
            while len(views) < max_templates:
                if len(views) > 0:
                    views.append(views[-1].clone())
                else:
                    views.append(torch.zeros(3, template_size[0], template_size[1]))

            templates_list.append(torch.stack(views))
        else:
            # Single image
            view = io.read_image(str(class_path)).float() / 255.0
            if view.shape[0] == 4:
                view = view[:3]
            view = TF.resize(view, template_size, antialias=True)
            views = [view] + [view.clone() for _ in range(max_templates - 1)]
            templates_list.append(torch.stack(views))

    # Pad to max_classes
    while len(templates_list) < max_classes:
        templates_list.append(
            torch.zeros(max_templates, 3, template_size[0], template_size[1])
        )

    templates = torch.stack(templates_list)
    return templates.unsqueeze(0)


def save_masks(
    masks: torch.Tensor,
    output_dir: str,
    threshold: float = 0.5,
) -> List[str]:
    """Save predicted masks to files.

    Args:
        masks: Predicted masks (num_classes, H, W)
        output_dir: Output directory
        threshold: Threshold for binarization

    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, mask in enumerate(masks):
        # Binarize
        mask_binary = (mask > threshold).float()

        # Convert to uint8
        mask_uint8 = (mask_binary * 255).to(torch.uint8)

        # Save
        output_path = output_dir / f"mask_{i:02d}.png"
        io.write_png(mask_uint8.unsqueeze(0), str(output_path))
        saved_paths.append(str(output_path))

        logger.info(f"Saved mask to {output_path}")

    return saved_paths


def save_visualization(
    image: torch.Tensor,
    masks: torch.Tensor,
    output_path: str,
    threshold: float = 0.5,
):
    """Save visualization with mask overlays.

    Args:
        image: Input image (3, H, W)
        masks: Predicted masks (num_classes, H, W)
        output_path: Output file path
        threshold: Threshold for binarization
    """
    from scripts.utils.visualization import create_mask_overlay

    # Color palette
    colors = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.5, 0.0),
        (0.5, 0.0, 1.0),
    ]

    overlay = image.clone()
    for i, mask in enumerate(masks):
        if mask.max() > threshold:
            mask_binary = (mask > threshold).float()
            overlay = create_mask_overlay(
                overlay, mask_binary, colors[i % len(colors)], alpha=0.4
            )

    # Convert to uint8 and save
    overlay_uint8 = (overlay * 255).clamp(0, 255).to(torch.uint8)
    io.write_png(overlay_uint8, output_path)
    logger.info(f"Saved visualization to {output_path}")


@torch.no_grad()
def run_inference(
    model: OCDiT,
    image: torch.Tensor,
    templates: torch.Tensor,
    device: torch.device,
    ensemble_size: int = 3,
) -> torch.Tensor:
    """Run inference on single image.

    Args:
        model: OC-DiT model
        image: Input image (1, 3, H, W)
        templates: Template images (1, num_classes, num_templates, 3, h, w)
        device: Device
        ensemble_size: Number of ensemble samples

    Returns:
        Predicted masks (num_classes, H, W)
    """
    image = image.to(device)
    templates = templates.to(device)

    pred_masks = model.generate_segmentations(
        image,
        templates,
        ensemble_size=ensemble_size,
    )

    return pred_masks[0].cpu()


def main():
    parser = argparse.ArgumentParser(description="OC-DiT inference demo")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--templates",
        type=str,
        nargs="+",
        required=True,
        help="Paths to template images or directories (one per class)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for masks",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="Number of ensemble samples",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights from checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for mask binarization",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization with overlays",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(args.checkpoint, config, device, use_ema=args.use_ema)

    # Get sizes from config
    model_config = config.get("model", {})
    image_size = tuple(model_config.get("image_size", [480, 640]))
    template_size = tuple(model_config.get("template_size", [128, 128]))

    # Load inputs
    image = load_image(args.image, image_size)
    templates = load_templates(
        args.templates,
        template_size,
        max_classes=config.get("data", {}).get("max_classes", 8),
        max_templates=config.get("data", {}).get("max_templates", 12),
    )

    logger.info(f"Image shape: {image.shape}")
    logger.info(f"Templates shape: {templates.shape}")

    # Run inference
    logger.info("Running inference...")
    pred_masks = run_inference(
        model,
        image,
        templates,
        device,
        ensemble_size=args.ensemble_size,
    )

    logger.info(f"Predicted masks shape: {pred_masks.shape}")

    # Save outputs
    output_dir = Path(args.output_dir)
    save_masks(pred_masks, output_dir / "masks", threshold=args.threshold)

    if args.visualize:
        save_visualization(
            image[0],
            pred_masks,
            str(output_dir / "visualization.png"),
            threshold=args.threshold,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
