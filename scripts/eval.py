"""Evaluation script for OC-DiT model."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from ocdit.models.ocdit import OCDiT
from data.dataset import SyntheticDataset, collate_fn
from data.transforms import EvalTransform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Compute segmentation metrics: IoU, Precision, Recall, F1."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        """Initialize metrics.

        Args:
            threshold: Threshold for binary prediction
            eps: Small value to avoid division by zero
        """
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.total_intersection = 0
        self.total_union = 0
        self.num_samples = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metrics with predictions and targets.

        Args:
            pred: Predicted masks (B, num_classes, H, W) or (B, H, W)
            target: Ground truth masks (B, num_classes, H, W) or (B, H, W)
        """
        # Binarize predictions
        pred_binary = (pred > self.threshold).float()
        target_binary = (target > 0.5).float()

        # Flatten spatial dimensions
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)

        # Compute TP, FP, FN, TN
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()

        # Compute intersection and union for IoU
        intersection = tp
        union = tp + fp + fn

        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn
        self.total_tn += tn
        self.total_intersection += intersection
        self.total_union += union
        self.num_samples += pred.shape[0]

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary with IoU, Precision, Recall, F1 scores
        """
        # IoU (Jaccard Index)
        iou = self.total_intersection / (self.total_union + self.eps)

        # Precision = TP / (TP + FP)
        precision = self.total_tp / (self.total_tp + self.total_fp + self.eps)

        # Recall = TP / (TP + FN)
        recall = self.total_tp / (self.total_tp + self.total_fn + self.eps)

        # F1 = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall + self.eps)

        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        total = self.total_tp + self.total_tn + self.total_fp + self.total_fn
        accuracy = (self.total_tp + self.total_tn) / (total + self.eps)

        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "num_samples": self.num_samples,
        }


def compute_per_class_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> Dict[str, torch.Tensor]:
    """Compute per-class segmentation metrics.

    Args:
        pred: Predicted masks (B, num_classes, H, W)
        target: Ground truth masks (B, num_classes, H, W)
        threshold: Threshold for binary prediction
        eps: Small value to avoid division by zero

    Returns:
        Dictionary with per-class metrics
    """
    batch_size, num_classes = pred.shape[:2]

    pred_binary = (pred > threshold).float()
    target_binary = (target > 0.5).float()

    # Compute metrics per class
    ious = []
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        pred_c = pred_binary[:, c].view(batch_size, -1)
        target_c = target_binary[:, c].view(batch_size, -1)

        # Skip empty classes
        if target_c.sum() == 0:
            continue

        intersection = (pred_c * target_c).sum(dim=1)
        union = pred_c.sum(dim=1) + target_c.sum(dim=1) - intersection

        iou = (intersection / (union + eps)).mean()
        ious.append(iou)

        tp = (pred_c * target_c).sum(dim=1)
        fp = (pred_c * (1 - target_c)).sum(dim=1)
        fn = ((1 - pred_c) * target_c).sum(dim=1)

        precision = (tp / (tp + fp + eps)).mean()
        recall = (tp / (tp + fn + eps)).mean()
        f1 = (2 * tp / (2 * tp + fp + fn + eps)).mean()

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    if len(ious) == 0:
        return {
            "mean_iou": torch.tensor(0.0),
            "mean_precision": torch.tensor(0.0),
            "mean_recall": torch.tensor(0.0),
            "mean_f1": torch.tensor(0.0),
        }

    return {
        "mean_iou": torch.stack(ious).mean(),
        "mean_precision": torch.stack(precisions).mean(),
        "mean_recall": torch.stack(recalls).mean(),
        "mean_f1": torch.stack(f1s).mean(),
    }


def load_checkpoint(
    path: str,
    model: nn.Module,
    use_ema: bool = True,
) -> None:
    """Load model checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        use_ema: Whether to use EMA weights if available
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

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

    logger.info(f"Loaded checkpoint from {path}")


def create_dataloader(config: Dict, transform=None) -> DataLoader:
    """Create evaluation DataLoader."""
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    dataset_type = data_config.get("dataset", "synthetic")

    if dataset_type == "synthetic":
        dataset = SyntheticDataset(
            num_samples=data_config.get("num_samples", 100),
            image_size=tuple(model_config.get("image_size", [480, 640])),
            template_size=tuple(model_config.get("template_size", [128, 128])),
            max_classes=data_config.get("max_classes", 8),
            max_templates=data_config.get("max_templates", 12),
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=data_config.get("batch_size", 4),
        shuffle=False,
        num_workers=data_config.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=config.get("device") == "cuda",
    )

    return dataloader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ensemble_size: int = 1,
) -> Dict[str, float]:
    """Run evaluation on dataset.

    Args:
        model: Model to evaluate
        dataloader: Evaluation dataloader
        device: Device to run on
        ensemble_size: Number of ensemble samples for prediction

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    metrics = SegmentationMetrics()

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        templates = batch["templates"].to(device)
        masks_gt = batch["masks"].to(device)

        # Generate predictions
        pred_masks = model.generate_segmentations(
            images,
            templates,
            ensemble_size=ensemble_size,
        )

        # Update metrics
        metrics.update(pred_masks, masks_gt)

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Evaluated {batch_idx + 1}/{len(dataloader)} batches")

    return metrics.compute()


def main():
    parser = argparse.ArgumentParser(description="Evaluate OC-DiT model")
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
        "--device",
        type=str,
        default=None,
        help="Override device from config",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="Ensemble size for prediction",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use EMA weights from checkpoint",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.device:
        config["device"] = args.device

    device = torch.device(config.get("device", "cpu"))
    logger.info(f"Using device: {device}")

    # Create model
    model_config = config.get("model", {})
    model = OCDiT(
        image_size=tuple(model_config.get("image_size", [480, 640])),
        template_size=tuple(model_config.get("template_size", [128, 128])),
        embed_dim=model_config.get("embed_dim", 1024),
        depth=model_config.get("depth", 12),
        num_heads=model_config.get("num_heads", 16),
    )

    # Load checkpoint
    load_checkpoint(args.checkpoint, model, use_ema=args.use_ema)
    model = model.to(device)

    # Create dataloader
    transform = EvalTransform(
        image_size=tuple(model_config.get("image_size", [480, 640])),
        template_size=tuple(model_config.get("template_size", [128, 128])),
    )
    dataloader = create_dataloader(config, transform=None)

    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluate(
        model,
        dataloader,
        device,
        ensemble_size=args.ensemble_size,
    )

    # Print results
    logger.info("=" * 50)
    logger.info("Evaluation Results:")
    logger.info(f"  IoU:       {results['iou']:.4f}")
    logger.info(f"  Precision: {results['precision']:.4f}")
    logger.info(f"  Recall:    {results['recall']:.4f}")
    logger.info(f"  F1:        {results['f1']:.4f}")
    logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
    logger.info(f"  Samples:   {results['num_samples']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
