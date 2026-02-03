"""Main training script for OC-DiT."""

import argparse
import copy
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocdit.models.ocdit import OCDiT
from data.dataset import SyntheticDataset, collate_fn
from data.transforms import TrainTransform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EMAModel:
    """Exponential Moving Average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """Initialize EMA model.

        Args:
            model: Model to track
            decay: EMA decay factor
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict["decay"]


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
):
    """Create cosine learning rate schedule with warmup.

    Args:
        optimizer: Optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        Learning rate scheduler
    """

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    config: Dict,
    ema: Optional[EMAModel] = None,
):
    """Save training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": config,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()

    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    ema: Optional[EMAModel] = None,
) -> int:
    """Load training checkpoint.

    Returns:
        Training step from checkpoint
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])

    logger.info(f"Loaded checkpoint from {path}, step {checkpoint['step']}")
    return checkpoint["step"]


def create_dataloader(config: Dict, transform=None) -> DataLoader:
    """Create DataLoader based on config."""
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    # For testing, use synthetic dataset
    dataset_type = data_config.get("dataset", "synthetic")

    if dataset_type == "synthetic":
        dataset = SyntheticDataset(
            num_samples=data_config.get("num_samples", 100),
            image_size=tuple(model_config.get("image_size", [480, 640])),
            template_size=tuple(model_config.get("template_size", [128, 128])),
            max_classes=data_config.get("max_classes", 8),
            max_templates=data_config.get("max_templates", 12),
            num_classes_per_sample=data_config.get("num_classes_per_sample", 4),
            num_templates_per_class=data_config.get("num_templates_per_class", 4),
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=data_config.get("batch_size", 8),
        shuffle=True,
        num_workers=data_config.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=config.get("device") == "cuda",
        drop_last=True,
    )

    return dataloader


def train(config: Dict):
    """Main training loop."""
    # Setup device
    device = torch.device(config.get("device", "cpu"))
    logger.info(f"Using device: {device}")

    # Set seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Create model
    model_config = config.get("model", {})
    model = OCDiT(
        image_size=tuple(model_config.get("image_size", [480, 640])),
        template_size=tuple(model_config.get("template_size", [128, 128])),
        embed_dim=model_config.get("embed_dim", 1024),
        depth=model_config.get("depth", 12),
        num_heads=model_config.get("num_heads", 16),
    )
    model = model.to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Freeze VAE and feature extractor
    for param in model.vae.parameters():
        param.requires_grad = False
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create dataloader
    transform = TrainTransform(
        image_size=tuple(model_config.get("image_size", [480, 640])),
        template_size=tuple(model_config.get("template_size", [128, 128])),
    )
    dataloader = create_dataloader(config, transform=None)  # Transform in dataset

    # Training config
    train_config = config.get("training", {})
    max_steps = train_config.get("max_steps", 400000)
    lr = train_config.get("learning_rate", 1e-4)
    weight_decay = train_config.get("weight_decay", 0.0)
    betas = tuple(train_config.get("betas", [0.9, 0.999]))
    warmup_steps = train_config.get("warmup_steps", 1000)
    gradient_clip_norm = train_config.get("gradient_clip_norm", 1.0)
    log_interval = train_config.get("log_interval", 100)
    save_interval = train_config.get("save_interval", 5000)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    # EMA
    ema_config = config.get("ema", {})
    ema = None
    if ema_config.get("enabled", False):
        ema = EMAModel(model, decay=ema_config.get("decay", 0.9999))
        logger.info(f"EMA enabled with decay {ema.decay}")

    # Mixed precision
    use_amp = config.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info(f"Mixed precision: {use_amp}")

    # Checkpoint directory
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_step = 0
    resume_path = config.get("resume_from")
    if resume_path and os.path.exists(resume_path):
        start_step = load_checkpoint(resume_path, model, optimizer, scheduler, ema)

    # Training loop
    model.train()
    step = start_step
    data_iter = iter(dataloader)

    logger.info(f"Starting training from step {start_step}")

    while step < max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        templates = batch["templates"].to(device)
        masks = batch["masks"].to(device)

        # Forward pass
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                loss_dict = model.forward_loss(images, templates, masks)
                loss = loss_dict["loss"].mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model.forward_loss(images, templates, masks)
            loss = loss_dict["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        scheduler.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        step += 1

        # Logging
        if step % log_interval == 0:
            lr_current = scheduler.get_last_lr()[0]
            loss_masks = loss_dict["loss_masks"].mean().item()
            logger.info(
                f"Step {step}/{max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"Loss_masks: {loss_masks:.4f} | "
                f"LR: {lr_current:.2e}"
            )

        # Save checkpoint
        if step % save_interval == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_{step:08d}.pt",
                model,
                optimizer,
                scheduler,
                step,
                config,
                ema,
            )

    # Save final checkpoint
    save_checkpoint(
        checkpoint_dir / "checkpoint_final.pt",
        model,
        optimizer,
        scheduler,
        step,
        config,
        ema,
    )

    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train OC-DiT model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override device if specified
    if args.device:
        config["device"] = args.device

    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Config: {config}")

    # Run training
    train(config)


if __name__ == "__main__":
    main()
