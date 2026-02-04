"""Logging utilities for training and evaluation."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


class BaseLogger(ABC):
    """Base class for logging."""

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalar values."""
        pass

    @abstractmethod
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image."""
        pass

    @abstractmethod
    def close(self):
        """Close the logger."""
        pass


class TensorBoardLogger(BaseLogger):
    """TensorBoard logger."""

    def __init__(self, log_dir: str):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        self.writer.add_scalars(tag, values, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log image tensor.

        Args:
            tag: Image tag
            image: Image tensor (C, H, W) or (H, W)
            step: Training step
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)
        self.writer.add_image(tag, image, step)

    def log_images(self, tag: str, images: torch.Tensor, step: int):
        """Log multiple images as grid.

        Args:
            tag: Image tag
            images: Image tensor (N, C, H, W)
            step: Training step
        """
        from torchvision.utils import make_grid

        grid = make_grid(images, normalize=True)
        self.writer.add_image(tag, grid, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        self.writer.add_histogram(tag, values, step)

    def close(self):
        self.writer.close()


class WandBLogger(BaseLogger):
    """Weights & Biases logger."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        resume: bool = False,
    ):
        """Initialize WandB logger.

        Args:
            project: WandB project name
            name: Run name
            config: Configuration dictionary
            resume: Whether to resume from previous run
        """
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            resume="allow" if resume else None,
        )

    def log_scalar(self, tag: str, value: float, step: int):
        self.wandb.log({tag: value}, step=step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        prefixed = {f"{tag}/{k}": v for k, v in values.items()}
        self.wandb.log(prefixed, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log image tensor.

        Args:
            tag: Image tag
            image: Image tensor (C, H, W) or (H, W)
            step: Training step
        """
        if image.dim() == 2:
            image = image.unsqueeze(0)
        # Convert to numpy and transpose to HWC
        img_np = image.permute(1, 2, 0).cpu().numpy()
        self.wandb.log({tag: self.wandb.Image(img_np)}, step=step)

    def log_images(self, tag: str, images: torch.Tensor, step: int):
        """Log multiple images.

        Args:
            tag: Image tag
            images: Image tensor (N, C, H, W)
            step: Training step
        """
        imgs = []
        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            imgs.append(self.wandb.Image(img_np))
        self.wandb.log({tag: imgs}, step=step)

    def close(self):
        self.wandb.finish()


class CompositeLogger(BaseLogger):
    """Composite logger that logs to multiple backends."""

    def __init__(self, loggers: list):
        """Initialize composite logger.

        Args:
            loggers: List of logger instances
        """
        self.loggers = loggers

    def log_scalar(self, tag: str, value: float, step: int):
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        for logger in self.loggers:
            logger.log_scalars(tag, values, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        for logger in self.loggers:
            logger.log_image(tag, image, step)

    def close(self):
        for logger in self.loggers:
            logger.close()


class DummyLogger(BaseLogger):
    """Dummy logger that does nothing (for testing)."""

    def log_scalar(self, tag: str, value: float, step: int):
        pass

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        pass

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        pass

    def close(self):
        pass


def create_logger(config: Dict) -> BaseLogger:
    """Create logger based on config.

    Args:
        config: Configuration dictionary with logging settings

    Returns:
        Logger instance
    """
    loggers = []

    # TensorBoard
    tensorboard_config = config.get("tensorboard", {})
    if tensorboard_config.get("enabled", False):
        log_dir = tensorboard_config.get("log_dir", "logs/tensorboard")
        loggers.append(TensorBoardLogger(log_dir))

    # WandB
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", False):
        loggers.append(
            WandBLogger(
                project=wandb_config.get("project", "oc-dit"),
                name=wandb_config.get("name"),
                config=config,
            )
        )

    if len(loggers) == 0:
        return DummyLogger()
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return CompositeLogger(loggers)
