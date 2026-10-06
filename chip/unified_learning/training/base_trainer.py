"""Base trainer interface."""

import torch
import torch.nn as nn
from typing import Dict, Any
from pathlib import Path


class BaseTrainer:
    """Base trainer class for all training tasks."""

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        lr = config.get('learning_rate', 0.001)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 0.0))

        # Scheduler
        if config.get('use_scheduler', True):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        else:
            self.scheduler = None

        self.epoch = 0
        self.best_loss = float('inf')

    def train_epoch(self, dataloader):
        """Train for one epoch - to be implemented by subclasses."""
        raise NotImplementedError

    def validate(self, dataloader):
        """Validate - to be implemented by subclasses."""
        raise NotImplementedError

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
