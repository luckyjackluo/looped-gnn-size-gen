"""Regression trainer."""

import torch
import torch.nn as nn
from tqdm import tqdm
from .base_trainer import BaseTrainer


class RegressionTrainer(BaseTrainer):
    """Trainer for regression tasks (e.g., shortest path, placement)."""

    def __init__(self, model: nn.Module, config: dict, device: str = 'cuda'):
        super().__init__(model, config, device)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {self.epoch}"):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred, _ = self.model(batch)

            # Get target
            if hasattr(batch, 'node_types') or 'inst' in batch:
                target = batch['inst'].y if hasattr(batch['inst'], 'y') else batch['inst'].pos
            else:
                target = batch.y if hasattr(batch, 'y') else batch.pos

            # Compute loss
            loss = self.criterion(pred, target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)

                pred, _ = self.model(batch)

                if hasattr(batch, 'node_types') or 'inst' in batch:
                    target = batch['inst'].y if hasattr(batch['inst'], 'y') else batch['inst'].pos
                else:
                    target = batch.y if hasattr(batch, 'y') else batch.pos

                loss = self.criterion(pred, target)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
