"""Diffusion trainer."""

import torch
import torch.nn as nn
from tqdm import tqdm
from .base_trainer import BaseTrainer
from .diffusion_modules import GraphDDPM


class DiffusionTrainer(BaseTrainer):
    """Trainer for diffusion models."""

    def __init__(self, model: nn.Module, config: dict, device: str = 'cuda'):
        super().__init__(model, config, device)

        # Create DDPM wrapper
        self.ddpm = GraphDDPM(
            denoiser=model,
            num_steps=config.get('num_diffusion_steps', 1000),
            beta_schedule=config.get('beta_schedule', 'cosine'),
            device=device
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {self.epoch}"):
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Get clean positions
            if hasattr(batch, 'node_types') or 'inst' in batch:
                x_0 = batch['inst'].pos
            else:
                x_0 = batch.pos

            # Sample timesteps
            batch_size = x_0.shape[0]
            t = torch.randint(0, self.ddpm.num_steps, (batch_size,), device=self.device)

            # Compute loss
            loss = self.ddpm.compute_loss(batch, x_0, t)

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

                if hasattr(batch, 'node_types') or 'inst' in batch:
                    x_0 = batch['inst'].pos
                else:
                    x_0 = batch.pos

                batch_size = x_0.shape[0]
                t = torch.randint(0, self.ddpm.num_steps, (batch_size,), device=self.device)

                loss = self.ddpm.compute_loss(batch, x_0, t)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
