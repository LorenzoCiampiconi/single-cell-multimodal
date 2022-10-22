from typing import Any

import torch
from torch import nn, distributions as td

from single_cell_multimodal_core.models.embedding.autoencoder.base import AutoEncoder


class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, *, lr: float, kl_coef: float, encoder: nn.Module, decoder: nn.Module):
        super().__init__(lr=lr, encoder=encoder, decoder=decoder)
        self.kl_coef = kl_coef
        self.save_hyperparameters()

    def main_training(self, batch):
        (x,), (y,) = batch
        q, z = self.encoder.forward_dist(x)
        y_hat = self.decoder(z)

        recon_loss = self.recon_loss(y_hat, y)

        prior = td.normal.Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))
        kl_loss = td.kl_divergence(q, prior).mean()

        return (q, z, y_hat), (recon_loss, kl_loss)

    def training_step(self, batch, batch_idx):
        _, (recon_loss, kl_loss) = self.main_training(batch)

        self.log("recon_loss/train", recon_loss, on_step=True, on_epoch=True)
        self.log("kl_loss/train", kl_loss, on_step=True, on_epoch=True)

        loss = recon_loss + self.kl_coef * kl_loss
        self.log("loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        self.log_dict(
            self.recon_metrics(y_hat.detach().cpu(), y.detach().cpu()),
            on_step=True,
            on_epoch=True,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self(*batch)


class ConditionalVAE(VariationalAutoEncoder):
    def forward(self, x, ctx):
        z = self.encoder(x, ctx)
        y_hat = self.decoder(z)
        return y_hat

    def main_training(self, batch):
        (x, ctx), (y,) = batch
        x, ctx, y = batch
        q, z = self.encoder.forward_dist(x, ctx)
        y_hat = self.decoder(z)

        recon_loss = self.recon_loss(y_hat, y)

        prior = td.normal.Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))
        kl_loss = td.kl_divergence(q, prior).mean()

        return (q, z, y_hat), (recon_loss, kl_loss)

    def validation_step(self, batch, batch_idx):
        x, ctx, y = batch
        y_hat = self.forward(x, ctx)

        self.log_dict(
            self.recon_metrics(y_hat.detach().cpu(), y.detach().cpu()),
            on_step=True,
            on_epoch=True,
        )
