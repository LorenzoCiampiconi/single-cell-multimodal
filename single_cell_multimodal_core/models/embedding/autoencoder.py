# -*- coding: utf-8 -*-

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import distributions as td
from torch import nn
from torch.optim import lr_scheduler


@torch.no_grad()
def init_fc_snn(layer):
    if not isinstance(layer, nn.Linear) or isinstance(layer, nn.LazyLinear):
        return
    nn.init.kaiming_normal_(layer.weight, nonlinearity="linear")
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(layer.bias, -bound, bound)


class Encoder(nn.Module):
    def __init__(self, *, input_length: int, latent_dim: int):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        self.conv_model = nn.Sequential(
            nn.Conv1d(1, 4, 5, stride=2),
            nn.SELU(),
            nn.Conv1d(4, 4, 3, stride=2),
            nn.SELU(),
            nn.Flatten(),
        )
        self.fc_model = nn.Sequential(
            nn.Linear(252, 40),
            nn.SELU(),
            nn.Linear(40, 20),
            nn.SELU(),
            nn.Linear(20, self.latent_dim),
            nn.SELU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_fc_snn)

    def forward(self, x):
        out = self.conv_model(x)
        return self.fc_model(out)

    @property
    def example_input_array(self):
        return torch.randn((1, 1, self.input_length))


class GaussianEncoder(Encoder):
    def __init__(self, *, input_length: int, latent_dim: int):
        super().__init__(input_length=input_length, latent_dim=latent_dim)
        self.fc_model = self.fc_model[:-2]

        self.fc_mu = nn.Sequential(
            nn.Linear(20, self.latent_dim),
            nn.SELU(),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(20, self.latent_dim),
            nn.SELU(),
        )

        self.reset_parameters()

    def forward_backbone(self, x):
        out = self.conv_model(x)
        out = self.fc_model(out)
        return out

    def forward(self, x):
        h = self.forward_backbone(x)
        mu = self.fc_mu(h)
        return mu

    def forward_dist(self, x):
        h = self.forward_backbone(x)
        return self.forward_sample(h)

    def forward_sample(self, h):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        q = td.Normal(mu, torch.exp(4 * logvar))
        z = q.rsample()

        return q, z


class GaussianCtxEncoder(GaussianEncoder):
    def __init__(self, *, input_length: int, latent_dim: int, ctx_dim: int):
        super().__init__(input_length=input_length, latent_dim=latent_dim)
        self.ctx_dim = ctx_dim

        self.fc_model = self.fc_model[:-2]
        self.ctx_model = nn.Sequential(
            nn.Linear(40 + self.ctx_dim, 20),
            nn.SELU(),
        )

        self.reset_parameters()

    def forward_backbone(self, x, ctx):
        out = self.conv_model(x)
        out = self.fc_model(out)
        out = torch.cat([out, ctx], dim=-1)
        out = self.ctx_model(out)
        return out

    def forward(self, x, ctx):
        h = self.forward_backbone(x, ctx)
        mu = self.fc_mu(h)
        return mu

    def forward_dist(self, x, ctx):
        h = self.forward_backbone(x, ctx)
        return self.forward_sample(h)

    @property
    def example_input_array(self):
        return torch.randn((1, 1, self.input_length)), torch.randn((1, self.ctx_dim))


class MainDecoder(nn.Module):
    def __init__(self, *, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc_model = nn.Sequential(
            nn.Linear(self.latent_dim, 20),
            nn.SELU(),
            nn.Linear(20, 40),
            nn.SELU(),
            nn.Linear(40, 252),
            nn.SELU(),
        )
        self.conv_model = nn.Sequential(
            nn.Unflatten(1, (4, 63)),
            nn.ConvTranspose1d(4, 4, 3, stride=2),
            nn.SELU(),
            nn.ConvTranspose1d(4, 1, 5, stride=2),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_fc_snn)

    def forward(self, x):
        out = self.fc_model(x)
        return self.conv_model(out)


class ExtraFeatureDecoder(nn.Module):
    def __init__(self, *, latent_dim: int, output_length: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_length = output_length
        self.fc_model = nn.Sequential(
            nn.Linear(self.latent_dim, self.output_length),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_fc_snn)

    def forward(self, x):
        return self.fc_model(x)


class JointDecoder(nn.Module):
    def __init__(self, *, latent_dim: int, output_dim: int):
        super().__init__()
        self.feature = ExtraFeatureDecoder(latent_dim, output_dim)
        self.spectrum = MainDecoder(latent_dim)

    def forward(self, x):
        return self.spectrum(x), self.feature(x)


class AutoEncoder(pl.LightningModule):
    def __init__(self, *, lr: float, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

        self.recon_loss = nn.SmoothL1Loss(beta=2e-1)
        self.recon_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanSquaredError(),
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.CosineSimilarity(reduction="mean"),
            ],
            prefix="recon_loss/val",
        )
        self.save_hyperparameters()

    def on_fit_start(self):
        self.recon_metrics = self.recon_metrics.cpu()

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat

    def training_step(self, batch, batch_idx):
        (x,), (y,) = batch
        y_hat = self.forward(x)

        loss = self.recon_loss(y_hat, y)
        self.log("loss/train", loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], [
            {
                "scheduler": lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    3,
                    2,
                    eta_min=1e-3 * self.lr,
                ),
                "interval": "step",  # epoch, step
                "frequency": 1,
            },
        ]

    @property
    def example_input_array(self):
        return self.encoder.example_input_array


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


class ConditionalMultiTaskEncoder(ConditionalVAE):
    def __init__(self, *, lr: float, kl_coef: float, input_coef: float, encoder: nn.Module, decoder: nn.Module):
        super().__init__(lr=lr, kl_coef=kl_coef, encoder=encoder, decoder=decoder)
        self.input_coef = input_coef

        self.recon_feat_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanSquaredError(),
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.CosineSimilarity(reduction="mean"),
            ],
            prefix="recon_feat/val",
        )
        self.save_hyperparameters()

    def forward(self, x, ctx):
        z = self.encoder(x, ctx)
        y_hat, feat_hat = self.decoder(z)
        return y_hat, feat_hat

    def main_training(self, batch):
        (x, ctx), (y, feat) = batch
        q, z = self.encoder.forward_dist(x, ctx)
        y_hat, feat_hat = self.decoder(z)

        input_loss = self.recon_loss(y_hat, y)
        feat_loss = self.recon_loss(feat_hat, feat)
        recon_loss = input_loss, feat_loss

        prior = td.normal.Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))
        kl_loss = td.kl_divergence(q, prior).mean()

        return (q, z, y_hat), (recon_loss, kl_loss)

    def training_step(self, batch, batch_idx):
        _, ((spt_loss, feat_loss), kl_loss) = self.main_training(batch)

        self.log("recon_loss/spectrum", spt_loss, on_step=True, on_epoch=True)
        self.log("recon_loss/feat", feat_loss, on_step=True, on_epoch=True)

        recon_loss = self.input_coef * spt_loss + feat_loss
        self.log("recon_loss/train", recon_loss, on_step=True, on_epoch=True)
        self.log("kl_loss/train", kl_loss, on_step=True, on_epoch=True)

        loss = recon_loss + self.kl_coef * kl_loss
        self.log("loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        (x, ctx), (y, feat) = batch
        y_hat, feat_hat = self.forward(x, ctx)

        self.log_dict(
            self.recon_metrics(y_hat.detach().cpu(), y.detach().cpu()),
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            self.recon_feat_metrics(feat_hat.detach().cpu(), feat.detach().cpu()),
            on_step=True,
            on_epoch=True,
        )
