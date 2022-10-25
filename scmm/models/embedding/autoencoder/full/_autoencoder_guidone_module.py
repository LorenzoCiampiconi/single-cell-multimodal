# -*- coding: utf-8 -*-

import torch
import torchmetrics
from torch import distributions as td
from torch import nn

from scmm.models.embedding.autoencoder.full.variational import ConditionalVAE
from scmm.models.embedding.autoencoder.decoder.base import MainDecoder
from scmm.models.embedding.autoencoder.encoder.gaussian import GaussianEncoder
from scmm.models.embedding.autoencoder.utils import init_fc_snn


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
