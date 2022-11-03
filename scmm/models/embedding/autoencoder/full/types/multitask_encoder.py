# -*- coding: utf-8 -*-

import torch
import torchmetrics
from torch import distributions as td
from torch import nn
from scmm.models.embedding.autoencoder.full.types.basic import BasicAutoEncoder


class MultiTaskEncoder(BasicAutoEncoder):
    def __init__(self, *, lr: float, input_coef: float, encoder: nn.Module, decoder: nn.Module):
        super().__init__(lr=lr, encoder=encoder, decoder=decoder)
        self.input_coef = input_coef

        self.recon_feat_loss = nn.HuberLoss()
        self.recon_feat_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanSquaredError(),
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.CosineSimilarity(reduction="mean"),
                torchmetrics.MeanAbsolutePercentageError(),
                torchmetrics.PearsonCorrCoef(),
            ],
            prefix="recon_feat/val",
        )
        self.save_hyperparameters(ignore=["encoder", 'decoder'])

    def forward(self, x):
        z = self.encoder(x)
        y_hat, feat_hat = self.decoder(z)
        return y_hat, feat_hat

    def main_training(self, batch):
        x, (y, feat) = batch
        z = self.encoder.forward(x)
        y_hat, feat_hat = self.decoder(z)

        input_loss = self.recon_loss(y_hat, y)
        feat_loss = self.recon_feat_loss(feat_hat, feat)

        return (z, y_hat), (input_loss, feat_loss)

    def training_step(self, batch, batch_idx):
        _, (input_loss, feat_loss) = self.main_training(batch)

        self.log("recon_loss/input", input_loss, on_step=True, on_epoch=True)
        self.log("recon_loss/feat", feat_loss, on_step=True, on_epoch=True)

        loss = self.input_coef * input_loss + feat_loss
        self.log("loss", loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, (y, feat) = batch
        y_hat, feat_hat = self.forward(x)

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
