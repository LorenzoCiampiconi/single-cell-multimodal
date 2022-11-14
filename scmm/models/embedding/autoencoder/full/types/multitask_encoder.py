# -*- coding: utf-8 -*-

import torchmetrics
from torch import nn
from scmm.models.embedding.autoencoder.full.types.basic import BasicAutoEncoder


class MultiTaskEncoder(BasicAutoEncoder):
    def __init__(
        self,
        *,
        lr: float,
        input_coef: float,
        encoder: nn.Module,
        decoder: nn.Module,
        loss: nn.Module,
        feat_loss: nn.Module,
    ):
        super().__init__(lr=lr, encoder=encoder, decoder=decoder, loss=loss)
        self.input_coef = input_coef

        self.recon_feat_loss = feat_loss
        self.recon_feat_metrics = torchmetrics.MetricCollection(
            [
                torchmetrics.MeanSquaredError(),
                torchmetrics.MeanAbsoluteError(),
                torchmetrics.CosineSimilarity(reduction="mean"),
                torchmetrics.MeanAbsolutePercentageError(),
            ],
            prefix="recon_feat/val/",
        )

    def save_filtered_hyperparameters(self):
        self.ignored_hparams = ["encoder", "decoder", "loss", "feat_loss"]
        self.save_hyperparameters(ignore=self.ignored_hparams)

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

        self.recon_metrics(y_hat, y)
        self.recon_feat_metrics(feat_hat, feat)

        return (z, y_hat, feat_hat), (input_loss, feat_loss)

    def training_step(self, batch, batch_idx):
        _, (input_loss, feat_loss) = self.main_training(batch)

        if isinstance(input_loss, dict):
            self.log_dict({f"recon_loss/input/{k}": v for k, v in input_loss.items()}, on_step=True, on_epoch=True)
            input_loss = sum(input_loss.values())
        self.log("recon_loss/input", input_loss, on_step=True, on_epoch=True)

        if isinstance(feat_loss, dict):
            self.log_dict({f"recon_loss/feat/{k}": v for k, v in feat_loss.items()}, on_step=True, on_epoch=True)
            feat_loss = sum(feat_loss.values())
        self.log("recon_loss/feat", feat_loss, on_step=True, on_epoch=True)

        self.log_dict(self.recon_metrics, on_step=True, on_epoch=True)
        self.log_dict(self.recon_feat_metrics, on_step=True, on_epoch=True)

        loss = self.input_coef * input_loss + feat_loss
        self.log("loss", loss, on_step=True, on_epoch=True)

        return loss
