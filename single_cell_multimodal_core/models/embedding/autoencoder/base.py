from typing import Optional

import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.optim import lr_scheduler

from single_cell_multimodal_core.models.embedding.encoder.base import EncoderABC


class AutoEncoder(pl.LightningModule):
    def __init__(self, *, lr: float, encoder: EncoderABC, decoder: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder

        self.decoder = decoder if decoder is not None else encoder.mirror_sequential_for_decoding()
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
        self.save_hyperparameters(ignore=["encoder", "decoder"])

    def on_fit_start(self):
        self.recon_metrics = self.recon_metrics.cpu()

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat

    def training_step(self, batch, batch_idx):
        x = y = batch
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
