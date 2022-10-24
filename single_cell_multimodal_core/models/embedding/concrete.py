from single_cell_multimodal_core.models.embedding.autoencoder.trainer import AutoEncoderTrainer
from single_cell_multimodal_core.models.embedding.base import Embedder


class AutoEncoderEmbedder(AutoEncoderTrainer, Embedder):
    ...