import logging

from torch import nn

from scmm.models.embedding.autoencoder.nn import FullyConnectedMixin, NNEntity
from scmm.models.embedding.autoencoder.utils import init_fc_snn

logger = logging.Logger(__name__)


class FullyConnectedDecoder(FullyConnectedMixin, NNEntity):
    def __init__(self, *, latent_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim
        self.sanitize_fc()
        self.reset_parameters()

    def validate_input_sequential(self, fully_connected_sequential) -> bool:
        return fully_connected_sequential[0].in_features == self.latent_dim

    def _build_fallback_fully_connected(self, shrinking_factors=(8, 2)):
        hidden_dim = self.latent_dim // shrinking_factors[-1]
        return nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            self.activation_function(),
            nn.Linear(hidden_dim, hidden_dim * shrinking_factors[-2]),
            self.activation_function(),
            nn.Linear(hidden_dim * shrinking_factors[-2], self.latent_dim),
            self.activation_function(),
        )
