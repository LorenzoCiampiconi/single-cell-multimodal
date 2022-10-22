import logging

from torch import nn

from single_cell_multimodal_core.models.embedding.utils import init_fc_snn

logger = logging.Logger(__name__)

class FullyConnectedDecoder(nn.Module):
    def __init__(self, *, latent_dim: int, fully_connected_sequential):
        super().__init__()

        self.latent_dim = latent_dim

        if fully_connected_sequential is None or not self.validate_input_sequential(fully_connected_sequential):  # todo
            logger.debug(
                'loading passed by argument sequential is not verified, a fallback fully connected layer will be built.')
            fully_connected_sequential = self._build_fully_connected()
        else:
            logger.debug('loading passed by argument sequential is verified and loaded.')

        self._fc = fully_connected_sequential
        self.reset_parameters()

    def validate_input_sequential(self, fully_connected_sequential) -> bool:
        return True #todo

    def reset_parameters(self):
        self.apply(init_fc_snn)

    def forward(self, x):
        return self._fc(x)

    def _build_fallback_fully_connected(self, shrinking_factors=(8,2)):
        hidden_dim = self.latent_dim // shrinking_factors[-1]
        return nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            self.activation_function(),
            nn.Linear(hidden_dim, hidden_dim * shrinking_factors[-2]),
            self.activation_function(),
            nn.Linear(hidden_dim * shrinking_factors[-2], self.latent_dim),
            self.activation_function(),
        )
