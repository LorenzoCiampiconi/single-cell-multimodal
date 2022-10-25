from scmm.models.embedding.autoencoder.base import AutoEncoder
from scmm.models.embedding.encoder.base import FullyConnectedEncoder

from torch import nn


input_dim = 2000
latent_dim = 64
activation_function = nn.SELU
shrinking_factors = (8, 2)
hidden_dim = input_dim // shrinking_factors[0]
lr = 1e-3

autoencoder = AutoEncoder(
    encoder=FullyConnectedEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        fully_connected_sequential=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, hidden_dim // (shrinking_factors[1])),
            activation_function(),
            nn.Linear(hidden_dim // shrinking_factors[1], latent_dim),
            activation_function(),
        ),
    ),
    lr=lr,
)
