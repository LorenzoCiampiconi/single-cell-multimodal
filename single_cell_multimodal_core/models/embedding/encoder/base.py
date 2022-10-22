import abc
import logging
from copy import deepcopy

import torch
from torch import nn

from single_cell_multimodal_core.models.embedding.decoder.base import FullyConnectedDecoder
from single_cell_multimodal_core.models.embedding.nn import NNEntity
from single_cell_multimodal_core.models.embedding.utils import init_fc_snn

logger = logging.getLogger(__name__)


class EncoderABC(NNEntity, metaclass=abc.ABCMeta):
    def __init__(self, *, input_dim, latent_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abc.abstractmethod
    def mirror_sequential_for_decoding(self):
        pass

    def validate_input_sequential(self, fully_connected_sequential):
        return True

    @property
    @abc.abstractmethod
    def hashing_parameters(self):
        pass

    @property
    @abc.abstractmethod
    def encoding_hash(self):
        pass


class FullyConnectedEncoder(EncoderABC):
    def __init__(self, *, fully_connected_sequential=None, **kwargs):
        super().__init__(**kwargs)

        self._fc = fully_connected_sequential
        self.reset_parameters()

    def encoding_hash(self):
        hash(str(self.__class__) + self.hashing_parameters)

    def forward(self, x):
        return self._fc(x)

    def reset_parameters(self):
        self.apply(init_fc_snn)

    def _build_fallback_fully_connected(self, shrinking_factors=(8, 2)):
        hidden_dim = self.input_dim // shrinking_factors[0]
        return nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            self.activation_function(),
            nn.Linear(hidden_dim, hidden_dim // (shrinking_factors[1])),
            self.activation_function(),
            nn.Linear(hidden_dim // shrinking_factors[1], self.latent_dim),
            self.activation_function(),
        )

    def _reverse_sequential_layers(self):
        sequential_as_list = list(self._fc)
        sequential_as_list[::2] = reversed(sequential_as_list[::2])
        return sequential_as_list

    def mirror_sequential_for_decoding(self):
        reversed_list_of_layer = self._reverse_sequential_layers()
        new_input_for_fresh_sequential = [deepcopy(e) for e in reversed_list_of_layer]
        mirrored_sequential = nn.Sequential(*new_input_for_fresh_sequential)
        return FullyConnectedDecoder(fully_connected_sequential=mirrored_sequential, latent_dim=self.latent_dim)


# class ConvolutionalEncoder(nn.Module):
#     def __init__(self, *, input_length: int, latent_dim: int):
#         super().__init__()
#         self.input_length = input_length
#         self.latent_dim = latent_dim
#         self.conv_model = nn.Sequential(
#             nn.Conv1d(1, 4, 5, stride=2),
#             nn.SELU(),
#             nn.Conv1d(4, 4, 3, stride=2),
#             nn.SELU(),
#             nn.Flatten(),
#         )
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.apply(init_fc_snn)
#
#     def forward(self, x):
#         out = self.conv_model(x)
#         return self.fc_model(out)
#
#     @property
#     def example_input_array(self):
#         return torch.randn((1, 1, self.input_length))
