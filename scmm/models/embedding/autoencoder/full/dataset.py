# -*- coding: utf-8 -*-

from typing import List, Tuple

from numpy.typing import NDArray
import torch
import xarray as xr
from torch.utils.data import Dataset


def as_numpy(t: torch.Tensor) -> NDArray:
    return t.detach().cpu().numpy()


def as_tensor(arr: NDArray) -> torch.Tensor:
    return torch.tensor(arr).float()


class BaseDataset(Dataset):
    def __init__(self, mat: NDArray) -> None:
        self.mat = mat
        self.shape = self.mat.shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index) -> torch.Tensor:
        arr = self.mat[index, :].ravel()
        return as_tensor(arr)


class IODataset(BaseDataset):
    def __init__(self, mat: NDArray, outmat: NDArray) -> None:
        super().__init__(mat)
        self.outmat = outmat

    def __getitem__(self, index) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        t_input = super().__getitem__(index)
        arr = self.outmat[index, :].ravel()
        t = as_tensor(arr)

        outputs = [t_input, t]

        return t_input, outputs
