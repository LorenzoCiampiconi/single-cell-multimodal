# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import torch
import xarray as xr
from scipy.sparse import csr_array
from torch.utils.data import Dataset


def as_numpy(t: torch.Tensor) -> np.array:
    return t.detach().cpu().numpy()


def as_tensor(arr: np.array) -> torch.Tensor:
    return torch.tensor(arr).float()


class BaseDataset(Dataset):
    def __init__(self, mat: csr_array) -> None:
        self.mat = mat
        self.mat.sort_indices()
        self.shape = self.mat.shape

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index) -> torch.Tensor:
        arr = self.mat.getrow(index).toarray().ravel()
        return as_tensor(arr)


class IODataset(BaseDataset):
    def __init__(self, mat: csr_array, ds: xr.Dataset, output_vars: List[str]) -> None:
        super().__init__(mat)
        self.ds = ds
        self.output_vars = output_vars

    def __getitem__(self, index) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        t_input = super().__getitem__(index)

        outputs = [t_input]
        for var in self.output_vars:
            arr = self.ds[var]
            t = as_tensor(arr.data)
            outputs.append(t)

        return t_input, outputs
