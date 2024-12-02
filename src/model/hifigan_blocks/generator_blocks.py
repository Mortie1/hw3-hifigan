from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from src.model.hifigan_blocks.conv_block import Conv1dBlock


class ResBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        dilations: List[int],
        n_channels: int,
        activation: nn.Module = nn.LeakyReLU(negative_slope=0.1),
        norm: Optional[Callable] = nn.utils.parametrizations.weight_norm,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dilations = dilations

        self.convs = nn.Sequential(
            *[
                Conv1dBlock(
                    in_channels=self.n_channels,
                    out_channels=self.n_channels,
                    kernel_size=kernel_size,
                    padding=dilation * (kernel_size - 1) // 2,
                    dilation=dilation,
                    activation=activation,
                    norm=norm,
                    pre_activation=True,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.convs(x)
        return out


class MRF(nn.Module):
    def __init__(
        self,
        n_channels: int,
        blocks_kernels: List[int] = [3, 7, 11],
        blocks_dilations: List[List[int]] = [[1, 3, 5] * 3],
        activation: nn.Module = nn.LeakyReLU(negative_slope=0.1),
        norm: Optional[Callable] = nn.utils.parametrizations.weight_norm,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.blocks_kernels = blocks_kernels
        self.blocks_dilations = blocks_dilations

        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    kernel_size=kernel,
                    dilations=dilations,
                    n_channels=self.n_channels,
                    activation=activation,
                    norm=norm,
                )
                for kernel, dilations in zip(self.blocks_kernels, self.blocks_dilations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for block in self.blocks:
            result = result + block(x)
        result = result / len(self.blocks)
        return result
