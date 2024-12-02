from typing import Callable, Optional, Tuple

import torch
from torch import nn


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        activation: nn.Module = nn.LeakyReLU(),
        norm: Optional[Callable] = None,
        groups: int = 1,
        bias=True,
        pre_activation: bool = False,
    ):
        super().__init__()
        if norm is None:
            norm = nn.Identity()
        self.conv_block = nn.Sequential(
            activation if pre_activation else nn.Identity(),
            norm(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    groups=groups,
                    bias=bias,
                )
            ),
            nn.Identity() if pre_activation else activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        dilation: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        activation: nn.Module = nn.LeakyReLU(),
        norm: Optional[Callable] = None,
        groups: int = 1,
        bias=True,
        pre_activation: bool = False,
    ):
        super().__init__()
        if norm is None:
            norm = nn.Identity()
        self.conv_block = nn.Sequential(
            activation if pre_activation else nn.Identity(),
            norm(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    groups=groups,
                    bias=bias,
                )
            ),
            nn.Identity() if pre_activation else activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)
