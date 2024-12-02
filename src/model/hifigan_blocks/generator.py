from typing import List, Tuple

import torch
from torch import nn

from src.model.hifigan_blocks.conv_block import Conv1dBlock


class ResBlock(nn.Module):
    def __init__(self, kernel_size: int, dilations: List[int], n_channels: int):
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
                    activation=nn.LeakyReLU(),
                    norm=nn.Identity(),
                    pre_activation=True,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convs(x)


class MRF(nn.Module):
    def __init__(
        self,
        n_channels: int,
        blocks_kernels: List[int] = [3, 7, 11],
        blocks_dilations: List[List[int]] = [[1, 3, 5] * 3],
    ):
        super().__init__()
        self.n_channels = n_channels
        self.blocks_kernels = blocks_kernels
        self.blocks_dilations = blocks_dilations

        self.blocks = nn.ModuleList(
            [
                ResBlock(
                    kernel_size=kernel, dilations=dilations, n_channels=self.n_channels
                )
                for kernel, dilations in zip(self.blocks_kernels, self.blocks_dilations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        for block in self.blocks:
            result = result + block(x)
        return result


class HiFiGanGenerator(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 512,
        upsampling_kernels: List[int] = [16, 16, 4, 4],
        mrf_kernels: List[int] = [3, 7, 11],
        mrf_dilations: List[List[Tuple[int, int]]] = [[1, 3, 5] * 3],
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.upsampling_kernels = upsampling_kernels

        self.in_conv = nn.Conv1d(
            in_channels=80,
            out_channels=self.hidden_channels,
            kernel_size=7,
            padding=3,
        )

        self.upsampling = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.ConvTranspose1d(
                        in_channels=self.hidden_channels // (2**i),
                        out_channels=self.hidden_channels // (2 ** (i + 1)),
                        kernel_size=kernel,
                        stride=kernel // 2,
                        padding=(kernel - kernel // 2) // 2,
                    ),
                    MRF(
                        n_channels=self.hidden_channels // (2 ** (i + 1)),
                        blocks_kernels=mrf_kernels,
                        blocks_dilations=mrf_dilations,
                    ),
                )
                for i, kernel in enumerate(self.upsampling_kernels)
            ]
        )

        self.out_conv = Conv1dBlock(
            in_channels=self.hidden_channels // (2 ** len(self.upsampling_kernels)),
            out_channels=1,
            kernel_size=7,
            padding=3,
            norm=nn.Identity(),
            activation=nn.LeakyReLU(),
            pre_activation=True,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spectrogram (torch.Tensor): input spectrogram. Shape: [B, T, F]
        Outputs:
            audio (torch.Tensor): generated audio. Shape: [B, Tnew]
        """
        x = self.in_conv(spectrogram)  # [B, C_hidden, T]
        x = self.upsampling(x)  # [B, C_final, T_new]

        x = self.out_conv(x)
        print(x.shape)
        audio = torch.nn.functional.tanh(x)
        audio = audio.reshape(audio.shape[0], -1)
        return audio
