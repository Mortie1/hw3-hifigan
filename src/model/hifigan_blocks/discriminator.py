from typing import List

import torch
from torch import nn

from src.model.hifigan_blocks.conv_block import Conv1dBlock, Conv2dBlock


class MPDBlock(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.in_conv = Conv2dBlock(
            in_channels=1,
            out_channels=64,
            kernel_size=(5, 1),
            stride=(3, 1),
            padding=(2, 0),
            activation=nn.LeakyReLU(),
            norm=nn.Identity(),
            pre_activation=False,
        )

        self.main_convs = nn.ModuleList(
            [
                Conv2dBlock(
                    in_channels=2 ** (5 + i),
                    out_channels=2 ** (5 + i + 1),
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    padding=(2, 0),
                    norm=nn.Identity(),
                    activation=nn.LeakyReLU(),
                    pre_activation=False,
                )
                for i in range(1, 4)
            ]
            + [
                Conv2dBlock(
                    in_channels=2 ** (5 + 4),
                    out_channels=1024,
                    kernel_size=(5, 1),
                    padding=(2, 0),
                    norm=nn.Identity(),
                    activation=nn.LeakyReLU(),
                    pre_activation=False,
                ),
            ]
        )
        self.out_conv = nn.Conv2d(
            in_channels=1024,
            out_channels=1,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            audio (torch.Tensor): (B, T)

        Returns:
            List[torch.Tensor]: (batch_size, 1, time_steps)
        """
        audio = torch.nn.functional.pad(
            audio, (0, self.period - audio.shape[1] % self.period), mode="reflect"
        )
        x = audio.reshape(audio.shape[0], 1, audio.shape[1] // self.period, self.period)

        features = [self.in_conv(x)]
        for conv in self.main_convs:
            features += [conv(features[-1])]
        features += [self.out_conv(x)]

        return features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [MPDBlock(period=i) for i in [2, 3, 5, 7, 11]]
        )

    def forward(self, audio: torch.Tensor) -> List[List[torch.Tensor]]:
        features = []
        for discriminator in self.discriminators:
            features += [discriminator(audio)]
        return features


class MSDBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_conv = Conv1dBlock(
            kernel_size=15,
            stride=1,
            in_channels=1,
            out_channels=16,
            padding=7,
            activation=nn.LeakyReLU(),
            norm=nn.Identity(),
            pre_activation=False,
        )
        self.downsample_convs = nn.ModuleList(
            [
                Conv1dBlock(
                    kernel_size=41,
                    stride=4,
                    in_channels=16 * (4**i),
                    out_channels=16 * (4 ** (i + 1)),
                    padding=20,
                    groups=4 ** (i + 1),
                    activation=nn.LeakyReLU(),
                    norm=nn.Identity(),
                    pre_activation=False,
                )
                for i in range(3)
            ]
            + [
                Conv1dBlock(
                    kernel_size=41,
                    stride=4,
                    in_channels=1024,
                    out_channels=1024,
                    padding=20,
                    groups=256,
                    activation=nn.LeakyReLU(),
                    norm=nn.Identity(),
                    pre_activation=False,
                )
            ]
        )
        self.out_conv1 = Conv1dBlock(
            in_channels=1024,
            out_channels=1024,
            kernel_size=5,
            stride=1,
            padding=2,
            norm=nn.Identity(),
            activation=nn.LeakyReLU(),
            pre_activation=False,
        )
        self.out_conv2 = Conv1dBlock(
            in_channels=1024,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=2,
            norm=nn.Identity(),
            activation=nn.Identity(),
            pre_activation=False,
        )

    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        features = [self.in_conv(audio)]
        for downsample_conv in self.downsample_convs:
            features += [downsample_conv(features[-1])]
        features += [self.out_conv1(features[-1])]
        features += [self.out_conv2(features[-1])]
        return features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([MSDBlock() for _ in range(3)])

        self.avg_pools = nn.ModuleList(
            [nn.Identity()]
            + [nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(2)]
        )

    def forward(self, audio: torch.Tensor) -> List[List[torch.Tensor]]:
        features = []
        for discriminator, avg_pool in zip(self.discriminators, self.avg_pools):
            features += [discriminator(avg_pool(audio))]
        return features
