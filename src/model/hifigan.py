from typing import Dict, List, Tuple

import torch
from torch import nn

from src.model.hifigan_blocks import (
    MRF,
    Conv1dBlock,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)


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

    def forward(self, spectrogram: torch.Tensor, **batch) -> Dict:
        """
        Args:
            spectrogram (torch.Tensor): input spectrogram. Shape: [B, T, F]
        Outputs:
            audio (torch.Tensor): generated audio. Shape: [B, Tnew]
        """
        x = self.in_conv(spectrogram)
        x = self.upsampling(x)

        x = self.out_conv(x)
        print(x.shape)
        audio = torch.nn.functional.tanh(x)
        audio = audio.reshape(audio.shape[0], -1)
        return {"output_audio": audio}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class HiFiGanDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()

    def forward(self, output_audio: torch.Tensor, audio: torch.Tensor, **batch) -> Dict:
        return {
            "msd_target_features": self.msd(audio),
            "mpd_target_features": self.mpd(audio),
            "msd_generated_features": self.msd(output_audio),
            "mpd_generated_features": self.mpd(output_audio),
        }
