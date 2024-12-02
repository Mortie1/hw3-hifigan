import torch
from torch import nn


class HiFiGAN(nn.Module):
    def __init__(self):
        super().__init__()

        # if self.training:

    def forward(self, spectrogram: torch.Tensor, audio: torch.Tensor, **batch):
        raise NotImplementedError

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
