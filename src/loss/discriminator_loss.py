import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(
        self,
        msd_target_features,
        mpd_target_features,
        msd_generated_features,
        mpd_generated_features,
        **batch,
    ):
        for feature_gen_msd, feature_true_msd, feature_gen_mpd, feature_true_mpd in zip(
            msd_generated_features[-1],
            msd_target_features[-1],
            mpd_generated_features[-1],
            mpd_target_features[-1],
        ):
            mpd_loss = self.loss(
                feature_true_mpd, torch.ones_like(feature_true_mpd)
            ) + self.loss(feature_gen_mpd, torch.zeros_like(feature_gen_mpd))

            msd_loss = self.loss(
                feature_true_msd, torch.ones_like(feature_true_msd)
            ) + self.loss(feature_gen_msd, torch.zeros_like(feature_gen_msd))

        return {"discriminator_loss": msd_loss + mpd_loss}
