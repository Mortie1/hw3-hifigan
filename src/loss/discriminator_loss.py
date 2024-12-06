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
        mpd_loss = 0
        msd_loss = 0
        for feature_gen_msd, feature_true_msd, feature_gen_mpd, feature_true_mpd in zip(
            msd_generated_features,
            msd_target_features,
            mpd_generated_features,
            mpd_target_features,
        ):
            mpd_loss = (
                mpd_loss
                + self.loss(feature_true_mpd[-1], torch.ones_like(feature_true_mpd[-1]))
                + self.loss(feature_gen_mpd[-1], torch.zeros_like(feature_gen_mpd[-1]))
            )

            msd_loss = (
                msd_loss
                + self.loss(feature_true_msd[-1], torch.ones_like(feature_true_msd[-1]))
                + self.loss(feature_gen_msd[-1], torch.zeros_like(feature_gen_msd[-1]))
            )

        return {
            "discriminator_loss": msd_loss + mpd_loss,
            "mpd_loss": mpd_loss,
            "msd_loss": msd_loss,
        }
