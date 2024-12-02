import torch
from torch import nn

from src.transforms import MelSpectrogram, MelSpectrogramConfig


class FMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        loss = 0
        for dscr_features_gen, dscr_features_true in zip(x, y):
            for feature_gen, feature_true in zip(dscr_features_gen, dscr_features_true):
                loss = loss + self.loss(feature_gen, feature_true)

        return loss


class GeneratorLoss(nn.Module):
    def __init__(
        self,
        alpha_fm: float = 2.0,
        alpha_mel: float = 45.0,
        melspec_config: MelSpectrogramConfig = None,
    ):
        super().__init__()
        self.adv_loss = nn.MSELoss()
        self.melspec_loss = nn.L1Loss()
        self.fm_loss = FMLoss()

        self.melspec_config = (
            melspec_config if melspec_config is not None else MelSpectrogramConfig()
        )
        self.melspec = MelSpectrogram(self.melspec_config)

        self.alpha_fm = alpha_fm
        self.alpha_mel = alpha_mel

    def forward(
        self,
        msd_target_features,
        mpd_target_features,
        msd_generated_features,
        mpd_generated_features,
        output_audio,
        audio,
        spectrogram,
        **batch,
    ):
        adv_loss = self.adv_loss(output_audio, torch.ones_like(output_audio))
        melspec_loss = self.melspec_loss(self.melspec(output_audio), spectrogram)
        msd_fm_loss = self.fm_loss(msd_target_features, msd_generated_features)
        mpd_fm_loss = self.fm_loss(mpd_target_features, mpd_generated_features)

        loss = (
            adv_loss
            + self.alpha_fm * (msd_fm_loss + mpd_fm_loss)
            + self.alpha_mel * melspec_loss
        )
        return {"loss": loss}
