from typing import List

import nltk
import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from torch import nn

from src.transforms.spectrogram import MelSpectrogramConfig


class FairSeqSpectrogramGenerator(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/fastspeech2-en-ljspeech",
        fp16: bool = False,
        device: str = "cuda",
        melspec_pad_value: float = MelSpectrogramConfig.pad_value,
    ):
        super().__init__()
        nltk.download("averaged_perceptron_tagger_eng")

        self.pad_value = melspec_pad_value

        models, self.cfg, self.task = load_model_ensemble_and_task_from_hf_hub(
            model_name,
            arg_overrides={"fp16": fp16},
        )
        self.device = device
        self.spec_model = models[0]
        self.spec_model.to(device)
        TTSHubInterface.update_cfg_with_data_cfg(self.cfg, self.task.data_cfg)
        self.generator = self.task.build_generator([self.spec_model], self.cfg)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """
        outputs = []
        for text in texts:
            sample = TTSHubInterface.get_model_input(self.task, text)
            sample["net_input"]["src_tokens"] = sample["net_input"]["src_tokens"].to(
                self.device
            )
            sample["net_input"]["src_lengths"] = sample["net_input"]["src_lengths"].to(
                self.device
            )
            sample["speaker"] = (
                sample["speaker"].to(self.device)
                if sample["speaker"] is not None
                else torch.tensor([[0]]).to(self.device)
            )

            outputs += [
                self.generator.generate(self.spec_model, sample)[0][
                    "feature"
                ].transpose(0, 1)
            ]
        mels = torch.nn.utils.rnn.pad_sequence(
            outputs, batch_first=True, padding_value=self.pad_value
        )
        return mels
