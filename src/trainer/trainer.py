from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

        # start = time.time()
        with torch.autocast(
            device_type=self.device,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        ):
            outputs = self.model(**batch)
            batch.update(outputs)

            discriminator_outputs = self.discriminator(**batch, detach_generated=True)
            batch.update(discriminator_outputs)

            all_discriminator_losses = self.discriminator_criterion(**batch)
            batch.update(all_discriminator_losses)

        if self.is_train:
            batch["discriminator_loss"].backward()
            self._clip_grad_norm()
            self.discriminator_optimizer.step()
            if self.discriminator_lr_scheduler is not None:
                self.discriminator_lr_scheduler.step()

        with torch.autocast(
            device_type=self.device,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        ):
            discriminator_outputs = self.discriminator(**batch, detach_generated=False)
            batch.update(discriminator_outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, output_audio, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        output_spectrogram = MelSpectrogram(MelSpectrogramConfig())(
            output_audio.detach().cpu()
        )[0]
        image = plot_spectrogram(spectrogram_for_plot)
        output_image = plot_spectrogram(output_spectrogram)
        self.writer.add_image("spectrogram", image)
        self.writer.add_image("output_spectrogram", output_image)

    def log_audio(self, audio, name):
        def _normalize_audio(audio: torch.Tensor):
            audio /= torch.max(torch.abs(audio))
            return audio.detach().cpu()

        audio = _normalize_audio(audio)
        self.writer.add_audio(
            name,
            audio.float(),
            sample_rate=self.config.writer.audio_sample_rate,
        )

    def log_predictions(self, output_audio, audio, examples_to_log=1, **batch):
        # columns = ["output audio", "ground truth"]
        # data = [
        #     [
        #         self.writer.convert_audio(output_audio[i, :]),
        #         self.writer.convert_audio(audio[i, :]),
        #     ]
        #     for i in range(examples_to_log)
        # ]

        # self.writer.add_wb_table("audios", columns=columns, data=data)
        for i in range(examples_to_log):
            self.log_audio(output_audio[i, :], f"output_audio_{i}")
            self.log_audio(audio[i, :], f"target_audio_{i}")
