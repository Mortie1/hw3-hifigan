from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
                entry["text"] = None
                entry["audio_len"] = 0
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
