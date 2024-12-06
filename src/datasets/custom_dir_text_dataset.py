from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class CustomDirTextDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        data = []
        transcription_dir = Path(data_dir) / "transcriptions"
        for path in Path(transcription_dir).iterdir():
            entry = {"path": path, "text": None}
            transc_path = Path(transcription_dir) / path
            if transc_path.exists():
                with transc_path.open() as f:
                    entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
