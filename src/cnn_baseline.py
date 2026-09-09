"""Baseline B2: CNN on mel-spectrograms -- no graph, no text (spec Section 8).

The point of comparison for Task 2: the CNN sees the same audio through the same
cached log-mel features and the same splits, so any difference against the GNN
is attributable to the structural representation rather than to the data or the
training budget.

Spectrograms are cropped to a fixed width (random crop while training, centre
crop at evaluation) because a convolutional stack needs a fixed input, whereas
the graph encoder handles variable-length tracks natively -- worth one sentence
in the report.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from audio_features import cache_file


class MelSpectrogramDataset(Dataset):
    """Fixed-width log-mel crops read from the Task 2 audio cache."""

    def __init__(self, frame, cache_dir: str, n_frames: int = 256, train: bool = True, seed: int = 42):
        self.cache_dir = cache_dir
        self.n_frames = n_frames
        self.train = train
        self.rng = np.random.default_rng(seed)

        # Only keep tracks whose features were successfully cached, so the CNN and
        # the GNN are scored on an identical set of tracks.
        self.rows = [
            row for row in frame.itertuples(index=False)
            if os.path.exists(cache_file(cache_dir, str(row.track_id)))
        ]
        missing = len(frame) - len(self.rows)
        if missing:
            print(f"[cnn] {missing} tracks missing from the audio cache and excluded")

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, mel: np.ndarray) -> np.ndarray:
        n_mels, total = mel.shape
        if total < self.n_frames:  # pad short tracks by wrapping
            reps = int(np.ceil(self.n_frames / total))
            mel = np.tile(mel, (1, reps))
            total = mel.shape[1]
        start = (
            int(self.rng.integers(0, total - self.n_frames + 1))
            if self.train
            else (total - self.n_frames) // 2
        )
        return mel[:, start : start + self.n_frames]

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        with np.load(cache_file(self.cache_dir, str(row.track_id))) as z:
            mel = z["log_mel"]
        crop = self._crop(mel).astype(np.float32)
        return {
            "x": torch.from_numpy(crop).unsqueeze(0),   # (1, n_mels, n_frames)
            "y": torch.tensor(int(row.label_idx), dtype=torch.long),
        }


class MelCNN(nn.Module):
    """Four-block VGG-style CNN over log-mel input, global-pooled to a classifier."""

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.3,
        in_channels: int = 1,
    ):
        super().__init__()
        blocks, dim = [], in_channels
        for out_dim in channels:
            blocks += [
                nn.Conv2d(dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            dim = out_dim

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(x))
