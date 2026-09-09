"""Audio front-end for Task 2: mel / chroma / MFCC extraction and segmentation.

Implements the preprocessing pipeline from spec Section 3:

  1. resample to 22,050 Hz
  2. log-mel spectrogram (128 bins) and chroma (12 bins), per-track normalised
  3. split into fixed windows (5-10 s) or beat-synchronous segments via librosa

Frame-level features are computed once per track and then aggregated over
segment frame ranges, so every descriptor shares one hop grid and the segment
boundaries stay exactly consistent across feature families.

Extraction is the slow part of Task 2, so results are cached to
`data/processed/audio_cache/<track_id>.npz` and reused by the graph builder and
the CNN baseline alike.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

try:
    import librosa
except ImportError as exc:  # pragma: no cover - dependency is documented in README
    raise ImportError(
        "librosa is required for Task 2: pip install librosa soundfile"
    ) from exc


@dataclass
class AudioConfig:
    """Mirrors the `task2.audio` block of config.yaml."""

    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 20
    n_chroma: int = 12
    duration: float | None = 30.0        # seconds of audio to read per track
    offset: float = 0.0
    segmentation: str = "fixed"           # fixed | beat
    segment_seconds: float = 5.0
    segment_overlap: float = 0.5          # fraction of a window; 0 => no overlap
    beats_per_segment: int = 8
    min_segments: int = 4                 # tracks yielding fewer are skipped
    normalize: bool = True                # per-track standardisation of mel/chroma

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- #
# loading + frame-level features
# --------------------------------------------------------------------------- #
def load_audio(path: str, cfg: AudioConfig) -> np.ndarray:
    """Mono waveform at cfg.sr. Raises on unreadable files (callers skip them)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(
            path, sr=cfg.sr, mono=True, offset=cfg.offset, duration=cfg.duration
        )
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")
    return y


def frame_features(y: np.ndarray, cfg: AudioConfig) -> dict[str, np.ndarray]:
    """All frame-level descriptors on a shared hop grid, shape (dim, T)."""
    kw = {"n_fft": cfg.n_fft, "hop_length": cfg.hop_length}

    mel_power = librosa.feature.melspectrogram(y=y, sr=cfg.sr, n_mels=cfg.n_mels, **kw)
    log_mel = librosa.power_to_db(mel_power, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=cfg.n_mfcc)
    # CQT-based chroma tracks pitch classes better than STFT chroma, which the
    # chord-transition graph in graph_builder.py depends on.
    chroma = librosa.feature.chroma_cqt(y=y, sr=cfg.sr, hop_length=cfg.hop_length, n_chroma=cfg.n_chroma)
    contrast = librosa.feature.spectral_contrast(y=y, sr=cfg.sr, **kw)
    centroid = librosa.feature.spectral_centroid(y=y, sr=cfg.sr, **kw)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=cfg.sr, **kw)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=cfg.sr, **kw)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)
    rms = librosa.feature.rms(y=y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)

    if cfg.normalize:
        log_mel = _standardize(log_mel)

    n_frames = min(
        x.shape[-1]
        for x in (log_mel, mfcc, chroma, contrast, centroid, bandwidth, rolloff, zcr, rms)
    )
    trim = lambda x: x[..., :n_frames]  # noqa: E731 - chroma_cqt can differ by a frame
    return {
        "log_mel": trim(log_mel),
        "mfcc": trim(mfcc),
        "chroma": trim(chroma),
        "contrast": trim(contrast),
        "scalars": np.vstack([trim(centroid), trim(bandwidth), trim(rolloff), trim(zcr), trim(rms)]),
    }


def _standardize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #
def fixed_boundaries(n_frames: int, cfg: AudioConfig) -> list[tuple[int, int]]:
    """Fixed windows of `segment_seconds`, hopped by (1 - overlap) of a window."""
    win = max(1, int(round(cfg.segment_seconds * cfg.sr / cfg.hop_length)))
    step = max(1, int(round(win * (1.0 - cfg.segment_overlap))))
    bounds = [(s, min(s + win, n_frames)) for s in range(0, max(1, n_frames - win // 2), step)]
    return [(a, b) for a, b in bounds if b - a >= max(2, win // 4)]


def beat_boundaries(y: np.ndarray, n_frames: int, cfg: AudioConfig) -> list[tuple[int, int]]:
    """Beat-synchronous segments: `beats_per_segment` beats per node.

    Falls back to fixed windows when beat tracking finds too few beats, which
    happens on ambient or heavily non-percussive material.
    """
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=cfg.sr, hop_length=cfg.hop_length, units="frames")
    except Exception:  # noqa: BLE001 - beat tracking is best-effort
        beats = np.array([], dtype=int)

    step = max(1, cfg.beats_per_segment)
    if len(beats) < step * (cfg.min_segments + 1):
        return fixed_boundaries(n_frames, cfg)

    edges = list(beats[::step]) + [n_frames]
    bounds = [(int(a), int(min(b, n_frames))) for a, b in zip(edges[:-1], edges[1:])]
    return [(a, b) for a, b in bounds if b - a >= 2]


def segment_boundaries(y: np.ndarray, n_frames: int, cfg: AudioConfig) -> list[tuple[int, int]]:
    if cfg.segmentation == "beat":
        return beat_boundaries(y, n_frames, cfg)
    return fixed_boundaries(n_frames, cfg)


# --------------------------------------------------------------------------- #
# segment-level node features
# --------------------------------------------------------------------------- #
def segment_node_features(
    feats: dict[str, np.ndarray], bounds: list[tuple[int, int]], cfg: AudioConfig
) -> np.ndarray:
    """One feature vector per segment -> the h_i^(0) of spec Algorithm 2, line 2.

    Layout (83 dims for the default config):
        MFCC mean/std (2 x n_mfcc) | chroma mean/std (2 x n_chroma)
        | spectral contrast mean (7) | centroid,bandwidth,rolloff,zcr,rms mean/std (10)
        | relative position in track (1) | segment duration in seconds (1)
    """
    n_frames = feats["mfcc"].shape[1]
    sec_per_frame = cfg.hop_length / cfg.sr
    rows = []

    for start, end in bounds:
        sl = slice(start, end)
        parts = [
            feats["mfcc"][:, sl].mean(axis=1),
            feats["mfcc"][:, sl].std(axis=1),
            feats["chroma"][:, sl].mean(axis=1),
            feats["chroma"][:, sl].std(axis=1),
            feats["contrast"][:, sl].mean(axis=1),
            feats["scalars"][:, sl].mean(axis=1),
            feats["scalars"][:, sl].std(axis=1),
            np.array([start / max(1, n_frames)]),          # temporal position
            np.array([(end - start) * sec_per_frame]),     # duration
        ]
        rows.append(np.concatenate(parts))

    return np.nan_to_num(np.vstack(rows).astype(np.float32))


def node_feature_dim(cfg: AudioConfig) -> int:
    return 2 * cfg.n_mfcc + 2 * cfg.n_chroma + 7 + 10 + 2


# --------------------------------------------------------------------------- #
# per-track extraction + cache
# --------------------------------------------------------------------------- #
def extract_track(path: str, cfg: AudioConfig) -> dict[str, np.ndarray]:
    """Full feature bundle for one track: node features, chroma, mel, boundaries."""
    y = load_audio(path, cfg)
    feats = frame_features(y, cfg)
    n_frames = feats["mfcc"].shape[1]
    bounds = segment_boundaries(y, n_frames, cfg)

    if len(bounds) < cfg.min_segments:
        raise ValueError(f"only {len(bounds)} segments (min {cfg.min_segments}): {path}")

    return {
        "node_features": segment_node_features(feats, bounds, cfg),
        "boundaries": np.asarray(bounds, dtype=np.int32),
        "chroma": feats["chroma"].astype(np.float32),
        "log_mel": feats["log_mel"].astype(np.float32),
        "sec_per_frame": np.float32(cfg.hop_length / cfg.sr),
    }


def cache_file(cache_dir: str, track_id: str) -> str:
    return os.path.join(cache_dir, f"{str(track_id).replace(os.sep, '_')}.npz")


def extract_and_cache(path: str, track_id: str, cache_dir: str, cfg: AudioConfig, overwrite: bool = False):
    """Extract with an on-disk cache. Returns the feature dict, or None on failure."""
    os.makedirs(cache_dir, exist_ok=True)
    dest = cache_file(cache_dir, track_id)

    if os.path.exists(dest) and not overwrite:
        with np.load(dest) as z:
            return {k: z[k] for k in z.files}

    try:
        bundle = extract_track(path, cfg)
    except Exception as exc:  # noqa: BLE001 - GTZAN ships at least one corrupt file
        print(f"[warn] skipping {track_id}: {exc}")
        return None

    np.savez_compressed(dest, **bundle)
    return bundle
