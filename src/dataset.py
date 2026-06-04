import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class AudioDeepfakeDataset(Dataset):
    """
    Expects the following folder structure:
        data/
          real/       <- label 0
          synthetic/  <- label 1
    """

    def __init__(
        self,
        real_dir: str,
        synthetic_dir: str,
        sample_rate: int = 16000,
        max_duration: float = 4.0,
        augment: bool = False,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.augment = augment

        self.samples = []  # list of (path, label)
        self._load_dir(real_dir, label=0)
        self._load_dir(synthetic_dir, label=1)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No audio files found in:\n  {real_dir}\n  {synthetic_dir}\n"
                f"Supported formats: {SUPPORTED_EXTENSIONS}"
            )

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_synth = sum(1 for _, l in self.samples if l == 1)
        print(f"Dataset loaded — real: {n_real}, synthetic: {n_synth}, total: {len(self.samples)}")

    def _load_dir(self, directory: str, label: int):
        path = Path(directory)
        if not path.exists():
            print(f"Warning: directory does not exist: {directory}")
            return
        for f in path.rglob("*"):
            if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                self.samples.append((str(f), label))

    def _load_audio(self, path: str) -> np.ndarray:
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        # Pad or trim to fixed length
        if len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))
        else:
            # Random crop during training, center crop otherwise
            if self.augment:
                start = np.random.randint(0, len(audio) - self.max_samples + 1)
            else:
                start = (len(audio) - self.max_samples) // 2
            audio = audio[start : start + self.max_samples]
        return audio

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        # Gaussian noise
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.0005, 0.002)
            audio = audio + noise_level * np.random.randn(*audio.shape)

        # Random gain
        if np.random.random() < 0.5:
            gain = np.random.uniform(0.7, 1.3)
            audio = audio * gain

        # Pitch shift (±2 semitones) — slow, disable if training is too slow
        # if np.random.random() < 0.3:
        #     steps = np.random.uniform(-2, 2)
        #     audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)

        return audio.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        audio = self._load_audio(path)

        if self.augment:
            audio = self._augment_audio(audio)

        # Normalize
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """Returns a sampler that balances classes per batch."""
        labels = [l for _, l in self.samples]
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(
    real_dir: str,
    synthetic_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.10,
    batch_size: int = 16,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_duration: float = 4.0,
):
    from sklearn.model_selection import train_test_split

    full_dataset = AudioDeepfakeDataset(
        real_dir, synthetic_dir,
        sample_rate=sample_rate,
        max_duration=max_duration,
        augment=False,
    )

    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels, test_size=val_split + test_split, stratify=labels, random_state=42
    )
    relative_test = test_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=relative_test, stratify=temp_labels, random_state=42
    )

    from torch.utils.data import Subset

    train_set = AudioDeepfakeDataset(
        real_dir, synthetic_dir,
        sample_rate=sample_rate,
        max_duration=max_duration,
        augment=True,
    )
    train_subset = Subset(train_set, train_idx)
    val_subset   = Subset(full_dataset, val_idx)
    test_subset  = Subset(full_dataset, test_idx)

    # Weighted sampler for training to handle imbalance
    train_labels = [full_dataset.samples[i][1] for i in train_idx]
    class_counts = np.bincount(train_labels)
    sample_weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    return train_loader, val_loader, test_loader