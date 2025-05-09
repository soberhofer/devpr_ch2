import torch
from torch.utils import data
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import os
import sys
from functools import partial
import numpy as np
import librosa

# import config # Removed old config import
from . import transforms
from audiomentations import Compose as AudiomentationsCompose, AddGaussianNoise, TimeStretch, PitchShift, Gain

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_extract_zip(url: str, file_path: str):
    #import wget
    import zipfile
    root = os.path.dirname(file_path)
    # wget.download(url, out=file_path, bar=download_progress)
    download_file(url=url, fname=file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)


# create this bar_progress method which is invoked automatically from wget
def download_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


class ESC50(data.Dataset):
    # Added sr, n_mels, hop_length, val_size, n_mfcc parameters
    def __init__(self, root, sr, n_mels, hop_length, val_size, n_mfcc=None,
                 test_folds=frozenset((1,)), subset="train", global_mean_std=(0.0, 0.0), download=False):
        audio = 'ESC-50-master/audio'
        root = os.path.normpath(root)
        audio = os.path.join(root, audio)
        # Store parameters
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.val_size = val_size
        self.n_mfcc = n_mfcc
        if subset in {"train", "test", "val"}:
            self.subset = subset
        else:
            raise ValueError
        # path = path.split(os.sep)
        if not os.path.exists(audio) and download:
            os.makedirs(root, exist_ok=True)
            file_name = 'master.zip'
            file_path = os.path.join(root, file_name)
            url = f'https://github.com/karoldvl/ESC-50/archive/{file_name}'
            download_extract_zip(url, file_path)

        self.root = audio
        self.cache_dict = dict()
        # getting name of all files inside the all the train_folds
        temp = sorted(os.listdir(self.root))
        folds = {int(v.split('-')[0]) for v in temp}
        self.test_folds = set(test_folds)
        self.train_folds = folds - test_folds
        train_files = [f for f in temp if int(f.split('-')[0]) in self.train_folds]
        test_files = [f for f in temp if int(f.split('-')[0]) in test_folds]
        # sanity check
        assert set(temp) == (set(train_files) | set(test_files))
        if subset == "test":
            self.file_names = test_files
        else:
            # Use self.val_size passed during init
            if self.val_size > 0:
                 # Use self.val_size
                train_files, val_files = train_test_split(train_files, test_size=self.val_size, random_state=0)
            else: # Handle case where val_size is 0 or None
                val_files = [] # Ensure val_files is defined even if not splitting
            if subset == "train":
                self.file_names = train_files
            elif subset == "val": # Explicitly check for "val" subset
                self.file_names = val_files
            # Removed the final 'else' as subset is validated earlier
        # the number of samples in the wave (=length) required for spectrogram
        # Use self.sr and self.hop_length
        out_len = int(((self.sr * 5) // self.hop_length) * self.hop_length)
        train = self.subset == "train"

        self.audiomentations_pipeline = None
        if train:
            # Define audiomentations pipeline for training
            self.audiomentations_pipeline = AudiomentationsCompose([
                Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.3),
                TimeStretch(min_rate=0.85, max_rate=1.15, p=0.3, leave_length_unchanged=False),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.3, sr=self.sr),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01, p=0.3),
            ])

            # Existing custom wave transforms (expect PyTorch Tensor)
            # RandomScale (time stretch) and RandomNoise are now handled by audiomentations_pipeline
            self.wave_transforms = transforms.Compose(
                torch.Tensor, # Converts NumPy array (from audiomentations or loaded wave) to Tensor
                transforms.RandomPadding(out_len=out_len),
                transforms.RandomCrop(out_len=out_len),
            )

            self.spec_transforms = transforms.Compose(
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
                transforms.TimeMask(max_width=3, numbers=2),
                transforms.FrequencyMask(max_width=3, numbers=2),
            )
        else:
            # For testing, no audiomentations, custom transforms are deterministic
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                transforms.RandomPadding(out_len=out_len, train=False),
                transforms.RandomCrop(out_len=out_len, train=False)
            )
            self.spec_transforms = transforms.Compose(
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
            )
        self.global_mean = global_mean_std[0]
        self.global_std = global_mean_std[1]
        # self.n_mfcc is already set in __init__

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        path = os.path.join(self.root, file_name)
        
        # identifying the label of the sample from its name
        temp = file_name.split('.')[0]
        class_id = int(temp.split('-')[-1])

        if index in self.cache_dict:
            # Retrieve cached float32 wave and class_id
            current_wave_np, cached_class_id = self.cache_dict[index]
            assert class_id == cached_class_id, f"Class ID mismatch for index {index}: file says {class_id}, cache says {cached_class_id}"
        else:
            # Load with librosa (returns float32, typically in [-1, 1])
            wave, rate = librosa.load(path, sr=self.sr, mono=False) # Load as potentially multi-channel

            # Ensure wave is (channels, samples)
            if wave.ndim == 1:
                wave = wave[np.newaxis, :] # Convert (N,) to (1, N)

            # Remove silent sections (from float32 wave)
            if np.any(wave): # Check if any non-zero values
                non_silent_indices = np.where(np.any(wave != 0, axis=0))[0]
                if len(non_silent_indices) > 0:
                    start = non_silent_indices[0]
                    end = non_silent_indices[-1]
                    wave = wave[:, start: end + 1]
                else: # All channels are silent
                    wave = np.zeros((wave.shape[0], 1), dtype=wave.dtype) # Keep C, make length 1
            else: # Already all zeros
                wave = np.zeros((wave.shape[0], 1), dtype=wave.dtype)

            current_wave_np = wave # This is float32, (C, N)
            self.cache_dict[index] = (current_wave_np, class_id) # Cache the processed float32 wave

        # Make a copy for stochastic augmentations
        processed_wave_np = np.copy(current_wave_np)

        # Apply audiomentations if training
        if self.subset == "train" and self.audiomentations_pipeline:
            # audiomentations expects samples to be float32.
            # It handles (channels, samples) or (samples,) for mono.
            processed_wave_np = self.audiomentations_pipeline(
                samples=processed_wave_np.astype(np.float32),
                sample_rate=self.sr
            )
            # Ensure shape is still (C,N) if audiomentations changed it (e.g. mono processing by mistake)
            if processed_wave_np.ndim == 1 and current_wave_np.shape[0] == 1: # Was mono (1,N), became (N,)
                processed_wave_np = processed_wave_np[np.newaxis, :]

        # Apply existing custom wave transforms (which expect Tensor)
        # The first transform in self.wave_transforms is torch.Tensor
        wave_tensor = self.wave_transforms(processed_wave_np)

        # Prepare for librosa feature extraction
        # librosa expects y as (N,) for mono, or (C,N) for multi-channel.
        if wave_tensor.shape[0] == 1: # Mono case
            processed_wave_for_librosa = wave_tensor.squeeze(0).numpy().astype(np.float32)
        else: # Multi-channel case
            processed_wave_for_librosa = wave_tensor.numpy().astype(np.float32)
        
        # Ensure processed_wave_for_librosa is not empty before feature extraction
        if processed_wave_for_librosa.size == 0:
             # If augmentations resulted in an empty array, create a silent one of minimal length
             # to prevent errors in librosa. This depends on how n_fft and hop_length are set.
             # A single frame of silence:
             min_len = self.hop_length 
             if processed_wave_for_librosa.ndim == 1: # Mono
                 processed_wave_for_librosa = np.zeros(min_len, dtype=np.float32)
             else: # Multi-channel (should be (C,N))
                 num_channels = current_wave_np.shape[0] # Get original number of channels
                 processed_wave_for_librosa = np.zeros((num_channels, min_len), dtype=np.float32)


        if self.n_mfcc:
            mfcc = librosa.feature.mfcc(y=processed_wave_for_librosa,
                                        sr=self.sr,
                                        n_mels=self.n_mels,
                                        n_fft=1024,
                                        hop_length=self.hop_length,
                                        n_mfcc=self.n_mfcc)
            feat = mfcc
        else:
            s = librosa.feature.melspectrogram(y=processed_wave_for_librosa,
                                            sr=self.sr,
                                            n_mels=self.n_mels,
                                            n_fft=1024,
                                            hop_length=self.hop_length,
                                            )
            log_s = librosa.power_to_db(s, ref=np.max)
            log_s = self.spec_transforms(log_s) # Apply random spectrogram augmentations
            feat = log_s

        # Normalize
        if self.global_mean != 0.0 or self.global_std != 0.0: # Check if stats are actually set
            feat = (feat - self.global_mean) / self.global_std
        
        return file_name, feat, class_id


# def get_global_stats(data_path):
#     # This function needs refactoring to load config (e.g., sr, n_mels, etc.)
#     # if it's intended to be run standalone. Commenting out for now as
#     # train_crossval.py uses hardcoded stats.
#     res = []
#     # Required params for ESC50: root, sr, n_mels, hop_length, val_size
#     # These need to be loaded from a config source here.
#     # Example placeholder: sr, n_mels, hop_length, val_size = load_defaults()
#     for i in range(1, 6):
#         # train_set = ESC50(subset="train", test_folds={i}, root=data_path, download=True,
#         #                   sr=sr, n_mels=n_mels, hop_length=hop_length, val_size=val_size)
#         # a = torch.concatenate([v[1] for v in tqdm(train_set)])
#         # res.append((a.mean(), a.std()))
#     # return np.array(res)
#     pass # Keep function defined but do nothing
