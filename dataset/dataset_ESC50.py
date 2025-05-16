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
from audiomentations import Compose as AudiomentationsCompose, AddGaussianNoise, TimeStretch, PitchShift, Gain, LoudnessNormalization, AddBackgroundNoise
import glob # For finding augmented files

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
    # Added use_preprocessed_data and preprocessed_data_root
    def __init__(self, root, sr, n_mels, hop_length, val_size, n_mfcc=None,
                 test_folds=frozenset((1,)), subset="train", global_mean_std=(0.0, 0.0), download=False,
                 use_preprocessed_data: bool = False, preprocessed_data_root: str = None):
        
        original_root_path = os.path.normpath(root) # Keep original root for potential download
        esc50_audio_subdir = 'ESC-50-master/audio'

        # Store parameters
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.val_size = val_size
        self.n_mfcc = n_mfcc
        self.use_preprocessed_data = use_preprocessed_data
        
        if subset in {"train", "test", "val"}:
            self.subset = subset
        else:
            raise ValueError

        # Handle download if original data not present
        original_audio_full_path = os.path.join(original_root_path, esc50_audio_subdir)
        if not os.path.exists(original_audio_full_path) and download:
            os.makedirs(original_root_path, exist_ok=True)
            file_name = 'master.zip'
            file_path = os.path.join(original_root_path, file_name)
            url = f'https://github.com/karoldvl/ESC-50/archive/{file_name}'
            download_extract_zip(url, file_path)

        self.cache_dict = dict()
        
        # Determine the root directory for listing files
        if self.use_preprocessed_data and preprocessed_data_root:
            self.root = os.path.join(os.path.normpath(preprocessed_data_root), esc50_audio_subdir)
            if not os.path.isdir(self.root):
                raise ValueError(f"Preprocessed data path not found or not a directory: {self.root}")
            print(f"Using preprocessed data from: {self.root}")
            # Get all .wav files, original and augmented
            all_available_files = [f for f in os.listdir(self.root) if f.endswith(".wav")]
        else:
            self.root = original_audio_full_path
            if not os.path.isdir(self.root):
                 raise ValueError(f"Original data path not found or not a directory: {self.root}. Try download=True.")
            all_available_files = [f for f in os.listdir(self.root) if f.endswith(".wav") and "_aug" not in f] # Only originals if not using preprocessed

        temp = sorted(all_available_files)
        
        # Folds are determined from the original file names (e.g., 1-xxxxx.wav)
        # This logic needs to correctly identify the original fold even for augmented files.
        # Augmented files are named like 1-xxxxx_augN.wav
        folds = {int(v.split('-')[0]) for v in temp} # This might include augmented files' prefixes
        self.test_folds = set(test_folds)
        self.train_folds = folds - self.test_folds # This assumes fold numbers are consistent

        # Filter files based on their original fold designation
        train_files_candidates = [f for f in temp if int(f.split('-')[0]) in self.train_folds]
        test_files_candidates = [f for f in temp if int(f.split('-')[0]) in self.test_folds]

        # If not using preprocessed data, ensure we only have original files
        if not self.use_preprocessed_data:
            train_files_candidates = [f for f in train_files_candidates if "_aug" not in f]
            test_files_candidates = [f for f in test_files_candidates if "_aug" not in f]
        
        # Sanity check (might fail if augmented files change total count significantly vs. manifest)
        # For now, we assume the file list `temp` is the source of truth for available files.
        # assert set(temp) == (set(train_files_candidates) | set(test_files_candidates)) # This might be too strict with augmentation

        if subset == "test":
            self.file_names = test_files_candidates
        else: # "train" or "val"
            if self.val_size > 0 and len(train_files_candidates) > 0 :
                # Split based on original file names to keep augmentations of the same original file together
                original_train_filenames = sorted(list(set([f.split('_aug')[0] + ".wav" if "_aug" in f else f for f in train_files_candidates])))
                
                if len(original_train_filenames) <=1 and self.val_size > 0: # handle case with very few files
                    print(f"Warning: Too few unique original files ({len(original_train_filenames)}) in train_files_candidates for val_split with val_size {self.val_size}. Using all for training.")
                    train_originals = original_train_filenames
                    val_originals = []
                elif self.val_size >= len(original_train_filenames): # val_size is too large
                    print(f"Warning: val_size {self.val_size} is >= number of original train files {len(original_train_filenames)}. Using all for training, no validation set.")
                    train_originals = original_train_filenames
                    val_originals = []
                else:
                    train_originals, val_originals = train_test_split(original_train_filenames, test_size=self.val_size, random_state=0)

                train_files = [f_cand for f_cand in train_files_candidates
                               for f_orig in train_originals if f_cand.startswith(f_orig.split('.')[0])]
                val_files = [f_cand for f_cand in train_files_candidates
                             for f_orig in val_originals if f_cand.startswith(f_orig.split('.')[0])]
            else:
                train_files = train_files_candidates
                val_files = []

            if subset == "train":
                self.file_names = train_files
            elif subset == "val":
                self.file_names = val_files
        
        if not self.file_names:
            print(f"Warning: No files found for subset '{self.subset}' with test_folds {self.test_folds}. Check data paths and split logic.")


        out_len = int(((self.sr * 5) // self.hop_length) * self.hop_length)
        
        # Determine if online augmentation should be applied
        apply_online_augmentation = (self.subset == "train") and (not self.use_preprocessed_data)

        self.audiomentations_pipeline = None
        if apply_online_augmentation:
            # Define audiomentations pipeline for training (only if not using preprocessed)
            self.audiomentations_pipeline = AudiomentationsCompose([
                Gain(min_gain_db=-12.0, max_gain_db=12.0, p=0.5),
                TimeStretch(min_rate=0.7, max_rate=1.3, p=0.5, leave_length_unchanged=False),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.02, p=0.5),
                LoudnessNormalization(min_lufs=-31.0, max_lufs=-13.0, p=0.5),
                # AddBackgroundNoise might be problematic if preprocessed_data_root is different from original sounds_path
                # Sounds path for AddBackgroundNoise should ideally point to original clean samples if used here.
                # For simplicity, if using preprocessed, this online AddBackgroundNoise is skipped.
                AddBackgroundNoise(sounds_path=original_audio_full_path, min_snr_db=0, max_snr_db=20, p=0.5)
            ])
        
        # Wave transforms (padding, cropping) are always needed
        self.wave_transforms = transforms.Compose(
            torch.Tensor, 
            transforms.RandomPadding(out_len=out_len, train=(self.subset == "train")), # train=False for val/test
            transforms.RandomCrop(out_len=out_len, train=(self.subset == "train"))    # train=False for val/test
        )

        # Spectrogram transforms
        spec_transform_list = [torch.Tensor, partial(torch.unsqueeze, dim=0)]
        # Apply SpecAugment (TimeMask, FrequencyMask) only to the training set,
        # regardless of whether use_preprocessed_data is true or false.
        # This allows applying spectrogram augmentations on top of potentially pre-augmented waveform data.
        if self.subset == "train":
            spec_transform_list.extend([
                transforms.TimeMask(max_width=3, numbers=2), # Parameters can be tuned
                transforms.FrequencyMask(max_width=3, numbers=2), # Parameters can be tuned
            ])
        self.spec_transforms = transforms.Compose(*spec_transform_list)
            
        self.global_mean = global_mean_std[0]
        self.global_std = global_mean_std[1]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        # self.root already points to the correct directory (original or augmented)
        path = os.path.join(self.root, file_name)
        
        # identifying the label of the sample from its name (works for original and augmented names like X-Y-Z_augN.wav)
        original_file_name_part = file_name.split('_aug')[0]
        temp = original_file_name_part.split('.')[0]
        class_id = int(temp.split('-')[-1])

        # Cache key should be unique per augmented file if caching is used with pre-augmented data
        # Or, disable caching if using pre-augmented data to avoid issues, as files are already on disk.
        # For simplicity, let's assume caching might still be beneficial for repeated epochs over the same augmented set.
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

        # Apply audiomentations if training AND it's configured (i.e., not using preprocessed data that's already augmented)
        if self.subset == "train" and self.audiomentations_pipeline: # self.audiomentations_pipeline is None if use_preprocessed_data
            temp_processed_wave_np = processed_wave_np.astype(np.float32)

            # Convert to mono by averaging channels if multichannel
            if temp_processed_wave_np.ndim > 1 and temp_processed_wave_np.shape[0] > 1:
                temp_processed_wave_np = np.mean(temp_processed_wave_np, axis=0)
            elif temp_processed_wave_np.ndim > 1 and temp_processed_wave_np.shape[0] == 1: # Already (1, N)
                temp_processed_wave_np = temp_processed_wave_np.squeeze(0) # Make it (N,) for audiomentations

            # Ensure minimum length for LoudnessNormalization (pyloudnorm expects > 0.4s)
            min_len_pyloudnorm = int(0.8 * self.sr)
            if temp_processed_wave_np.shape[0] < min_len_pyloudnorm:
                padding = np.zeros(min_len_pyloudnorm - temp_processed_wave_np.shape[0], dtype=temp_processed_wave_np.dtype)
                temp_processed_wave_np = np.concatenate((temp_processed_wave_np, padding))
            
            # Apply audiomentations pipeline (expects mono (N,) array)
            augmented_mono_wave_np = self.audiomentations_pipeline(
                samples=temp_processed_wave_np,
                sample_rate=self.sr
            )
            
            # For subsequent processing, ensure it's a 2D array (1, num_samples)
            if augmented_mono_wave_np.ndim == 1:
                processed_wave_np = augmented_mono_wave_np[np.newaxis, :]
            else: # Should not happen if pipeline outputs mono
                processed_wave_np = augmented_mono_wave_np

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
