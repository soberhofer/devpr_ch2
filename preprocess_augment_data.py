import os
import librosa
import soundfile as sf
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, LoudnessNormalization
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import hydra.utils as hyu
# Attempt to import the download utility from dataset_ESC50
try:
    from dataset.dataset_ESC50 import download_extract_zip
except ImportError:
    # Fallback or error if running in a context where dataset.dataset_ESC50 is not findable
    # This might happen if the script is run from a directory where 'dataset' is not a sibling module
    # For now, we'll assume it can be found. If not, sys.path manipulation might be needed
    # or the download functions could be copied here.
    print("Warning: Could not import download_extract_zip from dataset.dataset_ESC50. Download functionality will be missing.")
    def download_extract_zip(url: str, file_path: str): # Dummy function
        raise NotImplementedError("Download function not available due to import error.")

def create_augmentations(cfg_audio: DictConfig):
    """Creates an audiomentations augmentation pipeline from config."""
    pipeline_steps = []
    if cfg_audio.get("gain", False):
        pipeline_steps.append(Gain(min_gain_db=cfg_audio.gain.min_gain_db, max_gain_db=cfg_audio.gain.max_gain_db, p=cfg_audio.gain.p))
    if cfg_audio.get("time_stretch", False):
        pipeline_steps.append(TimeStretch(min_rate=cfg_audio.time_stretch.min_rate, max_rate=cfg_audio.time_stretch.max_rate, p=cfg_audio.time_stretch.p, leave_length_unchanged=False))
    if cfg_audio.get("pitch_shift", False):
        pipeline_steps.append(PitchShift(min_semitones=cfg_audio.pitch_shift.min_semitones, max_semitones=cfg_audio.pitch_shift.max_semitones, p=cfg_audio.pitch_shift.p))
    if cfg_audio.get("gaussian_noise", False):
        pipeline_steps.append(AddGaussianNoise(min_amplitude=cfg_audio.gaussian_noise.min_amplitude, max_amplitude=cfg_audio.gaussian_noise.max_amplitude, p=cfg_audio.gaussian_noise.p))
    if cfg_audio.get("loudness_normalization", False):
        pipeline_steps.append(LoudnessNormalization(min_lufs=cfg_audio.loudness_normalization.min_lufs, max_lufs=cfg_audio.loudness_normalization.max_lufs, p=cfg_audio.loudness_normalization.p))
    # Add other augmentations from your config as needed
    return Compose(pipeline_steps)

@hydra.main(config_path="conf", config_name="preprocess_config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    original_data_dir = hyu.to_absolute_path(cfg.paths.original_data_root)
    augmented_data_dir = hyu.to_absolute_path(cfg.paths.augmented_data_root)
    
    # Ensure the subdirectories exist in the augmented path
    # ESC-50 specific structure
    esc50_audio_subdir = "ESC-50-master/audio"
    original_audio_path = os.path.join(original_data_dir, esc50_audio_subdir)
    augmented_audio_path = os.path.join(augmented_data_dir, esc50_audio_subdir)

    os.makedirs(augmented_audio_path, exist_ok=True)

    augment_pipeline = create_augmentations(cfg.augmentations)
    num_augmentations_per_file = cfg.settings.num_augmentations_per_file
    target_sr = cfg.settings.target_sr
    download_if_missing = cfg.settings.get("download_if_missing", True) # Default to True

    if not os.path.isdir(original_audio_path):
        print(f"Original audio path not found: {original_audio_path}")
        if download_if_missing:
            print("Attempting to download ESC-50 dataset...")
            # original_data_dir is the root for ESC-50 (e.g., .../data/esc50)
            # master.zip will be downloaded into original_data_dir
            # and then extracted, creating ESC-50-master within original_data_dir
            os.makedirs(original_data_dir, exist_ok=True)
            file_name = 'master.zip'
            zip_file_path = os.path.join(original_data_dir, file_name)
            dataset_url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
            
            try:
                download_extract_zip(url=dataset_url, file_path=zip_file_path)
                print(f"Dataset downloaded and extracted to {original_data_dir}")
                # Verify if original_audio_path (which is original_data_dir/ESC-50-master/audio) now exists
                if not os.path.isdir(original_audio_path):
                    print(f"Error: Dataset downloaded, but expected audio path still not found: {original_audio_path}")
                    return
            except Exception as e:
                print(f"Error during download/extraction: {e}")
                print("Please ensure the ESC-50 dataset is manually downloaded and extracted to the specified location.")
                return
        else:
            print("Download_if_missing is false. Please ensure the dataset is present.")
            return
    else:
        print(f"Found original audio path: {original_audio_path}")

    all_files = [f for f in os.listdir(original_audio_path) if f.endswith(".wav")]
    
    print(f"Found {len(all_files)} .wav files in {original_audio_path}")
    print(f"Saving augmented files to: {augmented_audio_path}")

    for filename in tqdm(all_files, desc="Augmenting files"):
        original_filepath = os.path.join(original_audio_path, filename)
        
        try:
            y, sr = librosa.load(original_filepath, sr=target_sr, mono=True)
        except Exception as e:
            print(f"Error loading {original_filepath}: {e}")
            continue

        for i in range(num_augmentations_per_file):
            try:
                # Ensure 'y' is float32 for audiomentations
                augmented_samples = augment_pipeline(samples=y.astype(np.float32), sample_rate=sr)
            except Exception as e:
                print(f"Error augmenting {filename} (aug {i+1}): {e}")
                # Fallback: use original samples if augmentation fails catastrophically
                augmented_samples = y 
            
            base, ext = os.path.splitext(filename)
            # Naming convention: original_filename_augX.wav
            # Example: 1-100032-A-0_aug1.wav
            augmented_filename = f"{base}_aug{i+1}{ext}"
            augmented_filepath = os.path.join(augmented_audio_path, augmented_filename)
            
            try:
                sf.write(augmented_filepath, augmented_samples, sr)
            except Exception as e:
                print(f"Error writing {augmented_filepath}: {e}")

if __name__ == "__main__":
    main()
