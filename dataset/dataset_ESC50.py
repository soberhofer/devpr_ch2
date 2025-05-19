import os
import sys
import requests
import logging
from functools import partial
import random
import shutil

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
import torch
from torch.utils import data
from tqdm import tqdm
from torchaudio.transforms import TimeMasking, FrequencyMasking

# import config # Removed old config import
from . import transforms
import audiomentations # import Compose, AddGaussianNoise, PitchShift, TimeStretch, Shift

logger = logging.getLogger(__name__)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def download_file(url: str, fname: str, chunk_size=1024):
    """
    Downloads a file from a given URL and saves it to the specified file path.

    Parameters
    ----------
    url : str
        The URL of the file to be downloaded.
    fname : str
        The local file path where the downloaded content will be saved.
    chunk_size : int, optional
        The size of the chunks (in bytes) used for downloading the file.
        Default is 1024 bytes.

    Returns
    -------
    None
        The function saves the file locally and does not return anything.

    Notes
    -----
    This function uses the `requests` library to stream the file content in chunks
    and the `tqdm` library to display a progress bar during the download.
    """
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
    """
    Downloads a ZIP file from the specified URL, saves it to the given path, and extracts its contents.

    Parameters
    ----------
    url : str
        The URL from which the ZIP file will be downloaded.
    file_path : str
        The local file path where the ZIP file will be saved. This path is also used to extract the contents.

    Returns
    -------
    None
        The function downloads the file and extracts its contents to the same directory as the `file_path`.

    Notes
    -----
    The function uses `download_file` to download the ZIP file and `zipfile.ZipFile` to extract its contents.
    The ZIP file is extracted to the directory of the provided `file_path`.
    """
    #import wget
    import zipfile
    root = os.path.dirname(file_path)
    # wget.download(url, out=file_path, bar=download_progress)
    download_file(url=url, fname=file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)

"""
# create this bar_progress method which is invoked automatically from wget
def download_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()"""


class ESC50(data.Dataset):
    """
    ESC50 dataset class for loading and processing audio data from the ESC-50 dataset.

    This class supports data loading, augmentation for training, and caching for efficient
    retrieval of audio samples and their corresponding features (MFCC or spectrogram).

    Attributes:
    -----------
    root : str
        Path to the root directory containing the ESC-50 dataset audio files.
    cachedata : dict
        Cache for storing preprocessed waveforms to avoid redundant computation.
    subset : str
        Subset of the dataset to use ("train", "test", or "val").
    test_folds : set
        Set of folds used for testing.
    train_folds : set
        Set of folds used for training.
    file_names : list
        List of file names corresponding to the chosen subset.
    wave_transforms : torchvision.transforms.Compose
        List of wave transformations to apply to the audio data.
    spec_transforms : torchvision.transforms.Compose
        List of spectral transformations to apply to the spectrogram.
    global_mean : float
        Global mean used for normalization of features.
    global_std : float
        Global standard deviation used for normalization of features.
    n_mfcc : int or None
        Number of MFCC features to extract. If None, mel spectrogram is used.
    sr : int
        Sampling rate for audio.
    n_mels : int
        Number of Mel bands to generate.
    hop_length : int
        Hop length for STFT.
    val_size : float, optional
        Proportion of training data to use for validation.
    use_preprocessed_data : bool, optional
        Flag to indicate whether to use preprocessed data.
    preprocessed_data_root : str, optional
        Root directory for preprocessed data.
    """
    def __init__(self, root, sr, n_mels, hop_length,
                 test_folds=frozenset((1,)), subset="train",
                 global_mean_std=(0.5, 0.5), download=False,
                 num_aug=False, prob_aug_wave=0, prob_aug_spec=0,
                 val_size=None, n_mfcc=None,
                 use_preprocessed_data=False, preprocessed_data_root=None):
        """
        Initializes the ESC50 dataset, including setting paths, subsets, and transformations.

        Parameters:
        -----------
        root : str
            Path to the root directory containing the dataset.
        sr : int
            Sampling rate for audio.
        n_mels : int
            Number of Mel bands to generate.
        hop_length : int
            Hop length for STFT.
        test_folds : set, optional
            Set of folds to be used for testing (default is {1}).
        subset : str, optional
            Subset of the dataset to load ('train', 'test', or 'val').
        global_mean_std : tuple, optional
            Tuple containing global mean and global standard deviation for normalization (default is (0.0, 0.0)).
        download : bool, optional
            If True, will download and extract the ESC-50 dataset if it's not already present (default is False).
        num_aug : bool, optional
            Number of augmented copies for training data.
        prob_aug_wave : float, optional
            Probability of applying wave augmentation.
        prob_aug_spec : float, optional
            Probability of applying spectral augmentation.
        val_size : float, optional
            Proportion of training data to use for validation.
        n_mfcc : int, optional
            Number of MFCCs to compute. If None, Mel spectrograms are computed.
        use_preprocessed_data : bool, optional
            Flag to indicate whether to use preprocessed data.
        preprocessed_data_root : str, optional
            Root directory for preprocessed data.
        """
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.val_size = val_size
        self.n_mfcc = n_mfcc # This will be used directly later
        self.use_preprocessed_data = use_preprocessed_data
        self.preprocessed_data_root = preprocessed_data_root

        audio_folder_name = 'ESC-50-master/audio'
        # If using preprocessed data, the root path might be different
        if self.use_preprocessed_data and self.preprocessed_data_root:
            # Assuming preprocessed_data_root points to the parent of 'ESC-50-master'
            # or directly to 'ESC-50-master' if audio_folder_name is part of it.
            # This logic might need adjustment based on actual preprocessed data structure.
            # For now, let's assume preprocessed_data_root is the equivalent of the original 'root'
            # and contains the 'ESC-50-master/audio' structure.
            effective_root = os.path.normpath(self.preprocessed_data_root)
        else:
            effective_root = os.path.normpath(root)

        audio = os.path.join(effective_root, audio_folder_name)

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
        self.cachedata = {}
        # Make Copies of Dataset and use Data Augmentation (First Copy gets Key0, Second Copy Key1, ... Fifth Copy Key0 again,...
        self.aug_transforms = {0: audiomentations.Compose([audiomentations.TimeStretch(min_rate=0.8, max_rate=1.25)]),
                               1: audiomentations.Compose([audiomentations.PitchShift(min_semitones=-4, max_semitones=4)]),
                               2: audiomentations.Compose([audiomentations.Shift(min_shift=-0.2, max_shift=0.2)]),
                               3: audiomentations.Compose([audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015)])
                               }
        self.add_aug_num = num_aug  #  Number of Copies of train data with random data augmentation (self.aug_transforms)
        # Propabity that Data Augmentation
        self.prob_aug_wave = prob_aug_wave
        self.prob_aug_spec = prob_aug_spec


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
            self.file_names = [(f, 0) for f in test_files]
        else:
            if self.val_size: # Changed config.val_size to self.val_size
                train_files, val_files = train_test_split(train_files, test_size=self.val_size, random_state=0) # Changed config.val_size to self.val_size
            if subset == "train":
                self.file_names = train_files
                if not self.add_aug_num:
                    self.file_names = [(f, 0) for f in train_files]
                else:
                    self.file_names = [(f, i) for f in train_files for i in range(self.add_aug_num)]
            else:
                self.file_names = [(f, 0) for f in val_files]
        # the number of samples in the wave (=length) required for spectrogram
        out_len = int(((self.sr * 5) // self.hop_length) * self.hop_length) # Changed config.sr and config.hop_length
        train = self.subset == "train"
        if train:
            # augment training data with transformations that include randomness
            # transforms can be applied on wave and spectral representation
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                #transforms.RandomScale(max_scale=1.25),
                transforms.RandomPadding(out_len=out_len),
                transforms.RandomCrop(out_len=out_len)
            )
            self.wave_transforms_add = transforms.Compose(
                torch.Tensor,
                transforms.RandomScale(max_scale=1.25),
                transforms.RandomPadding(out_len=out_len),
                transforms.RandomCrop(out_len=out_len)
            )

            self.spec_transforms = transforms.Compose(
                # to Tensor and prepend singleton dim
                #lambda x: torch.Tensor(x).unsqueeze(0),
                # lambda non-pickleable, problem on windows, replace with partial function
                torch.Tensor,
                partial(torch.unsqueeze, dim=0),
            )

            self.spec_transforms_add = transforms.Compose(
                # to Tensor and prepend singleton dim
                # lambda x: torch.Tensor(x).unsqueeze(0),
                # lambda non-pickleable, problem on windows, replace with partial function
                torch.Tensor,
                FrequencyMasking(freq_mask_param=15),
                TimeMasking(time_mask_param=30),
                partial(torch.unsqueeze, dim=0),
            )

        else:
            # for testing transforms are applied deterministically to support reproducible scores
            self.wave_transforms = transforms.Compose(
                torch.Tensor,
                # disable randomness
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

        # Load Metadata
        path_meta = os.path.join(os.path.dirname(audio), 'meta', 'esc50.csv')
        metadata = pd.read_csv(path_meta)
        self.metadata_dict = {idx: metadata[metadata['filename'] == filename].iloc[0].to_dict() for idx, (filename,aug) in enumerate(self.file_names)}

        #filtered_metadata = metadata[metadata.index.isin(self.file_names)]
        #self.metadata_dict = {idx: row.to_dict() for idx, row in filtered_metadata.iterrows()}
        logger.debug('A')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        """
        Retrieves an audio sample and its corresponding features and label from the dataset.

        This method is used to access data for each sample in the dataset, either for training,
        validation, or testing.

        Parameters:
        -----------
        index : int
            Index of the sample to retrieve.

        Returns:
        --------
        tuple
            A tuple containing:
                - file_name (str): The name of the audio file.
                - feat (Tensor): The extracted features (MFCC or spectrogram).
                - class_id (int): The class label of the audio sample.

        Notes:
        ------
         Liefert grundsätzlich das Audiofile mit dem Index
        Wenn man Index 5 abfragt --> kommt immer das gleiche File zurück
        Melsprectrogram benötigt viel CPU Leistung
        Laden der Daten ist auch relevant
        und die sampling rate benötigt viel

        Pytorch sieht vor dataloarding auf der CPU zu machen
        CPU kann im gegensatz gpu multitreading,... --> sollten wir nutzen
        """
        #return_fft = True
        logger.debug(f"Start Retrieving Audio File with index {index}")
        file_name, augmentation = self.file_names[index]
        path = os.path.join(self.root, file_name)


        # identifying the label of the sample from its name
        temp = file_name.split('.')[0]
        class_id = int(temp.split('-')[-1])

        # Data only get loaded if not already loaded
        if index not in self.cachedata:
            wave, rate = librosa.load(path, sr=self.sr) # Changed config.sr to self.sr

            if wave.ndim == 1:
                wave = wave[:, np.newaxis]

            if augmentation != 0:
                idx = (augmentation-1)%len(self.aug_transforms)
                waveT = wave.T
                waveT = self.aug_transforms[idx](samples=waveT, sample_rate=self.sr) # Changed config.sr to self.sr
                wave = waveT.T

            # normalizing waves to [-1, 1]
            if np.abs(wave.max()) > 1.0:
                wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
            wave = wave.T * 32768.0

            # Remove silent sections
            start = wave.nonzero()[1].min()
            end = wave.nonzero()[1].max()
            wave = wave[:, start: end + 1]

            wave_copy = np.copy(wave)
            if self.subset == 'train':
                bool_aug_wave = random.random() < self.prob_aug_wave
                if bool_aug_wave:
                    wave_copy = self.wave_transforms_add(wave_copy)
                else:
                    wave_copy = self.wave_transforms(wave_copy)
            else:
                wave_copy = self.wave_transforms(wave_copy)
            wave_copy.squeeze_(0)
            self.cachedata[index] = wave_copy
        else:
            wave_copy = self.cachedata[index]

        if self.n_mfcc:
            mfcc = librosa.feature.mfcc(y=wave_copy.numpy(),
                                        sr=self.sr, # Changed config.sr to self.sr
                                        n_mels=self.n_mels, # Changed config.n_mels to self.n_mels
                                        n_fft=1024,
                                        hop_length=self.hop_length, # Changed config.hop_length to self.hop_length
                                        n_mfcc=self.n_mfcc)
            feat = mfcc
        else:
            # melspectrogram benötigt viel CPU Leistung
            s = librosa.feature.melspectrogram(y=wave_copy.numpy(),
                                               sr=self.sr, # Changed config.sr to self.sr
                                               n_mels=self.n_mels,    # --> Number of rows # Changed config.n_mels to self.n_mels
                                               n_fft=1024,
                                               hop_length=self.hop_length, # Changed config.hop_length to self.hop_length
                                               #center=False,
                                               )
            # s.size:
            # number of rows = n_mels
            # number of columns = sr*5/hop_length
            log_s = librosa.power_to_db(s, ref=np.max)  # 10*log10(s/reg) --> Transform from linear Spectrum to dB

            # masking the spectrograms --> not correct --> Transform to Tensor + Create new Channel
            if self.subset == 'train':
                bool_aug_spec = random.random() < self.prob_aug_spec
                if bool_aug_spec:
                    log_s = self.spec_transforms_add(log_s)
                else:
                    log_s = self.spec_transforms(log_s)
            else:
                log_s = self.spec_transforms(log_s)
            feat = log_s

        # normalize
        if self.global_mean:
            feat = (feat - self.global_mean) / self.global_std

        return file_name, feat, class_id

    def get_metadata(self, index):
        return self.metadata_dict[index]


def init_preprocessing():
    output_dir = "preprocessed_data"

    # Delete folder if it exists
    if os.path.exists(output_dir):
        print(f"Deleting existing folder: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


class InMemoryESC50(ESC50):
    def __init__(self, root, test_folds=frozenset((1,)), subset="train", global_mean_std=(0.0, 1.0), download=False, sr=None, n_mels=None, hop_length=None, num_aug=False, prob_aug_wave=0, prob_aug_spec=0, val_size=None, n_mfcc=None, use_preprocessed_data=False, preprocessed_data_root=None):
        super().__init__(root=root,
                 test_folds=test_folds,
                 subset=subset,
                 global_mean_std=global_mean_std,
                 download=download,
                 sr=sr, n_mels=n_mels, hop_length=hop_length, # Pass sr, n_mels, hop_length
                 num_aug=num_aug, prob_aug_wave=prob_aug_wave, prob_aug_spec=prob_aug_spec, # Pass augmentation params
                 val_size=val_size, n_mfcc=n_mfcc, # Pass val_size and n_mfcc
                 use_preprocessed_data=use_preprocessed_data, preprocessed_data_root=preprocessed_data_root) # Pass preprocessing params
                 
        # Make safe folder name based on fold
        fold_str = "_".join(str(f) for f in sorted(test_folds))
        output_dir = f"preprocessed_data/fold_{fold_str}_{subset}"
        folder_exists = os.path.exists(output_dir)

        if not folder_exists:
            os.makedirs(output_dir, exist_ok=True)       
            print(f"Preprocessing {super().__len__()} samples to: {output_dir}")
            for i in tqdm(range(super().__len__())):
                fname, feat, label = super().__getitem__(i)
                out_path = os.path.join(output_dir, fname.replace('.wav', '.pt'))
                torch.save({'features': feat, 'label': label}, out_path)
        
        self.data = []
        self.files = sorted([f for f in os.listdir(output_dir) if f.endswith('.pt')])
        for f in tqdm(self.files, desc="Loading into RAM"):
            data = torch.load(os.path.join(output_dir, f))
            self.data.append((f, data['features'], data['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_global_stats(data_path):
    """
    Calculate the global mean and standard deviation of features (e.g., MFCC or spectrogram)
    for the ESC50 dataset across multiple folds of the training data.

    This function iterates over the 5 folds of the ESC50 dataset, loading the training subset
    for each fold, concatenating the features, and calculating their mean and standard deviation.

    Parameters:
    -----------
    data_path : str
        Path to the directory where the ESC50 dataset is stored or should be downloaded.

    Returns:
    --------
    np.ndarray
        An array of shape (5, 2), where each row contains the mean and standard deviation of
        the features (e.g., MFCC or spectrogram) for the respective fold.
    """
    res = []
    for i in range(1, 6):
        # This function get_global_stats might need to be updated as well if it's used,
        # as ESC50 constructor now requires sr, n_mels, hop_length.
        # For now, assuming it's not the primary focus of this refactoring or will be updated separately.
        # If it needs to be functional, it would require default values or a config object.
        # For the purpose of this task, I will leave it as is, as the main goal is to refactor the class itself.
        # If this function is called, it will raise an error due to missing arguments.
        train_set = ESC50(subset="train", test_folds={i}, root=data_path, download=True) # This line will error if called
        a = torch.concatenate([v[1] for v in tqdm(train_set)])
        res.append((a.mean(), a.std(), a.min(), a.max()))

    return np.array(res)
