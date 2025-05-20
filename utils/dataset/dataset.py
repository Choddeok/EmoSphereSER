import numpy as np
import pickle as pk
import torch.utils as torch_utils
from . import normalizer

"""
All dataset should have the same order based on the utt_list
"""
def load_norm_stat(norm_stat_file):
    with open(norm_stat_file, 'rb') as f:
        wav_mean, wav_std = pk.load(f)
    return wav_mean, wav_std

def load_mel_norm_stat(norm_stat_file):
    with open(norm_stat_file, 'rb') as f:
        mel_mean, mel_std = pk.load(f)
    return mel_mean, mel_std

class CombinedSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CombinedSet, self).__init__()
        self.datasets = kwargs.get("datasets", args[0]) 
        self.data_len = len(self.datasets[0])
        for cur_dataset in self.datasets:
            assert len(cur_dataset) == self.data_len, "All dataset should have the same order based on the utt_list"
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        result = []
        for cur_dataset in self.datasets:
            result.append(cur_dataset[idx])
        return result

class MelSet(torch_utils.data.Dataset):
    def __init__(self, mel_list, mel_mean=None, mel_std=None, max_frames=None):
        """
        mel_list: List of mel spectrograms (Numpy arrays, shape: [frames, mel_bins])
        mel_mean: Precomputed mean of the mel spectrograms
        mel_std: Precomputed standard deviation of the mel spectrograms
        max_frames: Maximum number of frames for each mel spectrogram
        """
        super(MelSet, self).__init__()
        self.mel_list = mel_list  # List of mel spectrograms
        self.max_frames = max_frames if max_frames is not None else max([mel.shape[0] for mel in mel_list])
        
        # Compute or load normalization stats
        if mel_mean is None or mel_std is None:
            self.mel_mean, self.mel_std = self.compute_norm_stats()
        else:
            self.mel_mean, self.mel_std = mel_mean, mel_std
    
    def compute_norm_stats(self):
        """Compute mean and std for normalization"""
        all_mels = np.concatenate(self.mel_list, axis=0)  # Combine all frames
        mel_mean = np.mean(all_mels, axis=0)  # Mean over all mel bins
        mel_std = np.std(all_mels, axis=0)   # Std over all mel bins
        return mel_mean, mel_std
    
    def save_norm_stat(self, norm_stat_file):
        """Save normalization stats to a file"""
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.mel_mean, self.mel_std), f)
    
    def __len__(self):
        return len(self.mel_list)
    
    def __getitem__(self, idx):
        cur_mel = self.mel_list[idx]
        cur_frames = min(cur_mel.shape[0], self.max_frames)  # Limit to max_frames
        cur_mel = cur_mel[:cur_frames]  # Truncate to max_frames
        
        # Normalize mel spectrogram
        cur_mel = (cur_mel - self.mel_mean) / (self.mel_std + 1e-6)
        
        result = (cur_mel, cur_frames)
        return result

class WavSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(WavSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)

        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

        self.upper_bound_max_dur = kwargs.get("max_dur", 12)
        self.sampling_rate = kwargs.get("sr", 16000)

        # check max duration
        self.max_dur = np.min([np.max([len(cur_wav) for cur_wav in self.wav_list]), self.upper_bound_max_dur*self.sampling_rate])
        if self.wav_mean is None or self.wav_std is None:
            self.wav_mean, self.wav_std = normalizer. get_norm_stat_for_wav(self.wav_list)
    
    def save_norm_stat(self, norm_stat_file):
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.wav_mean, self.wav_std), f)
            
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = self.wav_list[idx][:self.max_dur]
        cur_dur = len(cur_wav)
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)
        
        result = (cur_wav, cur_dur)
        return result

class MelSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(MelSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0]) # (N, D, T)

        self.wav_mean = kwargs.get("wav_mean", None)
        self.wav_std = kwargs.get("wav_std", None)

        self.upper_bound_max_dur = kwargs.get("max_dur", 12)
        self.sampling_rate = kwargs.get("sr", 16000)

        # check max duration
        self.max_dur = np.min([np.max([len(cur_wav) for cur_wav in self.wav_list]), self.upper_bound_max_dur*self.sampling_rate])
        if self.wav_mean is None or self.wav_std is None:
            self.wav_mean, self.wav_std = normalizer. get_norm_stat_for_wav(self.wav_list)
    
    def save_norm_stat(self, norm_stat_file):
        with open(norm_stat_file, 'wb') as f:
            pk.dump((self.wav_mean, self.wav_std), f)
            
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_wav = self.wav_list[idx][:self.max_dur]
        cur_dur = len(cur_wav)
        cur_wav = (cur_wav - self.wav_mean) / (self.wav_std+0.000001)
        
        result = (cur_wav, cur_dur)
        return result

class MFCCSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(MFCCSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0])
    
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_mfcc = self.wav_list[idx][:self.max_dur]
        cur_dur = len(cur_mfcc)
        
        result = (cur_mfcc, cur_dur)
        return result

class TextSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(TextSet, self).__init__()
        self.wav_list = kwargs.get("wav_list", args[0])
        # self.max_dur = np.max([len(cur_text) for cur_text in self.wav_list])
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        cur_text = self.wav_list[idx]
        cur_dur = len(cur_text)
        
        result = (cur_text, cur_dur)
        return result

class ADV_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(ADV_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
        self.max_score = kwargs.get("max_score", 7)
        self.min_score = kwargs.get("min_score", 1)
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        cur_lab = (cur_lab - self.min_score) / (self.max_score-self.min_score)
        result = cur_lab
        return result

class ADV_EmoSet_SEV(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(ADV_EmoSet_SEV, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
        self.max_score = kwargs.get("max_score", 7)
        self.min_score = kwargs.get("min_score", 1)
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        cur_lab = 2 * (cur_lab - self.min_score) / (self.max_score - self.min_score) - 1
        result = cur_lab
        return result
    
class CAT_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(CAT_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        result = cur_lab
        return result

class SEV_EmoSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(SEV_EmoSet, self).__init__()
        self.lab_list = kwargs.get("lab_list", args[0])
    
    def __len__(self):
        return len(self.lab_list)

    def __getitem__(self, idx):
        cur_lab = self.lab_list[idx]
        result = cur_lab
        return result

class SpkSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(SpkSet, self).__init__()
        self.spk_list = kwargs.get("spk_list", args[0])
    
    def __len__(self):
        return len(self.spk_list)

    def __getitem__(self, idx):
        cur_lab = self.spk_list[idx]
        result = cur_lab
        return result    

class UttSet(torch_utils.data.Dataset): 
    def __init__(self, *args, **kwargs):
        super(UttSet, self).__init__()
        self.utt_list = kwargs.get("utt_list", args[0])
    
    def __len__(self):
        return len(self.utt_list)

    def __getitem__(self, idx):
        cur_lab = self.utt_list[idx]
        result = cur_lab
        return result
