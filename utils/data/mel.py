import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

def extract_mel(mel_path):
    mel = np.load(mel_path)
    return mel

def load_mel(audio_path, utts, nj=24):
    mel_paths = [os.path.join(audio_path, utt) for utt in utts]
    with Pool(nj) as p:
        mels = list(tqdm(p.imap(extract_mel, mel_paths), total=len(mel_paths)))
    return mels