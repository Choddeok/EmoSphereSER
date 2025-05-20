import os
import librosa
from tqdm import tqdm
from multiprocessing import Pool
import torch

# Load audio
def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav

# Load audio
def extract_mfcc(wav_path):
    mfcc_path = wav_path.replace("Audios", "MFCCs").replace(".wav", ".pt")
    raw_mfcc = torch.load(mfcc_path)
    return raw_mfcc

# def load_audio_mfcc(audio_path, utts, nj=4):
#     wav_paths = [os.path.join(audio_path, utt) for utt in utts]
#     with Pool(nj) as p:
#         wavs = list(tqdm(p.imap(extract_wav, wav_paths), total=len(wav_paths)))
#         mfccs = list(tqdm(p.imap(extract_mfcc, wav_paths), total=len(wav_paths)))
#     return wavs, mfccs


def load_audio_mfcc(audio_path, utts, nj=24):
    wav_paths = [os.path.join(audio_path, utt) for utt in utts]

    batch_size = 100
    wavs, mfccs = [], []
    for i in tqdm(range(0, len(wav_paths), batch_size)):
        batch_paths = wav_paths[i:i+batch_size]
        
        with Pool(min(nj, 4)) as p:
            batch_wavs = list(p.imap(extract_wav, batch_paths))
            batch_mfccs = list(p.imap(extract_mfcc, batch_paths))
        
        wavs.extend(batch_wavs)
        mfccs.extend(batch_mfccs)

    return wavs, mfccs