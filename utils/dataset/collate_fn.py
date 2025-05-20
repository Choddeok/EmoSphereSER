import torch
import torch.nn as nn
import numpy as np

def collate_fn_wav_lab_mask(batch):
    total_wav = []
    total_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        lab = wav_data[1]
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(np.array(total_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_lab, attention_mask, total_utt

def collate_fn_wav_lab_mask_bce(batch):
    total_wav = []
    total_lab = []
    total_x = []
    total_y = []
    total_z = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        lab = wav_data[1]
        lab_SEV = 2 * lab -1
        
        x, y, z = lab_SEV
        if x >= 0:
            x_label = 1
        else:
            x_label = 0

        if y >= 0:
            y_label = 1
        else:
            y_label = 0

        if z >= 0:
            z_label = 1
        else:
            z_label = 0
        x_labels = np.array([x_label])
        y_labels = np.array([y_label])
        z_labels = np.array([z_label])
        total_x.append(x_labels)
        total_y.append(y_labels)
        total_z.append(z_labels)
        # total_oct.append(label)
        
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    total_lab = torch.Tensor(np.array(total_lab))
    total_x = torch.Tensor(np.array(total_x))
    total_y = torch.Tensor(np.array(total_y))
    total_z = torch.Tensor(np.array(total_z))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
        
    return total_wav, total_lab, total_x, total_y, total_z, attention_mask, total_utt

def collate_fn_wav_lab_mask_text(batch):
    total_wav = []
    total_text = []
    total_lab = []
    total_wav_dur = []
    total_text_dur = []
    total_utt = []
    for wav_data in batch:

        wav, wav_dur = wav_data[0]   
        text, text_dur = wav_data[1]   
        lab = wav_data[2]
        total_wav.append(torch.Tensor(wav))
        total_text.append(torch.LongTensor(text))
        total_lab.append(lab)
        total_wav_dur.append(wav_dur)
        total_text_dur.append(text_dur)
        total_utt.append(wav_data[3])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    total_text = nn.utils.rnn.pad_sequence(total_text, batch_first=True)
    
    total_lab = torch.Tensor(np.array(total_lab))
    max_wav_dur = np.max(total_wav_dur)
    attention_wav_mask = torch.zeros(total_wav.shape[0], max_wav_dur)
    for data_idx, dur in enumerate(max_wav_dur):
        attention_wav_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_text, total_lab, attention_wav_mask, total_utt

def collate_fn_wav_lab_mask_mfcc(batch):
    total_wav = []
    total_mfcc = []
    total_lab = []
    total_wav_dur = []
    total_mfcc_dur = []
    total_utt = []
    for wav_data in batch:

        wav, wav_dur = wav_data[0]
        mfcc, mfcc_dur = wav_data[1]   
        lab = wav_data[2]
        total_wav.append(torch.Tensor(wav))
        total_mfcc.append(torch.Tensor(mfcc))
        total_lab.append(lab)
        total_wav_dur.append(wav_dur)
        total_mfcc_dur.append(mfcc_dur)
        total_utt.append(wav_data[3])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(np.array(total_lab))
    max_wav_dur = np.max(total_wav_dur)
    max_mfcc_dur = np.max(total_mfcc_dur)
    attention_wav_mask = torch.zeros(total_wav.shape[0], max_wav_dur)
    attention_mfcc_mask = torch.zeros(total_mfcc.shape[0], total_mfcc_dur)
    for data_idx, dur in enumerate(max_wav_dur):
        attention_wav_mask[data_idx,:dur] = 1
    for data_idx, dur in enumerate(max_mfcc_dur):
        attention_mfcc_mask[data_idx,:dur] = 1

    return total_wav, total_mfcc, total_lab, attention_wav_mask, attention_mfcc_mask, total_utt

def collate_fn_wav_lab_mask_SEV(batch):
    total_wav = []
    total_ran = []
    total_ang = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        lab = wav_data[1]
        r = np.sqrt(lab[0]**2 + lab[1]**2 + lab[2]**2)
        if r == 0:
            phi = 0
            theta = 0
        else:
            phi = np.arctan2(lab[1], lab[0])  # -pi ~ pi
            theta = np.arccos(lab[2] / r)    # 0 ~ pi
        radius = np.array([r])
        angles = np.array([phi, theta])
        total_wav.append(torch.Tensor(wav))
        total_ran.append(radius)
        total_ang.append(angles)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_ran = torch.Tensor(np.array(total_ran))
    total_ang = torch.Tensor(np.array(total_ang))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_ran, total_ang, attention_mask, total_utt

def collate_fn_wav_lab_mask_dual_VAD2SEV(batch):
    total_wav = []
    total_VAD_lab = []
    total_ran = []
    total_ang = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        SEV_lab = wav_data[1]
        VAD_lab = wav_data[2]
        r = np.sqrt(SEV_lab[0]**2 + SEV_lab[1]**2 + SEV_lab[2]**2)
        if r == 0:
            phi = 0
            theta = 0
        else:
            phi = np.arctan2(SEV_lab[1], SEV_lab[0])  # -pi ~ pi
            theta = np.arccos(SEV_lab[2] / r)    # 0 ~ pi
        radius = np.array([r])
        angles = np.array([phi, theta])
        total_wav.append(torch.Tensor(wav))
        total_VAD_lab.append(VAD_lab)
        total_ran.append(radius)
        total_ang.append(angles)
        total_dur.append(dur)
        total_utt.append(wav_data[3])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    total_VAD_lab = torch.Tensor(np.array(total_VAD_lab))
    total_ran = torch.Tensor(np.array(total_ran))
    total_ang = torch.Tensor(np.array(total_ang))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
        
    return total_wav, total_VAD_lab, total_ran, total_ang, attention_mask, total_utt

def collate_fn_wav_lab_mask_base_SEV(batch):
    total_wav = []
    total_lab = []
    total_ran = []
    total_ang = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        lab = wav_data[1]
        lab_SEV = 2 * lab -1
        r = np.sqrt(lab_SEV[0]**2 + lab_SEV[1]**2 + lab_SEV[2]**2)
        if r == 0:
            phi = 0
            theta = 0
        else:
            phi = np.arctan2(lab_SEV[1], lab_SEV[0])  # -pi ~ pi
            theta = np.arccos(lab_SEV[2] / r)    # 0 ~ pi
        radius = np.array([r])
        angles = np.array([phi, theta])
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_ran.append(radius)
        total_ang.append(angles)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    total_lab = torch.Tensor(np.array(total_lab))
    total_ran = torch.Tensor(np.array(total_ran))
    total_ang = torch.Tensor(np.array(total_ang))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
        
    return total_wav, total_lab, total_ran, total_ang, attention_mask, total_utt

def collate_fn_wav_lab_mask_dual_SEV(batch):
    total_wav = []
    total_lab = []
    total_ran = []
    total_ang = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        lab = wav_data[1]
        r = np.sqrt(lab[0]**2 + lab[1]**2 + lab[2]**2)
        if r == 0:
            phi = 0
            theta = 0
        else:
            phi = np.arctan2(lab[1], lab[0])  # -pi ~ pi
            theta = np.arccos(lab[2] / r)    # 0 ~ pi
        radius = np.array([r])
        angles = np.array([phi, theta])
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_ran.append(radius)
        total_ang.append(angles)
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    total_lab = torch.Tensor(np.array(total_lab))
    total_ran = torch.Tensor(np.array(total_ran))
    total_ang = torch.Tensor(np.array(total_ang))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
        
    return total_wav, total_lab, total_ran, total_ang, attention_mask, total_utt

def collate_fn_mel_lab_mask(batch):
    total_mel = []
    total_lab = []
    total_dur = []
    total_utt = []
    for mel_data in batch:
        mel, dur = mel_data[0]  # Mel spectrogram과 프레임 수
        lab = mel_data[1]       # Label
        total_mel.append(torch.Tensor(mel))  # Mel spectrogram 추가
        total_lab.append(lab)                # Label 추가
        total_dur.append(dur)                # Frame 수 추가
        total_utt.append(mel_data[2])        # Utterance 추가

    total_mel = nn.utils.rnn.pad_sequence(total_mel, batch_first=True)

    total_lab = torch.Tensor(np.array(total_lab))

    max_dur = np.max(total_dur)  # 최대 프레임 수
    attention_mask = torch.zeros(total_mel.shape[0], max_dur)  # (batch_size, max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx, :dur] = 1  # 유효 프레임에 1 할당

    return total_mel, total_lab, attention_mask, total_utt

def collate_fn_wav_lab_dur(batch):
    total_wav = []
    total_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:
        wav, dur = wav_data[0]   
        lab = wav_data[1]
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)  # 각 wav의 길이 저장
        total_utt.append(wav_data[2])

    # Pad wav sequences to the same length
    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    # Convert labels to Tensor
    total_lab = torch.Tensor(np.array(total_lab))
    total_dur = torch.tensor(total_dur, dtype=torch.int32)

    # Return wav_lengths as total_dur
    return total_wav, total_lab, total_dur, total_utt

def collate_fn_wav_lab_dur_SEV(batch):
    total_wav = []
    total_ran = []
    total_ang = []
    total_dur = []
    total_utt = []
    for wav_data in batch:
        wav, dur = wav_data[0]   
        lab = wav_data[1]
        r = np.sqrt(lab[0]**2 + lab[1]**2 + lab[2]**2)
        if r == 0:
            phi = 0
            theta = 0
        else:
            phi = np.arctan2(lab[1], lab[0])  # -pi ~ pi
            theta = np.arccos(lab[2] / r)    # 0 ~ pi
        radius = np.array([r])
        angles = np.array([phi, theta])
        total_wav.append(torch.Tensor(wav))
        total_ran.append(radius)
        total_ang.append(angles)
        total_dur.append(dur)  # 각 wav의 길이 저장
        total_utt.append(wav_data[2])

    # Pad wav sequences to the same length
    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    # Convert labels to Tensor
    total_ran = torch.Tensor(np.array(total_ran))
    total_ang = torch.Tensor(np.array(total_ang))
    total_dur = torch.tensor(total_dur, dtype=torch.int32)

    # Return wav_lengths as total_dur
    return total_wav, total_ran, total_ang, total_dur, total_utt


def collate_fn_wav_test3(batch):
    total_wav = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        total_wav.append(torch.Tensor(wav))
        total_dur.append(dur)
        total_utt.append(wav_data[1])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, attention_mask, total_utt

def collate_fn_wav_test3_dur(batch):
    total_wav = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        total_wav.append(torch.Tensor(wav))
        total_dur.append(dur)
        total_utt.append(wav_data[1])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    total_dur = torch.tensor(total_dur, dtype=torch.int32)

    return total_wav, total_dur, total_utt

def collate_fn_wav_eval(batch):
    total_wav = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        total_wav.append(torch.Tensor(wav))
        total_dur.append(dur)
        total_utt.append(wav_data[2])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, attention_mask, total_utt

def collate_fn_wav_lab_mask_dimcat(batch):
    total_wav = []
    total_dim_lab = []
    total_cat_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        dim_lab = wav_data[1]
        cat_lab = wav_data[2]
        total_wav.append(torch.Tensor(wav))
        total_dim_lab.append(dim_lab)
        total_cat_lab.append(cat_lab)
        total_dur.append(dur)
        total_utt.append(wav_data[3])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_dim_lab = torch.Tensor(np.array(total_dim_lab))
    total_cat_lab = torch.Tensor(np.array(total_cat_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_dim_lab, total_cat_lab, attention_mask, total_utt

def collate_fn_wav_lab_mask_dimcatsev(batch):
    total_wav = []
    total_dim_lab = []
    total_cat_lab = []
    total_sev_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        dim_lab = wav_data[1]
        cat_lab = wav_data[2]
        sev_lab = wav_data[3]
        total_wav.append(torch.Tensor(wav))
        total_dim_lab.append(dim_lab)
        total_cat_lab.append(cat_lab)
        total_sev_lab.append(sev_lab)
        total_dur.append(dur)
        total_utt.append(wav_data[4])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_dim_lab = torch.Tensor(np.array(total_dim_lab))
    total_cat_lab = torch.Tensor(np.array(total_cat_lab))
    total_sev_lab = torch.Tensor(np.array(total_sev_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_dim_lab, total_cat_lab, total_sev_lab, attention_mask, total_utt

def collate_fn_wav_lab_mask_dimsev(batch):
    total_wav = []
    total_dim_lab = []
    total_sev_lab = []
    total_dur = []
    total_utt = []
    for wav_data in batch:

        wav, dur = wav_data[0]   
        dim_lab = wav_data[1]
        sev_lab = wav_data[2]
        total_wav.append(torch.Tensor(wav))
        total_dim_lab.append(dim_lab)
        total_sev_lab.append(sev_lab)
        total_dur.append(dur)
        total_utt.append(wav_data[3])

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_dim_lab = torch.Tensor(np.array(total_dim_lab))
    total_sev_lab = torch.Tensor(np.array(total_sev_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_dim_lab, total_sev_lab, attention_mask, total_utt