
# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import AutoModel
import importlib
# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model_path", type=str, default="./temp")
parser.add_argument("--head_dim", type=int, default=1024)
args = parser.parse_args()

utils.set_deterministic(args.seed)
SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
BATCH_SIZE = args.batch_size
ACCUMULATION_STEP = args.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS=args.epochs
LR=args.lr
MODEL_PATH = args.model_path
os.makedirs(MODEL_PATH, exist_ok=True)


import json
from collections import defaultdict
config_path = "configs/emospehreser.json"
with open(config_path, "r") as f:
    config = json.load(f)
audio_path = config["wav_dir"]
label_path = config["label_path"]

# Load the CSV file
df = pd.read_csv(label_path)
# Filter out only 'Train' samples
train_df = df[df['Split_Set'] == 'Train']
# Total number of samples
total_samples = len(train_df)

###############
#     oct     #
###############
# Classes (emotions)
oct_classes = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

# Calculate class frequencies
oct_class_frequencies = train_df[oct_classes].sum().to_dict()

# Calculate class weights
oct_class_weights = {cls: total_samples / (len(oct_classes) * freq) if freq != 0 else 0 for cls, freq in oct_class_frequencies.items()}
print(oct_class_weights)
# Convert to list in the order of classes
oct_weights_list = [oct_class_weights[cls] for cls in oct_classes]
# Convert to PyTorch tensor
oct_class_weights_tensor = torch.tensor(oct_weights_list, device='cuda', dtype=torch.float)
# Print or return the tensor
print(oct_class_weights_tensor)

total_dataset=dict()
total_dataloader=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_dim_labs, cur_sev_labs = utils.load_sev_adv_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    if dtype == "train":
        cur_wav_set = utils.WavSet(cur_wavs)
        cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_dim_emo_set = utils.ADV_EmoSet(cur_dim_labs)
    cur_sev_emo_set = utils.SEV_EmoSet(cur_sev_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_dim_emo_set, cur_sev_emo_set, cur_utts])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask_dimsev
    )

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
ssl_model.freeze_feature_encoder()
ssl_model.eval(); ssl_model.cuda()

########## Implement pooling method ##########
feat_dim = ssl_model.config.hidden_size

pool_model = net.MelStyleEncoder()
print(pool_model)
pool_model.cuda()

dim_ser_model = net.EmotionRegression(feat_dim, args.head_dim, 1, 3, dropout=0.5)
sev_ser_model = net.EmotionRegression(feat_dim, args.head_dim, 1, 8, dropout=0.5)
##############################################
dim_ser_model.eval(); dim_ser_model.cuda()
sev_ser_model.eval(); sev_ser_model.cuda()

ssl_opt = torch.optim.AdamW(ssl_model.parameters(), LR)
dim_ser_opt = torch.optim.AdamW(dim_ser_model.parameters(), LR)
sev_ser_opt = torch.optim.AdamW(sev_ser_model.parameters(), LR)

scaler = GradScaler()
ssl_opt.zero_grad(set_to_none=True)
dim_ser_opt.zero_grad(set_to_none=True)
sev_ser_opt.zero_grad(set_to_none=True)

pool_opt = torch.optim.AdamW(pool_model.parameters(), LR)
pool_opt.zero_grad(set_to_none=True)

lm = utils.LogManager()
lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val"])
lm.alloc_stat_type_list(["dev_aro", "dev_dom", "dev_val"])
lm.alloc_stat_type_list(["train_sev_loss"])
lm.alloc_stat_type_list(["dev_ce_loss"])
lm.alloc_stat_type_list(["dev_sev_loss"])

min_epoch=0
min_loss=1e10

# TensorBoard SummaryWriter 초기화
writer = SummaryWriter(log_dir=os.path.join(MODEL_PATH, "tensorboard_logs"))

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    lm.init_stat()
    ssl_model.train()
    pool_model.train()
    dim_ser_model.train()    
    sev_ser_model.train()    
    batch_cnt = 0

    if epoch < 5:
        sev_weight = 1 - (epoch / 5) * (1 - 0.01)  # Linearly decrease from 1 to 0.001
    else:
        sev_weight = 0.01  # Maintain 0.001 after 5 epochs


    for xy_pair in tqdm(total_dataloader["train"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        dim_y = xy_pair[1]; dim_y=dim_y.cuda(non_blocking=True).float()
        sev_y = xy_pair[2]; sev_y=sev_y.max(dim=1)[1]; sev_y=sev_y.cuda(non_blocking=True).long()
        mask = xy_pair[3]; mask=mask.cuda(non_blocking=True).float()
        
        with autocast(enabled=True):
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state # (B, T, 1024)
            ssl = pool_model(ssl, mask)
            
            dim_emo_pred = dim_ser_model(ssl)
            if epoch < 5:
                sev_emo_pred = sev_ser_model(ssl)
            
            ccc = utils.CCC_loss(dim_emo_pred, dim_y)
            if epoch < 5:
                sev_loss = utils.CE_weight_category(sev_emo_pred, sev_y, oct_class_weights_tensor)
            loss = 1.0 - ccc
            if epoch >= 5:
                total_loss = torch.sum(loss) / ACCUMULATION_STEP
            else:
                total_loss = torch.sum(loss) / ACCUMULATION_STEP + sev_weight * sev_loss / ACCUMULATION_STEP
        scaler.scale(total_loss).backward()
        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(total_dataloader["train"]):
            scaler.step(ssl_opt)
            scaler.step(dim_ser_opt)
            if epoch < 5:
                scaler.step(sev_ser_opt)
            scaler.step(pool_opt)
            scaler.update()
            ssl_opt.zero_grad(set_to_none=True)
            dim_ser_opt.zero_grad(set_to_none=True)
            if epoch < 5:
                sev_ser_opt.zero_grad(set_to_none=True)
            pool_opt.zero_grad(set_to_none=True)
        batch_cnt += 1

        # Logging
        lm.add_torch_stat("train_aro", ccc[0])
        lm.add_torch_stat("train_dom", ccc[1])
        lm.add_torch_stat("train_val", ccc[2])
        if epoch < 5:   
            lm.add_torch_stat("train_sev_loss", sev_loss)

    ssl_model.eval()
    pool_model.eval()
    dim_ser_model.eval() 
    sev_ser_model.eval() 
    total_dim_pred = [] 
    total_sev_pred = [] 
    total_dim_y = []
    total_sev_y = []
    for xy_pair in tqdm(total_dataloader["dev"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        dim_y = xy_pair[1]; dim_y=dim_y.cuda(non_blocking=True).float()
        sev_y = xy_pair[2]; sev_y=sev_y.max(dim=1)[1]; sev_y=sev_y.cuda(non_blocking=True).long()
        mask = xy_pair[3]; mask=mask.cuda(non_blocking=True).float()
        
        with torch.no_grad():
            ssl = ssl_model(x, attention_mask=mask).last_hidden_state
            ssl = pool_model(ssl, mask)
            dim_emo_pred = dim_ser_model(ssl)
            total_dim_pred.append(dim_emo_pred)
            total_dim_y.append(dim_y)
            if epoch < 5:
                sev_emo_pred = sev_ser_model(ssl)
                total_sev_pred.append(sev_emo_pred)
                total_sev_y.append(sev_y)

    # CCC calculation
    total_dim_pred = torch.cat(total_dim_pred, 0)
    total_dim_y = torch.cat(total_dim_y, 0)
    ccc = utils.CCC_loss(total_dim_pred, total_dim_y)
    if epoch < 5:
        sev_loss = utils.CE_weight_category(sev_emo_pred, sev_y, oct_class_weights_tensor)
    
    # Logging
    lm.add_torch_stat("dev_aro", ccc[0])
    lm.add_torch_stat("dev_dom", ccc[1])
    lm.add_torch_stat("dev_val", ccc[2])
    if epoch < 5:
        lm.add_torch_stat("dev_sev_loss", sev_loss)

    # Save model
    lm.print_stat()
    
    dev_aro = lm.get_stat("dev_aro")
    dev_dom = lm.get_stat("dev_dom")
    dev_val = lm.get_stat("dev_val")
    if epoch < 5:
        sev_dev_loss = lm.get_stat("dev_sev_loss")

    writer.add_scalar("Loss/Dev_aro", dev_aro, epoch)
    writer.add_scalar("Loss/Dev_dom", dev_dom, epoch)
    writer.add_scalar("Loss/Dev_val", dev_val, epoch)
    if epoch < 5:
        writer.add_scalar("Loss/Dev_sev", sev_dev_loss, epoch)

    dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
    
    torch.save(dim_ser_model.state_dict(), \
        os.path.join(MODEL_PATH, str(epoch)+"_dim_ser.pt"))
    torch.save(sev_ser_model.state_dict(), \
        os.path.join(MODEL_PATH, str(epoch)+"_sev_ser.pt"))
    torch.save(ssl_model.state_dict(), \
        os.path.join(MODEL_PATH, str(epoch)+"_ssl.pt"))
    torch.save(pool_model.state_dict(), \
        os.path.join(MODEL_PATH, str(epoch)+"_pool.pt"))
    
    if min_loss > dev_loss:
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",min_loss)
        save_model_list = ["ser", "ssl"]
        save_model_list.append("pool")

        torch.save(dim_ser_model.state_dict(), \
            os.path.join(MODEL_PATH,  "best_dim_ser.pt"))
        torch.save(sev_ser_model.state_dict(), \
            os.path.join(MODEL_PATH,  "best_sev_ser.pt"))
        torch.save(ssl_model.state_dict(), \
            os.path.join(MODEL_PATH,  "best_ssl.pt"))
        torch.save(pool_model.state_dict(), \
            os.path.join(MODEL_PATH,  "best_pool.pt"))