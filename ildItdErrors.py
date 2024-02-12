import torch
import numpy as np
from DCNN.matFileGen import writeMatFile
from DCNN.writeMatFileIPD import writeMatFileIPD
from DCNN.datasets.test_dataset import BaseDataset
from DCNN.utils.eval_utils import ild_db, ipd_rad, speechMask
from glob import glob
import os
from DCNN.feature_extractors import Stft, IStft
import librosa.display 

parentDir = "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_evalset/*/"

SNRfolders = sorted(glob(parentDir, recursive=True))

win_len = 400
win_inc = 100
fft_len = 512
fbins = int(fft_len/2 + 1) - 50
fbins_ipd = 50

stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
# breakpoint()
SNRlist = [-3,-6,0,12,15,3,6,9]
SNRnames =  ['m3', 'm6', '0','12','15','3','6','9']

for i in range (len(SNRfolders)):
    print('Processed Signals of SNR ', SNRlist[i] )
    
    CLEAN_DATASET_PATH =  os.path.join(SNRfolders[i],"Clean_testset")
    NOISY_DATASET_PATH =  os.path.join(SNRfolders[i],"Noisy_testset")
    FCIM_DATASET_PATH =  os.path.join(SNRfolders[i],"FCIM")
    OMVDR_DATASET_PATH = os.path.join(SNRfolders[i],"OMVDR")
    E2E_MBCCTN_ISO_PATH = os.path.join(SNRfolders[i],"E2E_MBCCTN_ISO")
    GMBCCTN_OMVDR_PATH  = os.path.join(SNRfolders[i],"GMBCCTN_OMVDR")
    GMBCCTN_FCIM_PATH  = os.path.join(SNRfolders[i],"GMBCCTN_FCIM")
    
    dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
    dataset_omvdr = BaseDataset(OMVDR_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
    dataset_fcim = BaseDataset(FCIM_DATASET_PATH, CLEAN_DATASET_PATH, mono=False)
    dataset_e2e_mbcctn_iso = BaseDataset(E2E_MBCCTN_ISO_PATH, CLEAN_DATASET_PATH, mono=False)
    dataset_gmbcctn_omvdr = BaseDataset(GMBCCTN_OMVDR_PATH, CLEAN_DATASET_PATH, mono=False)
    dataset_gmbcctn_fcim = BaseDataset(GMBCCTN_FCIM_PATH, CLEAN_DATASET_PATH, mono=False)
    
    
    dataloader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                drop_last=False)
    
    dataloader_omvdr = torch.utils.data.DataLoader(
                                                dataset_omvdr,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                drop_last=False)
    
    
    dataloader_fcim = torch.utils.data.DataLoader( 
                                                dataset_fcim,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                drop_last=False)
    
    dataloader_e2e_mbcctn_iso = torch.utils.data.DataLoader(
                                                dataset_e2e_mbcctn_iso,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                drop_last=False)
    
    dataloader_gmbcctn_omvdr = torch.utils.data.DataLoader(
                                                dataset_gmbcctn_omvdr,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                drop_last=False)
    
    dataloader_gmbcctn_fcim = torch.utils.data.DataLoader(
                                                dataset_gmbcctn_fcim,
                                                batch_size=1,
                                                shuffle=False,
                                                pin_memory=True,
                                                drop_last=False)
    
        
    
    
    dataloader = iter(dataloader)
    dataloader_omvdr = iter(dataloader_omvdr)
    dataloader_fcim = iter(dataloader_fcim)
    dataloader_e2e_mbcctn_iso = iter(dataloader_e2e_mbcctn_iso)
    dataloader_gmbcctn_omvdr = iter(dataloader_gmbcctn_omvdr)
    dataloader_gmbcctn_fcim = iter(dataloader_gmbcctn_fcim)
    
    
    mie_omvdr = torch.zeros((375,fbins))
    mie_fcim = torch.zeros((375,fbins))
    mie_e2e_mbcctn_iso = torch.zeros((375,fbins))
    mie_gmbcctn_omvdr = torch.zeros((375,fbins))
    mie_gmbcctn_fcim = torch.zeros((375,fbins))
    
    
    mpe_omvdr = torch.zeros((375,fbins_ipd))
    mpe_fcim = torch.zeros((375,fbins_ipd))
    mpe_e2e_mbcctn_iso = torch.zeros((375,fbins_ipd))
    mpe_gmbcctn_omvdr = torch.zeros((375,fbins_ipd))
    mpe_gmbcctn_fcim = torch.zeros((375,fbins_ipd))
    
    for j in range(len(dataloader_e2e_mbcctn_iso)):
        
        try:
            batch_omvdr = next(dataloader_omvdr)
            batch_fcim = next(dataloader_fcim)
            batch_e2e_mbcctn_iso = next(dataloader_e2e_mbcctn_iso)
            batch_gmbcctn_omvdr = next(dataloader_gmbcctn_omvdr)
            batch_gmbcctn_fcim = next(dataloader_gmbcctn_fcim)
            
            

    

