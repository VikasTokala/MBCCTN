import torch
import numpy as np
from DCNN.matFileGen import writeMatFile
from DCNN.writeMatFileIPD import writeMatFileIPD
from DCNN.datasets.test_dataset import BaseDataset
from DCNN.utils.eval_utils import ild_db, ipd_rad, speechMask, gcc_phat_stft
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
fbins_ipd = 257

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
    E2E_MBCCTN_ISO_MINI_PATH = os.path.join(SNRfolders[i],"E2E_MBCCTN_ISO_MINI")
    GMBCCTN_OMVDR_PATH  = os.path.join(SNRfolders[i],"GMBCCTN_OMVDR")
    GMBCCTN_FCIM_PATH  = os.path.join(SNRfolders[i],"GMBCCTN_FCIM")
    GMBCCTN_OMVDR_MINI_PATH  = os.path.join(SNRfolders[i],"GMBCCTN_OMVDR_mini_2_8M")
    GMBCCTN_FCIM_MINI_PATH  = os.path.join(SNRfolders[i],"GMBCCTN_FCIM_mini_2_8M")
    
    dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH,OMVDR_DATASET_PATH,  mono=False)
    dataset_omvdr = BaseDataset(OMVDR_DATASET_PATH, CLEAN_DATASET_PATH,OMVDR_DATASET_PATH, mono=False)
    dataset_fcim = BaseDataset(FCIM_DATASET_PATH, CLEAN_DATASET_PATH,FCIM_DATASET_PATH, mono=False)
    dataset_e2e_mbcctn_iso = BaseDataset(E2E_MBCCTN_ISO_PATH, CLEAN_DATASET_PATH,OMVDR_DATASET_PATH, mono=False)
    dataset_gmbcctn_omvdr = BaseDataset(GMBCCTN_OMVDR_PATH, CLEAN_DATASET_PATH,OMVDR_DATASET_PATH, mono=False)
    dataset_gmbcctn_fcim = BaseDataset(GMBCCTN_FCIM_PATH, CLEAN_DATASET_PATH,FCIM_DATASET_PATH, mono=False)
    dataset_gmbcctn_fcim_mini = BaseDataset(GMBCCTN_FCIM_MINI_PATH, CLEAN_DATASET_PATH,FCIM_DATASET_PATH, mono=False)
    dataset_gmbcctn_omvdr_mini = BaseDataset(GMBCCTN_OMVDR_MINI_PATH, CLEAN_DATASET_PATH,OMVDR_DATASET_PATH, mono=False)
    dataset_e2e_mbcctn_iso_mini = BaseDataset(E2E_MBCCTN_ISO_MINI_PATH, CLEAN_DATASET_PATH,OMVDR_DATASET_PATH, mono=False)
    
    
    
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
    
    dataloader_gmbcctn_fcim_mini = torch.utils.data.DataLoader(
                                            dataset_gmbcctn_fcim_mini,  
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True,
                                            drop_last=False)
    
    dataloader_gmbcctn_omvdr_mini = torch.utils.data.DataLoader(
                                            dataset_gmbcctn_omvdr_mini, 
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True,
                                            drop_last=False)
    
    dataloader_e2e_mbcctn_iso_mini = torch.utils.data.DataLoader(
                                        dataset_e2e_mbcctn_iso_mini,
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
    dataloader_gmbcctn_fcim_mini = iter(dataloader_gmbcctn_fcim_mini)
    dataloader_gmbcctn_omvdr_mini = iter(dataloader_gmbcctn_omvdr_mini)
    dataloader_e2e_mbcctn_iso_mini = iter(dataloader_e2e_mbcctn_iso_mini)
    
    
    
    mie_omvdr = torch.zeros((375,fbins))
    mie_fcim = torch.zeros((375,fbins))
    mie_e2e_mbcctn_iso = torch.zeros((375,fbins))
    mie_gmbcctn_omvdr = torch.zeros((375,fbins))
    mie_gmbcctn_fcim = torch.zeros((375,fbins))
    mie_gmbcctn_omvdr_mini = torch.zeros((375,fbins))
    mie_gmbcctn_fcim_mini = torch.zeros((375,fbins))
    mie_e2e_mbcctn_iso_mini = torch.zeros((375,fbins))
    
    
    mpe_omvdr = torch.zeros((375,fbins_ipd))
    mpe_fcim = torch.zeros((375,fbins_ipd))
    mpe_e2e_mbcctn_iso = torch.zeros((375,fbins_ipd))
    mpe_gmbcctn_omvdr = torch.zeros((375,fbins_ipd))
    mpe_gmbcctn_fcim = torch.zeros((375,fbins_ipd))
    mpe_gmbcctn_omvdr_mini = torch.zeros((375,fbins_ipd))
    mpe_gmbcctn_fcim_mini = torch.zeros((375,fbins_ipd))
    mpe_e2e_mbcctn_iso_mini = torch.zeros((375,fbins_ipd))
    
    # breakpoint()exit
    
    for j in range(len(dataloader)):
        
        try:
            batch = next(dataloader)
            batch_omvdr = next(dataloader_omvdr)
            batch_fcim = next(dataloader_fcim)
            batch_e2e_mbcctn_iso = next(dataloader_e2e_mbcctn_iso)
            batch_gmbcctn_omvdr = next(dataloader_gmbcctn_omvdr)
            batch_gmbcctn_fcim = next(dataloader_gmbcctn_fcim)
            batch_gmbcctn_fcim_mini = next(dataloader_gmbcctn_fcim_mini)
            batch_gmbcctn_omvdr_mini = next(dataloader_gmbcctn_omvdr_mini)
            batch_e2e_mbcctn_iso_mini = next(dataloader_e2e_mbcctn_iso_mini)
            
        except StopIteration:
            break
        
        # Batch shape [noisy signal, clean signal, bf signal, noisy_audio_sample_path, clean_audio_sample_path, bf_audio_sample_path]
        clean_samples = (batch[1])[0] # [0] is used to remove the batch dimension
        noisy_samples = (batch[0])[0]
        
        omvdr_samples = (batch_omvdr[0])[0]
        fcim_samples = (batch_fcim[0])[0]
        e2e_mbcctn_iso_samples = (batch_e2e_mbcctn_iso[0])[0]
        gmbcctn_omvdr_samples = (batch_gmbcctn_omvdr[0])[0]
        gmbcctn_fcim_samples = (batch_gmbcctn_fcim[0])[0]
        gmbcctn_fcim_mini_samples = (batch_gmbcctn_fcim_mini[0])[0]
        gmbcctn_omvdr_mini_samples = (batch_gmbcctn_omvdr_mini[0])[0]
        e2e_mbcctn_iso_mini_samples = (batch_e2e_mbcctn_iso_mini[0])[0]
        
        clean_samples=(clean_samples)/(torch.max(clean_samples))
        
        target_stft_l = stft(clean_samples[0, :])
        target_stft_r = stft(clean_samples[3, :])
        
        noisy_stft_l = stft(noisy_samples[0, :])
        noisy_stft_r = stft(noisy_samples[3, :])
        
        omvdr_stft_l = stft(omvdr_samples[0, :])
        omvdr_stft_r = stft(omvdr_samples[1, :])
        
        fcim_stft_l = stft(fcim_samples[0, :])
        fcim_stft_r = stft(fcim_samples[1, :])
        
        e2e_mbcctn_iso_stft_l = stft(e2e_mbcctn_iso_samples[0, :])
        e2e_mbcctn_iso_stft_r = stft(e2e_mbcctn_iso_samples[1, :])
        
        gmbcctn_omvdr_stft_l = stft(gmbcctn_omvdr_samples[0, :])
        gmbcctn_omvdr_stft_r = stft(gmbcctn_omvdr_samples[1, :])
        
        gmbcctn_fcim_stft_l = stft(gmbcctn_fcim_samples[0, :])
        gmbcctn_fcim_stft_r = stft(gmbcctn_fcim_samples[1, :])
        
        gmbcctn_fcim_mini_stft_l = stft(gmbcctn_fcim_mini_samples[0, :])
        gmbcctn_fcim_mini_stft_r = stft(gmbcctn_fcim_mini_samples[1, :])
        
        gmbcctn_omvdr_mini_stft_l = stft(gmbcctn_omvdr_mini_samples[0, :])
        gmbcctn_omvdr_mini_stft_r = stft(gmbcctn_omvdr_mini_samples[1, :])
        
        e2e_mbcctn_iso_mini_stft_l = stft(e2e_mbcctn_iso_mini_samples[0, :])
        e2e_mbcctn_iso_mini_stft_r = stft(e2e_mbcctn_iso_mini_samples[1, :])
        
        mask = speechMask(target_stft_l,target_stft_r, threshold=20).squeeze(0)
        
        target_ild = ild_db(target_stft_l.abs(), target_stft_r.abs())
        omvdr_ild = ild_db(omvdr_stft_l.abs(), omvdr_stft_r.abs())
        fcim_ild = ild_db(fcim_stft_l.abs(), fcim_stft_r.abs())
        e2e_mbcctn_iso_ild = ild_db(e2e_mbcctn_iso_stft_l.abs(), e2e_mbcctn_iso_stft_r.abs())
        gmbcctn_omvdr_ild = ild_db(gmbcctn_omvdr_stft_l.abs(), gmbcctn_omvdr_stft_r.abs())
        gmbcctn_fcim_ild = ild_db(gmbcctn_fcim_stft_l.abs(), gmbcctn_fcim_stft_r.abs())
        gmbcctn_fcim_mini_ild = ild_db(gmbcctn_fcim_mini_stft_l.abs(), gmbcctn_fcim_mini_stft_r.abs())
        gmbcctn_omvdr_mini_ild = ild_db(gmbcctn_omvdr_mini_stft_l.abs(), gmbcctn_omvdr_mini_stft_r.abs())
        e2e_mbcctn_iso_mini_ild = ild_db(e2e_mbcctn_iso_mini_stft_l.abs(), e2e_mbcctn_iso_mini_stft_r.abs())
        
        
        
        omvdr_ild_error = (target_ild - omvdr_ild).abs()
        fcim_ild_error = (target_ild - fcim_ild).abs()
        e2e_mbcctn_iso_ild_error = (target_ild - e2e_mbcctn_iso_ild).abs()
        gmbcctn_omvdr_ild_error = (target_ild - gmbcctn_omvdr_ild).abs()
        gmbcctn_fcim_ild_error = (target_ild - gmbcctn_fcim_ild).abs()
        gmbcctn_fcim_mini_ild_error = (target_ild - gmbcctn_fcim_mini_ild).abs()
        gmbcctn_omvdr_mini_ild_error = (target_ild - gmbcctn_omvdr_mini_ild).abs()
        e2e_mbcctn_iso_mini_ild_error = (target_ild - e2e_mbcctn_iso_mini_ild).abs()
        
        
        mask = mask[50:,:]
        mask_sum=mask.sum(dim=1)
        mask_sum[mask_sum==0]=1
        
        omvdr_ild_error = omvdr_ild_error[50:,:]
        fcim_ild_error = fcim_ild_error[50:,:]
        e2e_mbcctn_iso_ild_error = e2e_mbcctn_iso_ild_error[50:,:]
        gmbcctn_omvdr_ild_error = gmbcctn_omvdr_ild_error[50:,:]
        gmbcctn_fcim_ild_error = gmbcctn_fcim_ild_error[50:,:]
        gmbcctn_fcim_mini_ild_error = gmbcctn_fcim_mini_ild_error[50:,:]
        gmbcctn_omvdr_mini_ild_error = gmbcctn_omvdr_mini_ild_error[50:,:]
        e2e_mbcctn_iso_mini_ild_error = e2e_mbcctn_iso_mini_ild_error[50:,:]
        
        
        mie_omvdr[j,:] = (omvdr_ild_error*mask).sum(dim=1)/mask_sum
        mie_fcim[j,:] = (fcim_ild_error*mask).sum(dim=1)/mask_sum
        mie_e2e_mbcctn_iso[j,:] = (e2e_mbcctn_iso_ild_error*mask).sum(dim=1)/mask_sum
        mie_gmbcctn_omvdr[j,:] = (gmbcctn_omvdr_ild_error*mask).sum(dim=1)/mask_sum
        mie_gmbcctn_fcim[j,:] = (gmbcctn_fcim_ild_error*mask).sum(dim=1)/mask_sum
        mie_gmbcctn_fcim_mini[j,:] = (gmbcctn_fcim_mini_ild_error*mask).sum(dim=1)/mask_sum
        mie_gmbcctn_omvdr_mini[j,:] = (gmbcctn_omvdr_mini_ild_error*mask).sum(dim=1)/mask_sum
        mie_e2e_mbcctn_iso_mini[j,:] = (e2e_mbcctn_iso_mini_ild_error*mask).sum(dim=1)/mask_sum
        
        ipd_target = ipd_rad(target_stft_l,target_stft_r).abs()
        ipd_omvdr = ipd_rad(omvdr_stft_l,omvdr_stft_r).abs()
        # breakpoint()
        ipd_fcim = ipd_rad(fcim_stft_l,fcim_stft_r).abs()
        ipd_e2e_mbcctn_iso = ipd_rad(e2e_mbcctn_iso_stft_l,e2e_mbcctn_iso_stft_r).abs()
        ipd_gmbcctn_omvdr = ipd_rad(gmbcctn_omvdr_stft_l,gmbcctn_omvdr_stft_r).abs()
        ipd_gmbcctn_fcim = ipd_rad(gmbcctn_fcim_stft_l,gmbcctn_fcim_stft_r).abs()
        ipd_gmbcctn_fcim_mini = ipd_rad(gmbcctn_fcim_mini_stft_l,gmbcctn_fcim_mini_stft_r).abs()
        ipd_gmbcctn_omvdr_mini = ipd_rad(gmbcctn_omvdr_mini_stft_l,gmbcctn_omvdr_mini_stft_r).abs()
        ipd_e2e_mbcctn_iso_mini = ipd_rad(e2e_mbcctn_iso_mini_stft_l,e2e_mbcctn_iso_mini_stft_r).abs()
        
        omvdr_ipd_error = (ipd_target - ipd_omvdr).abs()
        # omvdr_ipd_error = (torch.remainder(omvdr_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        omvdr_ipd_error = torch.rad2deg(omvdr_ipd_error)
        
        fcim_ipd_error = (ipd_target - ipd_fcim).abs()
        # fcim_ipd_error = (torch.remainder(fcim_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        fcim_ipd_error = torch.rad2deg(fcim_ipd_error)
        
        e2e_mbcctn_iso_ipd_error = (ipd_target - ipd_e2e_mbcctn_iso).abs()
        # e2e_mbcctn_iso_ipd_error = (torch.remainder(e2e_mbcctn_iso_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        e2e_mbcctn_iso_ipd_error = torch.rad2deg(e2e_mbcctn_iso_ipd_error)
        
        gmbcctn_omvdr_ipd_error = (ipd_target - ipd_gmbcctn_omvdr).abs()
        # gmbcctn_omvdr_ipd_error = (torch.remainder(gmbcctn_omvdr_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        gmbcctn_omvdr_ipd_error = torch.rad2deg(gmbcctn_omvdr_ipd_error)
        
        gmbcctn_fcim_ipd_error = (ipd_target - ipd_gmbcctn_fcim).abs()
        # gmbcctn_fcim_ipd_error = (torch.remainder(gmbcctn_fcim_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        gmbcctn_fcim_ipd_error = torch.rad2deg(gmbcctn_fcim_ipd_error)
        
        gmbcctn_fcim_mini_ipd_error = (ipd_target - ipd_gmbcctn_fcim_mini).abs()
        # gmbcctn_fcim_mini_ipd_error = (torch.remainder(gmbcctn_fcim_mini_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        gmbcctn_fcim_mini_ipd_error = torch.rad2deg(gmbcctn_fcim_mini_ipd_error)
        
        gmbcctn_omvdr_mini_ipd_error = (ipd_target - ipd_gmbcctn_omvdr_mini).abs()
        # gmbcctn_omvdr_mini_ipd_error = (torch.remainder(gmbcctn_omvdr_mini_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        gmbcctn_omvdr_mini_ipd_error = torch.rad2deg(gmbcctn_omvdr_mini_ipd_error)
        
        e2e_mbcctn_iso_mini_ipd_error = (ipd_target - ipd_e2e_mbcctn_iso_mini).abs()
        # e2e_mbcctn_iso_mini_ipd_error = (torch.remainder(e2e_mbcctn_iso_mini_ipd_error + torch.pi, 2 * torch.pi) - torch.pi).abs()*180/torch.pi
        e2e_mbcctn_iso_mini_ipd_error = torch.rad2deg(e2e_mbcctn_iso_mini_ipd_error)
        
        
        mask = speechMask(target_stft_l,target_stft_r,threshold=10).squeeze()
        # breakpoint()
        mask = mask[:fbins_ipd,:]
        mask_sum=mask.sum(dim=1)
        mask_sum[mask_sum==0]=1
        
        omvdr_ipd_error = omvdr_ipd_error[:fbins_ipd,:]
        fcim_ipd_error = fcim_ipd_error[:fbins_ipd,:]
        e2e_mbcctn_iso_ipd_error = e2e_mbcctn_iso_ipd_error[:fbins_ipd,:]
        gmbcctn_omvdr_ipd_error = gmbcctn_omvdr_ipd_error[:fbins_ipd,:]
        gmbcctn_fcim_ipd_error = gmbcctn_fcim_ipd_error[:fbins_ipd,:]
        gmbcctn_fcim_mini_ipd_error = gmbcctn_fcim_mini_ipd_error[:fbins_ipd,:]
        gmbcctn_omvdr_mini_ipd_error = gmbcctn_omvdr_mini_ipd_error[:fbins_ipd,:]
        e2e_mbcctn_iso_mini_ipd_error = e2e_mbcctn_iso_mini_ipd_error[:fbins_ipd,:]
        
        # breakpoint()
        mpe_omvdr[j,:] = (omvdr_ipd_error*mask).sum(dim=1)/mask_sum
        mpe_fcim[j,:] = (fcim_ipd_error*mask).sum(dim=1)/mask_sum
        mpe_e2e_mbcctn_iso[j,:] = (e2e_mbcctn_iso_ipd_error*mask).sum(dim=1)/mask_sum
        mpe_gmbcctn_omvdr[j,:] = (gmbcctn_omvdr_ipd_error*mask).sum(dim=1)/mask_sum
        mpe_gmbcctn_fcim[j,:] = (gmbcctn_fcim_ipd_error*mask).sum(dim=1)/mask_sum
        mpe_gmbcctn_fcim_mini[j,:] = (gmbcctn_fcim_mini_ipd_error*mask).sum(dim=1)/mask_sum
        mpe_gmbcctn_omvdr_mini[j,:] = (gmbcctn_omvdr_mini_ipd_error*mask).sum(dim=1)/mask_sum
        mpe_e2e_mbcctn_iso_mini[j,:] = (e2e_mbcctn_iso_mini_ipd_error*mask).sum(dim=1)/mask_sum
        
        print('Processed Signal ', j+1 , ' of ', len(dataloader_e2e_mbcctn_iso))
        
    writeMatFile(mie_omvdr,folPath=SNRfolders[i], method='omvdr_'+ SNRnames[i])
    writeMatFile(mie_fcim,folPath=SNRfolders[i], method='fcim_'+ SNRnames[i])
    writeMatFile(mie_e2e_mbcctn_iso,folPath=SNRfolders[i], method='e2e_mbcctn_iso_'+ SNRnames[i])
    writeMatFile(mie_gmbcctn_omvdr,folPath=SNRfolders[i], method='gmbcctn_omvdr_'+ SNRnames[i])
    writeMatFile(mie_gmbcctn_fcim,folPath=SNRfolders[i], method='gmbcctn_fcim_'+ SNRnames[i])
    writeMatFile(mie_gmbcctn_fcim_mini,folPath=SNRfolders[i], method='gmbcctn_fcim_mini_'+ SNRnames[i])
    writeMatFile(mie_gmbcctn_omvdr_mini,folPath=SNRfolders[i], method='gmbcctn_omvdr_mini_'+ SNRnames[i])
    writeMatFile(mie_e2e_mbcctn_iso_mini,folPath=SNRfolders[i], method='e2e_mbcctn_iso_mini_'+ SNRnames[i])

    writeMatFileIPD(mpe_omvdr,folPath=SNRfolders[i], method='omvdr_'+ SNRnames[i])
    writeMatFileIPD(mpe_fcim,folPath=SNRfolders[i], method='fcim_'+ SNRnames[i])
    writeMatFileIPD(mpe_e2e_mbcctn_iso,folPath=SNRfolders[i], method='e2e_mbcctn_iso_'+ SNRnames[i])
    writeMatFileIPD(mpe_gmbcctn_omvdr,folPath=SNRfolders[i], method='gmbcctn_omvdr_'+ SNRnames[i])
    writeMatFileIPD(mpe_gmbcctn_fcim,folPath=SNRfolders[i], method='gmbcctn_fcim_'+ SNRnames[i])
    writeMatFileIPD(mpe_gmbcctn_fcim_mini,folPath=SNRfolders[i], method='gmbcctn_fcim_mini_'+ SNRnames[i])
    writeMatFileIPD(mpe_gmbcctn_omvdr_mini,folPath=SNRfolders[i], method='gmbcctn_omvdr_mini_'+ SNRnames[i])
    writeMatFileIPD(mpe_e2e_mbcctn_iso_mini,folPath=SNRfolders[i], method='e2e_mbcctn_iso_mini_'+ SNRnames[i])
    
        
        
        
        
            
            

    

