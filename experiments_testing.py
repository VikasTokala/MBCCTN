import torch
import yaml
# !pip install tensorboardX
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
from DCNN.trainer import DCNNLightningModule
from glob import glob
import soundfile as sf
import os
from DCNN.datasets.test_dataset import BaseDataset

from DCNN.feature_extractors import Stft, IStft
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

config = {
    "defaults": [
        "model",
        "training",
        {"dataset": "speech_dataset"}
    ],
    "dataset":{
        "noisy_test_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Noisy_testset",
        "noisy_training_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Noisy_trainset",
        "noisy_validation_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Noisy_valset",
        "target_test_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Clean_testset",
        "target_training_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Clean_trainset",
        "target_validation_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Clean_valset",
        "bf_test_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Enhanced_omvdr_testset",
        "bf_training_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Enhanced_omvdr_trainset",
        "bf_validation_dataset_dir": "/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_dataset/Enhanced_omvdr_valset"

    },
    "model":{
        "attention": True,
        "ild_weight": 1,
        "ipd_weight": 10,
        "snr_loss_weight": 1,
        "stoi_weight": 10
    }

}

with open('config/config.yaml', 'w') as f_yaml:
    yaml.dump(config, f_yaml)
GlobalHydra.instance().clear()
initialize(config_path="./config")
config = compose("config")

test_name = 'FCIM'
MODEL_CHECKPOINT_PATH = '/Users/vtokala/Documents/Research/MBCCTN/outputs/GuidedBFenhancement.ckpt'
model = DCNNLightningModule(config)
model.eval()
torch.set_grad_enabled(False)
device = torch.device('cpu')
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
# checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
model.load_state_dict(checkpoint["state_dict"], strict=False)

paths=glob("/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_evalset/*/", recursive = True)
pathsEn=glob("/Users/vtokala/Documents/Research/Databases/Dataset_Binaural_2S/Multichannel_evalset/*/", recursive = True)


for j in range(len(paths)):
    
    paths = sorted(paths)
    pathsEn = sorted(pathsEn)
    NOISY_DATASET_PATH = os.path.join(paths[j],"Noisy_testset/")
    print(NOISY_DATASET_PATH)
    CLEAN_DATASET_PATH = os.path.join(paths[j],"Clean_testset/")
    BF_DATASET_PATH = os.path.join(paths[j],"FCIM/")
    ENHANCED_DATASET_PATH = os.path.join(pathsEn[j],"GMBCCTN_"+test_name+"/")
    dataset = BaseDataset(NOISY_DATASET_PATH, CLEAN_DATASET_PATH, BF_DATASET_PATH, mono=False)
    
    if os.path.isdir(ENHANCED_DATASET_PATH):
        print("Folder for Enhanced Signals Exists!")
        print(ENHANCED_DATASET_PATH)
        
        if os.path.exists(ENHANCED_DATASET_PATH):
    # Get a list of files and subdirectories in the folder
            contents = os.listdir(ENHANCED_DATASET_PATH)
            print(ENHANCED_DATASET_PATH)
            
    
        # Iterate over the contents and delete files or subdirectories
        for item in contents:
            item_path = os.path.join(ENHANCED_DATASET_PATH, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Delete files
            elif os.path.isdir(item_path):
                # Delete subdirectories' contents recursively
                for root, dirs, files in os.walk(item_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                # Remove the empty subdirectory
                os.rmdir(item_path)

            print(f"Contents of folder '{ENHANCED_DATASET_PATH}' have been deleted.")
    else:
        print("Folder for Enhanced Signals does not exist! - Creating Folder!!")
        os.mkdir(ENHANCED_DATASET_PATH)
    
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False
        

    )


    dataloader = iter(dataloader)
    # k = len(dataloader)

    for i in range (len(dataloader)): # Enhance 10 samples
        try:
            batch = next(dataloader)

        except StopIteration:
            break
    #     print(os.path.basename(batchEn[2][0]))

        noisy_samples = (batch[0])
        clean_samples = (batch[1])
        bf_output  = (batch[2])
        model_output = model(noisy_samples,bf_output)[0].detach().cpu()
        model_output = model_output/torch.max(model_output)
        # print(model_output.shape)

        # breakpoint()
    #     torchaudio.save(path, waveform, sample_rate)
        sf.write(ENHANCED_DATASET_PATH + os.path.basename(batch[3][0])[:len(os.path.basename(batch[3][0]))-4] + "_" + test_name + ".wav", model_output.numpy().transpose(), 16000) 
        # print(ENHANCED_DATASET_PATH + os.path.basename(batch[2][0])[:len(os.path.basename(batch[2][0]))-4] + "_DCCTN.wav")
        print(f"===== Computing Signal {i+1} of ", len(dataloader),"=====")



