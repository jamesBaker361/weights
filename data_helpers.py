import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch
import random
import csv
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from datasets import load_dataset
from huggingface_hub import hf_hub_download

import numpy as np
import torch.nn.functional as F

class PCADataset(Dataset):
    def __init__(self):
        super().__init__()
        file_path = hf_hub_download(
        repo_id="snap-research/weights2weights",
        filename="files/proj_1000pc.pt",   # <-- include the subfolder path here
        repo_type="model"               # or "dataset" if it's a dataset repo
        )


        self.data_tensor=torch.load(file_path,map_location="cpu")

        label_file_path=hf_hub_download(
        repo_id="snap-research/weights2weights",
        filename="files/identity_df.pt",   # <-- include the subfolder path here
        repo_type="model"               # or "dataset" if it's a dataset repo
        )

        self.label_df=torch.load(label_file_path,weights_only=False)



    def __len__(self):
        return self.data_tensor.size()[0]
    
    def __getitem__(self, index):
        row = self.label_df.iloc[index]  # or df.loc['some_index']
        cols_with_one = ", ".join(row[row == 1].index.tolist()).replace("_", " ")
        return { "weights": self.data_tensor[index],
                    "labels":cols_with_one
                     }
        
if __name__=="__main__":
    data=DataLoader(PCADataset(),batch_size=1)
    for row in data:
        break
    print(row)