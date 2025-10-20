import os
import argparse
from experiment_helpers.gpu_details import print_details
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

import torch
import accelerate
from accelerate import Accelerator
from huggingface_hub.errors import HfHubHTTPError
from accelerate import PartialState
import time
import torch.nn.functional as F
from PIL import Image
import random
import wandb
import numpy as np
import random
from diffusers import LCMScheduler,DiffusionPipeline,DEISMultistepScheduler,DDIMScheduler,SCMScheduler,AutoencoderDC
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from torchvision.transforms.v2 import functional as F_v2
from torchmetrics.image.fid import FrechetInceptionDistance
from data_helpers import PCADataset
from torch.utils.data import random_split, DataLoader
from models import LinearEncoder
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from transformers import AutoProcessor, CLIPModel
try:
    from torch.distributed.fsdp import register_fsdp_forward_method
except ImportError:
    print("cant import register_fsdp_forward_method")
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from huggingface_hub import create_repo,HfApi

parser=argparse.ArgumentParser()
parser.add_argument("--mixed_precision",type=str,default="fp16")
parser.add_argument("--project_name",type=str,default="person")
parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
parser.add_argument("--name",type=str,default="jlbaker361/model",help="name on hf")
parser.add_argument("--lr",type=float,default=0.0001)
parser.add_argument("--src_dataset",type=str,default="jlbaker361/idgaf")
parser.add_argument("--mode",type=str,default="pca",help="pca or vae")
parser.add_argument("--denoiser",type=str,default="linear")
parser.add_argument("--embedding_dim_internal",type=int,default=1024)
parser.add_argument("--n_layers",type=int,default=2)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--limit",type=int,default=-1)


def main(args):
    accelerator=Accelerator(log_with="wandb",mixed_precision=args.mixed_precision,gradient_accumulation_steps=args.gradient_accumulation_steps)
    print("accelerator device",accelerator.device)
    device=accelerator.device
    state = PartialState()
    print(f"Rank {state.process_index} initialized successfully")
    if accelerator.is_main_process or state.num_processes==1:
        accelerator.print(f"main process = {state.process_index}")
    if accelerator.is_main_process or state.num_processes==1:
        try:
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)
        except HfHubHTTPError:
            print("hf hub error!")
            time.sleep(random.randint(5,120))
            accelerator.init_trackers(project_name=args.project_name,config=vars(args))

            api=HfApi()
            api.create_repo(args.name,exist_ok=True)


    torch_dtype={
        "no":torch.float32,
        "fp16":torch.float16,
        "bf16":torch.bfloat16
    }[args.mixed_precision]

    if args.mode=="pca":
        dataset=PCADataset()
        

    test_size=args.batch_size
    train_size=len(dataset)-test_size

    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(42)

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    save_subdir=os.path.join(args.save_dir,args.name)
    os.makedirs(save_subdir,exist_ok=True)

    WEIGHTS_NAME="diffusion_pytorch_model.safetensors"
    CONFIG_NAME="config.json"
    
    save_path=os.path.join(save_subdir,WEIGHTS_NAME)
    config_path=os.path.join(save_subdir,CONFIG_NAME)


    for batch in train_loader:
        break

    input_dim=batch["weights"].size()[-1]

    if args.denoiser=="linear":
        denoiser=LinearEncoder(args.n_layers,args.embedding_dim_internal,input_dim)

    params=[p for p in denoiser.parameters()]
    optimizer=torch.optim.AdamW(params,args.lr)
    scheduler=DDIMScheduler()

    denoiser,optimizer,train_loader,test_loader,scheduler= accelerator.prepare(denoiser,optimizer,train_loader,test_loader,scheduler)

    start_epoch=1
    try:
        if args.load_hf:
            pretrained_weights_path=api.hf_hub_download(args.name,WEIGHTS_NAME,force_download=True)
            pretrained_config_path=api.hf_hub_download(args.name,CONFIG_NAME,force_download=True)
            denoiser.load_state_dict(torch.load(pretrained_weights_path,weights_only=True),strict=False)
            with open(pretrained_config_path,"r") as f:
                data=json.load(f)
            start_epoch=data["start_epoch"]+1
    except Exception as e:
        accelerator.print(e)

    state_dict=denoiser.state_dict()


    def save(state_dict:dict,e:int):
        #state_dict=???
        print("state dict len",len(state_dict))
        torch.save(state_dict,save_path)
        with open(config_path,"w+") as config_file:
            data={"start_epoch":e}
            json.dump(data,config_file, indent=4)
            pad = " " * 2048  # ~1KB of padding
            config_file.write(pad)
        print(f"saved {save_path}")
        try:
            api.upload_file(path_or_fileobj=save_path,
                                    path_in_repo=WEIGHTS_NAME,
                                    repo_id=args.name)
            api.upload_file(path_or_fileobj=config_path,path_in_repo=CONFIG_NAME,
                                    repo_id=args.name)
            print(f"uploaded {args.name} to hub")
        except Exception as e:
            accelerator.print("failed to upload")
            accelerator.print(e)

    for e in range(start_epoch,args.epochs+1):
        start=time.time()
        loss_buffer=[]
        train_loss=0.0
        for b,batch in enumerate(train_loader):
            if b==args.limit:
                break

            with accelerator.accumulate(params):
                batch=batch.unsqueeze(1)
                batch=batch.to(torch_dtype,device)
                t=torch.randint(0,len(scheduler),(len(batch)),device=device).long()
                noise=torch.randn_like(batch)

                noised=scheduler.add_noise(batch,noise,t)

                predicted=denoiser(noised,t)

                loss=F.mse_loss(batch.float(),predicted.float())

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                loss_buffer.append(loss.detach().cpu().detach())

                accelerator.backward(loss)
                '''if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)'''
                optimizer.step()
                #lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                accelerator.log({"train_loss": train_loss},)
                train_loss = 0.0

                
        accelerator.log({
            "avg_loss":np.mean(loss_buffer)
        })
        end=time.time()
        accelerator.print(f"epoch {e} elapsed {end-start} seconds ")
        save(denoiser.state_dict(),e)





if __name__=='__main__':
    print_details()
    start=time.time()
    args=parser.parse_args()
    print(args)
    main(args)
    end=time.time()
    seconds=end-start
    hours=seconds/(60*60)
    print(f"successful generating:) time elapsed: {seconds} seconds = {hours} hours")
    print("all done!")