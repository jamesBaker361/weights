# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
# https://github.com/snap-research/weights2weights/blob/main/lora_w2w.py
import os
import math
from typing import Optional, List, Type, Set, Literal
import tqdm

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel,DiffusionPipeline
from safetensors.torch import save_file
import numpy as np
import datasets
from huggingface_hub import hf_hub_download
from PIL import Image


UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
#     "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "DownBlock2D",
    "UpBlock2D",
    
]  # locon, 3clier

LORA_PREFIX_UNET = "lora_unet"

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        proj, 
        v,
        #weight_embedding,
       # mean, 
        #std,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dtype=torch.float16
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.in_dim = org_module.in_features
        self.out_dim = org_module.out_features

        self.proj = proj.to(dtype)
        '''self.mean1 = mean[0:self.in_dim].bfloat16()
        self.mean2 = mean[self.in_dim:].bfloat16()
        self.std1 = std[0:self.in_dim].bfloat16()
        self.std2 = std[self.in_dim:].bfloat16()'''
        self.v1 = v[0:self.in_dim].to(dtype)
        self.v2 = v[self.in_dim: ].to(dtype)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        #self.scale = self.scale.bfloat16()
        

        self.multiplier = multiplier
        self.org_module = org_module

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return self.org_forward(x) +\
            (x@((self.proj@self.v1.T)).T)@(((self.proj@self.v2.T)))*self.multiplier*self.scale

### basic inference to generate images conditioned on text prompts
@torch.no_grad
def inference(network, unet, vae, text_encoder, tokenizer, prompt, negative_prompt, guidance_scale,
              noise_scheduler, ddim_steps, seed, generator,
              device,dtype):
    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (1, unet.in_channels, 512 // 8, 512 // 8),
        generator = generator,
        device = device
    ).to(dtype)
   

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
                            [negative_prompt], padding="max_length", max_length=max_length, return_tensors="pt"
                        )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    noise_scheduler.set_timesteps(ddim_steps) 
    latents = latents * noise_scheduler.init_noise_sigma
    
    for i,t in enumerate(tqdm.tqdm(noise_scheduler.timesteps)):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)
        with network:
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, timestep_cond= None).sample
        #guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)

class LoRAw2w(nn.Module):
    def __init__(
        self,
        proj,
       # mean, 
       # std, 
        v,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier = 1.0,
        alpha = 1.0,
        train_method: TRAINING_METHODS = "full",
        torch_dtype=torch.float16

    ) -> None:
        super().__init__()
        self.dtype=torch_dtype
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha
        self.proj = torch.nn.Parameter(proj)
        '''self.register_buffer("mean", torch.tensor(mean)) 
        self.register_buffer("std", torch.tensor(std)) '''
        self.register_buffer("v", torch.tensor(v))
        #self.register_buffer("weight_embedding",torch.tensor(weight_embedding))
        
        self.module = LoRAModule

        self.unet_loras = self.create_modules(
            LORA_PREFIX_UNET,
            unet,
            DEFAULT_TARGET_REPLACE,
            self.lora_dim,
            self.multiplier,
            train_method=train_method,
        )
      

    
        self.lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in self.lora_names
            ), f"duplicated lora name: {lora.lora_name}. {self.lora_names}"
            self.lora_names.add(lora.lora_name)


        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del unet
        torch.cuda.empty_cache()

    
    def reset(self):
        for lora in self.unet_loras:
            lora.proj = torch.nn.Parameter(self.proj.to(self.dtype))
    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
    ) -> list:
        
        counter = 0


        mm = []
        nn = []
        for name, module in root_module.named_modules():
            nn.append(name)
            mm.append(module)


        midstart = 0
        upstart = 0
        for i in range(len(nn)):
            if "mid_block" in nn[i]:
                midstart = i
                break

        for i in range(len(nn)):
            if "up_block" in nn[i]:
                upstart = i
                break
        
        mm = mm[:upstart]+mm[midstart:]+mm[upstart:midstart]
        nn = nn[:upstart]+nn[midstart:]+nn[upstart:midstart]
        
        

        loras = []
        names = []

        for i in range(len(mm)):
            name = nn[i]
            module = mm[i]
            if train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":  # Cross Attention と Time Embed 以外学習
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":  # Cross Attention 
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":  # Self Attention 
                if "attn1" not in name:
                    continue
            elif train_method == "xattn" or train_method == "xattn-strict":  # Cross Attention 
                if "to_k" in name:
                    continue

            elif train_method == "full":  # 全部学習
                pass
            else:
                raise NotImplementedError(
                    f"train_method: {train_method} is not implemented."
                )
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                        if train_method == 'xattn-strict':
                            if 'out' in child_name:
                                continue
                            if "to_k" in child_name:
                                continue
                        if train_method == 'noxattn-hspace':
                            if 'mid_block' not in name:
                                continue
                        if train_method == 'noxattn-hspace-last':
                            if 'mid_block' not in name or '.1' not in name or 'conv2' not in child_name:
                                continue
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")


                        in_dim = child_module.in_features
                        out_dim = child_module.out_features
                        combined_dim = in_dim+out_dim

                        lora = self.module(
                            lora_name, self.proj, 
                            self.v[counter:counter+combined_dim], 
                            
                            #self.weight_embedding[counter:counter+combined_dim],

                              child_module, multiplier, rank, self.alpha)
                        counter+=combined_dim
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
                        

        return loras

    

    def prepare_optimizer_params(self):
        all_params = []

        if self.unet_loras:  # 実質これしかない
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0
            
def load_models(path:str,device,dtype)->tuple:
    pipe=DiffusionPipeline.from_pretrained(path).to(device,dtype)
    return pipe.unet.to(device,dtype), pipe.vae.to(device,dtype), pipe.text_encoder.to(device,dtype), pipe.tokenizer,pipe.scheduler
            
if __name__=="__main__":
    v_path=hf_hub_download("snap-research/weights2weights",
                           filename="files/V.pt")
    v = torch.load(v_path)
    #proj = torch.load("../files/proj_1000pc.pt")
    proj=torch.tensor([np.random.normal()]*1000)
    path="SimianLuo/LCM_Dreamshaper_v7"
    unet=DiffusionPipeline.from_pretrained(path).unet
    network=LoRAw2w(proj,v,unet)
    
    
    prompt = "sks person" 
    negative_prompt = "low quality, blurry, unfinished, cartoon" 
    batch_size = 1
    height = 128
    width = 128
    guidance_scale = 3.0
    seed = 5
    ddim_steps = 10
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # random seed generator
    generator = torch.Generator(device=device)
    dtype=torch.float16
    unet, vae, text_encoder, tokenizer,scheduler =load_models(path,device,dtype)
    
    

    #run inference
    image = inference(network, unet, vae, text_encoder, tokenizer, prompt,
                      negative_prompt, guidance_scale, scheduler, ddim_steps, seed, generator, device,dtype)

    ### display image
    image = image.detach().cpu().float().permute(0, 2, 3, 1).numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))
    
    image.save("test.png")
    
    