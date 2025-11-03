import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from accelerate import Accelerator


def infer_proj(denoiser:torch.nn.Module,
               scheduler:DDIMScheduler,
               text_prompt:str,
               dim_proj:int,
               accelerator: Accelerator,
               device="cpu",
               dtype=torch.float32,
               num_inference_steps:int =10,
               n_samples:int=1,):
    
    noise=torch.randn([n_samples,dim_proj],device=device) #,dtype=dtype)
    
    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device,
    )
    
    latents=noise
    
    
    
    for i, t in enumerate(timesteps):
        latent_model_input = scheduler.scale_model_input(latents, t)
        t=torch.tensor([t]*n_samples).unsqueeze(-1).to(device,latents.dtype)
        if i==0:
            print("t",t.size(),t.dtype,t,t.device)
            print("latent model ",latent_model_input.size())
        
        
        noise_pred = denoiser(latent_model_input,t,)[0]
        
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        
    return latents