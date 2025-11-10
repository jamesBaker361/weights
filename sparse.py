from torch import nn
import torch
import json
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, d_hidden:int,
                 l1_coeff:float,
                 act_size:int,
                 save_dir:str):
        super().__init__()
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(act_size, d_hidden, )))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, act_size, )))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, ))
        self.b_dec = nn.Parameter(torch.zeros(act_size, ))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.save_dir=save_dir
        self.act_size=act_size
    
    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed
        
    def get_version(self):
        return 1
    

    def save(self):
        version = self.get_version()
        torch.save(self.state_dict(), self.save_dir/(str(version)+".pt"))
        cfg={
            "d_hidden":self.d_hidden,
            "l1_coeff":self.l1_coeff,
            "act_size":self.act_size,
            "save_dir":self.save_dir
        }
        with open(self.save_dir/(str(version)+"_cfg.json"), "w") as f:
            json.dump(cfg, f)
        print("Saved as version", version)
    
    @classmethod
    def load(cls, version):
        cfg = (json.load(open(self.save_dir/(str(version)+"_cfg.json"), "r")))
        print(cfg)
        self = cls(**cfg)
        self.load_state_dict(torch.load(self.save_dir/(str(version)+".pt")))
        return self