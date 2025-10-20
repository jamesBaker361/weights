import torch
from torch.nn.modules import Linear, Module,ModuleList,Dropout1d,MultiheadAttention

class LinearEncoder(Module):
    def __init__(self,n_layers:int
                 ,embedding_dim:int,
                 input_dim:int, 
               #  residual:bool,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers=n_layers
        self.embedding_dim=embedding_dim
        step=(input_dim-embedding_dim)/self.n_layers
        #self.residual=residual
        dim_list=[input_dim]
        self.down_block_list=ModuleList([
            Linear(int(input_dim - (step*k)),int(input_dim-(step*(k+1))))
              for k in range(n_layers)])
        self.down_attention_list=ModuleList([MultiheadAttention(int(input_dim-(step*(k+1))),1) for k in range(n_layers)])
        self.down_time_emb_list=ModuleList([Linear(1, int(input_dim-(step*(k+1)))) for k in range(n_layers)])

        self.up_block_list=ModuleList([
            Linear(int(embedding_dim + (step*k)),int(embedding_dim+(step*(k+1))))
              for k in range(n_layers)])
        self.up_attention_list=ModuleList([MultiheadAttention(int(embedding_dim+(step*(k+1))),1) for k in range(n_layers)])
        self.up_time_emb_list=ModuleList([Linear(1, int(embedding_dim+(step*(k+1)))) for k in range(n_layers)])
        self.droput=Dropout1d()

    def forward(self,x,t):
        self.res_list=[]
        for layer,attention,time_emb in zip(self.down_block_list,self.down_attention_list, self.down_time_emb_list):
            x=layer(x)
            x=self.droput(x)
            x=torch.nn.LeakyReLU(x)
            _t=time_emb(t)
            x=attention(x,_t,_t)


        for layer,attention,time_emb in zip(self.up_block_list,self.up_attention_list,self.up_time_emb_list):
            x=layer(x)
            #x=self.droput(x)
            x=torch.nn.LeakyReLU()
            _t=time_emb(t)
            x=attention(x,_t,_t)

        return x

