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

    def forward(self,x,t,*args):
        self.res_list=[]
        for layer,attention,time_emb in zip(self.down_block_list,self.down_attention_list, self.down_time_emb_list):
            x=layer(x)
            x=self.droput(x)
            x=torch.nn.LeakyReLU()(x)
            _t=time_emb(t).unsqueeze(1)
            #print("_t",_t.size())
            x=attention(x,_t,_t)[0]


        for layer,attention,time_emb in zip(self.up_block_list,self.up_attention_list,self.up_time_emb_list):
            x=layer(x)
            #x=self.droput(x)
            x=torch.nn.LeakyReLU()(x)
            _t=time_emb(t).unsqueeze(1)
            x=attention(x,_t,_t)[0]

        return x

class LinearEncoderText(Module):
    def __init__(self,n_layers:int
                 ,embedding_dim:int,
                 input_dim:int, 
                 text_dim:int,
               #  residual:bool,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers=n_layers
        self.text_dim=text_dim
        self.embedding_dim=embedding_dim
        step=(input_dim-embedding_dim)/self.n_layers
        #self.residual=residual
        dim_list=[input_dim]
        self.down_block_list=ModuleList([
            Linear(int(input_dim - (step*k)),int(input_dim-(step*(k+1))))
              for k in range(n_layers)])
        self.down_attention_list=ModuleList([MultiheadAttention(int(input_dim-(step*(k+1))),1, batch_first=True) for k in range(n_layers)])
        self.down_text_attention_list=ModuleList([MultiheadAttention(int(input_dim-(step*(k+1))),1,batch_first=True) for k in range(n_layers)])
        self.down_time_emb_list=ModuleList([Linear(1, int(input_dim-(step*(k+1)))) for k in range(n_layers)])
        self.down_text_emb_list=ModuleList([Linear(text_dim, int(input_dim-(step*(k+1)))) for k in range(n_layers)])

        self.up_block_list=ModuleList([
            Linear(int(embedding_dim + (step*k)),int(embedding_dim+(step*(k+1))))
              for k in range(n_layers)])
        self.up_attention_list=ModuleList([MultiheadAttention(int(embedding_dim+(step*(k+1))),1,batch_first=True) for k in range(n_layers)])
        self.up_text_attention_list=ModuleList([MultiheadAttention(int(input_dim+(step*(k+1))),1,batch_first=True) for k in range(n_layers)])
        self.up_time_emb_list=ModuleList([Linear(1, int(embedding_dim+(step*(k+1)))) for k in range(n_layers)])
        self.up_text_emb_list=ModuleList([Linear(text_dim, int(embedding_dim+(step*(k+1)))) for k in range(n_layers)])
        self.droput=Dropout1d()

    def forward(self,x,t,text):
        self.res_list=[]
        for layer,attention,time_emb,text_attention,text_emb in zip(self.down_block_list,
                                            self.down_attention_list, 
                                            self.down_time_emb_list,
                                            self.down_text_attention_list,
                                            self.down_text_emb_list):
            x=layer(x)
            x=self.droput(x)
            x=torch.nn.LeakyReLU()(x)
            _t=time_emb(t).unsqueeze(1)
            print("_t size ",_t.size())
            x=attention(x,_t,_t)[0]
            _text=text_emb(text)
            print("_text ",_text.size())
            x=text_attention(x,_text,_text)[0]


        for layer,attention,time_emb,text_attention,text_emb in zip(self.up_block_list,
                                            self.up_attention_list,
                                            self.up_time_emb_list,
                                            self.up_text_attention_list,
                                            self.up_text_emb_list):
            x=layer(x)
            #x=self.droput(x)
            x=torch.nn.LeakyReLU()(x)
            _t=time_emb(t).unsqueeze(1)
            x=attention(x,_t,_t)[0]
            _text=text_emb(text)
            x=text_attention(x,_text,_text)[0]

        return x

