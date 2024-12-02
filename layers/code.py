class SGU(nn.Module):
    def __init__(self,d_model,d_ff,c_in,dropout):
        super().__init__()
        self.in_layer=nn.Linear(d_model,d_ff)
        self.out_layer=nn.Linear(d_ff//2,d_model)
        self.spatial=nn.Linear(c_in,c_in)
        self.activation=nn.GELU()
        self.norm = nn.LayerNorm(d_ff // 2)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        res=x
        x=self.in_layer(x.permute(0,1,3,2)).permute(0,1,3,2)
        x=self.activation(x)
        x=self.dropout(x)
        
        x=torch.split(x,x.size(2)//2,dim=2)
        u=x[0]
        v=x[1]
        # print(u.shape,v.shape)
        v=self.norm(v.permute(0,1,3,2)).permute(0,1,3,2)
        v=self.spatial(v)
        x=u*v
        # print(x.shape)
        # exit()
        x=self.out_layer(x.permute(0,1,3,2)).permute(0,1,3,2)
        return x+res