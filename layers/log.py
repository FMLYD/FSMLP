
import math
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import torch_dct as dct
from einops import rearrange
from models.iTransformer import iTransformer
# from collections import OrderedDict
from layers.layers import *
from layers.RevIN import RevIN
from mamba_ssm import Mamba
import random
# Cell

class SimplexLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=False):
        super(SimplexLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        
        self.weight = nn.Parameter(torch.randn(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # Apply softmax to the weight along the input feature dimension
        # weight = F.softmax(self.weight, dim=1)
        weight=torch.log(self.weight.abs()+1)
        # if self.input_features==self.output_features:
        #     weight=weight+weight.T
        # weight=F.relu(self.weight)+1
        # weight=self.weight
        # weight=weight*weight.detach()
        weight=weight/weight.sum(1,keepdim=True)
        # # weight=torch.randn(self.output_features,self.input_features).to(softmax_weight.device)
        # weight=weight*self.var+softmax_weight
        
        # Perform the linear transformation
        output = F.linear(input, weight, self.bias)
        return output
    def loss(self):
        return self.weight.abs().sum()
def spherical_gaussian_kernel_matrix(input_tensor, sigma):
    # 计算单位向量
    x_unit = input_tensor / input_tensor.norm(dim=1, keepdim=True)  # shape (b, d, n)
    
    # 计算球面夹角
    cos_theta = torch.bmm(x_unit.permute(0, 2, 1), x_unit)  # shape (b, n, n)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # 确保在范围内
    theta = torch.acos(cos_theta)  # 计算夹角

    # 计算球面高斯核
    similarity_matrix = torch.exp(-torch.sin(theta)**2 / (2 * sigma**2))  # shape (b, n, n)
    return similarity_matrix
def local_weighted_regression(input_tensor,linear,random_layer):
    
    b, d, n = input_tensor.shape
    
    ori_embedding=input_tensor
    
    # input_tensor=linear(input_tensor.detach())+input_tensor
    # 计算余弦相似度
    # input_tensor=random_layer(input_tensor)
    normalized_input = input_tensor / input_tensor.norm(dim=1, keepdim=True)
    similarity_matrix = torch.bmm(normalized_input.permute(0, 2, 1), normalized_input)  # shape (b, n, n)
    # similarity_matrix=spherical_gaussian_kernel_matrix(input_tensor,1)
    # distances = torch.norm(input_tensor.unsqueeze(3) - input_tensor.unsqueeze(2), dim=1)
    
    # similarity_matrix = 1 / (1 + distances)  # 使用距离计算相似度，避免零除

    # 计算权重（使用softmax）
    # weights = F.leaky_relu(similarity_matrix)  # shape (b, n, n)
    # similarity_matrix=torch.abs(similarity_matrix)
    weights = torch.softmax(similarity_matrix**2, dim=-1)  # shape (b, n, n)
    # weights=F.sigmoid(similarity_matrix)
    # 加权聚合
    output = torch.bmm(weights.detach(), ori_embedding.permute(0, 2, 1))  # shape (b, n, d)
    return output.permute(0, 2, 1)  # shape (b, d, n)
class ReuseLinear(nn.Module):
    def __init__(self, d,n):
        super().__init__()
        self.linear_layers =nn.ModuleList([nn.Linear(n,1) for _ in range(d)])  # d 个线性层的权重，形状 (d, n, m)
        self.random_permutation_matrix = torch.empty((n, n), dtype=torch.int64)

        for i in range(n):
            
            self.random_permutation_matrix[i] = torch.randperm(n)
    def forward(self, x):

        b, p, d, n = x.shape
        

        # 先将 x 的形状变换为 (b * p * d, n, n) 以便于进行矩阵乘法
        res=x.clone()
        for i in range(d):
            for j in range(n):
                res[:,:,i,j]=self.linear_layers[i](x[:,:,i,self.random_permutation_matrix[i].to(x.device)]).reshape(-1,1)
        

        # 使用 einsum 计算输出
        # 这里是对每个 n 的输入矩阵应用线性层，输出将是 (b * p * d, n, m)

        # 将输出调整为 (b, p, d, n, 1)
        return res  # 返回形状为 (b, p, d, n, 1)

def gaussian_kernel(tokens, sigma=1.0):
    # tokens: shape (b, d, n)
    
    # 计算每对 token 之间的平方距离
    pairwise_sq_dists = torch.cdist(tokens.permute(0, 2, 1), tokens.permute(0, 2, 1)) ** 2  # shape (b, n, n)

    # 计算高斯核相似性矩阵
    similarity_matrix = torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))  # shape (b, n, n)
    
    return similarity_matrix
class HyperNetwork(nn.Module):
    def __init__(self, d, hidden_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(d, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim * output_dim)  # 生成权重矩阵
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 生成偏置
        self.output_dim=output_dim
    def forward(self, embedding):
        x = self.fc1(embedding)
        weights = self.fc2(x)
        biases = self.fc3(x)  
        output_dim=self.output_dim
        return weights.view(-1, output_dim, output_dim), biases.view(-1, output_dim) 
from .FAN import FAN
def randomly_fix_parameters(layer, fixed_ratio=0.8):
    # 固定权重参数
    weight_np = layer.weight.data.cpu().numpy()
    total_weight_params = weight_np.size
    num_fixed_weight_params = int(total_weight_params * fixed_ratio)
    flat_weight_indices = np.random.choice(total_weight_params, num_fixed_weight_params, replace=False)
    multi_dim_weight_indices = np.unravel_index(flat_weight_indices, weight_np.shape)
    weight_mask = np.ones(weight_np.shape, dtype=bool)
    weight_mask[multi_dim_weight_indices] = False
    layer.register_buffer('weight_mask', torch.tensor(weight_mask, dtype=torch.bool))
    
    # 将固定的权重参数设置为0，并禁止它们的梯度更新
    with torch.no_grad():
        layer.weight[~layer.weight_mask] = 0
    layer.weight.requires_grad = True

    def weight_hook(grad):
        grad[~layer.weight_mask] = 0
        return grad

    layer.weight.register_hook(weight_hook)

    # 固定偏置参数（如果存在偏置）
    if layer.bias is not None:
        bias_np = layer.bias.data.cpu().numpy()
        total_bias_params = bias_np.size
        num_fixed_bias_params = int(total_bias_params * fixed_ratio)
        flat_bias_indices = np.random.choice(total_bias_params, num_fixed_bias_params, replace=False)
        bias_mask = np.ones(bias_np.shape, dtype=bool)
        bias_mask[flat_bias_indices] = False
        layer.register_buffer('bias_mask', torch.tensor(bias_mask, dtype=torch.bool))

        # 将固定的偏置参数设置为0，并禁止它们的梯度更新
        with torch.no_grad():
            layer.bias[~layer.bias_mask] = 0
        layer.bias.requires_grad = True

        def bias_hook(grad):
            grad[~layer.bias_mask] = 0
            return grad

        layer.bias.register_hook(bias_hook)
class SparseLinear(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        # 定义用于将 d 维度降到 n 的 Linear 层
        self.linear = nn.Linear(d, n)
    def forward(self, x):
        # 假设输入 x 的形状为 (b, p, d, n)
        b, p, d, n = x.shape
        
        # 将 d 维度降到 n 维度
        # 重塑 x 以适配 Linear 层的输入要求，然后再变回原来的形状
        x_reshaped = x.permute(0, 1, 3, 2)  # 变成 (b, p, n, d) 以便将 d 转换为 n
        x_transformed = self.linear(x_reshaped)  # 应用 Linear 层
        x_transformed = x_transformed.permute(0, 1, 3, 2)  # 恢复到 (b, p, n, n)

        # 计算 softmax，生成 (b, p, n, n) 的权重矩阵
        softmax_weights = torch.softmax(x_transformed, dim=-1)
        
        # 使用 softmax 权重转变原始矩阵 (b, p, d, n)
        output = torch.einsum('bpqn,bpdn->bpdn', softmax_weights, x)
        
        return output
class RandomLinear(nn.Module):
    def __init__(self, n, d):
        super(RandomLinear, self).__init__()
        self.n = n
        self.d = d
        
        # 创建一个包含 d 个(n, n)线性层的权重矩阵
        self.linears = nn.Parameter(torch.randn(d, n, n))  # (d, n, n)
        self.bias = nn.Parameter(torch.randn(d, n))  # (d, n)，偏置

    def forward(self, x):
        # x 的形状为 (b, p, d, n)
        b, p, d, _ = x.shape

        # 进行矩阵乘法以获取线性变换
        # 将 x 的维度调整为 (b * p, d, n) 以进行批量矩阵乘法
        x_reshaped = x.view(b , p, d, self.n)  # (b*p, d, n)
        res=x_reshaped.clone()
        # 进行线性变换: (b*p, d, n) @ (d, n, n) + (d, n) => (b*p, d, n)
        
        for i in range(d):
            for j in range(b):
                res[j,:,i,:] = torch.matmul(x_reshaped[j,:,i,:], self.linears[i]) + self.bias[i].unsqueeze(0)
        # 将输出的形状调整为 (b, p, d, n)
        return res  # (b, p, d, n)
def print_zero_weights(layer):
    # 确保输入是线性层
    if not isinstance(layer, nn.Linear):
        raise ValueError("Input must be a nn.Linear layer")
    
    # 获取权重
    weight = layer.weight.data
    
    # 计算零值的数量
    num_zero_weights = (weight == 0).sum().item()
    total_weights = weight.numel()  # 权重的总数
    
    # 打印结果
    print(f"Total weights: {total_weights}")
    print(f"Number of zero weights: {num_zero_weights}")
    print(f"Percentage of zero weights: {num_zero_weights / total_weights * 100:.2f}%")
class CompressedEigenLinear(nn.Module):
    def __init__(self, n, k):
        super(CompressedEigenLinear, self).__init__()
        self.n = n
        self.k = k  # 要保留的特征值数量

        # 初始化权重矩阵 W
        self.W = nn.Parameter(torch.randn(n, n))  # 权重矩阵 W

        # 计算特征值和特征向量
        self.eigenvalues, self.eigenvectors = self.compute_eigen_decomposition()
        self.eigenvalues=nn.Parameter(self.eigenvalues)
        self.eigenvectors=nn.Parameter(self.eigenvectors)
    def compute_eigen_decomposition(self):
        # 计算特征值分解
        eigenvalues, eigenvectors = torch.linalg.eig(self.W)

        # 提取实部的特征值
        eigenvalues = eigenvalues.real  # 只保留实部的特征值

        # 找到前 k 个最大的特征值及其对应的特征向量
        _, indices = torch.topk(eigenvalues, self.k)

        # 返回最大的 k 个特征值和对应的特征向量
        return eigenvalues[indices], eigenvectors[:, indices].real  # 返回实部的特征向量

    def forward(self, x):
        # 生成对角特征值矩阵
        Lambda_diag = torch.diag(self.eigenvalues)  # 形状 (k, k)

        # 计算 W_k = Q_k * Lambda_diag * Q_k^T
        W_k = torch.matmul(self.eigenvectors, torch.matmul(Lambda_diag, self.eigenvectors.t()))  # 形状 (n, n)

        # 使用批量矩阵乘法
        output = torch.matmul(x, W_k.unsqueeze(0))  # x 的形状为 (b, p, d, n)，W_k 需要扩展为 (1, n, n)

        return output  # 输出形状为 (b, p, d, n)
class channel_mix(nn.Module):
    def __init__(self,c_in,d_model,m_model,f_model,e_layers,dropout):
        super().__init__()
        self.f_model=f_model
        print(f_model)
        if f_model!=0:
            self.emd_time=nn.Linear(d_model,self.f_model)
            self.out_time=nn.Linear(self.f_model,d_model)
        self.e_layers=e_layers
        self.emd=SimplexLinear(c_in,m_model)
        self.cin=c_in
        self.activation=nn.SELU()
        self.m_model=m_model
        self.hypernetwork = HyperNetwork(64, 128, c_in)
        self.embeddings = nn.Parameter(torch.randn(f_model, 64)) 
        self.trans_layer=nn.ModuleList([nn.Linear(m_model,m_model) for _ in range(f_model)])
        self.out_layers=SimplexLinear(m_model,c_in)
        self.random_emd=nn.Linear(c_in,m_model)
        self.random_up=nn.Linear(m_model,c_in)
        self.cos_layers=nn.ModuleList([nn.Sequential(nn.Linear(f_model, 2*f_model),
                                nn.SELU(),
                                nn.Dropout(dropout),
                                nn.Linear(f_model*2, f_model)) for _ in range(e_layers)])
        self.down=nn.ModuleList([nn.Linear(m_model,m_model) for _ in range(f_model)])
        # self.down=nn.ModuleList([nn.Linear(m_model,m_model) for _ in range(f_model)])

        self.ffn=nn.Linear(10,10)
        for param in self.random_emd.parameters():
            param.requires_grad = False
        # for param in self.out_layers.parameters():
        #     param.requires_grad = False
        # randomly_fix_parameters(self.random_emd,0.9)
        # randomly_fix_parameters(self.out_layers,0.9)

        self.random_layers=nn.ModuleList([nn.Linear(m_model,m_model) for _ in range(e_layers)])
        for layer in self.random_layers:
            randomly_fix_parameters(layer,0.995)
        self.dw_conv=nn.ModuleList([nn.Conv1d(f_model,f_model,kernel_size=71,stride=1,padding='same',groups=f_model) for _ in range(e_layers)])
        with torch.no_grad():
            conv=self.dw_conv[0]
            conv.weight.fill_(1/71)  # 将权重设置为全1
            if conv.bias is not None:
                conv.bias.fill_(0.0)
        # for layer in self.trans_layer:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        self.layers=nn.ModuleList([SimplexLinear(m_model,m_model) for _ in range(e_layers)])
        # self.layers=nn.ModuleList([CompressedEigenLinear(m_model,4) for _ in range(e_layers)])
        # self.layers=nn.ModuleList([RandomLinear(m_model,f_model) for _ in range(e_layers)])

        self.time_layers=nn.ModuleList([nn.Linear(f_model,f_model) for _ in range(e_layers)])
        # self.time_layers=nn.ModuleList([nn.Sequential(nn.Linear(f_model, 2*f_model),
        #                         nn.SELU(),
        #                         nn.Dropout(dropout),
        #                         nn.Linear(f_model*2, f_model)) for _ in range(e_layers)])
        self.dropout=nn.Dropout(dropout)
        self.up=nn.Linear(10,m_model)
        self.layer_norms1=nn.ModuleList([nn.LayerNorm(c_in)for _ in range(e_layers)])
        self.layer_norms2=nn.ModuleList([nn.LayerNorm(f_model)for _ in range(e_layers)])
        self.perm=[torch.randperm(c_in) for _ in range(10)]

        self.row_layers=nn.ModuleList([nn.Linear(30,30) for _ in range(e_layers)])
        self.col_layers=nn.ModuleList([nn.Linear(30,30) for _ in range(e_layers)])
        # self.col_layers=nn.ModuleList([nn.Conv1d(f_model,f_model,kernel_size=30,stride=1,padding='same',groups=f_model) for _ in range(e_layers)])

        self.prototypes=torch.randn(5,f_model)
        self.mask = (torch.rand(256, 1, f_model, c_in) < 0.05).float()
    def loss(self):
        loss=0
        for layer in self.layers:
            loss+=layer.loss()
        loss+=self.emd.loss()
        loss+=self.out_layers.loss()
        return loss*5e-4
    def forward(self,x):
        self.prototypes=self.prototypes.to(x.device)
        if self.f_model!=0:
            embedding=self.emd_time(x.permute(0,1,3,2)).permute(0,1,3,2)

            # embedding=embedding.permute(0,1,3,2)
            # for i in range(len(self.layers)):
            #     embedding=embedding+self.dropout(self.activation(self.time_layers[i](embedding)))
            # embedding=embedding.permute(0,1,3,2)
            # embedding=embedding+self.random_emd(embedding).detach()
            embedding=self.emd(embedding)


        
        else:
            embedding=self.emd(x)
        # embedding=embedding.mean(dim=-1,keepdim=True)
        # b,p,d,n=embedding.shape
        # embedding=embedding.reshape(b,d,n)
        
        
        
        
        
        """
        CNN
        """
        # embedding=embedding.squeeze()
        # embedding=embedding+self.dropout(self.activation(self.dw_conv[0](embedding)))
        # # print(torch.var(self.dw_conv[0].weight,dim=0))
        # embedding=embedding.unsqueeze(1)


        # B, P, f_model, N = embedding.shape

        # self.mask = self.mask.to(embedding.device)
        # embedding = embedding * self.mask

        
        """
        Kernel methods
        """
    

# 计算高斯核相似性矩阵
        # similarity_matrix = gaussian_kernel(embedding.squeeze())

        # # 计算权重并聚合特征
        # weights = similarity_matrix / similarity_matrix.sum(dim=-1, keepdim=True)  # 归一化
        # # 注意：在这里，我们需要将输入张量的形状调整，以便与相似性矩阵进行乘法运算
        # # 输入的形状为 (b, d, n)，需要转置为 (b, n, d)，并在最后一维进行加权
        # output = torch.matmul(weights, embedding.squeeze().permute(0, 2, 1))  # shape (b, n, d)

        # # 最后再转置回 (b, d, n)
        # embedding = output.permute(0, 2, 1).unsqueeze(1)+embedding
        
        # embedding=embedding.mean(-1,keepdim=True)
        # embedding=embedding.permute(0,2,1,3)
        # embedding=embedding.reshape(b,p,d,n)
        
            

            # embedding=embedding.permute(0,1,2,4,3)
            
            # embedding=embedding.permute(0,1,3,4,2)
            # embedding=embedding+self.dropout(self.activation(self.post_layers[i](embedding)))
            # embedding=embedding.permute(0,1,4,2,3)
        # embedding=embedding+self.random_layers[0](embedding)
        # weights, biases = self.hypernetwork(self.embeddings)  # (1, D, N, N) 和 (1, D, N)
        # print(weights.shape);exit()
        # # 对于每个 d 的 n 应用对应的权重和偏置
        # output = []
        # for i in range(self.f_model):
        #     linear_transform = weights.unsqueeze(0).repeat(embedding.shape[0],1,1,1)[:,i,:,:]  # (D, N, N)
        #     bias = biases.unsqueeze(0).repeat(embedding.shape[0],1,1)[:,i,:]  # (B, 1, N) 进行广播
        #     transformed = torch.matmul(embedding[:, :, i,:], linear_transform) + bias  # (B, P, N)
        #     output.append(transformed)

        # embedding = torch.stack(output, dim=2)
        for i in range(len(self.layers)):
            # b,p,d,n=embedding.shape
            # embedding = torch.nn.functional.pad(embedding, (0, 900 - 862))
            # embedding = embedding.view(b, p, d, 30, 30)
            # w1,w2,w3,w4,w5=embedding.shape
            # embedding=res1=embedding.reshape(-1,w4,w5)
            # embedding=self.col_layers[i](embedding[...,torch.randperm(30)])
            # embedding=embedding.permute(0,2,1)
            # embedding=self.dropout(self.activation(self.col_layers[i](embedding[...,torch.randperm(30)])))
            # print(torch.var(self.col_layers[0].weight))
            # embedding=embedding.permute(0,2,1)+res1
            
            
            
            
            # embedding=embedding.reshape(w1,w2,w3,w4,w5)
            # embedding=embedding.permute(0,1,3,2,4)
            # embedding=embedding.reshape(b,p,d,900)[:,:,:,:862]

            # embedding=self.layer_norms1[i](embedding)
            # embedding=embedding+local_weighted_regression(embedding.squeeze(),self.cos_layers[i],self.random_emd).unsqueeze(1)
            # embedding=embedding.squeeze()
            # embedding=embedding+self.dropout(self.activation(self.dw_conv[0](embedding)))
            # embedding=embedding.unsqueeze(1)

            
            # norm=embedding.norm(dim=2, keepdim=True)
            # normalized_input = embedding / norm

            # norm_out=
            # norm_out=norm_out/norm_out.norm(dim=2, keepdim=True)
            # for j in range(d):
            #     embedding[:,:,j,:]=embedding[:,:,j,self.perm[random.randint(0, 9)]]
            # self.layers[i].weight=F.softmax(self.layers[i].weight,dim=-1)
            embedding=embedding+self.dropout(self.activation(self.layers[i](embedding)))
            # embedding=embedding/embedding.norm(dim=-1, keepdim=True)
            
            
            # print_zero_weights(self.random_layers[i])
            # embedding1=embedding.clone()
            # for j in range(embedding.size(2)):
            #     if random.random() < 0.5:
            #         embedding1[:,0,j,:]=embedding[:,0,j,self.perm]
            # embedding=embedding1
            
            embedding=embedding.permute(0,1,3,2)

            embedding=embedding+self.dropout(self.activation(self.time_layers[i](embedding)))
            # embedding=self.layer_norms2[i](embedding)

            embedding=embedding.permute(0,1,3,2)
        # embedding=embedding.repeat(1,1,1,self.cin)
        if self.f_model!=0:
            out=self.out_time(embedding.permute(0,1,3,2)).permute(0,1,3,2)
            out=self.out_layers(out)
        return out

        
# class Variable(nn.Module):
#     def __init__(self,context_window,target_window,m_layers,d_model,dropout,c_in):
#         super(Variable,self).__init__()
#         self.mambas=nn.ModuleList([Mamba(d_model=d_model,  # Model dimension d_model
#             d_state=2,  # SSM state expansion factor
#             d_conv=2,  # Local convolution width
#             expand=1,  # Block expansion factor)
#             )for _ in range(m_layers)])
#         self.convs=nn.ModuleList([nn.Sequential(nn.Linear(d_model,d_model))for _ in range(m_layers)])
#         self.pwconvs=nn.ModuleList([nn.Sequential(nn.Conv1d(c_in,c_in,1,1))for _ in range(m_layers)])

#         self.layers=m_layers
#         self.up=nn.Linear(context_window,d_model)
#         self.down=nn.Linear(d_model,target_window)
#         self.bns=nn.ModuleList([nn.LayerNorm(d_model)for _ in range(m_layers)])
#         self.bnv=nn.ModuleList([nn.BatchNorm1d(c_in)for _ in range(m_layers)])

#         self.act=nn.SELU()
#         self.dropout=nn.Dropout(dropout)
#         self.Linears=nn.ModuleList([nn.Sequential(nn.Linear(d_model,d_model*2),nn.SELU(),nn.Linear(d_model*2,d_model),nn.LayerNorm(d_model))for _ in range(m_layers)])
    
#     def forward(self,x):
#         x=dct.dct(x)
#         for i in range(self.layers):
#             if i==0:
#                 x=self.up(x)
#             x=self.convs[i](x)
#             x=self.dropout(x)+x
#             x=self.bns[i](x)
#             x=self.pwconvs[i](x)
#             x=self.dropout(x)+x
#             x=self.bnv[i](x)
#             if i==self.layers-1:
#                 x=self.down(x)
#                 x=dct.idct(x)
#         return x if self.layers >0 else 0
class backbone(nn.Module):
    def get_para(self):
        weights=self.linear.weight.data.detach().cpu()
        # weights=F.softmax(weights,dim=0)
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
        import seaborn as sns
        import matplotlib.pyplot as plt
        cmap = LinearSegmentedColormap.from_list(
    'custom_cmap',
    [(0, 'blue'), (0.5, 'white'), (1, 'red')]
)

        ax = sns.heatmap(weights,cmap=cmap,center=0, linewidth=0)
        plt.savefig('time.pdf',format='pdf')

    def __init__(self, c_in: int, context_window: int, target_window: int,
                 period, patch_len, stride, kernel_list, serial_conv=False, wo_conv=False, add=False,
                 max_seq_len: Optional[int] = 1024,m_model=512,
                 n_layers: int = 3, d_model=128, n_heads=16, d_k: Optional[int] = None,
                 d_v: Optional[int] = None,v_dropout=0.9,
                 d_ff: int = 256, norm: str = 'BatchNorm', attn_dropout: float = 0., dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = False,
                 pre_norm: bool = False, store_attn: bool = False,
                 pe: str = 'zeros', learn_pe: bool = True, fc_dropout: float = 0., head_dropout=0, padding_patch=None,
                 pretrain_head: bool = False, head_type='flatten', individual=False, revin=True, affine=True,
                 subtract_last=False,f_model=0,
                 verbose: bool = False,m_layers=1,configs=None, **kwargs):
        super().__init__()
        self.n=3
        #self.skip=nn.Linear(context_window,target_window)
        self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.period_list = period
        self.period_len = [math.ceil(context_window / i) for i in self.period_list]
        self.kernel_list = [(n, patch_len[i]) for i, n in enumerate(self.period_len)]
        self.stride_list = [(n , m // 2 if stride is None else stride[i]) for i, (n, m) in enumerate(self.kernel_list)]
        self.d_model=d_model
        self.cin=c_in
        self.dim_list = [ k[0] * k[1] for k in self.kernel_list]
        self.tokens_list = [
            (self.period_len[i] // s[0]) *
            ((math.ceil(self.period_list[i] / k[1]) * k[1] - k[1]) // s[1] + 1)
            for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        ]
        # self.var=Variable(context_window,target_window,m_layers,m_model,v_dropout,c_in)
        self.pad_layer = nn.ModuleList([nn.ModuleList([
            nn.ConstantPad1d((0, p-context_window%p), 0)if context_window % p != 0 else nn.Identity(),
            nn.ConstantPad1d((0, k[1] - p % k[1]), 0) if p % k[1] != 0 else nn.Identity()
        ]) for p, (k, s) in zip(self.period_list, zip(self.kernel_list, self.stride_list))
        ])
        # self.FAN=FAN(context_window,target_window,c_in)
        # self.embedding = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s),
        #     nn.Flatten(start_dim=2)
        # ) for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        # ])
        # self.embedding1 = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(1, self.dim_list[i], kernel_size=k, stride=s),
        #     nn.Flatten(start_dim=2)
        # ) for i, (k, s) in enumerate(zip(self.kernel_list, self.stride_list))
        # ])
        self.backbone =nn.Sequential( TSTiEncoder(c_in, patch_num=sum(self.period_len), patch_len=1, max_seq_len=max_seq_len,
                        n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                        norm=norm, attn_dropout=attn_dropout, dropout=dropout, act=act,
                        key_padding_mask=key_padding_mask, padding_var=padding_var,
                        attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                        store_attn=store_attn,individual=individual,
                        pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs),nn.Flatten(start_dim=-2),nn.Linear(sum(self.period_len)*d_model,context_window)
            )
        # self.clinear1=nn.Linear(target_window,target_window*10).to(torch.cfloat)
        
        self.last=nn.Linear(context_window,target_window)
        self.wo_conv = wo_conv
        
        self.serial_conv = serial_conv
        self.compensate=(context_window+target_window)/context_window
        if not self.wo_conv:
            self.conv = nn.Sequential(*[
                nn.Sequential(nn.Conv1d(self.n+1, self.n+1, kernel_size=i, groups=self.n+1, padding='same'), nn.SELU(),nn.Dropout(fc_dropout),nn.BatchNorm1d(self.n+1))
                for i in kernel_list],
                nn.Flatten(start_dim=-2),
                nn.Linear(context_window*(self.n+1),context_window)
            )
            
  
            self.conv1 = nn.ModuleList([nn.Sequential(*[
                nn.Sequential(nn.Conv1d(n, n, kernel_size=i, groups=n, padding='same'), nn.SELU(),nn.BatchNorm1d(n))
                for i in kernel_list],
                nn.Dropout(fc_dropout),
            ) for n in self.period_len])
        # self.dual=nn.Linear(context_window,target_window)
        self.conv_drop=nn.Dropout(fc_dropout)
        self.glo=nn.ModuleList([nn.Linear(context_window,context_window) for i in range(len(period))])
        # self.proj=nn.ModuleList([nn.Linear(context_window,context_window) for _ in range(len(period))])
       # self.mamba = Variable(context_window,m_model
        #,m_layers,dropout)
        # self.pre_emd=nn.Linear(context_window,context_window)
        # self.linear=nn.Linear(context_window,target_window)
        self.dropout=nn.Dropout(dropout)
        self.mix=channel_mix(c_in,d_model,m_model,f_model,m_layers,fc_dropout)
        # self.mix=iTransformer(configs)
        # self.mix=nn.Linear(c_in,c_in)

        self.individual=individual
        if individual==False:
            self.W_P=nn.ModuleList([nn.Linear(self.period_list[i],d_model)for i in range(len(self.period_list))])
            self.W_P1=nn.ModuleList([nn.Linear(self.period_list[i],d_model)for i in range(len(self.period_list))])

        else:
            self.W_P1=nn.ModuleList([nn.Linear(self.period_list[i],d_model)for i in range(len(self.period_list))])
            self.loc_W_p1=nn.ModuleList([nn.ModuleList([nn.Linear(self.period_list[i],d_model) for _ in range(c_in)]) for i in range(len(self.period_list))])

            self.W_P=nn.ModuleList([nn.Linear(self.period_list[i],d_model)for i in range(len(self.period_list))])
            self.loc_W_p=nn.ModuleList([nn.ModuleList([nn.Linear(self.period_list[i],d_model) for _ in range(c_in)]) for i in range(len(self.period_list))])
        
        self.head = Head(context_window, 1, target_window, head_dropout=head_dropout, Concat=not add)
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        # self.bn=nn.ModuleList([nn.BatchNorm1d(self.period_len[i])  for i in range(len(self.period_len)) ])
        self.linears=nn.ModuleList([ nn.Linear( context_window//((n+1)*2) ,d_model//2) for n in range(2)])
        self.linear_all=[nn.Linear(d_model*3,d_model//4)]
    def decouple(self,z,linear_all,linears,n):
        store=[]
        def sub_decouple(z,linears,n,store):
            if n==0:return 
            n=n-1
            index_tensor = torch.arange(z.size(-1))
            odd_indices = index_tensor % 2 != 0
            z_odd=z[:,:,odd_indices]
            z_even=z[:,:,~odd_indices]
            
            sub_decouple(z_odd,linears,n,store)
            sub_decouple(z_even,linears,n,store)
   
            z1=torch.cat([self.linears[n](dct.dct(z_odd)),self.linears[n](dct.dct(z_even))],dim=-1)
            
            store.append(z1)
            if n==0:return
        sub_decouple(z,linears,n,store)
        res=torch.cat(store,dim=-1)
        #res=F.leaky_relu(res)
        return res
    # def decouple1(self,z,n):
    #     def sub_decouple(z,n):
    #         if n==0:return None
    #         n=n-1
    #         index_tensor = torch.arange(z.size(-1))
    #         odd_indices = index_tensor % 2 != 0
    #         z_odd=z[:,:,odd_indices]
    #         z_even=z[:,:,~odd_indices]

    #         tmp1=sub_decouple(z_odd,n)
    #         if tmp1==None:
    #             #z_odd=dct.dct(z_odd)
    #             #z_even=dct.dct(z_even)

    #             z1=dct.dct(torch.cat([z_odd,z_even],dim=-1)).unsqueeze(-1)
    #             return z1
    #         tmp2=sub_decouple(z_even,n)
    #         z_odd=dct.dct(z_odd)
    #         z_even=dct.dct(z_even)
    #         z1=dct.dct(torch.cat([z_odd,z_even],dim=-1)).unsqueeze(-1)
    #         tmp=torch.cat([tmp1,tmp2],dim=-2)
    #         z1=torch.cat([z1,tmp],dim=-1)
    #         #z1=self.linears[n](z_odd)+self.linears[n](z_even)
    #         #z1=z_odd-z_even
    #         try:
    #         #z1=linears[n](z1)
    #             pass
    #         except:
    #             print(n)
    #             exit()

    #         return z1
    #     z1=sub_decouple(z,n)
    #     res=torch.cat([dct.dct(z).unsqueeze(-1),z1],dim=-1)
    #     #res=torch.cat(store,dim=-1)
    #     #res=F.leaky_relu(res)
    #     #res=linear_all(res)
    #     return res
   
   
    def forward(self, z):  # z: [bs x nvars x seq_len]
        z = z.permute(0, 2, 1)
        z=self.revin_layer(z,'norm')
        z = z.permute(0, 2, 1)
        res = []
        #loc1=dct.idct((self.mamba(dct.dct(z))))#.reshape(z.shape[0] * z.shape[1], -1, period)
        #loc1=self.var_down(loc1)
        
        time_z=z
        z=dct.dct(z)
        for i, period in enumerate(self.period_list):
            

            time_z = self.pad_layer[i][0](time_z).reshape(z.shape[0],  -1,z.shape[1], period).permute(0,1,3,2)

            x = self.pad_layer[i][0](z).reshape(z.shape[0] * z.shape[1], -1, period)
            
            
            
            
            
            glo=x #+loc*F.sigmoid(x)#+loc*F.sigmoid(x)
            # glo = self.pad_layer[i][1](glo)
            # loc=self.pad_layer[i][1](loc)
            # loc=loc.unsqueeze(-3)
            # glo=glo.unsqueeze(-3)
            # glo = self.embedding[i](glo)
            # loc=self.embedding1[i](loc)
            # glo=dct.dct_2d(glo)
            # loc=dct.dct_2d(loc)
            glo = rearrange(glo, "(b m) d n -> b m d n", b=z.shape[0]).contiguous()
            # glo=glo.permute(0,1,3,2)
            # loc=loc.permute(0,1,3,2)
            glo1=glo.permute(0,2,3,1)
            if not self.individual:
                glo = self.W_P[i](glo)  # x: [bs x nvars x patch_num x d_model]
            else:
                tmp=[]
                tmp=torch.zeros((glo.shape[0],glo.shape[1],glo.shape[2],self.d_model)).to(glo.dtype).to(glo.device)
                for j in range(self.cin):
                    
                    tmp[:,i,:,:]=self.loc_W_p[i][j](glo[:,i,:,:])
                glo=self.W_P[i](glo)+tmp
            
            glo=glo.permute(0,2,3,1)
            # time_z=time_z.permute(0,2,3,1)

            # b,p,d,n=glo.shape
            # glo=glo.squeeze()
            glo=glo+(self.mix(glo))
            # glo=glo.reshape(b,p,d,n)
            glo=glo.permute(0,3,2,1)
            
            res.append(glo)
        glo=torch.cat(res,dim=-1)
        glo=self.backbone(glo)
        # glo=glo+F.leaky_relu(self.mix(glo.permute(0,2,1)).permute(0,2,1))
        z=self.last(glo)
        #+skip
        #z=F.sigmoid(skip)*z+F.sigmoid(z)*skip#*self.compensate#+skip
        #z=dct.idct(z)
    #+loc1
        # z=dct.idct(z)
        #z=z.to(torch.cfloat)
        #z=self.clinear1(z)
        #z=torch.fft.ifft(z,dim=-1).float()
        #*self.compensate
        # z = self.last(glo)
        z=dct.idct(z)
        
        z = z.permute(0, 2, 1)
        # z = self.FAN.denormalize(z)
        z=self.revin_layer(z,'denorm')
        z = z.permute(0, 2, 1)
        return z

class Head(nn.Module):
    def __init__(self, context_window, num_period, target_window, head_dropout=0,
                 Concat=True):
        super().__init__()
        self.Concat = Concat
        self.linear = nn.Linear(context_window * (num_period if Concat else 1), target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.Concat:
            x = torch.cat(x, dim=-1)
            x = self.linear(x)
        else:
            x = torch.stack(x, dim=-1)
            x = torch.mean(x, dim=-1)
            x = self.linear(x)
        x = self.dropout(x)
        return x

class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=False, pre_norm=False,
                 pe='zeros',individual=False, learn_pe=True, verbose=False, **kwargs):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        res_attention=False
        # Input encoding
        q_len = patch_num
        if individual==False:
            self.W_P = nn.Linear(patch_len, d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space
        else:
            self.W_P=nn.Linear(patch_len,d_model)
            self.loc_W_p=nn.ModuleList([nn.Linear(patch_len,d_model) for _ in range(c_in)])
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(0.1)
        self.indivi=individual
        self.cin=c_in
        # Encoder
        self.d_model=d_model
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                  attn_dropout=attn_dropout, dropout=dropout,
                                  pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers,
                                  store_attn=store_attn, pos=self.W_pos)
        
    def forward(self, x) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        #x=dct.dct(x)
        
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs * nvars x patch_num x d_model]
        u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x d_model]

        # Encoder
        z=u
        z = self.encoder(u)  # z: [bs * nvars x patch_num x d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))  # z: [bs x nvars x patch_num x d_model]
        #z=dct.idct(z)
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                 pos=None
                 ):
        super().__init__()

        self.layers = nn.ModuleList(
            [TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                             attn_dropout=attn_dropout, dropout=dropout,
                             activation=activation, res_attention=res_attention,
                             pre_norm=pre_norm, store_attn=store_attn, pos=pos) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src: Tensor, key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        output = src
        scores = None
        
        for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False,
                 pre_norm=False, pos=None):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout,
                                             proj_dropout=dropout, res_attention=res_attention, pos=pos)
        # Multi-Head attention
        self.res_attention = res_attention
        
        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm_attn2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                nn.SELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))
        
        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            self.norm_ffn2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.dw_conv=nn.Conv1d(d_model,d_model,kernel_size=1,stride=1,padding='same',groups=d_model)
        self.conv1=nn.Linear(d_model,d_model)
        self.conv2=nn.Linear(d_model,d_model)
        
        self.pre_norm = pre_norm
        self.store_attn = store_attn
        self.activation=nn.SELU()
    def forward(self, src: Tensor, prev: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None) -> Tensor:

        # Multi-Head attention sublayer
        
        # src = self.norm_attn(src)
        ## Multi-Head attention
        # if self.res_attention:
        #     src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask,
        #                                         attn_mask=attn_mask)
        # else:
        #     src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # if self.store_attn:
        #     self.attn = attn
        #
        # src2=self.mamba(src)
        # src=dct.dct(src)
        # src2=self.dw_conv(src.permute(0,2,1)).permute(0,2,1)
        # # src2,_=self.attn(src)
        # src2=self.activation(src2)
        # ## Add & Norm
        # src2 = src + self.dropout_attn(src2)  # Add: residual connection with residual dropout
        # src=src2
        
        # src = self.norm_attn2(src)
        
        # # Feed-forward sublayer
        
        # src = self.norm_ffn(src)
        # Position-wise Feed-Forward
        
        src2 = self.ff(src)
        ## Add & Norm
        src2 = src + self.dropout_ffn(src2)  # Add: residual connection with residual dropout
        src=src2#*F.sigmoid(self.conv2(src))
        
        # src = self.norm_ffn(src)
        return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0.,
                 qkv_bias=True, lsa=False, pos=None):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.pos = pos
        self.P_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.P_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout,
                                                   res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,
                                                                         2)  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3,
                                                                       1)  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s    : [bs x n_heads x q_len x d_v]

        q_p = self.P_Q(self.pos).view(1, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_p = self.P_K(self.pos).view(1, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev,
                                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                                                              q_p=q_p, k_p=k_p)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1,
                                                          self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor((head_dim * 1) ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None,
                q_p=None, k_p=None):
        '''
        
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]
        # attn_scores += torch.matmul(q_p, k) * self.scale
        # attn_scores += torch.matmul(q, k_p) * self.scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        
        return output, attn_weights
