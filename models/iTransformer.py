import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import torch_dct as dct

class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.f_model
        self.pred_len = configs.f_model
        self.output_attention = False
        self.use_norm = True
        # Embedding
        self.embd=nn.Linear(configs.d_model,configs.f_model)
        self.down=nn.Linear(configs.f_model,configs.d_model)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.fc_dropout,
                                      output_attention=False), configs.f_model, configs.n_heads),
                    configs.f_model,
                    configs.f_model*2,
                    dropout=configs.fc_dropout,
                    activation=configs.activation
                ) for l in range(configs.m_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.f_model)
        )
        self.projector = nn.Linear(configs.f_model, configs.d_model, bias=True)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        
        x_enc=x_enc.permute(0,2,1)
        
        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out =self.embd( x_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, _ = self.encoder(enc_out)

        # B N E -> B N S -> B S N 
        dec_out = (self.projector(enc_out)).permute(0, 2, 1)[:, :, :N] # filter the covariates

        

        return dec_out


    def forward(self, x_enc):
        
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]