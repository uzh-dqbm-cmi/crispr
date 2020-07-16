import numpy as np
import torch
from torch import nn

class SH_SelfAttention(nn.Module):
    """ single head self-attention module
    """
    def __init__(self, input_size):
        
        super().__init__()
        # define query, key and value transformation matrices
        # usually input_size is equal to embed_size
        self.embed_size = input_size
        self.Wq = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wk = nn.Linear(input_size, self.embed_size, bias=False)
        self.Wv = nn.Linear(input_size, self.embed_size, bias=False)
        self.softmax = nn.Softmax(dim=2) # normalized across feature dimension
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        X_q = self.Wq(X) # queries
        X_k = self.Wk(X) # keys
        X_v = self.Wv(X) # values
        
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4))
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        
        return z, attn_w_normalized
    

class MH_SelfAttention(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads):
        
        super().__init__()
        
        layers = [SH_SelfAttention(input_size) for i in range(num_attn_heads)]
        
        self.multihead_pipeline = nn.ModuleList(layers)
        embed_size = input_size
        self.Wz = nn.Linear(num_attn_heads*embed_size, embed_size)
        
    
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        
        out = []
        attn_dict = {}
        for count, SH_layer in enumerate(self.multihead_pipeline):
            z, attn_w = SH_layer(X)
            out.append(z)
            attn_dict[f'h{count}'] = attn_w
        # concat on the feature dimension
        out = torch.cat(out, -1) 
        
        # return a unified vector mapping of the different self-attention blocks
        return self.Wz(out), attn_dict
        

class TransformerUnit(nn.Module):
    
    def __init__(self, input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout):
        
        super().__init__()
        
        embed_size = input_size
        self.multihead_attn = MH_SelfAttention(input_size, num_attn_heads)
        
        self.layernorm_1 = nn.LayerNorm(embed_size)

        # also known as position wise feed forward neural network
        self.MLP = nn.Sequential(
            nn.Linear(embed_size, embed_size*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_size*mlp_embed_factor, embed_size)
        )
        
        self.layernorm_2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(p=pdropout)
                
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        # z is tensor of size (batch, sequence length, input_size)
        z, attn_mhead_dict = self.multihead_attn(X)
        # layer norm with residual connection
        z = self.layernorm_1(z + X)
        z = self.dropout(z)
        z_ff= self.MLP(z)
        z = self.layernorm_2(z_ff + z)
        z = self.dropout(z)
        
        return z, attn_mhead_dict

"""
      implement position encoder based on cosine and sine approach proposed 
      by original Transformers paper ('Attention is all what you need')
"""
class NucleoPosEncoding(nn.Module):
    def __init__(self, num_nucleotides, seq_len, embed_dim, pdropout=0.1):
        super().__init__()
        self.nucleo_emb = nn.Embedding(num_nucleotides, embed_dim)
        self.dropout = nn.Dropout(p=pdropout)
        # positional encoding matrix
        base_pow = 10000
        PE_matrix = torch.zeros((1, seq_len, embed_dim))
        i_num = torch.arange(0., seq_len).reshape(-1, 1) # i iterates over sequence length (i.e. sequence items)
        j_denom = torch.pow(base_pow, torch.arange(0., embed_dim, 2.) / embed_dim) # j iterates over embedding dimension
        PE_matrix[:, :, 0::2] = torch.sin(i_num/j_denom)
        PE_matrix[:, :, 1::2] = torch.cos(i_num/j_denom)
        self.register_buffer('PE_matrix', PE_matrix)
        
        
    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        X_emb = self.nucleo_emb(X)
        # (batch, sequence length, embedding dim)
        X_embpos = X_emb + self.PE_matrix
        return self.dropout(X_embpos)

class NucleoPosEmbedder(nn.Module):
    def __init__(self, num_nucleotides, seq_length, embedding_dim):
        super().__init__()
        self.nucleo_emb = nn.Embedding(num_nucleotides, embedding_dim)
        self.pos_emb = nn.Embedding(seq_length, embedding_dim)

    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        X_emb = self.nucleo_emb(X)
        bsize, seqlen, featdim = X_emb.size()
        device = X_emb.device
        positions = torch.arange(seqlen).to(device)
        positions_emb = self.pos_emb(positions)[None, :, :].expand(bsize, seqlen, featdim)
        # (batch, sequence length, embedding dim)
        X_embpos = X_emb + positions_emb
        return X_embpos

class PerBaseFeatureEmbAttention(nn.Module):
    """ Per base feature attention module
    """
    def __init__(self, input_dim, seq_len):
        
        super().__init__()
        # define query, key and value transformation matrices
        # usually input_size is equal to embed_size
        self.embed_size = input_dim
        self.Q = nn.Parameter(torch.randn((seq_len, input_dim), dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1) # normalized across feature dimension
    
    def forward(self, X):
        """
        Args:
            X: tensor, (batch, sequence length, input_size)
        """
        bsize, seqlen, featdim = X.shape
        X_q = self.Q[None, :, :].expand(bsize, seqlen, featdim) # queries
        X_k = X
        X_v = X
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4))
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        # print(X_q_scaled.shape)
        # print(X_k_scaled.shape)
        
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        # attn_w = X_q_scaled.matmul(X_k_scaled.transpose(1,0))
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        # print('z.shape', z.shape)
        
        return z, attn_w_normalized

class FeatureEmbAttention(nn.Module):
    def __init__(self, input_dim):
        '''
        Args:
            input_dim: int, size of the input vector (i.e. feature vector)
        '''

        super().__init__()
        self.input_dim = input_dim
        # use this as query vector against the transformer outputs
        self.queryv = nn.Parameter(torch.randn(input_dim, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1) # normalized across seqlen

    def forward(self, X):
        '''Performs forward computation
        Args:
            X: torch.Tensor, (bsize, seqlen, feature_dim), dtype=torch.float32
        '''

        X_scaled = X / (self.input_dim ** (1/4))
        queryv_scaled = self.queryv / (self.input_dim ** (1/4))
        # using  matmul to compute tensor vector multiplication
        
        # (bsize, seqlen)
        attn_weights = X_scaled.matmul(queryv_scaled)

        # softmax
        attn_weights_norm = self.softmax(attn_weights)

        # reweighted value vectors (in this case reweighting the original input X)
        # unsqueeze attn_weights_norm to get (bsize, 1, seqlen)
        # perform batch multiplication with X that has shape (bsize, seqlen, feat_dim)
        # result will be (bsize, 1, feat_dim)
        # squeeze the result to obtain (bsize, feat_dim)
        z = attn_weights_norm.unsqueeze(1).bmm(X).squeeze(1)
        
        # returns (bsize, feat_dim), (bsize, seqlen)
        return z, attn_weights_norm

class Categ_CrisCasTransformer(nn.Module):

    def __init__(self, input_size=64, num_nucleotides=4, 
                 seq_length=20, num_attn_heads=8, 
                 mlp_embed_factor=2, nonlin_func=nn.ReLU(), 
                 pdropout=0.3, num_transformer_units=12, 
                 pooling_mode='attn', num_classes=2, per_base=False):
        
        super().__init__()
        
        embed_size = input_size

        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, embed_size)
        # self.nucleopos_embedder = NucleoPosEncoding(num_nucleotides, seq_length, embed_size)
        
        trfunit_layers = [TransformerUnit(input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout) 
                          for i in range(num_transformer_units)]
        # self.trfunit_layers = trfunit_layers
        self.trfunit_pipeline = nn.ModuleList(trfunit_layers)
        # self.trfunit_pipeline = nn.Sequential(*trfunit_layers)
        self.per_base = per_base
        self.Wy = nn.Linear(embed_size, num_classes)
        if not per_base:
            self.pooling_mode = pooling_mode
            if pooling_mode == 'attn':
                self.pooling = FeatureEmbAttention(input_size)
            elif pooling_mode == 'mean':
                self.pooling = torch.mean
        else:
            self.pooling_mode = pooling_mode
            if pooling_mode == 'attn':
                self.pooling = PerBaseFeatureEmbAttention(input_size, seq_length)
        # perform log softmax on the feature dimension
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self._init_params_()
        
    def _init_params_(self):
        for p_name, p in self.named_parameters():
            param_dim = p.dim()
            if param_dim > 1: # weight matrices
                nn.init.xavier_uniform_(p)
            elif param_dim == 1: # bias parameters
                if p_name.endswith('bias'):
                    nn.init.uniform_(p, a=-1.0, b=1.0)

    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        # (batch, seqlen, embedding dim)
        X_embpos = self.nucleopos_embedder(X)
        # z is tensor of size (batch,  seqlen, embedding dim)
        # z = self.trfunit_pipeline(X_embpos)
        attn_mlayer_mhead_dict = {}
        xinput = X_embpos
        for count, trfunit in enumerate(self.trfunit_pipeline):
            z, attn_mhead_dict = trfunit(xinput)
            attn_mlayer_mhead_dict[f'l{count}'] = attn_mhead_dict
            xinput = z

         # pool across seqlen vectors
        if not self.per_base:
            if self.pooling_mode == 'attn':
                z, fattn_w_norm = self.pooling(z)
            # Note: z.mean(dim=1) or self.pooling(z, dim=1) will change shape of z to become (batch, embedding dim)
            # we can keep dimension by running z.mean(dim=1, keepdim=True) to have (batch, 1, embedding dim)
            elif self.pooling_mode == 'mean':
                z = self.pooling(z, dim=1)
                fattn_w_norm = None
        else:
            if self.pooling_mode == 'attn':
                z, fattn_w_norm = self.pooling(z)
        
        y = self.Wy(z) 
        
        return self.log_softmax(y), fattn_w_norm, attn_mlayer_mhead_dict

