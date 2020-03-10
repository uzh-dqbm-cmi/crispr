import torch
from torch import nn

class SH_SelfAttention(nn.Module):
    """ single head self-attention module
    """
    def __init__(self, input_size):
        
        super(SH_SelfAttention, self).__init__()
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
        
        attn_w_normalized = self.softmax(attn_w)
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        
        return z, attn_w_normalized
    

class MH_SelfAttention(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads):
        
        super(MH_SelfAttention, self).__init__()
        
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
        for SH_layer in self.multihead_pipeline:
            z, __ = SH_layer(X)
            out.append(z)
        # concat on the feature dimension
        out = torch.cat(out, -1) 
        
        # return a unified vector mapping of the different self-attention blocks
        return self.Wz(out)
        

class TransformerUnit(nn.Module):
    
    def __init__(self, input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout):
        
        super(TransformerUnit, self).__init__()
        
        embed_size = input_size
        self.multihead_attn = MH_SelfAttention(input_size, num_attn_heads)
        
        self.layernorm_1 = nn.LayerNorm(embed_size)
        
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
        # z is tensor of size (batch, ddi similarity type vector, input_size)
        z = self.multihead_attn(X)
        # layer norm with residual connection
        z = self.layernorm_1(z + X)
        z = self.dropout(z)
        z_ff= self.MLP(z)
        z = self.layernorm_2(z_ff + z)
        z = self.dropout(z)
        
        return z

# TODO: implement position encoder based on cosine and sine approach by original Transformers paper ('Attention is all what you need')
        
class NucleoPosEmbedder(nn.Module):
    def __init__(self, num_nucleotides, seq_length, embedding_dim):
        super(NucleoPosEmbedder, self).__init__()
        self.nucleo_emb = nn.Embedding(num_nucleotides, embedding_dim)
        self.pos_emb = nn.Embedding(seq_length, embedding_dim)

    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        X_emb = self.nucleo_emb(X)
        bsize, seqlen, featdim = X_emb.size()
        positions = torch.arange(seqlen)
        positions_emb = self.pos_emb(positions)[None, :, :].expand(bsize, seqlen, featdim)
        # (batch, sequence length, embedding dim)
        X_embpos = X_emb + positions_emb
        return X_embpos

class Categ_CrisCasTransformer(nn.Module):

    def __init__(self, input_size=64, num_nucleotides=4, seq_length=20, 
                num_attn_heads=8, mlp_embed_factor=2, 
                nonlin_func=nn.ReLU(), pdropout=0.3, num_transformer_units=12, num_classes=2):
        
        super(Categ_CrisCasTransformer, self).__init__()
        
        embed_size = input_size

        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, embed_size)
        
        trfunit_layers = [TransformerUnit(input_size, num_attn_heads, mlp_embed_factor, nonlin_func, pdropout) for i in range(num_transformer_units)]
        self.trfunit_pipeline = nn.Sequential(*trfunit_layers)

        self.Wy = nn.Linear(embed_size, num_classes)
        # perform log softmax on the feature dimension
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    
    def forward(self, X):
        """
        Args:
            X: tensor, int64,  (batch, sequence length)
        """
        # (batch, seqlen, embedding dim)
        X_embpos = self.nucleopos_embedder(X)
        # z is tensor of size (batch,  seqlen, embedding dim)
        z = self.trfunit_pipeline(X_embpos)
        
        # TODO: add global attention layer or other pooling strategy
        # we currently perform mean pooling 
        # pool across similarity type vectors
        # Note: z.mean(dim=1) will change shape of z to become (batch, embedding dim)
        # we can keep dimension by running z.mean(dim=1, keepdim=True) to have (batch, 1, embedding dim)
        
        y = self.Wy(z.mean(dim=1)) 
        
        return self.log_softmax(y)