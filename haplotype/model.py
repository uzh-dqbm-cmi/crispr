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
        self.neginf = -1e6
    
    def forward(self, Xin_q, Xin_k, Xin_v, mask=None):
        """
        Args:
            Xin_q: query tensor, (batch, sequence length, input_size)
            Xin_k: key tensor, (batch, sequence length, input_size)
            Xin_v: value tensor, (batch, sequence length, input_size)
            mask: tensor, (batch, sequence length, sequence length) with 0/1 entries
                  (default None)
                  
        .. note:
            
            mask has to have at least one element in a row that is equal to one otherwise a uniform distribution
            will be genertaed when computing attn_w_normalized!
            
        """
        X_q = self.Wq(Xin_q) # queries
        X_k = self.Wk(Xin_k) # keys
        X_v = self.Wv(Xin_v) # values
        
        # print('---- SH layer ----')
        # print('X_q.shape', X_q.shape)
        # print('X_k.shape', X_k.shape)
        # print('X_v.shape', X_v.shape)
        # print('mask.shape', mask.shape)
        
        # scaled queries and keys by forth root 
        X_q_scaled = X_q / (self.embed_size ** (1/4))
        X_k_scaled = X_k / (self.embed_size ** (1/4))
        
        # (batch, sequence length, sequence length)
        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        # print('attn_w.shape:', attn_w.shape)
        # print()
        
        if mask is not None:
            # (batch, seqlen, seqlen)
            # if mask.dim() == 2: # assumption mask.shape = (seqlen, seqlen)
            #     mask = mask.unsqueeze(0) # add batch dimension
            # fill with neginf where mask == 0  
            attn_w = attn_w.masked_fill(mask == 0, self.neginf)
            # print('attn_w masked:\n', attn_w)
        
        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        # print('attn_w_normalized masked:\n', attn_w_normalized)

        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        
        return z, attn_w_normalized
    
class MH_SelfAttentionWide(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads):
        
        super().__init__()
        
        embed_size = input_size
        
        layers = [SH_SelfAttention(embed_size) for i in range(num_attn_heads)]
        self.multihead_pipeline = nn.ModuleList(layers)
        
        self.Wz = nn.Linear(num_attn_heads*embed_size, embed_size, bias=True)
    
    def forward(self, Xin_q, Xin_k, Xin_v, mask=None):
        """
        Args:
            Xin_q: query tensor, (batch, sequence length, input_size)
            Xin_k: key tensor, (batch, sequence length, input_size)
            Xin_v: value tensor, (batch, sequence length, input_size)
            mask: tensor, (batch, sequence length) with 0/1 entries
                  (default None)
        """
        out = []
        attn_dict = {}
        for count, SH_layer in enumerate(self.multihead_pipeline):
            z, attn_w = SH_layer(Xin_q, Xin_k, Xin_v, mask=mask)
            out.append(z)
            attn_dict[f'h{count}'] = attn_w
        # concat on the feature dimension
        out = torch.cat(out, -1) 
        
        # return a unified vector mapping of the different self-attention blocks
        return self.Wz(out), attn_dict
    
class MH_SelfAttentionNarrow(nn.Module):
    """ multi head self-attention module
    """
    def __init__(self, input_size, num_attn_heads):
        
        super().__init__()
        
        assert input_size%num_attn_heads == 0
        
        embed_size = input_size
        
        self.num_attn_heads = num_attn_heads
        self.head_dim = embed_size//num_attn_heads
        
        layers = [SH_SelfAttention(self.head_dim) for i in range(self.num_attn_heads)]
        self.multihead_pipeline = nn.ModuleList(layers)
        
        self.Wz = nn.Linear(embed_size, embed_size, bias=True)
        
        
    def forward(self, Xin_q, Xin_k, Xin_v, mask=None):
        """
        Args:
            Xin_q: query tensor, (batch, sequence length, input_size)
            Xin_k: key tensor, (batch, sequence length, input_size)
            Xin_v: value tensor, (batch, sequence length, input_size)
            mask: tensor, (batch, sequence length) with 0/1 entries
                  (default None)        """
        out = []
        attn_dict = {}
        bsize, q_seqlen, inputsize = Xin_q.size()
        kv_seqlen = Xin_k.size(1)

        # print('Xin_q.shape', Xin_q.shape)
        # print('Xin_k.shape', Xin_k.shape)
        # print('Xin_v.shape', Xin_v.shape)
        # print('mask.shape', mask.shape)

        Xq_head = Xin_q.view(bsize, q_seqlen, self.num_attn_heads, self.head_dim)
        Xk_head = Xin_k.view(bsize, kv_seqlen, self.num_attn_heads, self.head_dim)
        Xv_head = Xin_v.view(bsize, kv_seqlen, self.num_attn_heads, self.head_dim)

        for count, SH_layer in enumerate(self.multihead_pipeline):
            z, attn_w = SH_layer(Xq_head[:,:,count,:],
                                 Xk_head[:,:,count,:],
                                 Xv_head[:,:,count,:],
                                 mask=mask)
            out.append(z)
            attn_dict[f'h{count}'] = attn_w
        # concat on the feature dimension
        out = torch.cat(out, -1)         
        # return a unified vector mapping of the different self-attention blocks
        return self.Wz(out), attn_dict

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
        # usually input_size is equal to embed_size
        self.embed_size = input_dim
        self.Q = nn.Parameter(torch.randn((seq_len, self.embed_size), dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1) # normalized across feature dimension
        self.neginf = -1e6
    
    def forward(self, X, mask=None):
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
        # print('---- PosFeatureAttn -----')
        # print('X_q_scaled.shape:', X_q_scaled.shape)
        # print('X_k_scaled.shape:',X_k_scaled.shape)
        # print('X_v.shape:', X_v.shape)
        # print('mask.shape:', mask.shape)

        attn_w = torch.bmm(X_q_scaled, X_k_scaled.transpose(1,2))
        # print('attn_w.shape:', attn_w.shape)
        if mask is not None:
            #assert mask.dim() == 2
            # fill with neginf where mask == 0    
            attn_w = attn_w.masked_fill(mask == 0, self.neginf)
            
        # attn_w = X_q_scaled.matmul(X_k_scaled.transpose(1,0))
        # (batch, sequence length, sequence length)
        attn_w_normalized = self.softmax(attn_w)
        # print('attn_w_normalized.shape', attn_w_normalized.shape)
        
        # reweighted value vectors
        z = torch.bmm(attn_w_normalized, X_v)
        # print('z.shape', z.shape)
        
        return z, attn_w_normalized

####### Encoder specification #######
class EncoderBlock(nn.Module):
            
    def __init__(self,
                 input_size, 
                 num_attn_heads, 
                 mlp_embed_factor, 
                 nonlin_func, 
                 pdropout,
                 multihead_type='Wide'):
        
        super().__init__()
        
        embed_size = input_size
        
        if multihead_type == 'Wide':
            self.multihead_attn = MH_SelfAttentionWide(input_size, num_attn_heads)
        elif multihead_type == 'Narrow':
            self.multihead_attn = MH_SelfAttentionNarrow(input_size, num_attn_heads)

        self.layernorm_1 = nn.LayerNorm(embed_size)

        # also known as position wise feed forward neural network
        self.MLP = nn.Sequential(
            nn.Linear(embed_size, embed_size*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_size*mlp_embed_factor, embed_size)
        )
        
        self.layernorm_2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(p=pdropout)
                
    
    def forward(self, X, mask=None):
        """
        Args:
            X: input tensor, (batch, sequence length, input_size)
            mask: tensor, (batch, sequence length, sequence length) with 0/1 entries
        """
        # z is tensor of size (batch, sequence length, input_size)
        z, attn_mhead_dict = self.multihead_attn(X, X, X, mask)
        # layer norm with residual connection
        z = self.layernorm_1(z + X)
        z = self.dropout(z)
        z_ff= self.MLP(z)
        z = self.layernorm_2(z_ff + z)
        z = self.dropout(z)
        
        return z, attn_mhead_dict

class Encoder(nn.Module):

    def __init__(self, 
                 input_size=64, 
                 num_nucleotides=4, 
                 seq_length=20,
                 num_attn_heads=8, 
                 mlp_embed_factor=2, 
                 nonlin_func=nn.ReLU(), 
                 pdropout=0.3, 
                 num_encoder_units=12, 
                 pooling_mode='attn',
                 multihead_type='Wide'):
        
        super().__init__()
        
        embed_size = input_size

        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, embed_size)           
                    
        encunit_layers = [EncoderBlock(embed_size, 
                                       num_attn_heads, 
                                       mlp_embed_factor, 
                                       nonlin_func, 
                                       pdropout, 
                                       multihead_type) 
                          for i in range(num_encoder_units)]
        self.encunit_pipeline = nn.ModuleList(encunit_layers)
        
        self.pooling = PerBaseFeatureEmbAttention(input_size, seq_length)
        
        self._init_params_()
        
    def _init_params_(self):
        for p_name, p in self.named_parameters():
            param_dim = p.dim()
            if param_dim > 1: # weight matrices
                nn.init.xavier_uniform_(p)
            elif param_dim == 1: # bias parameters
                if p_name.endswith('bias'):
                    nn.init.uniform_(p, a=-1.0, b=1.0)

    def forward(self, X, mask=None):
        """
        Args:
            X: tensor, int64, (batch, sequence length), numeric encoding of nucleotides in target sequence
            mask: tensor, (batch, sequence length, sequence length) with 0/1 entries
        """
        # X_embpos (batch, seqlen, embedding dim)
        X_embpos = self.nucleopos_embedder(X)
        # z is tensor of size (batch,  seqlen, embedding dim)
        attn_mlayer_mhead_dict = {}
        xinput = X_embpos
        # print('--- Encoder ---')
        # print('xinput.shape:',xinput.shape)
        # print('mask.shape:',mask.shape)
        for count, encunit in enumerate(self.encunit_pipeline):
            z, attn_mhead_dict = encunit(xinput, mask)
            attn_mlayer_mhead_dict[f'l{count}'] = attn_mhead_dict
            xinput = z

        # pooling using another attention layer
        # z, fattn_w_norm = self.pooling(z, mask)
        z, fattn_w_norm = self.pooling(z)
        return z, fattn_w_norm, attn_mlayer_mhead_dict

####### Decoder specification #######
class DecoderBlock(nn.Module):
                    
    def __init__(self,
                 input_size, 
                 num_attn_heads, 
                 mlp_embed_factor, 
                 nonlin_func, 
                 pdropout,
                 multihead_type='Wide'):
        
        super().__init__()
        
        embed_size = input_size
        
        if multihead_type == 'Wide':
            self.decoder_attn = MH_SelfAttentionWide(input_size, num_attn_heads)
            self.encdec_attn = MH_SelfAttentionWide(input_size, num_attn_heads)
        elif multihead_type == 'Narrow':
            self.decoder_attn = MH_SelfAttentionNarrow(input_size, num_attn_heads)
            self.encdec_attn = MH_SelfAttentionNarrow(input_size, num_attn_heads)


        self.layernorm_1 = nn.LayerNorm(embed_size)
        self.layernorm_2 = nn.LayerNorm(embed_size)

        # also known as position wise feed forward neural network
        self.MLP = nn.Sequential(
            nn.Linear(embed_size, embed_size*mlp_embed_factor),
            nonlin_func,
            nn.Linear(embed_size*mlp_embed_factor, embed_size)
        )
        
        self.layernorm_3 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(p=pdropout)
                
    
    def forward(self, Xin_dec, Zout_enc, mask_dec, mask_encdec):
        """
        Args:
            Xin_dec: decoder input tensor, (batch, sequence length, embed_dim)
            Zout_enc: encoder output tensor, (batch, sequence length, embed_dim)
            mask_dec: decoder mask, tensor, (batch, sequence length, sequence length) with 0/1 entries
            mask_enc: encoder mask, tensor, (batch, sequence length, sequence length) with 0/1 entries
            
        """
        ### Decoder self-attention ##
        # z is tensor of size (batch, sequence length, embed_dim)
        z, attn_mhead_dec_dict = self.decoder_attn(Xin_dec,Xin_dec,Xin_dec,mask_dec)
        # layer norm with residual connection
        z = self.layernorm_1(z + Xin_dec)
        z_dec = self.dropout(z)
        # print('--- decoder_unit ---')
        # print('z_dec.shape:', z_dec.shape)
        # print('Zout_enc.shape', Zout_enc.shape)
        # print('mask_encdec.shape', mask_encdec.shape)
        ### Decoder encoder attention ##
        # z_dec acts as query
        # Zout_enc acts as key and values
        z, attn_mhead_encdec_dict = self.encdec_attn(z_dec,Zout_enc,Zout_enc,mask_encdec)
        # layer norm with residual connection
        z = self.layernorm_2(z + z_dec)
        z = self.dropout(z)
        
        ## pointwise feed fowrward network
        z_ff= self.MLP(z)
        # layer norm with residual connection
        z = self.layernorm_3(z_ff + z)
        z = self.dropout(z)
        
        return z, attn_mhead_dec_dict, attn_mhead_encdec_dict

class Decoder(nn.Module):

    def __init__(self, 
                 input_size=64,
                 num_nucleotides=4, 
                 seq_length=20, 
                 num_attn_heads=8, 
                 mlp_embed_factor=2, 
                 nonlin_func=nn.ReLU(), 
                 pdropout=0.3, 
                 num_decoder_units=12, 
                 pooling_mode='attn', 
                 multihead_type='Wide', 
                 num_classes=2):
        
        super().__init__()
        
        embed_size = input_size

        self.nucleopos_embedder = NucleoPosEmbedder(num_nucleotides, seq_length, embed_size)           
                    
        decunit_layers = [DecoderBlock(embed_size, 
                                       num_attn_heads,
                                       mlp_embed_factor, 
                                       nonlin_func, 
                                       pdropout, 
                                       multihead_type) 
                          for i in range(num_decoder_units)]
        self.decunit_pipeline = nn.ModuleList(decunit_layers)
        
        self.pooling = PerBaseFeatureEmbAttention(input_size, seq_length)
        
        # self.Wy = nn.Linear(embed_size, num_classes)

        self.bias = nn.Parameter(torch.randn((seq_length, num_classes), dtype=torch.float32), requires_grad=True)
        self.Wy = nn.Linear(embed_size, num_classes, bias=False)

        self._init_params_()
        
    def _init_params_(self):
        for p_name, p in self.named_parameters():
            param_dim = p.dim()
            if param_dim > 1: # weight matrices
                nn.init.xavier_uniform_(p)
            elif param_dim == 1: 
                # bias parameters
                if p_name.endswith('bias'):
                    nn.init.uniform_(p, a=-1.0, b=1.0)

    def forward(self, Xin_dec, Zout_enc, mask_dec, mask_encdec):
        """
        Args:
            Xin_dec: decoder input tensor, (batch, sequence length)
            Zout_enc: encoder output tensor, (batch, sequence length, embed_dim)
            mask_dec: decoder mask, tensor, (batch, sequence length, sequence length) with 0/1 entries
            mask_encdec: encoder mask, tensor, (batch, sequence length, sequence length) with 0/1 entries
        """
        
        # (batch, seqlen, embedding dim)
        X_embpos = self.nucleopos_embedder(Xin_dec)
        
        # z is tensor of size (batch,  seqlen, embedding dim)
        attn_mlayer_mhead_dec_dict = {}
        attn_mlayer_mhead_encdec_dict = {}
        
        xinput = X_embpos
        # print('--- decoder ----')
        # print('decoder xinput.shape:', xinput.shape)
        # print('decoder zout_enc.shape:', Zout_enc.shape)
        # print('mask_encdec.shape', mask_encdec.shape)
        # print('mask_dec.shape', mask_dec.shape)
        for count, decunit in enumerate(self.decunit_pipeline):
            z, attn_mhead_dec_dict, attn_mhead_encdec_dict = decunit(xinput, Zout_enc, mask_dec, mask_encdec)
            attn_mlayer_mhead_dec_dict[f'l{count}'] = attn_mhead_dec_dict
            attn_mlayer_mhead_encdec_dict[f'l{count}'] = attn_mhead_encdec_dict
            # update xinput to the newly compute z representation
            xinput = z

        # pooling using another attention layer
        # print('z.shape', z.shape)
        z, fattn_w_norm = self.pooling(z, mask_dec)

        # y = self.Wy(z) 
        y = self.Wy(z) + self.bias

        return y, fattn_w_norm, attn_mlayer_mhead_dec_dict, attn_mlayer_mhead_encdec_dict


#### Haplotype Encoder-Decoder model ####
class HaplotypeEncoderDecoder(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.enc = encoder
            self.dec = decoder
            self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

        def forward(self, Xin_enc, Xin_dec, mask_enc, mask_dec, mask_encdec):
            """
            Args:
                Xin_enc: decoder input tensor,  (1, sequence length)
                Xin_dec: encoder output tensor, (num_haplotypes, sequence length)
                mask_dec: decoder mask, tensor, (1, sequence length, sequence length) with 0/1 entries
                mask_enc: encoder mask, tensor, (1, sequence length, sequence length) with 0/1 entries
                mask_encdec: encoder mask, tensor, (1, dec sequence length, enc sequence length) with 0/1 entries

            """

            # compute encoder representation
            num_haplotypes = Xin_dec.shape[0]
            # z_enc (1, encoder sequene length, embed_dim)
            z_enc, fattn_norm_enc, attn_mlayer_mhead_enc_dict  = self.enc(Xin_enc, mask_enc)
            # print()
            # print('out encoder -> z_enc.shape:', z_enc.shape)
            # expand z_enc encoding to number of haplotypes
            # # (num_haplotypes, encoder sequence length, embed_dim)
            z_enc = z_enc.expand(num_haplotypes,z_enc.shape[1], z_enc.shape[2])
            # print('expanded z_enc.shape:', z_enc.shape)
            out = self.dec(Xin_dec, z_enc, mask_dec, mask_encdec)
            y, fattn_norm_dec, attn_mlayer_mhead_encdec_dict, attn_mlayer_mhead_dec_dict = out
            
            return self.logsoftmax(y), fattn_norm_dec, attn_mlayer_mhead_encdec_dict

        