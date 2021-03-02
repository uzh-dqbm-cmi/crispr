import os
from .utilities import ReaderWriter

class Haplotype_Trf_HyperparamConfig:
    def __init__(self, embed_dim, num_attn_heads, num_transformer_units, 
                p_dropout, nonlin_func, mlp_embed_factor, multihead_type, 
                l2_reg, batch_size, num_epochs):
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.num_transformer_units = num_transformer_units
        self.p_dropout = p_dropout
        self.nonlin_func = nonlin_func
        self.mlp_embed_factor = mlp_embed_factor
        self.multihead_type = multihead_type
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def __repr__(self):
        desc = " embed_dim:{}\n num_attn_heads:{}\n num_transformer_units:{}\n p_dropout:{} \n " \
               "nonlin_func:{} \n mlp_embed_factor:{} \n multihead_type:{} \n" \
               "l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.embed_dim,
                                                                     self.num_attn_heads,
                                                                     self.num_transformer_units,
                                                                     self.p_dropout, 
                                                                     self.nonlin_func,
                                                                     self.mlp_embed_factor,
                                                                     self.multihead_type,
                                                                     self.l2_reg, 
                                                                     self.batch_size,
                                                                     self.num_epochs)
        return desc



def get_saved_config(config_dir):
    options = ReaderWriter.read_data(os.path.join(config_dir, 'exp_options.pkl'))
    mconfig = ReaderWriter.read_data(os.path.join(config_dir, 'mconfig.pkl'))
    return mconfig, options






