import os
import numpy as np
import pandas as pd


def filter_attn_rows(attn_w_df, base_w):
    t = attn_w_df[[f'Attn{i}'for i in range(20)]].max(axis=1)
    # attn score >= 15% (0.15)
    cond = ((t - base_w) >= (2*base_w))
    print('number of rows below threshold:', len(attn_w_df)- cond.sum())
    attn_w_df = attn_w_df.loc[cond].copy()
    return attn_w_df