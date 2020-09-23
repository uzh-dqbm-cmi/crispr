#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[1]:


import os
import datetime
import numpy as np
import scipy
import pandas as pd
import torch
from torch import nn
import criscas
from criscas.utilities import create_directory, get_device, report_available_cuda_devices
from criscas.predict_model import *


# In[2]:


base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))


# ### Read sample data

# In[3]:


seq_df = pd.read_csv(os.path.join(base_dir, 'sample_data', 'abemax_sampledata.csv'), header=0)


# In[4]:


seq_df


# The models expect sequences (i.e. target sites) to be wrapped in a `pandas.DataFrame` with a header that includes `ID` of the sequence and `seq` columns.
# The sequences should be of length 20 (i.e. 20 bases) and represent the protospacer target site.

# In[5]:


# create a directory where we dump the predictions of the models
csv_dir = create_directory(os.path.join(base_dir, 'sample_data', 'predictions'))


# ### Specify device (i.e. CPU or GPU) to run the models on

# Specify device to run the model on. The models can run on `GPU` or `CPU`. We can instantiate a device by running `get_device(to_gpu,gpu_index)` function. 
# 
# - To run on GPU we pass `to_gpu = True` and specify which card to use if we have multiple cards `gpu_index=int` (i.e. in case we have multiple GPU cards we specify the index counting from 0). 
# - If there is no GPU installed, the function will return a `CPU` device.

# We can get a detailed information on the GPU cards installed on the compute node by calling `report_available_cuda_devices` function.

# In[6]:


report_available_cuda_devices()


# In[7]:


# instantiate a device using the only one available :P
device = get_device(True, 0)
device


# ### Create a BE-DICT model by sepcifying the target base editor 

# We start `BE-DICT` model by calling `BEDICT_CriscasModel(base_editor, device)` where we specify which base editor to use (i.e. `ABEmax`, `BE4`, `ABE8e`, `AID`) and the `device` we create earlier to run on.

# In[9]:


base_editor = 'ABEmax'
bedict = BEDICT_CriscasModel(base_editor, device)


# We generate predictions by calling `predict_from_dataframe(seq_df)` where we pass the data frame wrapping the target sequences. The function returns two objects:
# 
# - `pred_w_attn_runs_df` which is a data frame that contains predictions per target base and the attentions scores across all positions.
# 
# - `proc_df` which is a data frame that represents the processed sequence data frame we passed (i.e. `seq_df`)

# In[10]:


pred_w_attn_runs_df, proc_df = bedict.predict_from_dataframe(seq_df)


# `pred_w_attn_runs_df` contains predictions from 5 trained models for `ABEmax` base editor (we have 5 runs trained per base editor). For more info, see our [paper](https://www.biorxiv.org/content/10.1101/2020.07.05.186544v1) on biorxiv.

# Target positions in the sequence reported in `base_pos` column in `pred_w_attn_runs_df` uses 0-based indexing (i.e. 0-19)

# In[11]:


pred_w_attn_runs_df


# In[12]:


proc_df


# Given that we have 5 predictions per sequence, we can further reduce to one prediction by either `averaging` across all models, or taking the `median` or `max` prediction based on the probability of editing scores. For this we use `select_prediction(pred_w_attn_runs_df, pred_option)` where `pred_w_attn_runs_df` is the data frame containing predictions from 5 models for each sequence. `pred_option` can be assume one of {`mean`, `median`, `max`}.

# In[13]:


pred_option = 'mean'
pred_w_attn_df = bedict.select_prediction(pred_w_attn_runs_df, pred_option)


# In[14]:


pred_w_attn_df


# We can dump the prediction results on a specified directory on disk. We will dump the predictions with all 5 runs `pred_w_attn_runs_df` and the one average across runs `pred_w_attn_df`.

# Under `sample_data` directory we will have the following tree:
# 
# <code>
# sample_data
# └── predictions
#     ├── predictions_allruns.csv
#     └── predictions_predoption_mean.csv
# </code>

# In[15]:


pred_w_attn_runs_df.to_csv(os.path.join(csv_dir, f'predictions_allruns.csv'))


# In[16]:


pred_w_attn_df.to_csv(os.path.join(csv_dir, f'predictions_predoption_{pred_option}.csv'))


# ### Generate attention plots

# We can generate attention plots for the prediction of each target base in the sequence using `highlight_attn_per_seq` method that takes the following arguments:
# 
# - `pred_w_attn_runs_df`: data frame that contains model's predictions (5 runs) for each target base of each sequence (see above).
# - `proc_df`: data frame that represents the processed sequence data frame we passed (i.e. seq_df)
# - `seqid_pos_map`: dictionary `{seq_id:list of positions}` where `seq_id` is the ID of the target sequence, and list of positions that we want to generate attention plots for. Users can specify a `position from 1 to 20` (i.e. length of protospacer sequence)
# - `pred_option`: selection option for aggregating across 5 models' predictions. That is we can average the predictions across 5 runs, or take `max`, `median`, `min` or `None` (i.e. keep all 5 runs) 
# - `apply_attnscore_filter`: boolean (`True` or `False`) to further apply filtering on the generated attention scores. This filtering allow to plot only predictions where the associated attention scores have a maximum that is >= 3 times the base attention score value <=> (3 * 1/20)
# - `fig_dir`: directory where to dump the generated plots or `None` (to return the plots inline)

# In[17]:


# create a dictionary to specify target sequence and the position I want attention plot for
# we are targeting position 5 in the sequence
seqid_pos_map = {'CTRL_HEKsiteNO1':[5], 'CTRL_HEKsiteNO2':[5]}
pred_option = 'mean'
apply_attn_filter = False
bedict.highlight_attn_per_seq(pred_w_attn_runs_df, 
                              proc_df,
                              seqid_pos_map=seqid_pos_map,
                              pred_option=pred_option, 
                              apply_attnscore_filter=apply_attn_filter, 
                              fig_dir=None)


# We can save the plots on disk without returning them by specifing `fig_dir`

# In[18]:


# create a dictionary to specify target sequence and the position I want attention plot for
# we are targeting position 5 in the sequence
seqid_pos_map = {'CTRL_HEKsiteNO1':[5], 'CTRL_HEKsiteNO2':[5]}
pred_option = 'mean'
apply_attn_filter = False
fig_dir =  create_directory(os.path.join(base_dir, 'sample_data', 'fig_dir'))
bedict.highlight_attn_per_seq(pred_w_attn_runs_df, 
                              proc_df,
                              seqid_pos_map=seqid_pos_map,
                              pred_option=pred_option, 
                              apply_attnscore_filter=apply_attn_filter, 
                              fig_dir=create_directory(os.path.join(fig_dir, pred_option)))


# We will generate the following files:
# 
# <code>
# sample_data
# ├── abemax_sampledata.csv
# ├── fig_dir
# │   └── mean
# │       ├── ABEmax_seqattn_CTRL_HEKsiteNO1_basepos_5_predoption_mean.pdf
# │       └── ABEmax_seqattn_CTRL_HEKsiteNO2_basepos_5_predoption_mean.pdf
# └── predictions
#     ├── predictions_allruns.csv
#     └── predictions_predoption_mean.csv
# </code>

# Similarly we can change the other arguments such as `pred_option` `apply_attnscore_filter` and so on to get different filtering options - We leave this as an exercise for the user/reader :D

# In[ ]:




