import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

class CrisCASDataTensor(Dataset):

    def __init__(self, X_feat, y_score, y_categ, indx_seqid_map):
        # B: batch elements; T: sequence length
        self.X_feat = X_feat  # tensor.float32, B x T, (sequence characters are mapped to 0-3)
        self.y_score = y_score  # tensor.float32, B, (efficiency score)
        self.y_categ = y_categ # tensor.int64, B, (categorized efficiency score)
        self.indx_seqid_map = indx_seqid_map
        self.num_samples = self.X_feat.size(0)  # int, number of sequences

    def __getitem__(self, indx):
        if self.y_score is None:
            return(self.X_feat[indx], indx, self.indx_seqid_map[indx])
            
        return(self.X_feat[indx], self.y_score[indx], self.y_categ[indx], indx, self.indx_seqid_map[indx])

    def __len__(self):
        return(self.num_samples)

class CrisCASSeqDataTensor(Dataset):

    def __init__(self, X_feat, y_score, y_categ, y_overall_categ, mask, indx_seqid_map):
        # B: batch elements; T: sequence length
        self.X_feat = X_feat  # tensor.float32, B x T, (sequence characters are mapped to 0-3)
        self.y_score = y_score  # tensor.float32, B x T, (efficiency score)
        self.y_categ = y_categ # tensor.int64, B x T, (categorized efficiency score)
        self.y_overall_categ = y_overall_categ # tensor.int64, B, (categorized efficiency score)
        self.mask = mask # tensor.int64, B x T, (boolean mask for indicating target base)
        self.indx_seqid_map = indx_seqid_map
        self.num_samples = self.X_feat.size(0)  # int, number of sequences

    def __getitem__(self, indx):

        if self.y_score is None:
            return(self.X_feat[indx], self.mask[indx], indx, self.indx_seqid_map[indx])

        return(self.X_feat[indx], self.y_score[indx], self.y_categ[indx], self.mask[indx], indx, self.indx_seqid_map[indx])

    def __len__(self):
        return(self.num_samples)

class PartitionDataTensor(Dataset):

    def __init__(self, criscas_datatensor, partition_ids, dsettype, run_num):
        self.criscas_datatensor = criscas_datatensor  # instance of :class:`CrisCASDataTensor` or :`CrisCasSeqDataTensor`
        self.partition_ids = partition_ids  # list of sequence indices
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.run_num = run_num  # int, run number
        self.num_samples = len(self.partition_ids[:])  # int, number of docs in the partition

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        return self.criscas_datatensor[target_id]

    def __len__(self):
        return(self.num_samples)


def create_datatensor(df, per_base=False, refscore_available=True):
    """create a instance of DataTensor from processeed/cleaned dataframe
    
    Args:
        df: pandas.DataFrame, processed data by :func:`generate_perbase_df`
    """
    # index -> sequence id map
    # X_tensor -> B x T (sequence characters are mapped to 0-3)
    # mask -> B x T (mask where A or C characters exist)
    # y -> B (efficiency score or efficiency score categ )
    if refscore_available:
        if not per_base:
            X_tensor = torch.from_numpy(df[[f'B{i}' for  i in range(1, 21)]].values)
            # y_score = torch.from_numpy(df['efficiency_score_prank'].values)
            y_score = torch.from_numpy(df['efficiency_score'].values)
            y_categ = torch.from_numpy(df['edited_seq_categ'].values)
            seqs_id = df['ID']
            indx_seqid_map = {i:seqs_id[i] for i in df.index.tolist()}
            dtensor = CrisCASDataTensor(X_tensor, y_score, y_categ, indx_seqid_map)
        else:
            X_tensor = torch.from_numpy(df[[f'B{i}' for  i in range(1, 21)]].values)
            y_score = torch.from_numpy(df[[f'ES{i}' for  i in range(1, 21)]].values)
            # we change type since it is categorical dtype
            # y_categ = torch.from_numpy(df[[f'ESpr{i}' for  i in range(1, 21)]].values)
            y_categ = torch.from_numpy(df[[f'ECi{i}' for  i in range(1, 21)]].astype(np.int64).values)
            y_overall_categ = torch.from_numpy(df['edited_seq_categ'].values)
            mask = torch.from_numpy(df[[f'M{i}' for  i in range(1, 21)]].values)
            seqs_id = df['ID']
            indx_seqid_map = {i:seqs_id[i] for i in df.index.tolist()}
            dtensor = CrisCASSeqDataTensor(X_tensor, y_score, y_categ, y_overall_categ, mask, indx_seqid_map)
    else:
        if not per_base:
            X_tensor = torch.from_numpy(df[[f'B{i}' for  i in range(1, 21)]].values)
            y_score = None
            y_categ = None
            seqs_id = df['ID']
            indx_seqid_map = {i:seqs_id[i] for i in df.index.tolist()}
            dtensor = CrisCASDataTensor(X_tensor, y_score, y_categ, indx_seqid_map)
        else:
            X_tensor = torch.from_numpy(df[[f'B{i}' for  i in range(1, 21)]].values)
            y_score = None
            y_categ = None
            y_overall_categ = None
            mask = torch.from_numpy(df[[f'M{i}' for  i in range(1, 21)]].values)
            seqs_id = df['ID']
            indx_seqid_map = {i:seqs_id[i] for i in df.index.tolist()}
            dtensor = CrisCASSeqDataTensor(X_tensor, y_score, y_categ, y_overall_categ, mask, indx_seqid_map)
    
    return dtensor