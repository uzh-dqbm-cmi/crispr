import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from .utilities import CategModelScore, ContModelScore

class CrisCASDataTensor(Dataset):

    def __init__(self, X_feat, y_score, y_categ, indx_seqid_map):
        # B: batch elements; T: sequence length
        self.X_feat = X_feat  # tensor.float32, B x T, (sequence characters are mapped to 0-3)
        self.y_score = y_score  # tensor.float32, B, (efficiency score)
        self.y_categ = y_categ # tensor.int64, B, (categorized efficiency score)
        self.indx_seqid_map = indx_seqid_map
        self.num_samples = self.X_feat.size(0)  # int, number of sequences

    def __getitem__(self, indx):

        return(self.X_feat[indx], self.y_score[indx], self.y_categ[indx], indx, self.indx_seqid_map[indx])

    def __len__(self):
        return(self.num_samples)


class PartitionDataTensor(Dataset):

    def __init__(self, criscas_datatensor, partition_ids, dsettype, run_num):
        self.criscas_datatensor = criscas_datatensor  # instance of :class:`CrisCASDataTensor`
        self.partition_ids = partition_ids  # list of sequence indices
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.run_num = run_num  # int, run number
        self.num_samples = len(self.partition_ids)  # int, number of docs in the partition

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        return self.criscas_datatensor[target_id]

    def __len__(self):
        return(self.num_samples)

def create_datatensor(df):
    """create a instance of DataTensor from processeed/cleaned dataframe
    
    Args:
        df: pandas.DataFrame, processed data by :func:`generate_perbase_df`
    """
    # index -> sequence id map
    # X_tensor -> B x T (sequence characters are mapped to 0-3)
    # mask -> B x T (mask where A or C characters exist)
    # y -> B (efficiency score or efficiency score categ )
    
    X_tensor = torch.from_numpy(df[[f'B{i}' for  i in range(1, 21)]].values)
    y_score = torch.from_numpy(df['efficiency_score'].values)
    y_categ = torch.from_numpy(df['edited_seq_categ'].values)
    seqs_id = df['ID']
    indx_seqid_map = {i:seqs_id[i] for i in range(df.shape[0])}
    
    return CrisCASDataTensor(X_tensor, y_score, y_categ, indx_seqid_map)

def get_stratified_partitions(df, num_splits=5, val_set_portion=0.5, random_state=42):
    """Generate multi-run stratified sample of sequence ids based on the efficiency score categ
    Args:
        df: pandas.DataFrame, dataframe processed by :func:`generate_clean_df` or `generate_perbase_df`
        num_splits: int, number of runs
        val_set_portion: float, % of validation set from val/set portion
        random_state: int, 
    """
    
    sss_valtest = StratifiedShuffleSplit(n_splits=num_splits, random_state=random_state, train_size=val_set_portion)
    data_partitions = {}
    train_index = df.loc[df['seq_type']==0].index.tolist()
    print('train data')
    report_label_distrib(df.loc[df['seq_type']==0, 'edited_seq_categ'].values)
    print()
    
    df_index = df.loc[df['seq_type']==1].index # index corresponding to val/test sequences
    
    y = df.loc[df['seq_type']==1, 'edited_seq_categ'].values
    X = np.zeros(len(y))
    
    run_num = 0
    
    for val_index, test_index in sss_valtest.split(X,y):
    
        data_partitions[run_num] = {'train': train_index,
                                    'validation': df_index[val_index].tolist(),
                                    'test': df_index[test_index].tolist()}
        print("run_num:", run_num)
        print('validation data')
        report_label_distrib(y[val_index])
        print('test data')
        report_label_distrib(y[test_index])
        print()
        run_num += 1
        print("-"*25)
    return(data_partitions)

def validate_partitions(data_partitions, seqs_id, val_set_portion=0.5):
    
    if(not isinstance(seqs_id, set)):
        seqs_id = set(seqs_id)
    for run_num in data_partitions:
        print('run_num', run_num)
        tr_ids = data_partitions[run_num]['train']
        val_ids = data_partitions[run_num]['validation']
        te_ids = data_partitions[run_num]['test']

        tr_val = set(tr_ids).intersection(val_ids)
        tr_te = set(tr_ids).intersection(te_ids)
        te_val = set(te_ids).intersection(val_ids)

        valset_size = len(te_ids) + len(val_ids)
        num_seqs = len(tr_ids) + len(val_ids) + len(te_ids)
        print('expected validation set size:', val_set_portion*valset_size, '; actual validation set size:', len(val_ids))
        print('expected test set size:', (1-val_set_portion)*valset_size, '; actual test set size:', len(te_ids))
        print()
        assert np.abs(val_set_portion*valset_size - len(val_ids)) <= 2  # valid difference range
        assert np.abs(((1-val_set_portion)*valset_size)- len(te_ids)) <= 2
        # assert there is no overlap among train, val and test partition within a fold
        for s in (tr_val, tr_te, te_val):
            assert len(s) == 0

        s_union = set(tr_ids).union(val_ids).union(te_ids)
        assert len(s_union) == num_seqs
    print('-'*25)
    print("passed intersection and overlap test (i.e. train, validation and test sets are not",
          "intersecting in each fold and the concatenation of test sets from each fold is",
          "equivalent to the whole dataset)")

def report_label_distrib(labels):
    classes, counts = np.unique(labels, return_counts=True)
    norm_counts = counts/counts.sum()
    for i, label in enumerate(classes):
        print("class:", label, "norm count:", norm_counts[i])

def generate_partition_datatensor(criscas_datatensor, data_partitions):
    datatensor_partitions = {}
    for run_num in data_partitions:
        datatensor_partitions[run_num] = {}
        for dsettype in data_partitions[run_num]:
            target_ids = data_partitions[run_num][dsettype]
            datatensor_partition = PartitionDataTensor(criscas_datatensor, target_ids, dsettype, run_num)
            datatensor_partitions[run_num][dsettype] = datatensor_partition
    compute_class_weights_per_run_(datatensor_partitions)
    return(datatensor_partitions)

def compute_class_weights(labels_tensor):
    classes, counts = np.unique(labels_tensor, return_counts=True)
    # print("classes", classes)
    # print("counts", counts)
    class_weights = compute_class_weight('balanced', classes, labels_tensor.numpy())
    return class_weights


def compute_class_weights_per_run_(datatensor_partitions):
    """computes inverse class weights and updates the passed dictionary
    Args:
        datatensor_partitions: dictionary, {run_num, int: {datasettype, string:{datapartition, instance of
        :class:`PartitionDataTensor`}}}}
    Example:
        datatensor_partitions
            {0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                 'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                 'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>
                }, ..
            }
        is updated after computation of class weights to
            {0: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                 'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                 'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                 'class_weights': tensor([0.6957, 1.7778]),
                 }, ..
            }
    """

    for run_num in datatensor_partitions:  # looping over the numbered folds
        dpartition = datatensor_partitions[run_num]['train']
        partition_ids = dpartition.partition_ids
        labels = dpartition.criscas_datatensor.y_categ[partition_ids]
        datatensor_partitions[run_num]['class_weights'] = torch.from_numpy(compute_class_weights(labels)).float()

def construct_load_dataloaders(dataset_fold, dsettypes, score_type, config, wrk_dir):
    """construct dataloaders for the dataset for one run or fold
       Args:
            dataset_fold: dictionary,
                          example: {'train': <neural.dataset.PartitionDataTensor at 0x1cec95c96a0>,
                                    'validation': <neural.dataset.PartitionDataTensor at 0x1cec95c9208>,
                                    'test': <neural.dataset.PartitionDataTensor at 0x1cec95c9240>,
                                    'class_weights': tensor([0.6957, 1.7778])
                                   }
            score_type:  str, either {'categ', 'continuous'}
            dsettype: list, ['train', 'validation', 'test']
            config: dict, {'batch_size': int, 'num_workers': int}
            wrk_dir: string, folder path
    """

    # setup data loaders
    data_loaders = {}
    epoch_loss_avgbatch = {}
    epoch_loss_avgsamples = {}
    flog_out = {}
    score_dict = {}
    class_weights = {}
    for dsettype in dsettypes:
        if(dsettype == 'train'):
            shuffle = True
            class_weights[dsettype] = dataset_fold['class_weights']
        else:
            shuffle = False
            class_weights[dsettype] = None
        data_loaders[dsettype] = DataLoader(dataset_fold[dsettype],
                                            batch_size=config['batch_size'],
                                            shuffle=shuffle,
                                            num_workers=config['num_workers'])

        epoch_loss_avgbatch[dsettype] = []
        epoch_loss_avgsamples[dsettype] = []
        
        if(score_type == 'categ'):
            #  (best_epoch, binary_f1, macro_f1, accuracy, auc)
            score_dict[dsettype] = CategModelScore(0, 0.0, 0.0, 0.0, 0.0)
        elif(score_type == 'continuous'):
            #  (best_epoch, spearman_correlation)
            score_dict[dsettype] = ContModelScore(0, 0.0)
        if(wrk_dir):
            flog_out[dsettype] = os.path.join(wrk_dir, dsettype + ".log")
        else:
            flog_out[dsettype] = None

    return (data_loaders, epoch_loss_avgbatch, epoch_loss_avgsamples, score_dict, class_weights, flog_out)