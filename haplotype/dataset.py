import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class SeqMaskGenerator(object):
    def __init__(self, seqconfig):

        self.seqconfig = seqconfig

    def create_enc_mask(self, enc_inp): 
        #enc_inp = [N, inp_seq_len]
        # N is total number of input sequences
        bsize, seq_len = enc_inp.shape
#         #enc_mask.shape = [1, 1, inp_seq_len, inp_seq_len]
#         enc_mask = np.ones((1, 1, seq_len, seq_len))
#         #enc_mask.shape = [1, 1, inp_seq_len, inp_seq_len]
#         # enc_mask = enc_mask.reshape(1, 1, seq_len, seq_len)
#         #enc_mask.shape = [bsize, 1, inp_seq_len, inp_seq_len]
#         enc_mask = np.repeat(enc_mask, bsize, axis=0)
        enc_mask = np.full((bsize,1, seq_len, seq_len), 1)
        return enc_mask

    def create_enc_dec_mask(self, num_samples):
        inp_seqlen = self.seqconfig.seq_len
        outp_seqlen = self.seqconfig.ewindow_end+1
#         enc_dec_mask = np.ones((1,1, outp_seqlen, inp_seqlen))
#         enc_dec_mask = np.repeat(enc_dec_mask, num_samples, axis=0)
        enc_dec_mask = np.full((num_samples, 1, outp_seqlen, inp_seqlen), 1)
        return enc_dec_mask


    def create_dec_mask(self, mask_targetbase):
        # dec_inp = [num_haplotypes, outcome_seq_len]
        # outcome_seq_len is length of haplotype outcome sequence
        # mask_targetbase = [num_haplotyptes, outcome_seq_len]

        # generate causal mask
        seqconfig = self.seqconfig
        num_haplotypes = mask_targetbase.shape[0]
        ewindow_st, ewindow_end = seqconfig.ewindow_st, seqconfig.ewindow_end
#         ewindow_st = 0
        # 6-13
#         print('ewindow_st:', ewindow_st, 'ewindow_end:', ewindow_end)
        
        
        tm = mask_targetbase[:, ewindow_st:ewindow_end+1]
        tindx = np.where(tm.astype(np.bool))
        
#         print('tindx:\n', tindx)
        # tindx (array(), array()) representing row and column indices where mask has 1 entries
        target_pos_st = tindx[1][0] # give the start of target base occurence in the sequence
        
        ew_seqlen = ewindow_end - (target_pos_st + ewindow_st) + 1
#         print('ew_seqlen:', ew_seqlen)
        sub_mask = np.ones((ew_seqlen, ew_seqlen))
        sub_mask_ind = np.triu_indices(ew_seqlen, k=0)
        sub_mask[sub_mask_ind[0], sub_mask_ind[1]] = 0
        
        dec_causal_mask = np.ones((ewindow_end+1,ewindow_end+1))
#         print('dec_causal_mask.shape', dec_causal_mask.shape)
        
        offset = target_pos_st + ewindow_st
#         print('offset:',offset)
        for i in range(ewindow_end+1):
            if i < offset:
                dec_causal_mask[i, offset:] = 0
            else:
                dec_causal_mask[i, offset:] = sub_mask[i-offset,:]
#         print('dec_causal_mask:\n', dec_causal_mask)
        
        #dec_causal_mask.shape = [1, 0:ewindow_end+1, 0:ewindow_end+1]
        dec_causal_mask = dec_causal_mask.reshape(1, dec_causal_mask.shape[0], dec_causal_mask.shape[1])
        dec_causal_mask = np.repeat(dec_causal_mask, num_haplotypes, axis=0)
        return dec_causal_mask

class HaplotypeDataTensor(Dataset):

    def __init__(self, seqconfig):
        self.seqconfig = seqconfig
        
    # def _encode_to_one_hot(self, mask, n_dims=None):
    #     """ turn matrix with labels into one-hot encoding using the max number of classes detected"""
    #     original_mask_shape = mask.shape
    #     mask = mask.type(torch.LongTensor).view(-1, 1)
    #     if n_dims is None:
    #         n_dims = int(torch.max(mask)) + 1
    #     one_hot = torch.zeros(mask.shape[0], n_dims).scatter_(1, mask, 1)
    #     one_hot = one_hot.view(*original_mask_shape, -1)
    #     return one_hot

    def generate_tensor_from_df(self, proc_df, tb_cb_nucl, outcome_prop_col):
        # create the tensors we need
        # N is total number of input sequences
        print('Generating tensors using sequence config:\n', self.seqconfig)
        Xinp_enc = [] # tensor, (N x inp_sequence_len)
        Xinp_dec = [] # list of tensors, (N x num_haplotypes x outp_sequence_len)
        mask_inp_targetbase = [] # list of tensors, (N x num_haplotypes x outp_sequence_len)
        target_conv = [] # list of tensors, (N x num_haplotypes x outp_sequence_len)
        target_conv_onehot = [] # list of tensors (i.e. one-hot encoding), (N x num_haplotypes x outp_sequence_len x 2 x 1)
        target_prob = [] # list of tensors, (N x num_haplotypes)
        mask_dec = []
        indx_seqid_map = {} # dict, int_id:(seqid, target_seq)
        inpseq_outpseq_map = {} # dict([]), int_id:[outp_seq1, out_seq2, ....]

        seqconfig = self.seqconfig
        mask_generator = SeqMaskGenerator(seqconfig)
        seq_len = seqconfig.seq_len
        tb_nucl, cb_nucl = tb_cb_nucl # target base, conversion base (i.e. A->G for ABE base editor)
                                      #                                    C->T for CBE base editor
        
        # output sequence will be from 0:end of editable window indx

        for gr_name, gr_df in tqdm(proc_df.groupby(by=['seq_id', 'Inp_seq'])):
            
            Xinp_enc.append(gr_df[[f'Inp_B{i}' for i in range(1,seq_len+1)]].values[0,:])
            Xinp_dec.append(gr_df[[f'Outp_B{i}' for i in range(1,seq_len+1)]].values[:,0:seqconfig.ewindow_end+1])
            
            mask_inp_targetbase.append(gr_df[[f'Inp_M{i}' for i in range(1,seq_len+1)]].values[:,0:seqconfig.ewindow_end+1])
            conv = gr_df[[f'conv{tb_nucl}{cb_nucl}_{i}' for i in range(1,seq_len+1)]].values[:,0:seqconfig.ewindow_end+1]
            target_conv.append(conv)
            if outcome_prop_col is not None:
                target_prob.append(gr_df[outcome_prop_col].values)
#             print(target_prob[-1])

            # compute mask_enc and mask_dec
#             print(mask_targetbase[-1])
            mask_dec.append(mask_generator.create_dec_mask(mask_inp_targetbase[-1]))
            
            inpseq_id = len(indx_seqid_map)
            indx_seqid_map[inpseq_id] = gr_name
            inpseq_outpseq_map[inpseq_id] = gr_df['Outp_seq'].values.tolist()
        
        mask_enc = None
        mask_encdec = None

        # tensorize
        print('--- tensorizing ---')
        device_cpu = torch.device('cpu')
        self.Xinp_enc = torch.tensor(Xinp_enc).long().to(device_cpu)
        self.Xinp_enc = self.Xinp_enc.reshape(self.Xinp_enc.shape[0], 1, self.Xinp_enc.shape[1])
        self.Xinp_dec = [torch.from_numpy(arr).long().to(device_cpu) for arr in Xinp_dec]
        self.mask_inp_targetbase = [torch.from_numpy(arr).long().to(device_cpu) for arr in mask_inp_targetbase]
        self.target_conv_onehot = [torch.nn.functional.one_hot(torch.from_numpy(arr).long().to(device_cpu), num_classes=2)
                                   for arr in target_conv]
        if outcome_prop_col is not None:
            self.target_prob = [torch.from_numpy(arr).float().to(device_cpu) for arr in target_prob]
        else:
            self.target_prob = None

        self.mask_enc = mask_enc
        self.mask_encdec = mask_encdec

        self.mask_dec = [torch.from_numpy(arr).long().to(device_cpu) for arr in mask_dec]
        
        self.num_samples = len(self.Xinp_enc) # int, number of sequences
        self.indx_seqid_map = indx_seqid_map
        self.inpseq_outpseq_map = inpseq_outpseq_map
        print('--- end ---')

    def hap_collate(self, batch):
        # pack batches in a list for now
        # to be used in dataloader object
        return [item for item in batch]

    def __getitem__(self, indx):

        if self.target_prob is None:
            return_target_prob = None
        else:
            return_target_prob = self.target_prob[indx]

        return(self.Xinp_enc[indx], 
               self.Xinp_dec[indx],
               self.mask_enc,
               self.mask_dec[indx],
               self.mask_encdec,
               self.mask_inp_targetbase[indx],
               self.target_conv_onehot[indx],
               return_target_prob,
               indx,
               self.indx_seqid_map[indx],
               self.inpseq_outpseq_map[indx])     

    def __len__(self):
        return(self.num_samples)

class PartitionDataTensor(Dataset):

    def __init__(self, dtensor, partition_ids, dsettype, run_num):
        self.dtensor = dtensor  # instance of :class:`HaplotypeDataTensor`
        self.partition_ids = partition_ids  # list of sequence indices
        self.dsettype = dsettype  # string, dataset type (i.e. train, validation, test)
        self.run_num = run_num  # int, run number
        self.num_samples = len(self.partition_ids[:])  # int, number of docs in the partition

    def __getitem__(self, indx):
        target_id = self.partition_ids[indx]
        return self.dtensor[target_id]

    def __len__(self):
        return(self.num_samples)

def print_data_example(elm):
    Xinp_enc, Xinp_dec, mask_enc, mask_dec, mask_encdec, mask_targetbase_enc, target_conv_onehot, target_prob, indx, seqid = elm
    print('Xinp_enc:\n', Xinp_enc, 'shape:', Xinp_enc.shape)
    print('Xinp_dec:\n',Xinp_dec, 'shape:',Xinp_dec.shape)
    if mask_enc is not None:
        print('mask_enc:\n', mask_enc, 'shape:',mask_enc.shape)
    print('mask_dec:\n',mask_dec, 'shape:',mask_dec.shape)
    if mask_encdec is not None:
        print('mask_encdec:\n', mask_encdec, 'shape:',mask_encdec.shape)

    print('mask_targetbase_enc:\n', mask_targetbase_enc,'shape:', mask_targetbase_enc.shape)
    print('target_conv_onehot:\n',target_conv_onehot, 'shape:',target_conv_onehot.shape)
    if target_prob is not None:
        print('target_prob:\n',target_prob, 'shape:',target_prob.shape)
    else:
        print('target_prob:None')
    print('indx:', indx)
    print('seqid:', seqid)

def hap_collate(batch):
    # pack batches in a list for now
    # to be used in dataloader object
    return [item for item in batch]
