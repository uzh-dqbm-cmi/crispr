import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
from .model import HaplotypeEncoderDecoder, Encoder, Decoder
from .data_preprocess import validate_df, VizInpOutp_Haplotype
from .dataset import HaplotypeDataTensor, hap_collate
from .hyperparam import get_saved_config
from .utilities import ReaderWriter, build_predictions_df, check_na
from tqdm import tqdm


class BEDICT_HaplotypeModel:
    def __init__(self, seq_processor, seqconfig, device):
        self.seq_processor = seq_processor
        self.seqconfig = seqconfig
        self.device = device

    def _process_df(self, df, inpseq_cols, outpseq_col=None, outcome_col=None, renormalize=True):
        """
        Args:
            df: pandas dataframe
            inpseq_cols: list of column names containing `sequence id` and `input sequence` such as ['seq_id', 'Inp_seq']
            outpseq_col: string, column name of output sequence (i.e. edit combination)
            outcome_col: string, column name of outcome propotion
            renormalize: boolean, for renormalizing outcome proportion
        """
        print('--- processing input data frame ---')
        df = df.copy()
        seqid_col = inpseq_cols[0] # seqid column
        inpseq_col = inpseq_cols[-1] # Inp_seq column
        if outpseq_col is None:
            # case where the model is given list of sequeneces to generate haplotype for
            df.drop_duplicates(subset=inpseq_cols, ignore_index=True, inplace=True, keep='first')
            validate_df(df)
            df = self.seq_processor.remove_viol_seqs(df, inpseq_col)
            proc_df = self.seq_processor.process_inp_outp_df(df, seqid_col, inpseq_col, None, None)
            comb_df = self.seq_processor.generate_combinatorial_outcome(proc_df)
            proc_df = self.seq_processor.process_inp_outp_df(comb_df, 'seq_id', 'Inp_seq', 'Outp_seq', None)
        else:
            # case where the model is given list of input sequences and their outcome sequences to evaluate
            # probability of each outcome sequence
            df = self.seq_processor.preprocess_df(df, inpseq_cols, outpseq_col)
            df = self.seq_processor.remove_viol_seqs(df, inpseq_col)
            validate_df(df)
            # TODO: add flag to check if to apply renormalization routine
            if renormalize:
                df = self.seq_processor.renormalize_outcome_prop(df, inpseq_cols, outcome_col)
            proc_df = self.seq_processor.process_inp_outp_df(df, seqid_col, inpseq_col, outpseq_col, outcome_col)

        return proc_df

    def _construct_datatensor(self, proc_df, outcome_col=None):
        dtensor = HaplotypeDataTensor(self.seqconfig)
        conv_nucl = self.seq_processor.conversion_nucl
        dtensor.generate_tensor_from_df(proc_df, conv_nucl, outcome_col)
        return dtensor

    def _construct_dloader(self, dtensor, batch_size):
        print('--- creating datatensor ---')
        dloader = DataLoader(dtensor,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=hap_collate,
                            sampler=None)
        return dloader

    def _load_model_config(self, mconfig_dir):
        print('--- loading model config ---')
        mconfig, options = get_saved_config(mconfig_dir)
        return mconfig, options

    def _build_base_model(self, config):
        print('--- building model ---')
        mconfig, options = config
        model_config = mconfig['model_config']
        inp_seqlen = options.get('inp_seqlen')
        encoder = Encoder(model_config.embed_dim,
                          num_nucleotides=4, 
                          seq_length=inp_seqlen, 
                          num_attn_heads=model_config.num_attn_heads, 
                          mlp_embed_factor=model_config.mlp_embed_factor,
                          nonlin_func=model_config.nonlin_func, 
                          pdropout=model_config.p_dropout, 
                          num_encoder_units=model_config.num_transformer_units, 
                          pooling_mode='attn', 
                          multihead_type=model_config.multihead_type)

        outp_seqlen = options.get('outp_seqlen')
        decoder = Decoder(model_config.embed_dim, 
                          num_nucleotides=4, 
                          seq_length=outp_seqlen,
                          num_attn_heads=model_config.num_attn_heads, 
                          mlp_embed_factor=model_config.mlp_embed_factor,
                          nonlin_func=model_config.nonlin_func, 
                          pdropout=model_config.p_dropout, 
                          num_decoder_units=model_config.num_transformer_units, 
                          pooling_mode='attn', 
                          multihead_type=model_config.multihead_type,
                          num_classes=2)

        encoder_decoder = HaplotypeEncoderDecoder(encoder, decoder)
        return encoder_decoder

    def _load_model_statedict_(self, model, model_dir):
        print('--- loading trained model ---')
        device = self.device

        model_name = 'HaplotypeTransformer'
        models = [(model, model_name)]

        # load state_dict pth
        state_dict_dir = os.path.join(model_dir, 'model_statedict')
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

        # update model's fdtype and move to device
        for m, m_name in models:
            m.type(torch.float32).to(device)
            m.eval()
        return model

    def _run_prediction(self, model, dloader, refscore_avail):

        device = self.device
        pred_score = []
        ref_score = []
        seqs_ids_lst = []
        outpseqs_ids_lst = []

        # going over batches
        for indx_batch, sbatch in tqdm(enumerate(dloader)):
            # print('batch indx:', indx_batch)

            # going over examples in a current batch
            for dexample in sbatch:
                Xinp_enc, Xinp_dec, mask_enc, mask_dec, mask_encdec, mask_targetbase_enc, target_conv_onehot, target_prob, indx, seqid, outpseqs_lst = dexample
                
                num_haplotypes = Xinp_dec.shape[0]
                hap_seqlen = Xinp_dec.shape[1]

                Xinp_enc = Xinp_enc.to(device)
                Xinp_dec = Xinp_dec.to(device)
                mask_dec = mask_dec.to(device)

                mask_targetbase_enc = mask_targetbase_enc.to(device)
                target_conv_onehot = target_conv_onehot.to(device)
                if refscore_avail:
                    target_prob = target_prob.to(device)
                with torch.set_grad_enabled(False):
                    pred_logprob, fattn_norm_dec, attn_mlayer_mhead_encdec_dict= model(Xinp_enc, 
                                                                                       Xinp_dec, 
                                                                                       mask_enc, 
                                                                                       mask_dec, 
                                                                                       mask_encdec)
                
                    # (num_haplotypes, seqlen, 1, 2)
                #  print('pred_logprob.shape:', pred_logprob.shape)
                    pred_logprob_resh = pred_logprob.unsqueeze(2)
                #  print('pred_logprob_resh.shape:', pred_logprob_resh.shape)
                    # (num_haplotypes, seqlen, 2, 1)
                #  print('target_conv_onehot.shape:', target_conv_onehot.shape)
                    conv_onehot_resh = target_conv_onehot.unsqueeze(3)
                    out = pred_logprob_resh.matmul(conv_onehot_resh.type(torch.float32))
                    # (num_haplotypes, seqlen)
                    out = out.reshape(num_haplotypes, hap_seqlen)
                    tindx = torch.where(mask_targetbase_enc.type(torch.bool))

                    pred_hap_logprob = out[tindx].view(num_haplotypes, -1).sum(axis=1)

                    pred_score.extend(torch.exp(pred_hap_logprob).view(-1).tolist())
                    if refscore_avail:
                        ref_score.extend(target_prob.view(-1).tolist())
                    seqs_ids_lst.extend([seqid]*num_haplotypes)
                    outpseqs_ids_lst.extend(outpseqs_lst)
        if refscore_avail:
            #TODO: we can ommit the check by simply passing empty array and check in build_predictions
            predictions_df = build_predictions_df(seqs_ids_lst, outpseqs_ids_lst, ref_score, pred_score)
        else:
            predictions_df = build_predictions_df(seqs_ids_lst, outpseqs_ids_lst, None, pred_score)

        return predictions_df
        
    def prepare_data(self, df, inpseq_cols, outpseq_col=None, outcome_col=None, renormalize=True, batch_size=500):
        """
        Args:
            df: pandas dataframe
            inpseq_cols: list of column names containing `sequence id` and `input sequence` such as ['seq_id', 'Inp_seq']
            outpseq_col: string, column name of output sequence (i.e. edit combination)
            outcome_col: string, column name of outcome propotion
            renormalize: boolean, for renormalizing outcome proportion
            batch_size: int, number of samples to process per batch
        """
        proc_df = self._process_df(df, inpseq_cols, outpseq_col, outcome_col, renormalize=renormalize)
        dtensor = self._construct_datatensor(proc_df, outcome_col)
        dloader = self._construct_dloader(dtensor, batch_size)
        return dloader

    def predict_from_dloader(self, dloader, model_dir, outcome_col=None):
        refscore_avail = outcome_col
        mconfig_dir = os.path.join(model_dir, 'config')
        mconfig = self._load_model_config(mconfig_dir)
    
        model = self._build_base_model(mconfig)

        model = self._load_model_statedict_(model, model_dir)
        print(f'running prediction for base_editor: {self.seq_processor.base_editor} | model_dir: {model_dir}')
        pred_df = self._run_prediction(model, dloader, refscore_avail)
        return pred_df

    def predict_from_dataframe(self, df, inpseq_cols, model_dir, outpseq_col=None, outcome_col=None, renormalize=True, batch_size=500):
        """
        Args:
            df: pandas dataframe
            inpseq_cols: list of column names containing `sequence id` and `input sequence` such as ['seq_id', 'Inp_seq']
            model_dir: string, path to trained model files
            outpseq_col: string, column name of output sequence (i.e. edit combination) such as ['Outp_seq']
            outcome_col: string, column name of outcome propotion such as ['proportion']
            renormalize: boolean, for renormalizing outcome proportion
            batch_size: int, number of samples to process per batch
        """
        refscore_avail = outcome_col
        proc_df = self._process_df(df, inpseq_cols, outpseq_col, outcome_col, renormalize=renormalize)
        dtensor = self._construct_datatensor(proc_df, outcome_col)
        dloader = self._construct_dloader(dtensor, batch_size)
        mconfig_dir = os.path.join(model_dir, 'config')
        mconfig = self._load_model_config(mconfig_dir)
    
        model = self._build_base_model(mconfig)

        model = self._load_model_statedict_(model, model_dir)
        print(f'running prediction for base_editor: {self.seq_processor.base_editor} | model_dir: {model_dir}')
        pred_df = self._run_prediction(model, dloader, refscore_avail)

        return pred_df

    def compute_avg_predictions(self, df):
        agg_df = df.groupby(by=['seq_id','Inp_seq', 'Outp_seq']).mean()
        agg_df.reset_index(inplace=True)
        for colname in ('run_num', 'Unnamed: 0'):
            if colname in agg_df:
                del agg_df[colname]
        return agg_df

    def visualize_haplotype(self, df, seqsids_lst, inpseq_cols, outpseq_col, outcome_col, predscore_thr=0.):
        """
        Args:
            df: pandas dataframe
            inpseq_cols: list of column names containing `sequence id` and `input sequence` such as ['seq_id', 'Inp_seq']
            outpseq_col: string, column name of output sequence (i.e. edit combination) such as ['Outp_seq']
            outcome_col: string, column name of outcome propotion such as ['proportion'] or None
            predscore_thr: float, probability threshold
        """
        seqid_col = inpseq_cols[0] # seqid column
        inpseq_col = inpseq_cols[-1] # Inp_seq column
        proc_df = self.seq_processor.process_inp_outp_df(df, seqid_col, inpseq_col, outpseq_col, outcome_col)
        conv_nucl = self.seq_processor.conversion_nucl
        seqconfig = self.seq_processor.seqconfig
        out_tb = {}
        tseqids = set(seqsids_lst).intersection(set(proc_df['seq_id'].unique()))
        for tseqid in tqdm(tseqids):
            out_tb[tseqid] = VizInpOutp_Haplotype().viz_align_haplotype(proc_df, 
                                                                 tseqid,
                                                                 outcome_col, 
                                                                 seqconfig, 
                                                                 conv_nucl, 
                                                                 predscore_thr=predscore_thr)
        return out_tb