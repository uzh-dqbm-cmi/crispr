import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import seaborn as sns
from .model import Categ_CrisCasTransformer
from .data_preprocess import process_perbase_df
from .dataset import create_datatensor
from .utilities import ReaderWriter, build_probscores_df, check_na
from .attnetion_analysis import filter_attn_rows


class BEDICT_CriscasModel:
    def __init__(self, base_editor, device):
        self.base_editor = base_editor
        self.device = device

    def _process_df(self, df, target_base):
        print('--- processing input data frame ---')
        df = process_perbase_df(df, target_base)
        return df

    def _construct_datatensor(self, df, refscore_available=False):
        dtensor = create_datatensor(df, per_base=True, refscore_available=refscore_available)
        return dtensor

    def _construct_dloader(self, dtensor, batch_size):
        print('--- creating datatensor ---')
        dloader = DataLoader(dtensor,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             sampler=None)
        return dloader

    def _build_base_model(self):
        print('--- building model ---')
        embed_dim = 64
        num_attn_heads = 8
        num_trf_units = 2
        pdropout = 0.1
        activ_func = nn.ReLU()
        mlp_embed_factor = 2
        num_classes = 2
        model = Categ_CrisCasTransformer(input_size=embed_dim,
                                         num_nucleotides=4,
                                         seq_length=20,
                                         num_attn_heads=num_attn_heads,
                                         mlp_embed_factor=mlp_embed_factor,
                                         nonlin_func=activ_func,
                                         pdropout=pdropout,
                                         num_transformer_units=num_trf_units,
                                         pooling_mode='attn',
                                         num_classes=num_classes,
                                         per_base=True)
        return model

    def _load_model_statedict_(self, model, run_num):
        print('--- loading trained model ---')
        base_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        run_pth = os.path.join(base_dir, 'trained_models', self.base_editor, 'train_val', f'run_{run_num}')
        print(run_pth)
        device = self.device

        model_name = 'Transformer'
        models = [(model, model_name)]

        # load state_dict pth
        state_dict_dir = os.path.join(run_pth, 'model_statedict')
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

        # update model's fdtype and move to device
        for m, m_name in models:
            m.type(torch.float32).to(device)
            m.eval()
        return model

    def _run_prediction(self, model, dloader):

        device = self.device
        prob_scores = []
        seqs_ids_lst = []
        base_pos_lst = []
        seqid_fattnw_map = {}

        for i_batch, samples_batch in enumerate(dloader):
            X_batch, mask, b_seqs_indx, b_seqs_id = samples_batch

            X_batch = X_batch.to(device)
            mask = mask.to(device)

            with torch.set_grad_enabled(False):
                logsoftmax_scores, fattn_w_scores, attn_mlayer_mhead_dict = model(X_batch)

                seqid_fattnw_map.update({seqid: fattn_w_scores[c].detach().cpu() for c, seqid in enumerate(b_seqs_id)})
                # seqid_mlhattnw_map[dsettype].update(process_multilayer_multihead_attn(attn_mlayer_mhead_dict, b_seqs_id))

                # __, y_pred_clss = torch.max(logsoftmax_scores, -1)

                # print('y_pred_clss.shape', y_pred_clss.shape)
                # use mask to retrieve relevant entries
                tindx = torch.where(mask.type(torch.bool))

                # pred_class.extend(y_pred_clss[tindx].view(-1).tolist())
                prob_scores.append((torch.exp(logsoftmax_scores[tindx].detach().cpu())).numpy())

                seqs_ids_lst.extend([b_seqs_id[i] for i in tindx[0].tolist()])
                base_pos_lst.extend(tindx[1].tolist())  # positions of target base

                # torch.cuda.ipc_collect()
                # torch.cuda.empty_cache()
        # end of epoch
        # print("+"*35)
        prob_scores_arr = np.concatenate(prob_scores, axis=0)
        predictions_df = build_probscores_df(seqs_ids_lst, prob_scores_arr, base_pos_lst)

        return seqid_fattnw_map, predictions_df

        # dump attention weights
        # if wrk_dir:
        #     ReaderWriter.dump_data(seqid_fattnw_map, os.path.join(wrk_dir, 'seqid_fattnw_map.pkl'))
        #     predictions_df.to_csv(os.path.join(wrk_dir, f'predictions.csv'))

    def _join_prediction_w_attn(self, pred_df, seqid_fattnw_map):

        attn_w_lst = []
        for seq_id in seqid_fattnw_map:
            bpos = pred_df.loc[pred_df['id'] == seq_id, 'base_pos'].values
            attn_w = seqid_fattnw_map[seq_id][bpos].numpy()
            attn_w_lst.append(attn_w)

        attn_w_df = pd.DataFrame(np.concatenate(attn_w_lst, axis=0))
        attn_w_df.columns = [f'Attn{i}' for i in range(20)]

        pred_w_attn_df = pd.concat([pred_df, attn_w_df], axis=1)
        check_na(pred_w_attn_df)

        return pred_w_attn_df

    def predict_from_dataframe(self, df, batch_size=500):
        if self.base_editor in {'ABEmax', 'ABE8e'}:
            target_base = 'A'
        elif self.base_editor in {'BE4', 'AID'}:
            target_base = 'C'

        proc_df = self._process_df(df, target_base)
        dtensor = self._construct_datatensor(proc_df)
        dloader = self._construct_dloader(dtensor, batch_size)

        pred_w_attn_runs_df = pd.DataFrame()

        model = self._build_base_model()

        for run_num in range(5):
            self._load_model_statedict_(model, run_num)
            print(f'running prediction for base_editor: {self.base_editor} | run_num: {run_num}')
            seqid_fattnw_map, pred_df = self._run_prediction(model, dloader)
            pred_w_attn_df = self._join_prediction_w_attn(pred_df, seqid_fattnw_map)
            pred_w_attn_df['run_id'] = f'run_{run_num}'
            pred_w_attn_df['model_name'] = self.base_editor
            pred_w_attn_runs_df = pd.concat([pred_w_attn_runs_df, pred_w_attn_df])
        # reset index
        pred_w_attn_runs_df.reset_index(inplace=True, drop=True)

        return pred_w_attn_runs_df, proc_df

    def _filter_attn_rows(self, pred_w_attn_df, base_w):
        filtered_df = filter_attn_rows(pred_w_attn_df, base_w)
        return filtered_df

    def _select_prediction_run(self, gr_df, option):
        gr_df['diff'] = (gr_df['prob_score_class1'] - gr_df['prob_score_class0']).abs()
        if option == 'median':
            choice = gr_df['diff'].median()
        elif option == 'max':
            choice = gr_df['diff'].max()
        elif option == 'min':
            choice = gr_df['diff'].min()
        cond = gr_df['diff'] == choice
        t_indx = np.where(cond)[0][0]

        return gr_df.iloc[t_indx]

    def select_prediction(self, pred_w_attn_runs_df, option):
        assert option in {'mean', 'median', 'max', 'min'}, "selection option should be in {mean, median, min, max}!"
        if option == 'mean':
            pred_w_attn_df = pred_w_attn_runs_df.groupby(['id', 'base_pos', 'model_name']).mean().reset_index()
        else:
            pred_w_attn_df = pred_w_attn_runs_df.groupby(['id', 'base_pos', 'model_name']).apply(
                self._select_prediction_run, option).reset_index(drop=True)
        return pred_w_attn_df

    def _highlight_attn_scores(self, df, pred_option, model_name, cmap='YlOrRd', fig_dir=None):
        # we index these axes from 0 subscript
        fig, ax = plt.subplots(figsize=(11, 3),
                               nrows=1,
                               constrained_layout=True)
        seq_id = df['id']
        attn_vars = [f'Attn{i}' for i in range(20)]
        letter_vars = [f'L{i}' for i in range(1, 21)]
        prob = df['prob_score_class1']
        base_pos = df['base_pos'] + 1
        attn_scores = df[[f'Attn{i}' for i in range(20)]].values.astype(np.float).reshape(1, -1)
        max_score = df[[f'Attn{i}' for i in range(20)]].max()
        base_letters = df[letter_vars].values.reshape(1, -1).tolist()
        cbar_kws = {'label': 'Attention score', 'orientation': 'horizontal'}
        #     cmap='YlOrRd'
        g = sns.heatmap(attn_scores, cmap=cmap, annot=base_letters, fmt="", linewidths=.5, cbar_kws=cbar_kws, ax=ax)
        ax.set_xticklabels(list(range(1, 21)))
        ax.set(xlabel='Base position', ylabel='')
        ax.set_yticklabels([''])
        ax.text(20.4, 0.2, 'base position = {}'.format(base_pos),
                bbox={'facecolor': 'orange', 'alpha': 0.2, 'edgecolor': 'none', 'pad': 9},
                fontsize=12)
        ax.text(20.4, 0.65, r'Edit $probability=$' + '{:.2f}'.format(prob),
                bbox={'facecolor': 'magenta', 'alpha': 0.2, 'edgecolor': 'none', 'pad': 8},
                fontsize=12)
        ax.text(0.2, -0.2, r'$seqid=${}'.format(seq_id),
                bbox={'facecolor': 'grey', 'alpha': 0.2, 'edgecolor': 'none', 'pad': 10},
                fontsize=12)
        ax.tick_params(left=False, labelbottom=True)
        if fig_dir:
            fig.savefig(
                os.path.join(fig_dir, f'{model_name}_seqattn_{seq_id}_basepos_{base_pos}_predoption_{pred_option}.pdf'),
                bbox_inches='tight')
            plt.close()
        return ax

    def highlight_attn_per_seq(self, pred_w_attn_runs_df, proc_df,
                               seqid_pos_map=None,
                               pred_option='mean',
                               apply_attnscore_filter=False,
                               fig_dir=None):

        letter_vars = [f'L{i}' for i in range(1, 21)]
        if pred_option in {'mean', 'median', 'min', 'max'}:
            pred_w_attn_df = self.select_prediction(pred_w_attn_runs_df, pred_option)
        else:
            pred_w_attn_df = pred_w_attn_runs_df

        if apply_attnscore_filter:
            base_w = 1.0 / 20
            pred_w_attn_df = self._filter_attn_rows(pred_w_attn_df, base_w)
            pred_w_attn_df.reset_index(inplace=True, drop=True)
            check_na(pred_w_attn_df)

        res_df = pd.merge(left=pred_w_attn_df,
                          right=proc_df[['ID'] + letter_vars],
                          how='left',
                          left_on=['id'],
                          right_on=['ID'])
        check_na(res_df)

        if seqid_pos_map:
            for seqid, t_pos in seqid_pos_map.items():
                print('seq_id:', seqid)
                if t_pos:  # if list of positions are supplied
                    # subtract 1 since base position indexing is from 0-19
                    t_pos_upd = [bpos - 1 for bpos in t_pos]
                    cond = (res_df['id'] == seqid) & (res_df['base_pos'].isin(t_pos_upd))
                    t_df = res_df.loc[cond].copy()
                    for rname, row in t_df.iterrows():
                        print(f"highlighting seqid:{row['id']}, pos:{row['base_pos']}")
                        self._highlight_attn_scores(row, pred_option, self.base_editor, cmap='YlOrRd', fig_dir=fig_dir)
                else:
                    cond = res_df['id'] == seqid
                    t_df = res_df.loc[cond]
                    for rname, row in t_df.iterrows():
                        print(f"highlighting seqid:{row['id']}, pos:{row['base_pos'] + 1}")
                        self._highlight_attn_scores(row, pred_option, self.base_editor, cmap='YlOrRd', fig_dir=fig_dir)
        else:
            for gr_name, gr_df in res_df.groupby(['id', 'base_pos']):
                print(f"highlighting seqid: {gr_name[0]}, pos: {gr_name[1] + 1}")
                for rname, row in gr_df.iterrows():
                    self._highlight_attn_scores(row, pred_option, self.base_editor, cmap='YlOrRd', fig_dir=fig_dir)