import os
import itertools
from .utilities import get_device, create_directory, ReaderWriter, perfmetric_report_categ, plot_loss
from .model import Categ_CrisCasTransformer
from .dataset import construct_load_dataloaders
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.multiprocessing as mp



class TrfHyperparamConfig:
    def __init__(self, embed_dim, num_attn_heads, num_transformer_units, 
                p_dropout, nonlin_func, mlp_embed_factor, 
                l2_reg, batch_size, num_epochs):
        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.num_transformer_units = num_transformer_units
        self.p_dropout = p_dropout
        self.nonlin_func = nonlin_func
        self.mlp_embed_factor = mlp_embed_factor
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs


    def __repr__(self):
        desc = " embed_dim:{}\n num_attn_heads:{}\n num_transformer_units:{}\n p_dropout:{} \n " \
               "nonlin_func:{} \n mlp_embed_factor:{} \n " \
               "l2_reg:{} \n batch_size:{} \n num_epochs: {}".format(self.embed_dim,
                                                                     self.num_attn_heads,
                                                                     self.num_transformer_units,
                                                                     self.p_dropout, 
                                                                     self.nonlin_func,
                                                                     self.mlp_embed_factor,
                                                                     self.l2_reg, 
                                                                     self.batch_size,
                                                                     self.num_epochs)
        return desc

def generate_models_config(hyperparam_config, experiment_desc, model_name, run_num, fdtype):


    # currently generic_config is shared across all models
    # leaving it as placeholder such that custom generic configs could be passed :)



    generic_config = {'fdtype':fdtype, 'to_gpu':True}
    dataloader_config = {'batch_size': hyperparam_config.batch_size,
                         'num_workers': 0}
    config = {'dataloader_config': dataloader_config,
              'model_config': hyperparam_config,
              'generic_config': generic_config
             }

    options = {'experiment_desc': experiment_desc,
               'run_num': run_num,
               'model_name': model_name,
               'num_epochs': hyperparam_config.num_epochs,
               'weight_decay': hyperparam_config.l2_reg}

    return config, options

def build_custom_config_map(experiment_desc, model_name):
    if(model_name == 'Transformer'):
        hyperparam_config = TrfHyperparamConfig(32, 8, 12, 0.3, nn.ReLU(), 2, 0, 200, 20)
    run_num = -1 
    fdtype = torch.float32
    mconfig, options = generate_models_config(hyperparam_config, experiment_desc, model_name, run_num, fdtype)
    return mconfig, options

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)

def process_multilayer_multihead_attn(attn_dict, seqs_id):
    attn_dict_perseq = {}
    for l in attn_dict:
        for h in attn_dict[l]:
            tmp = attn_dict[l][h].detach().cpu()
            for count, seq_id in enumerate(seqs_id):
                if(seq_id not in attn_dict_perseq):
                    attn_dict_perseq[seq_id] = {} 
                if(l in attn_dict_perseq[seq_id]):
                    attn_dict_perseq[seq_id][l].update({h:tmp[count]})
                else:
                    attn_dict_perseq[seq_id][l] = {h:tmp[count]}
    return attn_dict_perseq

def run_categTrf(data_partition, dsettypes, config, options, wrk_dir, state_dict_dir=None, to_gpu=True, gpu_index=0):
    pid = "{}".format(os.getpid())  # process id description
    # get data loader config
    dataloader_config = config['dataloader_config']
    cld = construct_load_dataloaders(data_partition, dsettypes, 'categ', dataloader_config, wrk_dir)
    # dictionaries by dsettypes
    data_loaders, epoch_loss_avgbatch, epoch_loss_avgsamples, score_dict, class_weights, flog_out = cld
    # print(flog_out)
    # print(class_weights)
    device = get_device(to_gpu, gpu_index)  # gpu device
    generic_config = config['generic_config']
    fdtype = generic_config['fdtype']

    if('train' in class_weights):
        class_weights = class_weights['train'].type(fdtype).to(device)  # update class weights to fdtype tensor
    else:
        class_weights = torch.tensor([1, 1]).type(fdtype).to(device)  # weighting all casess equally

    print("class weights", class_weights)
    loss_func = torch.nn.NLLLoss(weight=class_weights, reduction='mean')  # negative log likelihood loss

    num_epochs = options.get('num_epochs', 50)
    run_num = options.get('run_num')

    # parse config dict
    model_config = config['model_config']
    model_name = options['model_name']

    if(model_name == 'Transformer'):
        criscas_categ_model = Categ_CrisCasTransformer(input_size=model_config.embed_dim, 
                                                        num_nucleotides=4, 
                                                        seq_length=20, 
                                                        num_attn_heads=model_config.num_attn_heads, 
                                                        mlp_embed_factor=model_config.mlp_embed_factor, 
                                                        nonlin_func=model_config.nonlin_func, 
                                                        pdropout=model_config.p_dropout, 
                                                        num_transformer_units=model_config.num_transformer_units,
                                                        pooling_mode='attn',
                                                        num_classes=2)
    

    # define optimizer and group parameters
    models_param = list(criscas_categ_model.parameters())
    models = [(criscas_categ_model, model_name)]

    if(state_dict_dir):  # load state dictionary of saved models
        for m, m_name in models:
            m.load_state_dict(torch.load(os.path.join(state_dict_dir, '{}.pkl'.format(m_name)), map_location=device))

    # update models fdtype and move to device
    for m, m_name in models:
        m.type(fdtype).to(device)

    if('train' in data_loaders):
        weight_decay = options.get('weight_decay', 1e-3)
        optimizer = torch.optim.RMSprop(models_param, weight_decay=weight_decay, lr=1e-4)
        # see paper Cyclical Learning rates for Training Neural Networks for parameters' choice
        # `https://arxive.org/pdf/1506.01186.pdf`
        # pytorch version >1.1, scheduler should be called after optimizer
        # for cyclical lr scheduler, it should be called after each batch update
        num_iter = len(data_loaders['train'])  # num_train_samples/batch_size
        c_step_size = int(np.ceil(5*num_iter))  # this should be 2-10 times num_iter
        base_lr = 3e-4
        max_lr = 5*base_lr  # 3-5 times base_lr
        cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=c_step_size,
                                                          mode='triangular', cycle_momentum=True)

    if ('validation' in data_loaders):
        m_state_dict_dir = create_directory(os.path.join(wrk_dir, 'model_statedict'))

    if(num_epochs > 1):
        fig_dir = create_directory(os.path.join(wrk_dir, 'figures'))

    # dump config dictionaries on disk
    config_dir = create_directory(os.path.join(wrk_dir, 'config'))
    ReaderWriter.dump_data(config, os.path.join(config_dir, 'mconfig.pkl'))
    ReaderWriter.dump_data(options, os.path.join(config_dir, 'exp_options.pkl'))
    # store attention weights for validation and test set
    seqid_fattnw_map = {dsettype: {} for dsettype in data_loaders if dsettype in {'test'}}
    seqid_mlhattnw_map = {dsettype: {} for dsettype in data_loaders if dsettype in {'test'}}

    for epoch in range(num_epochs):
        # print("-"*35)
        for dsettype in dsettypes:
            print("device: {} | experiment_desc: {} | run_num: {} | epoch: {} | dsettype: {} | pid: {}"
                  "".format(device, options.get('experiment_desc'), run_num, epoch, dsettype, pid))
            pred_class = []
            ref_class = []
            prob_scores = []
            seqs_ids_lst = []
            data_loader = data_loaders[dsettype]
            # total_num_samples = len(data_loader.dataset)
            epoch_loss = 0.
            epoch_loss_deavrg = 0.

            if(dsettype == 'train'):  # should be only for train
                for m, m_name in models:
                    m.train()
            else:
                for m, m_name in models:
                    m.eval()

            for i_batch, samples_batch in enumerate(data_loader):
                print('batch num:', i_batch)

                # zero model grad
                if(dsettype == 'train'):
                    optimizer.zero_grad()

                X_batch, __ , y_batch, b_seqs_indx, b_seqs_id = samples_batch

                X_batch = X_batch.to(device)
                y_batch = y_batch.type(torch.int64).to(device)

                with torch.set_grad_enabled(dsettype == 'train'):
                    num_samples_perbatch = X_batch.size(0)
                    logsoftmax_scores, fattn_w_scores, attn_mlayer_mhead_dict = criscas_categ_model(X_batch)

                    if(dsettype in seqid_fattnw_map):
                        seqid_fattnw_map[dsettype].update({seqid:fattn_w_scores[c].detach().cpu() for c, seqid in enumerate(b_seqs_id)})
                        # seqid_mlhattnw_map[dsettype].update(process_multilayer_multihead_attn(attn_mlayer_mhead_dict, b_seqs_id))

                    __, y_pred_clss = torch.max(logsoftmax_scores, 1)

                    # print('y_pred_clss.shape', y_pred_clss.shape)

                    pred_class.extend(y_pred_clss.view(-1).tolist())
                    ref_class.extend(y_batch.view(-1).tolist())
                    prob_scores.append((torch.exp(logsoftmax_scores.detach().cpu())).numpy())
                    seqs_ids_lst.extend(list(b_seqs_id))

                    loss = loss_func(logsoftmax_scores, y_batch)
                    if(dsettype == 'train'):
                        # print("computing loss")
                        # backward step (i.e. compute gradients)
                        loss.backward()
                        # optimzer step -- update weights
                        optimizer.step()
                        # after each batch step the scheduler
                        cyc_scheduler.step()
                    epoch_loss += loss.item()
                    # deaverage the loss to deal with last batch with unequal size
                    epoch_loss_deavrg += loss.item() * num_samples_perbatch

                    # torch.cuda.ipc_collect()
                    # torch.cuda.empty_cache()
            # end of epoch
            # print("+"*35)
            epoch_loss_avgbatch[dsettype].append(epoch_loss/len(data_loader))
            epoch_loss_avgsamples[dsettype].append(epoch_loss_deavrg/len(data_loader.dataset))
            prob_scores_arr = np.concatenate(prob_scores, axis=0)
            modelscore = perfmetric_report_categ(pred_class, ref_class, prob_scores_arr[:, 1], epoch, flog_out[dsettype])
            perf = modelscore.auc
            if(perf > score_dict[dsettype].auc):
                score_dict[dsettype] = modelscore
                if(dsettype == 'validation'):
                    for m, m_name in models:
                        torch.save(m.state_dict(), os.path.join(m_state_dict_dir, '{}.pkl'.format(m_name)))
                    # dump attention weights for the validation data for the best peforming model
                    # dump_dict_content(seqid_fattnw_map, ['validation'], 'seqid_fattnw_map', wrk_dir)
                    # dump_dict_content(seqid_mlhattnw_map, ['validation'], 'seqid_mlhattnw_map', wrk_dir)
                elif(dsettype == 'test'):
                    # dump attention weights for the validation data
                    dump_dict_content(seqid_fattnw_map, ['test'], 'seqid_fattnw_map', wrk_dir)
                    # dump_dict_content(seqid_mlhattnw_map, ['test'], 'seqid_mlhattnw_map', wrk_dir)
                    # save predictions for test
                if dsettype in {'test', 'validation'}:
                    predictions_df = build_predictions_df(seqs_ids_lst, ref_class, pred_class, prob_scores_arr)
                    predictions_path = os.path.join(wrk_dir, f'predictions_{dsettype}.csv')
                    predictions_df.to_csv(predictions_path)

    if(num_epochs > 1):
        plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, fig_dir)

    # dump_scores
    dump_dict_content(score_dict, list(score_dict.keys()), 'score', wrk_dir)


def build_predictions_df(ids, true_class, pred_class, prob_scores):
    df_dict = {
        'id': ids,
        'true_class': true_class,
        'pred_class': pred_class,
        'prob_score_class1': prob_scores[:,1],
        'prob_scores_class0': prob_scores[:,0]
    }
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df


def generate_hyperparam_space(model_name):
    if(model_name == 'Transformer'):
        # TODO: add the possible options for transformer model
        embed_dim = [16,32,64,128]
        num_attn_heads = [4,6,8,12]
        num_transformer_units = [2,4,6,8]
        p_dropout = [0.1, 0.3, 0.5]
        nonlin_func = [nn.ReLU, nn.ELU]
        mlp_embed_factor = [2]
        l2_reg = [1e-4, 1e-3, 0.]
        batch_size = [200, 400, 600]
        num_epochs = [30]
        opt_lst = [embed_dim, num_attn_heads, 
                   num_transformer_units, p_dropout,
                   nonlin_func, mlp_embed_factor,
                   l2_reg, batch_size, num_epochs]

    hyperparam_space = list(itertools.product(*opt_lst))

    return hyperparam_space

def compute_numtrials(prob_interval_truemax, prob_estim):
    """ computes number of trials needed for random hyperparameter search
        see `algorithms for hyperparameter optimization paper
        <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`__
        Args:
            prob_interval_truemax: float, probability interval of the true optimal hyperparam,
                i.e. within 5% expressed as .05
            prob_estim: float, probability/confidence level, i.e. 95% expressed as .95
    """
    n = np.log(1-prob_estim)/np.log(1-prob_interval_truemax)
    return(int(np.ceil(n))+1)


def get_hyperparam_options(prob_interval_truemax, prob_estim, model_name, random_seed=42):
    np.random.seed(random_seed)
    num_trials = compute_numtrials(prob_interval_truemax, prob_estim)
    hyperparam_space = generate_hyperparam_space(model_name)
    if(num_trials > len(hyperparam_space)):
        num_trials = len(hyperparam_space)
    indxs = np.random.choice(len(hyperparam_space), size=num_trials, replace=False)
    if(model_name == 'Transformer'):
        hyperconfig_class = TrfHyperparamConfig
    # encoder_dim, num_layers, encoder_approach, attn_method, p_dropout, l2_reg, batch_size, num_epochs
    return [hyperconfig_class(*hyperparam_space[indx]) for indx in indxs]



def get_saved_config(config_dir):
    options = ReaderWriter.read_data(os.path.join(config_dir, 'exp_options.pkl'))
    mconfig = ReaderWriter.read_data(os.path.join(config_dir, 'mconfig.pkl'))
    return mconfig, options


def get_index_argmax(score_matrix, target_indx):
    argmax_indx = np.argmax(score_matrix, axis=0)[target_indx]
    return argmax_indx


def train_val_run(datatensor_partitions, config_map, train_val_dir, run_gpu_map, num_epochs=20):
    dsettypes = ['train', 'validation']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    for run_num in datatensor_partitions:
        # update options run num to the current run
        options['run_num'] = run_num
        data_partition = datatensor_partitions[run_num]
        # tr_val_dir = create_directory(train_val_dir)
        path = os.path.join(train_val_dir, 'train_val', 'run_{}'.format(run_num))
        wrk_dir = create_directory(path)
        # print(wrk_dir)
        run_categTrf(data_partition, dsettypes, mconfig, options, wrk_dir,
                     state_dict_dir=None, to_gpu=True, gpu_index=run_gpu_map[run_num])


def test_run(datatensor_partitions, config_map, train_val_dir, test_dir, run_gpu_map, num_epochs=1):
    dsettypes = ['test']
    mconfig, options = config_map
    options['num_epochs'] = num_epochs  # override number of epochs using user specified value
    for run_num in datatensor_partitions:
        # update options fold num to the current fold
        options['run_num'] = run_num
        data_partition = datatensor_partitions[run_num]
        train_dir = create_directory(os.path.join(train_val_dir, 'train_val', 'run_{}'.format(run_num)))
        if os.path.exists(train_dir):
            # load state_dict pth
            state_dict_pth = os.path.join(train_dir, 'model_statedict')
            path = os.path.join(test_dir, 'test', 'run_{}'.format(run_num))
            test_wrk_dir = create_directory(path)
            run_categTrf(data_partition, dsettypes, mconfig, options, test_wrk_dir,
                         state_dict_dir=state_dict_pth, to_gpu=True, gpu_index=run_gpu_map[run_num])
        else:
            print('WARNING: train dir not found: {}'.format(path))