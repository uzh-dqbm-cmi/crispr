import os
import shutil
import pickle
import string
import torch
import numpy as np
import scipy
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_curve, \
                            precision_recall_curve, accuracy_score, \
                            recall_score, precision_score, roc_auc_score, \
                            auc, average_precision_score
from matplotlib import pyplot as plt
import seaborn as sns

class CategModelScore:
    def __init__(self, best_epoch_indx, binary_f1, macro_f1, aupr, auc):
        self.best_epoch_indx = best_epoch_indx
        self.binary_f1 = binary_f1
        self.macro_f1 = macro_f1
        self.aupr = aupr
        self.auc = auc

    def __repr__(self):
        desc = " best_epoch_indx:{}\n binary_f1:{}\n macro_f1:{}\n aupr:{}\n auc:{}\n" \
               "".format(self.best_epoch_indx, self.binary_f1, self.macro_f1, self.aupr, self.auc)
        return desc


def get_performance_results(target_dir, num_runs, dsettype, ref_run, task_type='categ'):

    if task_type == 'categ':
        metric_names = ('auc', 'aupr', 'macro_f1')

    num_metrics = len(metric_names)
    all_perf = {}
    perf_dict = [{} for i in range(num_metrics)]

    if dsettype in {'train', 'validation'} and ref_run is None:
        prefix = 'train_val'
    else:
        prefix = 'test'

    for run_num in range(num_runs):
        
        if ref_run is not None:
            runname = 'run_{}_{}'.format(ref_run, run_num)

        else:
            runname = 'run_{}'.format(run_num)

        run_dir = os.path.join(target_dir,
                               '{}'.format(prefix),
                               runname)

        score_file = os.path.join(run_dir, 'score_{}.pkl'.format(dsettype))
        # print(score_file)
        if os.path.isfile(score_file):
            mscore = ReaderWriter.read_data(score_file)

            if task_type == 'categ':
                perf_dict[0][runname] = mscore.auc
                perf_dict[1][runname] = mscore.aupr
                perf_dict[2][runname] = mscore.macro_f1

    perf_df_lst = []
    for i in range(num_metrics):
        all_perf = perf_dict[i]
        all_perf_df = pd.DataFrame(all_perf, index=[metric_names[i]])
        median = all_perf_df.median(axis=1)
        mean = all_perf_df.mean(axis=1)
        stddev = all_perf_df.std(axis=1)
        all_perf_df['mean'] = mean
        all_perf_df['median'] = median
        all_perf_df['stddev'] = stddev
        perf_df_lst.append(all_perf_df.sort_values('mean', ascending=False))
    
    return pd.concat(perf_df_lst, axis=0)


def build_performance_dfs(target_dir, num_runs, dsettype, task_type, ref_run=None):
    target_dir = create_directory(target_dir)
    return get_performance_results(target_dir, num_runs, dsettype, ref_run, task_type=task_type)

def update_Adamoptimizer_lr_momentum_(optm, lr, momen):
    """in-place update for learning rate and momentum for Adam optimizer"""
    for pg in optm.param_groups:
        pg['lr'] = lr
        pg['betas'] = (momen, pg['betas'][-1])

def compute_lr_scheduler(l0, lmax, num_iter, annealing_percent):
    num_annealing_iter = np.floor(annealing_percent * num_iter)
    num_iter_upd = num_iter - num_annealing_iter
    
    x = [0, np.ceil(num_iter_upd/2.0), num_iter_upd, num_iter]
    y = [l0, lmax, l0, l0/100.0]
    tck = scipy.interpolate.splrep(x, y, k=1, s=0)
    
    xnew = np.arange(0, num_iter)
    lrates = scipy.interpolate.splev(xnew, tck, der=0)
    return lrates

def compute_momentum_scheduler(momen_0, momen_max, num_iter, annealing_percent):
    num_annealing_iter = np.floor(annealing_percent * num_iter)
    num_iter_upd = num_iter - num_annealing_iter
    
    x = [0, np.ceil(num_iter_upd/2.0), num_iter_upd, num_iter]
    y = [momen_max, momen_0, momen_max, momen_max]
    tck = scipy.interpolate.splrep(x, y, k=1, s=0)
    
    xnew = np.arange(0, num_iter)
    momentum_vals = scipy.interpolate.splev(xnew, tck, der=0)
    return momentum_vals

def build_classification_df(ids, true_class, pred_class, prob_scores, base_pos=None):

    prob_scores_dict = {}
    for i in range (prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    if not base_pos:
        df_dict = {
            'id': ids,
            'true_class': true_class,
            'pred_class': pred_class
        }
    else:
        df_dict = {
            'id': ids,
            'base_pos':base_pos,
            'true_class': true_class,
            'pred_class': pred_class
        }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df

def build_probscores_df(ids, prob_scores, base_pos=None):

    prob_scores_dict = {}
    for i in range (prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    if not base_pos:
        df_dict = {
            'id': ids
        }
    else:
        df_dict = {
            'id': ids,
            'base_pos':base_pos
        }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    # predictions_df.set_index('id', inplace=True)
    return predictions_df

def dump_dict_content(dsettype_content_map, dsettypes, desc, wrk_dir):
    for dsettype in dsettypes:
        path = os.path.join(wrk_dir, '{}_{}.pkl'.format(desc, dsettype))
        ReaderWriter.dump_data(dsettype_content_map[dsettype], path)

class ReaderWriter(object):
    """class for dumping, reading and logging data"""
    def __init__(self):
        pass

    @staticmethod
    def dump_data(data, file_name, mode="wb"):
        """dump data by pickling
           Args:
               data: data to be pickled
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            pickle.dump(data, f)

    @staticmethod
    def read_data(file_name, mode="rb"):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               mode: specify writing options i.e. binary or unicode
        """
        with open(file_name, mode) as f:
            data = pickle.load(f)
        return(data)

    @staticmethod
    def dump_tensor(data, file_name):
        """
        Dump a tensor using PyTorch's custom serialization. Enables re-loading the tensor on a specific gpu later.
        Args:
            data: Tensor
            file_name: file path where data will be dumped
        Returns:
        """
        torch.save(data, file_name)

    @staticmethod
    def read_tensor(file_name, device):
        """read dumped/pickled data
           Args:
               file_name: file path where data will be dumped
               device: the gpu to load the tensor on to
        """
        data = torch.load(file_name, map_location=device)
        return data

    @staticmethod
    def write_log(line, outfile, mode="a"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(outfile, mode) as f:
            f.write(line)

    @staticmethod
    def read_log(file_name, mode="r"):
        """write data to a file
           Args:
               line: string representing data to be written out
               outfile: file path where data will be written/logged
               mode: specify writing options i.e. append, write
        """
        with open(file_name, mode) as f:
            for line in f:
                yield line


def create_directory(folder_name, directory="current"):
    """create directory/folder (if it does not exist) and returns the path of the directory
       Args:
           folder_name: string representing the name of the folder to be created
       Keyword Arguments:
           directory: string representing the directory where to create the folder
                      if `current` then the folder will be created in the current directory
    """
    if directory == "current":
        path_current_dir = os.path.dirname(__file__)  # __file__ refers to utilities.py
    else:
        path_current_dir = directory
    path_new_dir = os.path.join(path_current_dir, folder_name)
    if not os.path.exists(path_new_dir):
        os.makedirs(path_new_dir)
    return(path_new_dir)


def get_device(to_gpu, index=0):
    is_cuda = torch.cuda.is_available()
    if(is_cuda and to_gpu):
        target_device = 'cuda:{}'.format(index)
    else:
        target_device = 'cpu'
    return torch.device(target_device)


def report_available_cuda_devices():
    if(torch.cuda.is_available()):
        n_gpu = torch.cuda.device_count()
        print('number of GPUs available:', n_gpu)
        for i in range(n_gpu):
            print("cuda:{}, name:{}".format(i, torch.cuda.get_device_name(i)))
            device = torch.device('cuda', i)
            get_cuda_device_stats(device)
            print()
    else:
        print("no GPU devices available!!")

def get_cuda_device_stats(device):
    print('total memory available:', torch.cuda.get_device_properties(device).total_memory/(1024**3), 'GB')
    print('total memory allocated on device:', torch.cuda.memory_allocated(device)/(1024**3), 'GB')
    print('max memory allocated on device:', torch.cuda.max_memory_allocated(device)/(1024**3), 'GB')
    print('total memory cached on device:', torch.cuda.memory_cached(device)/(1024**3), 'GB')
    print('max memory cached  on device:', torch.cuda.max_memory_cached(device)/(1024**3), 'GB')

# def compute_spearman_corr(pred_score, ref_score):
#     return scipy.stats.kendalltau(pred_score, ref_score)
#     # return scipy.stats.spearmanr(pred_score, ref_score)

def restrict_grad_(mparams, mode, limit):
    """clamp/clip a gradient in-place
    """
    if(mode == 'clip_norm'):
        __, maxl = limit
        torch.nn.utils.clip_grad_norm_(mparams, maxl, norm_type=2) # l2 norm clipping
    elif(mode == 'clamp'): # case of clamping
        minl, maxl = limit
        for param in mparams:
            if param.grad is not None:
                param.grad.data.clamp_(minl, maxl)
def check_na(df):
    assert df.isna().any().sum() == 0

def perfmetric_report_categ(pred_target, ref_target, probscore, epoch, outlog):

    # print(ref_target.shape)
    # print(pred_target.shape)
    #
    # print("ref_target \n", ref_target)
    # print("pred_target \n", pred_target)
    
    lsep = "\n"
    report = "Epoch: {}".format(epoch) + lsep
    report += "Classification report on all events:" + lsep
    report += str(classification_report(ref_target, pred_target)) + lsep

    report += "binary f1:" + lsep
    s_binaryf1 = f1_score(ref_target, pred_target, average='binary')
    report += str(s_binaryf1) + lsep

    report += "macro f1:" + lsep
    macro_f1 = f1_score(ref_target, pred_target, average='macro')
    report += str(macro_f1) + lsep

    report += "accuracy:" + lsep
    accuracy = accuracy_score(ref_target, pred_target)
    report += str(accuracy) + lsep

    auc_multi_class = 'raise'
    auc_lst = []
    for opt in ('macro', 'weighted'):
        s_auc = roc_auc_score(ref_target, probscore, average=opt, multi_class=auc_multi_class)
        auc_lst.append(s_auc)
        report += f"{opt} AUC:\n" + str(s_auc) + lsep
    s_auc = auc_lst[0] # keep macro

    pr, rec, __ = precision_recall_curve(ref_target, probscore)
    s_aupr = auc(rec, pr)
    report += "AUPR:\n" + str(s_aupr) + lsep
    s_avgprecision = average_precision_score(ref_target, probscore)
    report += "AP:\n"+ str(s_avgprecision) + lsep
    report += "-"*30 + lsep

    # (best_epoch_indx, binary_f1, macro_f1, aupr, auc)
    modelscore = CategModelScore(epoch, s_binaryf1, macro_f1, s_aupr, s_auc)
    ReaderWriter.write_log(report, outlog)
    return modelscore


def plot_precision_recall_curve(ref_target, prob_poslabel, figname, outdir):
    pr, rec, thresholds = precision_recall_curve(ref_target, prob_poslabel)
    avg_precision = average_precision_score(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(rec, pr, 'b+', label=f'Average Precision (AP):{avg_precision:.2}')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs. recall curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('precisionrecall_curve_{}'.format(figname) + ".pdf")))
    plt.close()


def plot_roc_curve(ref_target, prob_poslabel, figname, outdir):
    fpr, tpr, thresholds = roc_curve(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, 'b+', label='TPR vs FPR')
    plt.plot(fpr, thresholds, 'r-', label='thresholds')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('roc_curve_{}'.format(figname) + ".pdf")))
    plt.close()


def plot_loss(epoch_loss_avgbatch, wrk_dir):
    dsettypes = epoch_loss_avgbatch.keys()
    for dsettype in dsettypes:
        plt.figure(figsize=(9, 6))
        plt.plot(epoch_loss_avgbatch[dsettype], 'r')
        plt.xlabel("number of epochs")
        plt.ylabel("negative loglikelihood cost")
        plt.legend(['epoch batch average loss'])
        plt.savefig(os.path.join(wrk_dir, os.path.join(dsettype + ".pdf")))
        plt.close()


def plot_xy(x, y, xlabel, ylabel, legend, fname, wrk_dir):
    plt.figure(figsize=(9, 6))
    plt.plot(x, y, 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend([legend])
    plt.savefig(os.path.join(wrk_dir, os.path.join(fname + ".pdf")))
    plt.close()

def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)

def find_youdenj_threshold(ref_target, prob_poslabel, fig_dir=None):
    fpr, tpr, thresholds = roc_curve(ref_target, prob_poslabel)
    s_auc = roc_auc_score(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(fpr, tpr, 'b+', label=f'TPR vs FPR => AUC:{s_auc:.2}')
#     plt.plot(fpr, thresholds, 'r-', label='thresholds')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    # youden index is max(sensitivity + specificity - 1)
    # max(tpr + (1-fpr) - 1)
    # max(tpr - fpr)
    youden_indx = np.argmax(tpr - fpr) # the index where the difference between tpr and fpr is max
    optimal_threshold = thresholds[youden_indx]
    plt.plot(fpr[youden_indx], tpr[youden_indx], marker='o', markersize=3, color="red", label=f'optimal probability threshold:{optimal_threshold:.2}')
    plt.legend(loc='best')
    if fig_dir:
        plt.savefig(f'{fig_dir}.pdf')
        plt.close()
    return fpr, tpr, thresholds, optimal_threshold

def analyze_precision_recall_curve(ref_target, prob_poslabel, fig_dir=None):
    pr, rec, thresholds = precision_recall_curve(ref_target, prob_poslabel)
    avg_precision = average_precision_score(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(rec, pr, 'b+', label=f'Precision vs Recall => Average Precision (AP):{avg_precision:.2}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. recall curve')
    indx = np.argmax(pr + rec)
    print('indx', indx)
    optimal_threshold = thresholds[indx]
    plt.plot(rec[indx], pr[indx], marker='o', markersize=3, color="red", label=f'optimal probability threshold:{optimal_threshold:.2}')
    plt.legend(loc='best')
    if fig_dir:
        plt.savefig(f'{fig_dir}.pdf')
        plt.close()
    return pr, rec, thresholds, optimal_threshold

def analyze_performance(pred_target, ref_target, probscore):
    report=''
    lsep = "\n"
    report += "Classification report on all events:" + lsep
    report += str(classification_report(ref_target, pred_target)) + lsep

    
    report += "binary f1:" + lsep
    s_binaryf1 = f1_score(ref_target, pred_target)
    report += str(s_binaryf1) + lsep

    report += "macro f1:" + lsep
    macro_f1 = f1_score(ref_target, pred_target, average='macro')
    report += str(macro_f1) + lsep

    report += "accuracy:" + lsep
    accuracy = accuracy_score(ref_target, pred_target)
    report += str(accuracy) + lsep
        
    s_auc = roc_auc_score(ref_target, probscore)
    report += "AUC:\n" + str(s_auc) + lsep
    
    pr, rec, __ = precision_recall_curve(ref_target, probscore)
    s_aupr = auc(rec, pr)
    report += "AUPR:\n" + str(s_aupr) + lsep
    
    s_avgprecision = average_precision_score(ref_target, probscore)
    report += "AP:\n" + str(s_avgprecision) + lsep

    report += "-"*30 + lsep
    return report, s_auc, s_aupr, macro_f1, accuracy


