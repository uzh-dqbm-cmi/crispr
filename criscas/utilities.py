import os
import shutil
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_curve, precision_recall_curve, accuracy_score, \
                            recall_score, precision_score, roc_auc_score, auc
from matplotlib import pyplot as plt

class CategModelScore:
    def __init__(self, best_epoch_indx, binary_f1, macro_f1, accuracy, auc):
        self.best_epoch_indx = best_epoch_indx
        self.binary_f1 = binary_f1
        self.macro_f1 = macro_f1
        self.accuracy = accuracy
        self.auc = auc

    def __repr__(self):
        desc = " best_epoch_indx:{}\n binary_f1:{}\n macro_f1:{}\n accuracy:{}\n auc:{}\n" \
               "".format(self.best_epoch_indx, self.binary_f1, self.macro_f1, self.accuracy, self.auc)
        return desc

class ContModelScore:
    def __init__(self, best_epoch_indx, spearman_corr_score):
        self.best_epoch_indx = best_epoch_indx
        self.spearman_corr_score = spearman_corr_score

    def __repr__(self):
        desc = " best_epoch_indx:{}\n spearman_corr_score:{}\n" \
               "".format(self.best_epoch_indx, self.spearman_corr_score)
        return desc

def get_performance_results(target_dir, num_runs, dsettype):
    all_perf = {}
    num_metrics = 3 # number of metrics to focus on
    perf_dict = [{} for i in range(num_metrics)]  # track auc, accuracy, macro_f1
    for run_num in range(num_runs):

        run_dir = os.path.join(target_dir,
                               'run_{}'.format(run_num))

        score_file = os.path.join(run_dir, 'score_{}.pkl'.format(dsettype))
        # print(score_file)
        if os.path.isfile(score_file):
            mscore = ReaderWriter.read_data(score_file)
            perf_dict[0]['run{}'.format(run_num)] = mscore.auc
            perf_dict[1]['run{}'.format(run_num)] = mscore.accuracy
            perf_dict[2]['run{}'.format(run_num)] = mscore.macro_f1
    perf_df = []
    for i in range(num_metrics):
        all_perf = perf_dict[i]
        all_perf_df = pd.DataFrame(all_perf)
        median = all_perf_df.median(axis=1)
        mean = all_perf_df.mean(axis=1)
        stddev = all_perf_df.std(axis=1)
        all_perf_df['mean'] = mean
        all_perf_df['median'] = median
        all_perf_df['stddev'] = stddev
        perf_df.append(all_perf_df.sort_values('mean', ascending=False))
    return perf_df


def build_performance_dfs(target_dir, num_runs, dsettype):
    auc_df = pd.DataFrame()
    accuracy_df = pd.DataFrame()
    macro_f1_df = pd.DataFrame()
    target_dir = create_directory(target_dir)
    print(target_dir)
    s_auc, s_accuracy, s_macro_f1 = get_performance_results(target_dir, num_runs, dsettype)
    return s_auc, s_accuracy, s_macro_f1


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


def perfmetric_report_categ(pred_target, ref_target, probscore, epoch, outlog, plot_roc=True):

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

    report += "-"*30 + lsep
    # (best_epoch_indx, binary_f1, macro_f1, accuracy, auc)
    modelscore = CategModelScore(epoch, s_binaryf1, macro_f1, accuracy, s_auc)
    ReaderWriter.write_log(report, outlog)
    return modelscore


def plot_precision_recall_curve(ref_target, prob_poslabel, figname, outdir):
    pr, rec, thresholds = precision_recall_curve(ref_target, prob_poslabel)
    thresholds[0] = 1
    plt.figure(figsize=(9, 6))
    plt.plot(pr, rec, 'bo', label='Precision vs Recall')
    # plt.plot(np.arange(0,len(thresholds)), thresholds, 'r-', label='thresholds')
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
    plt.plot(fpr, tpr, 'bo', label='TPR vs FPR')
    plt.plot(fpr, thresholds, 'r-', label='thresholds')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(outdir, os.path.join('roc_curve_{}'.format(figname) + ".pdf")))
    plt.close()


def plot_loss(epoch_loss_avgbatch, epoch_loss_avgsamples, wrk_dir):
    dsettypes = epoch_loss_avgbatch.keys()
    for dsettype in dsettypes:
        plt.figure(figsize=(9, 6))
        plt.plot(epoch_loss_avgbatch[dsettype], 'r', epoch_loss_avgsamples[dsettype], 'b')
        plt.xlabel("number of epochs")
        plt.ylabel("negative loglikelihood cost")
        plt.legend(['epoch batch average loss', 'epoch training samples average loss'])
        plt.savefig(os.path.join(wrk_dir, os.path.join(dsettype + ".pdf")))
        plt.close()


def delete_directory(directory):
    if(os.path.isdir(directory)):
        shutil.rmtree(directory)