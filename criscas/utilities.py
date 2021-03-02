import os
import shutil
import pickle
import torch
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


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


def check_na(df):
    assert df.isna().any().sum() == 0

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




