__author__ = 'XF'
__date__ = '2023/07/12'


'''
Frequently used tools.
'''

import os
import numpy as np
import torch
import random
import json
import time
from builtins import print as b_print
from os import path as osp
import pickle
from scipy.stats import chi2
from math import sqrt

ROOT_DIR = osp.dirname(osp.abspath(__file__))

def distribution_sampling(sample_dim, size=10000, p=0.9):
        
        
        Sampler = torch.randn
        r = sqrt(chi2.ppf(p, sample_dim))

        targets = None
        while size > 0:

            sample = Sampler(sample_dim)
            sample_norm = torch.sqrt(torch.sum(sample ** 2))

            if sample_norm < r:
                if targets is None:
                    targets = sample.unsqueeze(0)
                else:
                    targets = torch.cat((targets, sample.unsqueeze(0)))
                size -= 1
       
        
        return targets


def generate_filename(suffix, *args, sep='_', timestamp=False):

    '''

    :param suffix: suffix of file
    :param sep: separator, default '_'
    :param timestamp: add timestamp for uniqueness
    :param args:
    :return:
    '''

    filename = sep.join(args).replace(' ', '_')
    if timestamp:
        filename += time.strftime('_%Y%m%d%H%M%S')
    if suffix[0] == '.':
        filename += suffix
    else:
        filename += ('.' + suffix)

    return filename


def split_train_test(normal_data, abnormal_data, n_train_data=10000, imbalanced=1):

    assert len(normal_data) > n_train_data
    if n_train_data > 0:
        idx_train_data = list(range(0, n_train_data))
        idx_remains_normal_data = list(range(n_train_data, len(normal_data)))

    else:
        idx_remains_normal_data = list(range(0, len(abnormal_data)))
        idx_train_data = list(range(len(abnormal_data), len(normal_data)))

    train_data = normal_data[idx_train_data]
    train_lab = np.ones(len(train_data))

    test_abnormal_data = abnormal_data
    num_test_normal_data = len(test_abnormal_data) * imbalanced
    assert len(idx_remains_normal_data) >= num_test_normal_data
    test_normal_data = normal_data[idx_remains_normal_data[:num_test_normal_data]]
    test_data = np.concatenate((test_normal_data, test_abnormal_data))
    test_lab = np.concatenate((np.zeros(len(test_normal_data)), np.ones(len(test_abnormal_data))))
    
    return train_data, train_lab, test_data, test_lab



def json_dump(path, dict_obj):

    with open(path, 'a+', encoding='utf-8') as f:
        json.dump(dict_obj, f, indent=4, ensure_ascii=False)


def print(*args, end='\n', _file=None):

    b_print(*args)
    if _file:
        with open(file=_file, mode='a+', encoding='utf-8') as console:
            b_print(*args, file=console, end=end)
    


def obj_save(path, obj):

    if obj is not None:
        with open(path, 'wb') as file:
            pickle.dump(obj, file)
    else:
        print('object is None!')


def obj_load(path):

    if osp.exists(path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    else:
        raise OSError('no such path:%s' % path)


def new_dir(father_dir, mk_dir=None):

    if mk_dir is None:
        new_path = osp.join(father_dir, time.strftime('%Y%m%d%H%M%S'))
    else:
        new_path = osp.join(father_dir, mk_dir)
    if not osp.exists(new_path):
        os.makedirs(new_path)
    return new_path
