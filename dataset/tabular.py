__author__ = 'XF'
__date__ = '2023/07/12'

'''
the script is for tabular datasets.
'''

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.base_dataset import TorchvisionDataset
from utils import obj_save, obj_load, ROOT_DIR, generate_filename, new_dir

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = torch.from_numpy(labels).type(torch.FloatTensor)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        _data = torch.from_numpy(self.data[idx]).type(torch.FloatTensor)
        return _data, (self.labels[idx]), idx


class TabularDataset(TorchvisionDataset):
      def __init__(self, train_data, train_lab, test_data, test_lab, mode='train', mean=None, std=None):
        super().__init__('')
        self.name='tabular'

        if mode == 'train':


            ## preprocessing 
            mean = np.mean(train_data, 0)
            std = np.std(train_data, 0)
    
            train_data = (train_data - mean) / (std + 1e-6)
            test_data = (test_data - mean) / (std + 1e-6)      
            print('============ data set ================')
            print(f'train data: {train_data.shape}')
            print(f'test data: {test_data.shape}')
            print('============ data set ================')

            self.train_set = CustomDataset(train_data, train_lab)

            self.test_set = CustomDataset(test_data, test_lab)
            mean_std_file_name = generate_filename('.pkl', *['mean', 'std'], timestamp=True)
            save_path = new_dir(ROOT_DIR, mk_dir='results')
            obj_save(os.path.join(save_path, mean_std_file_name), {'mean': mean, 'std': std})

        elif mode == 'test':
            print('============ data set ================')
            print(f'test data: {test_data.shape}')
            print('============ data set ================')

            test_data = (test_data - mean) / (std + 1e-6)
            self.test_set = CustomDataset(test_data, test_lab)
        
        else:
            raise Exception(f'Unknown mode [{mode}]!')
