__author__ = 'XF'
__date__ = '2024/07/01'


'''
Data preparation.
'''

import click
import sys
import numpy as np
from os import path as osp
from scipy.io import loadmat
sys.path.append('..')
from utils import new_dir

DATA_DIR = osp.dirname(osp.abspath(__file__))


@click.command()
@click.option('--dataset', type=click.Choice(['thyroid']))

def main(dataset):

    print(f'Preprocessing dataset [{dataset}] ======================')
    data_dir = osp.join(DATA_DIR, f'{dataset}')
    data_path = osp.join(data_dir, f'{dataset.lower()}.mat')

    data = loadmat(data_path)

    normal_data = data['X'][data['y'][:, 0] == 0]
    abnormal_data = data['X'][data['y'][:, 0] == 1]
    
    print('============= data info ==============')
    print(f'normal data: {normal_data.shape}')
    print(f'abnormal data: {abnormal_data.shape}')
    save_dir = new_dir(data_dir, mk_dir='processed')
    # saving formatted data
    np.save(osp.join(save_dir, 'normal_data.npy'), normal_data)
    np.save(osp.join(save_dir, 'abnormal_data.npy'), abnormal_data)
    print('Saving data successfully!')
    


if __name__ == '__main__':

    main()
