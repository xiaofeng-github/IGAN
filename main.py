__author__ = 'XF'
__date__ = '2023/07/12'

'''
the entrance of the project.
'''

from os import path as osp
import click
import torch
import numpy as np

from network.GANs import Discriminator, Generator
from optim.trainer import GANsTrainer
from dataset.tabular import TabularDataset
from dataset.load_data import load_data
from utils import split_train_test


ROOT_DIR = osp.dirname(osp.abspath(__file__))

@click.command()
@click.option('--dataset', type=click.Choice(['thyroid', 'adult', 'abalone', 'arrhythmia', 'Speech', 'Vowels', 'Vertebral', 'Cardio', 'Satellite']))
@click.option('--optimizer_name', type=str, default='adam', help='')
@click.option('--epochs', type=int, default=100, help='The iteration number')
@click.option('--batch_size', type=int, default=32, help='')
@click.option('--lr_d', type=float, default=1e-4, help='the learning rate of discriminator.')
@click.option('--lr_g', type=float, default=1e-4, help='the learning rate of generator.')
@click.option('--latent_dim', type=int, default=4, help='the data dimension in latent space.')
@click.option('--seed', type=int, default=-1, help='the random seed.')
@click.option('--n_train_data', type=int, default=-1, help='the sample size of training data.')
@click.option('--train', type=bool, default=True, help='when it is True, training mode, otherwise testing mode')
@click.option('--repeat', type=int, default=1, help='The repeat time of the training process.')

def main(dataset, optimizer_name, epochs, batch_size, lr_d, lr_g, latent_dim, seed, n_train_data, train, repeat):
    
    
    print(f'=============================== IGAN ====================================== ')
    
    if train: # train mode
        # experimental settings
        print('experiment settings')
        print(f'Dataset: {dataset}')
        print(f'Optimizer: {optimizer_name}')
        print(f'Epochs: [{epochs}]')
        print(f'Batch size: [{batch_size}]')
        print(f'Latent dimension: [{latent_dim}]')
        print(f'Learning rate of discriminator: [{lr_d}]')
        print(f'Learning rate of genertor: [{lr_g}]')
        
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        print(f'Computation device: {device}')

        for i in range(1, repeat + 1):
            print(f'######################## the {i}-th repeat ########################')
            if seed != -1:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.backends.cudnn.deterministic = True
                print('Set seed to %d.' % seed)
            
            
            # load dataset =============================

            normal_data, abnormal_data = load_data(dataset=dataset)
            data_dim = normal_data.shape[1]
            train_data, train_lab, test_data, test_lab = split_train_test(normal_data, abnormal_data, n_train_data=n_train_data)
            dataset = TabularDataset(
                                    train_data=train_data,
                                    train_lab=train_lab,
                                    test_data=test_data,
                                    test_lab=test_lab)
    
            # model =====================================

            PBAD = GANsTrainer(
                                optimizer_name=optimizer_name,
                                lr_d=lr_d,
                                lr_g=lr_g,
                                epochs=epochs,
                                batch_size=batch_size,
                                device=device,
                                latent_dim=latent_dim)

            PBAD.build_networks(generator=Generator(data_dim, latent_dim), discriminator=Discriminator(latent_dim, 1))
            
            # train ======================================
            PBAD.train(dataset=dataset)

    print(f'===============================  End  =============================== ')


if __name__  == '__main__':
    
    main()




