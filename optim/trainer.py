__author__ = 'XF'
__date__ = '2023/07/12'


'''
The optmization process of model.
'''
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from utils import distribution_sampling
from sklearn.metrics import roc_auc_score,  precision_recall_curve, auc
from torch import optim
import time


class GANsTrainer(object):
    
    def __init__(self, optimizer_name: str = 'adam', lr_d: float = 1e-3, lr_g: float = 1e-3, epochs: int = 100,
                batch_size: int = 128, latent_dim: int = 4, device: str = 'cuda'):
        
        self.optimizer_name = optimizer_name
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.epochs = epochs

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.device = device
        
        self.generator = None
        self.discriminator = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.results = {
            'AUROC': 0,
            'AUPRC': 0,
        }
        
    def build_networks(self, generator, discriminator):
        
        self.generator = generator
        self.discriminator = discriminator
        
    
    def train(self, dataset):

        # Set device for network
        discriminator = self.discriminator.to(self.device)
        generator = self.generator.to(self.device)

        # Set optimizer
        if self.optimizer_name == 'adam':
            self.optimizer_g = optim.Adam(generator.parameters(), lr=self.lr_g)
            self.optimizer_d = optim.Adam(discriminator.parameters(), lr=self.lr_d)
        elif self.optimizer_name == 'sgd':
            self.optimizer_g = optim.SGD(generator.parameters(), lr=self.lr_g)
            self.optimizer_d = optim.SGD(discriminator.parameters(), lr=self.lr_d)
        else:
            raise Exception(f'Unknown optimizer name [{self.optimizer_name}].')
        
        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size)

        # Training
        print(f'Start train ===================================')
        discriminator.train()
        generator.train()
        start = time.time()
        with trange(1, self.epochs + 1) as pbar:
            for epoch in pbar:
                g_loss_epoch = 0.0
                d_loss_epoch = 0.0
                n_batches = 0
                flag = True
                all_mid_repre = None
                for data in train_loader:
                    real_samples, real_labels, _ = data

                    real_samples = real_samples.to(self.device)
                    real_labels = real_labels.to(self.device)

                    gene_samples = distribution_sampling(sample_dim=self.latent_dim, size = self.batch_size)
                    gene_samples = gene_samples.to(self.device)
                    gene_labels = torch.zeros(len(gene_samples)).to(self.device)
          
                    labels = torch.cat((real_labels, gene_labels))
                   
                    # train discriminator ===============================
                    if flag:
                        self.optimizer_d.zero_grad()
                        outputs = generator(real_samples)

                        pred_labels = discriminator(torch.cat((outputs.detach(), gene_samples)))

                        d_loss = F.binary_cross_entropy(pred_labels.squeeze(1), labels)
                        d_loss.backward()
                        self.optimizer_d.step()
                        flag = False
                        d_loss_epoch += d_loss.item()
    
                    # train generator ====================================
                    else:
                        self.optimizer_g.zero_grad()
                        outputs = generator(real_samples)
                        g_loss = F.binary_cross_entropy(discriminator(outputs).squeeze(1), gene_labels)
                        g_loss.backward()
                        self.optimizer_g.step()
                        if all_mid_repre is None:
                            all_mid_repre = outputs
                        else:
                            all_mid_repre = torch.concat((all_mid_repre, outputs))
                        flag = True
                        g_loss_epoch += g_loss.item()
                    n_batches += 1

                pbar.set_description(f'Epoch[{epoch}]\t loss: {((g_loss_epoch + d_loss_epoch) / n_batches):.4f}\t g_loss: {(g_loss_epoch / n_batches):.4f}\t d_loss: {(d_loss_epoch / n_batches):.4f} ')

                # testing
                if epoch % 1 == 0:

                    print(f'\nEpoch[{epoch}] ################### testing ######################')
                    self.test(dataset)
        print(f'train time: {(time.time() - start) / 60:.2f} Min')
        return self.results
    
    def test(self, dataset, device='cuda'):

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size)
        self.device = device

        self.generator.eval()
        self.discriminator.eval()
        idx_label_score = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                scores = self.discriminator(self.generator(inputs))

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))


        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        
        auroc = roc_auc_score(labels, scores)
        print(f'Test set AUROC: [{auroc * 100:.2f}]%')
        precision, recall, _ = precision_recall_curve(labels, scores)
        auprc = auc(recall, precision)
        print('Test set AUPRC: [{:.2f}%]'.format(100. * auprc))

        if auroc > self.results['AUROC']:
            self.results['AUROC'] = auroc
            self.results['AUPRC'] = auprc




