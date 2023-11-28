__author__ = 'XF'
__date__ = '2023/07/12'

'''
generative adversarial networks.

'''

import torch
import torch.nn as nn



class Discriminator(nn.Module):
    
    
    def __init__(self, input_dim, output_dim):
        
        super(Discriminator, self).__init__()

        

        self.discriminator = nn.Sequential(
                            nn.Linear(input_dim, 64),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(64, 32),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(32, 16),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(16, 8),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(8, 4),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(4, output_dim))
    
    def forward(self, x):
        
        output = torch.sigmoid(self.discriminator(x))

        return output
    
    

class Generator(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        super(Generator, self).__init__()
        

        self.generator = nn.Sequential(
                            nn.Linear(input_dim, 256),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(256, 200),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(200, 128),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(128, 100),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(100, 64),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(64, 32),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(32, 16),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(16, output_dim))
    

    def forward(self, x):
   
        output = self.generator(x)
        return output