
'''
'''


import os
import numpy as np
from utils import ROOT_DIR

def load_data(dataset):

   
    if dataset in ['arrhythmia', 'thyroid', 'abalone', 'adult', 'Cardio', 'Satellite', 'Speech', 'Vowels', 'Vertebral']:
            normal_data_path = os.path.join(ROOT_DIR, f'dataset/{dataset}/normal_data.npy')
            abnormal_data_path = os.path.join(ROOT_DIR, f'dataset/{dataset}/abnormal_data.npy')
    else:
        raise Exception(f'Unknown dataset [{dataset}]!')

    normal_data = np.load(normal_data_path)
    abnormal_data = np.load(abnormal_data_path)

    print(f'normal data: {len(normal_data)}')
    print(f'abnormal data: {len(abnormal_data)}')
  

    return np.array(normal_data, dtype=np.float32), np.array(abnormal_data, dtype=np.float32)
