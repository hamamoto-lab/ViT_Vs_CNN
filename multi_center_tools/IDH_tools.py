from typing import Callable
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.optim as optim
from PIL import Image
import numpy as np
import os
import time
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
#from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

import pickle

seed =1
torch.manual_seed(seed)


from torchvision import transforms, utils
import torchvision.models as models
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from multi_center_tools.multi_tools import load_JC_arrays, load_TCGA_arrays, translate_mask
#from multi_center_tools.Cycle_tool import get_converted_arrys
import copy

dict_g = {'FEMALE': 0, 'MALE': 1}
dict_h = {'Astrocytoma': 0,
 'Glioblastoma': 1,
 'Oligoastrocytoma': 2,
 'Oligodendroglioma': 3}
dict_r = {'BLACK OR AFRICAN AMERICAN': 0, 'WHITE': 1, '[Not Available]': 2}


dict_gn = {'F': 0, 'M': 1}
dict_hn = {'DA': 0,
 'GBM': 1,
 'OL': 3,
 'AA': 4,
 'AO': 5, 
 'FALSE':6}
dict_rn = {'BLACK OR AFRICAN AMERICAN': 0, 'WHITE': 1, '[Not Available]': 2}
dict_inst = {'TMDU':0, 'SMU':1, 'Kyorin':2, 'Dokkyo':3, 'OU':4, 'KNBTG':5, 'YCU':6, 'KU':7,
       'KYU':8, 'NCC':9}

class TCGA_Dataset_With_sampling_from_array(Dataset):
    """TCGA_Dataset"""
    
    def __init__(self, df, image_dir, sampling_tech,transform = None, z_score = False, sz=False,inst=None, double = False):
        """
        Args:
           csv_file(string):Path to the csv files with annotations
           image_dir(string):Directory with all images.
           mask_dir(string):Directory with mask
           sampling_tech:Type of Sampling techniques, 
                         axial_sampling or coronal_sampling or sagital_sampling
           transform(callable, optional):Optional transform to be applied on a sample
        """
        
        self.Training_frames = df
        self.image_dir = image_dir
        self.sampling_tech = sampling_tech
        self.transform = transform
        self.z_score =z_score
        self.sz = sz
        self.inst = inst
        self.double = double

    def __len__(self):
        return len(self.Training_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        target = self.Training_frames.iloc[idx,:]
        img_name =target['BraTS_2019_subject_ID']
        FLAIR, T1, T1CE, T2, mask = load_TCGA_arrays(self.image_dir, img_name,Z_score=self.z_score, SZ=self.sz,inst=self.inst, double = self.double)
 
        
        con_image = np.concatenate([FLAIR, T1, T1CE,T2])
        
        T2_mask = translate_mask(mask,translate_type='T2')
        T2_mask_r = T2_mask.reshape(T2_mask.shape[1],T2_mask.shape[2],T2_mask.shape[3])
        
        sampled_image = self.sampling_tech(con_image, T2_mask_r)
        sampled_image = sampled_image.transpose((3,0,1,2))
        
        sample = {'Sampled_image':sampled_image,'IDH_status':target['IDH1_2'],
                 'Age':target['age_at_initial_pathologic_diagnosis'],
                 'Gender':target['gender'],'Race':target['race'],
                 'Patho':target['histological_type'],
                 'Name':img_name}
        if self.transform:
            sample = self.transform(sample)
        
        return sample 

class TCGAToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sampled_image, IDH_status,age= sample['Sampled_image'],sample['IDH_status'],sample['Age']
        Gender, Race, Patho,Name=dict_g[sample['Gender']], dict_r[sample['Race']],dict_h[sample['Patho']],sample['Name'] 
        
        images = torch.from_numpy(sampled_image)
        imag_size =images.size()
        images = images.reshape(imag_size[0]*imag_size[1], imag_size[2],imag_size[3]).float()
        
        return images,(torch.tensor(float(age)).log_()/4.75,torch.tensor(Gender),torch.tensor(Patho),torch.tensor(IDH_status)) 

class JC_Dataset_With_sampling_from_array(Dataset):
    """JC_Dataset"""
    
    def __init__(self, df, image_dir, sampling_tech,transform = None, z_score = False, sz=False, inst=None, double = False):
        """
        Args:
           csv_file(string):Path to the csv files with annotations
           image_dir(string):Directory with all images.
           mask_dir(string):Directory with mask
           sampling_tech:Type of Sampling techniques, 
                         axial_sampling or coronal_sampling or sagital_sampling
           transform(callable, optional):Optional transform to be applied on a sample
        """
        
        self.Training_frames = df
        self.image_dir = image_dir
        self.sampling_tech = sampling_tech
        self.transform = transform
        self.z_score =z_score
        self.sz = sz
        self.inst = inst
        self.double =double
        
    def __len__(self):
        return len(self.Training_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        target = self.Training_frames.iloc[idx,:]
        img_name =target['ID']
        img_name = str(img_name)
        FLAIR, T1, T1CE, T2, T2_mask,GD_mask = load_JC_arrays(self.image_dir, img_name,Z_score=self.z_score, SZ=self.sz,inst=self.inst, double = self.double)
        
        con_image = np.concatenate([FLAIR, T1, T1CE,T2])
        
        T2_mask_r = T2_mask.reshape(T2_mask.shape[1],T2_mask.shape[2],T2_mask.shape[3])
        
        sampled_image = self.sampling_tech(con_image, T2_mask_r)
        sampled_image = sampled_image.transpose((3,0,1,2))
        
        sample = {'Sampled_image':sampled_image,'IDH_status':target['IDH1_2'],
                 'Age':target['age_at_initial_pathologic_diagnosis'],
                 'Gender':target['gender'],
                 'Patho':target['histological_type'],'Institution':target['Institute'],
                 'Name':img_name}
        if self.transform:
            sample = self.transform(sample)
        
        return sample 
    
    def get_labels(self):
        #for  ImbalancedDatasetSampler
        return self.Training_frames['IDH1_2']
    
    def get_institute(self):
        return np.array([dict_inst[inst] for inst in self.Training_frames['Institute']])

class JCToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sampled_image, IDH_status,age= sample['Sampled_image'],sample['IDH_status'],sample['Age']
        Gender, Race, Patho,inst, Name = dict_gn[sample['Gender']], 2, dict_hn[sample['Patho']], dict_inst[sample['Institution']], sample['Name'] 
        
        images = torch.from_numpy(sampled_image)
        imag_size =images.size()
        images = images.reshape(imag_size[0]*imag_size[1], imag_size[2],imag_size[3]).float()
        return images,(torch.tensor(float(age)).log_()/4.75,torch.tensor(Gender),torch.tensor(Patho),torch.tensor(IDH_status),
        torch.tensor(inst))

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """https://github.com/ufoym/imbalanced-dataset-sampler
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class ImbalanceInstitutetSampler(torch.utils.data.sampler.Sampler):
    """https://github.com/ufoym/imbalanced-dataset-sampler
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices: list = None, num_samples: int = None, callback_get_label: Callable = None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        return dataset.get_institute()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples