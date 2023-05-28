import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import os
import time
import random
from torch.utils.data import Dataset, DataLoader
from multi_center_tools.multi_tools import institute_zscore, sample_z
import pandas as pd
import pickle
from tensorboardX import SummaryWriter
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
from multi_center_tools.multi_tools import load_TCGA_arrays, translate_mask

def load_npz(dirc):
    return np.load(dirc)['arr_0']
class ResNetBlock(nn.Module):
    
    def __init__(self,dim):
        super(ResNetBlock,self).__init__()
        conv_block =[]
        conv_block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(dim,dim,kernel_size=3),
                      nn.InstanceNorm2d(dim),
                      nn.ReLU(True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(dim,dim,kernel_size=3),
                      nn.InstanceNorm2d(dim)]
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self,x):
        out = x + self.conv_block(x)
        return out
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
        nn.ReflectionPad2d(3),
            
        nn.Conv2d(1,64,kernel_size=7),
        nn.InstanceNorm2d(64),
        nn.ReLU(True),
        
        nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(True),
            
        nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
        nn.InstanceNorm2d(256),
        nn.ReLU(True),
        
        ResNetBlock(256),
        ResNetBlock(256),
        ResNetBlock(256),
        ResNetBlock(256),
        ResNetBlock(256),
        ResNetBlock(256),
        ResNetBlock(256),
        ResNetBlock(256),
        ResNetBlock(256),
            
        nn.ConvTranspose2d(256,128,kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(True),
            
        nn.ConvTranspose2d(128,64,kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.InstanceNorm2d(64),
        nn.ReLU(True),
            
        nn.ReflectionPad2d(3),
        nn.Conv2d(64,1,kernel_size=7, stride=1, padding=0),
        nn.Tanh()
        
        )
        
        # initialize weights
        self.model.apply(self._init_weights)
        
    def forward(self, input):
        return self.model(input)
    
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)

class TCGA_Dataset_With_sampling_and_convert(Dataset):
    """TCGA_Dataset"""
    
    def __init__(self, df, image_dir, sampling_tech,converter,transform = None,Z_score=False, SZ=False, inst = None, double = False,image_calibration=True):
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
        self.converter = converter
        self.Z_score = Z_score
        self.SZ =SZ
        self.inst =inst
        self.double =double
        self.image_calibration = image_calibration
        
    def __len__(self):
        return len(self.Training_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        target = self.Training_frames.iloc[idx,:]
        img_name =target['BraTS_2019_subject_ID']

        #Before Scaling all scaling is False
        FLAIR, T1, T1CE, T2, mask = load_TCGA_arrays(self.image_dir, img_name,Z_score=False, SZ=False)
 
        
        con_image = np.concatenate([FLAIR, T1, T1CE,T2])
        
        T2_mask = translate_mask(mask,translate_type='T2')
        T2_mask_r = T2_mask.reshape(T2_mask.shape[1],T2_mask.shape[2],T2_mask.shape[3])
        
        sampled_image = self.sampling_tech(con_image, T2_mask_r)
        #images reshape for converter
        sampled_image = sampled_image.transpose([0,3,1,2]) #(imagetype, slice, h, w) example(4,3,176,192)
        # make image mask for noise cacelling
        FLAIR_imask = np.where(sampled_image[0]==0,0,1)
        T1_imask = np.where(sampled_image[1]==0,0,1)
        T1CE_imask = np.where(sampled_image[2]==0,0,1)
        T2_imask = np.where(sampled_image[3]==0,0,1)
        cFLAIR,cT1, cT1CE, cT2 = self.converter(sampled_image) #need coverters separatrion is True
        #back groud noise cancelling
        cFLAIR = cFLAIR * FLAIR_imask
        cT1 = cT1 * T1_imask
        cT1CE = cT1CE * T1CE_imask
        cT2 = cT2 * T2_imask
        if self.image_calibration == True:
            cFLAIR = cFLAIR / np.max(cFLAIR)
            cT1= cT1/ np.max(cT1)
            cT1CE= cT1CE/ np.max(cT1CE)
        if self.Z_score == True:
            cFLAIR, cT1, cT1CE, cT2 = institute_zscore(cFLAIR, cT1, cT1CE, cT2, self.inst)
        if self.SZ ==True:
            cFLAIR = sample_z(cFLAIR)
            cT1 = sample_z(cT1)
            cT1CE = sample_z(cT1CE)
            cT2= sample_z(cT2)
        if self.double == True:
            cFLAIR = cFLAIR *2 -1
            cT1 = cT1 *2 -1
            cT1CE = cT1CE*2 -1
            cT2 = cT2 *2 -1
        stacked = np.stack([cFLAIR, cT1, cT1CE, cT2]) #(imagetype, slice, h, w) example(4,3,176,192)
        stacked = stacked.transpose([1,0,2,3]) #(slice,image_tyoe, h, w) example(3,4,176,192)
        
        sample = {'Sampled_image':stacked,'IDH_status':target['IDH1_2'],
                 'Age':target['age_at_initial_pathologic_diagnosis'],
                 'Gender':target['gender'],'Race':target['race'],
                 'Patho':target['histological_type'],
                 'Name':img_name}
        if self.transform:
            sample = self.transform(sample)
        
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        dict_g = {'FEMALE': 0, 'MALE': 1}
        dict_h = {'Astrocytoma': 0,'Glioblastoma': 1,'Oligoastrocytoma': 2,'Oligodendroglioma': 3}
        dict_r = {'BLACK OR AFRICAN AMERICAN': 0, 'WHITE': 1, '[Not Available]': 2}
        sampled_image, IDH_status,age= sample['Sampled_image'],sample['IDH_status'],sample['Age']
        Gender, Race, Patho,Name=dict_g[sample['Gender']], dict_r[sample['Race']],dict_h[sample['Patho']],sample['Name'] 
        
        images = torch.from_numpy(sampled_image)
        imag_size =images.size()
        images = images.reshape(imag_size[0]*imag_size[1], imag_size[2],imag_size[3]).float()
        #images = torch.rot90(images, 2,(1,2)) # for vizualization rotate 180 degree
        return images,(torch.tensor(float(age)).log_()/4.75,torch.tensor(Gender),torch.tensor(Patho),torch.tensor(IDH_status)) 


def getJCArray(data_path, sample_ID, slice_n, itype):
    array = loadArray(data_path, sample_ID, slice_n,itype)
    array = np.squeeze(array)
    array = torch.from_numpy(array.astype(np.float32)).clone()
    array = torch.unsqueeze(array,0)
    return array

def loadArray(data_path, sample_ID, slice_n, itype):
    if itype  == "FLAIR":
        img= load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
    elif itype == 'T1':
        img = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
    elif itype == 'T1CE':
        img= load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
    elif itype == 'T2':
        img = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
    else:
        img = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_mask.npz')
    return img[:,:,:,slice_n]

def convert_TCGA_style(net_name,baseimg,imgType):
    '''
    input tensor (1,176,192)
    output tensor(1,176,192)
    '''
    gen = Generator()
    gen.load_state_dict(torch.load(net_name))
    gen.eval()
    tensor = baseimg.view(1,1,176,192)
    #tensor = torch.from_numpy(baseimg.reshape(1,1,176,192).astype(np.float32))
    conv_img = gen(tensor)
    conv_img = conv_img.view(1,176,192)
    return conv_img


def convert_style(baseimg,base_dir,imgType, instName,suffix,net_name):
    gen = Generator()
    gen.load_state_dict(torch.load(os.path.join(base_dir,imgType,'log_' + instName + imgType + suffix, net_name)))
    gen.eval()
    baseimg = baseimg*2 -1 #0to1 -> -1to1
    tensor = torch.from_numpy(baseimg.astype(np.float32))
    conv_img = gen(tensor)
    conv_img = (conv_img +1)/2 #-1 to1 -> 0to1
    return conv_img.to('cpu').detach().numpy().copy()

def get_converted_array_for_save(in_array,base_dir, imgType, inst, suffix, net_name ):
    array = in_array.transpose(3,0,1,2)
    mask = np.where(in_array ==0, 0, 1)
    converted = convert_style(array,base_dir, imgType, inst, suffix, net_name )
    t_converted = converted.transpose(1,2,3,0)
    masked = t_converted * mask
    masked = masked /np.max(masked)
    return masked