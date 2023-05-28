from curses import beep
from json import load
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import pytorch_lightning as pl
import torch.optim as optim
from multi_center_tools.multi_tools import load_npz, loadArray
from multi_center_tools.Cycle_tool import Generator
from skimage.transform import rescale
import pandas as pd
from multi_center_tools.multi_tools import load_oneSlice_JC_arrays, load_oneSlice_TCGA_arrays
from multi_center_tools.Cycle_tool import Generator
act_fn_by_name = {
    "tann":nn.Tanh,
    "relu":nn.ReLU,
    "leakyrelu":nn.LeakyReLU,
    "gelu":nn.GELU
}

class DL_base_feature_extraction_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_dir,image_list,itype, institute = 'JC', to_3ch = True, transform = None):
        super(torch.utils.data.Dataset,self).__init__()
        self.data_dir = data_dir
        self.image_list = image_list
        self.itype =itype
        self.institute =institute
        self.transform = transform
        self.to_3ch = to_3ch
        
    def __getitem__(self, index):
        case = self.image_list[index]
        if self.institute =="JC":
            images = load_oneSlice_JC_arrays(self.data_dir,case[0],case[1])
        else:
            images = load_oneSlice_TCGA_arrays(self.data_dir, case[0], case[1])

        if self.itype =="FLAIR":
            base = np.squeeze(images[0])
        elif self.itype =='T1':
            base = np.squeeze(images[1])
        elif self.itype =='T1CE':
            base = np.squeeze(images[2])
        elif self.itype =='T2':
            base = np.squeeze(images[3])
        if self.to_3ch ==True :
            base = rescale(base, 1.75) # to (308, 336)
            base = base[:299, :299]
            for_tensor=np.stack([base, base, base], axis=0)
        else:
            for_tensor = np.expand_dims(base,0)
        intensor = torch.from_numpy(for_tensor).float()
        
        if self.transform:
            intensor = self.transform(intensor)
        return intensor
    
    def __len__(self):
        return  len(self.image_list)

class DL_base_feature_extraction_Dataset_use_converter(torch.utils.data.Dataset):
    
    def __init__(self, data_dir,image_list,itype, converter,institute = 'JC', transform = None):
        super(torch.utils.data.Dataset,self).__init__()
        self.data_dir = data_dir
        self.image_list = image_list
        self.itype =itype
        self.converter = converter #instance of  convert_image_style()
        self.institute =institute
        self.transform = transform
        
    def __getitem__(self, index):
        case = self.image_list[index]
        if self.institute =="JC":
            images = load_oneSlice_JC_arrays(self.data_dir,case[0],case[1])
        else:
            images = load_oneSlice_TCGA_arrays(self.data_dir, case[0], case[1])

        if self.itype =="FLAIR":
            base = np.squeeze(images[0])
        elif self.itype =='T1':
            base = np.squeeze(images[1])
        elif self.itype =='T1CE':
            base = np.squeeze(images[2])
        elif self.itype =='T2':
            base = np.squeeze(images[3])
        for_tensor = np.expand_dims(base,0)
        in_tensor = torch.from_numpy(for_tensor).float()
        converted = self.converter(in_tensor)
        
        if self.transform:
            converted = self.transform(converted)
        return converted
    
    def __len__(self):
        return  len(self.image_list)

def load_npz(dirc):
    return np.load(dirc)['arr_0']

def get_feature_df(target_data, target_list):
    features=  target_data[1:, :] # remove start array
    case_name = [case[0] for case in target_list]
    slice_numbers = np.array([case[1] for case in target_list])
    slice_numbers = slice_numbers.reshape(len(target_list), 1)
    input_data = np.concatenate([slice_numbers, features], axis =1)
    result_df = pd.DataFrame(input_data, index=case_name)
    result_df = result_df.rename(columns={0:'slice_number'})
    return result_df


'''def convert_image_style(base_dir,instName,imgType, suffix,net_name, baseimg):
    image = torch.unsqueeze(baseimg, dim=0) #torch.Size([1, 1, 176, 192])
    image = image *2 -1  #0to1 -> -1to1
    gen = Generator()
    gen.load_state_dict(torch.load(os.path.join(base_dir,'log_' + instName + imgType + suffix, net_name)))
    gen.eval()
    conv_img = gen(image)
    conv_img = conv_img.squeeze(0) #torch.Size([1, 176, 192])
    conv_img = (conv_img +1)/2  #-1 to1 -> 0to1
    return conv_img 
'''



class convert_image_style():
    def __init__(self,base_dir,instName,imgType, suffix,net_name):
        self.base_dir = base_dir
        self.instName =instName
        self.imgType = imgType
        self.suffix = suffix
        self.net_name = net_name

    def __call__(self, baseimg):
        image = torch.unsqueeze(baseimg, dim=0) #torch.Size([1, 1, 176, 192])
        image = image *2 -1  #0to1 -> -1to1
        gen = Generator()
        gen.load_state_dict(torch.load(os.path.join(self.base_dir,'log_' + self.instName + self.imgType + self.suffix, self.net_name)))
        gen.eval()
        conv_img = gen(image)
        conv_img = conv_img.squeeze(0) #torch.Size([1, 176, 192])
        conv_img = (conv_img +1)/2  #-1 to1 -> 0to1
        return conv_img 




'''def load_oneSlice_TCGA_arrays(data_path, sample_ID, sliceN,Z_score = False, SZ=False):
    FLAIR_Mean = 84.399592
    T1_Mean = 131.312326
    T1CE_Mean = 163.123228
    T2_Mean = 162.879965
    
    FLAIIR_SD = 156.820304
    T1_SD = 235.835906
    T1CE_SD = 308.169945
    T2_SD = 313.665391
    
    FLAIR= load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
    T1 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
    T1CE = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
    T2 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
    mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_mask.npz')
    if Z_score == True:
        FLAIR = zscore(FLAIR, FLAIR_Mean, FLAIIR_SD)
        T1 = zscore(T1, T1_Mean, T1_SD)
        T1CE = zscore(T1CE, T1CE_Mean, T1CE_SD)
        T2= zscore(T2, T2_Mean,T2_SD)
    
    if SZ ==True:
        FLAIR = sample_z(FLAIR)
        T1 = sample_z(T1)
        T1CE = sample_z(T1CE)
        T2= sample_z(T2)
    
    FLAIR = FLAIR[:,:,:,sliceN]
    T1 = T1[:,:,:,sliceN]
    T1CE = T1CE[:,:,:,sliceN]
    T2 = T2[:,:,:,sliceN]
    mask = mask[:,:,:,sliceN]
    
    return FLAIR, T1, T1CE, T2, mask

def load_oneSlice_JC_arrays(data_path, sample_ID, sliceN,Z_score = False, SZ=False):
    
    FLAIR_Mean = 84.399592
    T1_Mean = 131.312326
    T1CE_Mean = 163.123228
    T2_Mean = 162.879965
    
    FLAIIR_SD = 156.820304
    T1_SD = 235.835906
    T1CE_SD = 308.169945
    T2_SD = 313.665391
    
    FLAIR= load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
    T1 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
    T1CE = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
    T2 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
    T2_mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2_mask.npz')
    GD_mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_GD_mask.npz')
    if Z_score == True:
        FLAIR = zscore(FLAIR, FLAIR_Mean, FLAIIR_SD)
        T1 = zscore(T1, T1_Mean, T1_SD)
        T1CE = zscore(T1CE, T1CE_Mean, T1CE_SD)
        T2= zscore(T2, T2_Mean,T2_SD)
    
    if SZ ==True:
        FLAIR = sample_z(FLAIR)
        T1 = sample_z(T1)
        T1CE = sample_z(T1CE)
        T2= sample_z(T2)

    FLAIR = FLAIR[:,:,:,sliceN]
    T1 = T1[:,:,:,sliceN]
    T1CE = T1CE[:,:,:,sliceN]
    T2 = T2[:,:,:,sliceN]
    T2_mask = T2_mask[:,:,:,sliceN]
    GD_mask = GD_mask[:,:,:,sliceN]

    return FLAIR, T1, T1CE, T2, T2_mask,GD_mask'''