import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
#import SimpleITK as sitk

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

def ch3_to_GD(label_3ch):
    GD=label_3ch[0,:,:,:]+label_3ch[2,:,:,:]
    GD=np.where(GD>0.9, 1, 0)
    return GD

def ch3_to_t2(label_3ch):
    t2=label_3ch[0,:,:,:]+label_3ch[1,:,:,:]+label_3ch[2,:,:,:]
    t2=np.where(t2>0.9, 1, 0)
    return t2

'''def axial_sampling(img,mask,n_sample=3):
    """
    sampling axial cutting image that contatin top n tumor leasion.
    retun: numpy array (Dim,W,D,n_sample)
    img:numpy array (Dim, W, D,H )
    mask numpy array (W,D,H)
    """
    a=np.sum(mask, axis=0)
    a= np.sum(a,axis=0)

    base = []
    for i in range(1,n_sample+1):
        r_i= -1*i
        idexs=np.argsort(a)
        idex = idexs[r_i]
        #print(idex)
        sampled = img[:,:,:,idex]
        base.append(sampled)
        #print(r_i)
    return np.stack(base, -1)'''

def axial_sampling(img,mask, need_number=3):
    """
    sampling axial cutting image that contatin top 1 and plus minus1 (total 3)
    retun: numpy array (Dim,W,D,n_sample)
    img:numpy array (Dim, W, D,H )
    mask numpy array (W,D,H)
    """
    a=np.sum(mask, axis=0)
    a= np.sum(a,axis=0)
    slice_width = need_number//2
    idexs=np.argsort(a)
    idex = idexs[-1]
    start = idex -(slice_width +1 )
    end  = idex +slice_width
    sampled = img[:,:,:,start:end]
    #print(start)
    #print(end)
    return sampled

'''def coronal_sampling(img,mask,n_sample=3):
    """
    sampling axial cutting image that contatin top n tumor leasion.
    retun: numpy array (Dim,W,H, n_sample)
    img:numpy array (Dim, W, D,H )
    mask numpy array (W,D,H)
    """
    a=np.sum(mask, axis=0)
    a= np.sum(a,axis=1)

    base = []
    for i in range(1,n_sample+1):
        r_i= -1*i
        idex=np.argsort(a)[r_i]
        sampled = img[:,:,idex,:]
        base.append(sampled)
    return np.stack(base, -1)

def sagital_sampling(img,mask,n_sample=3):
    """
    sampling axial cutting image that contatin top n tumor leasion.
    retun: numpy array (Dim,D,H, n_sample)
    img:numpy array (Dim, W, D,H )
    mask numpy array (W,D,H)
    """
    a=np.sum(mask, axis=1)
    a= np.sum(a,axis=1)

    base = []
    for i in range(1,n_sample+1):
        r_i= -1*i
        idex=np.argsort(a)[r_i]
        sampled = img[:,idex,:,:]
        base.append(sampled)
    return np.stack(base, -1)'''


class AxiSampling:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, img, mask):
        a=np.sum(mask, axis=0)
        a= np.sum(a,axis=0)

        base = []
        for i in range(1,self.n_sample+1):
            r_i= -1*i
            idexs=np.argsort(a)
            idex = idexs[r_i]
            #print(idex)
            sampled = img[:,:,:,idex]
            base.append(sampled)
            #print(r_i)
        return np.stack(base, -1)

class CorSampling:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, img, mask):
        a=np.sum(mask, axis=0)
        a= np.sum(a,axis=1)

        base = []
        for i in range(1,self.n_sample+1):
            r_i= -1*i
            idex=np.argsort(a)[r_i]
            sampled = img[:,:,idex,:]
            base.append(sampled)
        return np.stack(base, -1)

class SagSampling:
    def __init__(self, n_sample):
        self.n_sample = n_sample

    def __call__(self, img, mask):
        a=np.sum(mask, axis=1)
        a= np.sum(a,axis=1)

        base = []
        for i in range(1,n_sample+1):
            r_i= -1*i
            idex=np.argsort(a)[r_i]
            sampled = img[:,idex,:,:]
            base.append(sampled)
        return np.stack(base, -1)

def open_pickle(loc):
    with open(loc, 'rb') as f:
        return pickle.load(f)
def select_df(df,names,select_list):
    common_list =set(select_list) & set(df[names])
    return df[df[names].isin(common_list)]

def load_npz(dirc):
    return np.load(dirc)['arr_0']

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

#画像のzスコア化
def zscore(img, mean,std):
    return (img - mean) / std

def institute_zscore(FLAIR, T1, T1CE, T2, inst):
    if inst == 'NCC':
        FLAIR_Mean = 0.154518
        T1_Mean = 0.161404
        T1CE_Mean = 0.131060
        T2_Mean = 0.110935

        FLAIIR_SD = 0.252240
        T1_SD = 0.257210
        T1CE_SD = 0.210441
        T2_SD = 0.183151
    elif inst == 'JC':
        FLAIR_Mean = 0.142227
        T1_Mean = 0.156767
        T1CE_Mean = 0.125968
        T2_Mean = 0.103641

        FLAIIR_SD = 0.237769
        T1_SD = 0.254498
        T1CE_SD = 0.206420
        T2_SD = 0.174886
    elif inst == 'TCGA':
        FLAIR_Mean = 0.124959
        T1_Mean = 0.179980
        T1CE_Mean = 0.123096
        T2_Mean = 0.085109

        FLAIIR_SD = 0.222700
        T1_SD = 0.308447
        T1CE_SD = 0.214064
        T2_SD = 0.154901
    else:
        print('you need appropriate inst')
    
    zFLAIR = zscore(FLAIR, FLAIR_Mean, FLAIIR_SD)
    zT1 = zscore(T1, T1_Mean, T1_SD)
    zT1CE = zscore(T1CE, T1CE_Mean, T1CE_SD)
    zT2= zscore(T2, T2_Mean,T2_SD)

    return zFLAIR, zT1, zT1CE, zT2




def sample_z(img):
    mean = img.mean()
    std = img.std()
    return (img - mean) / std

def tensor_show(img):
    npimg = img.numpy()
    #npimg = 0.5 * (npimg + 1) # [-1,1] => [0,1] changed! scaled at converter(get_coverted_array)
    npimg = np.squeeze(npimg)
    plt.imshow(npimg,cmap='gray')

'''def load_JC_arrays(data_path, sample_ID, Z_score = False, SZ=False):
    
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
    
    return FLAIR, T1, T1CE, T2, T2_mask,GD_mask'''

def load_JC_arrays(data_path, sample_ID, Z_score = False, SZ=False,
                        inst =None, double = False, padding=False, paded_size =224):
    
    FLAIR= load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
    T1 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
    T1CE = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
    T2 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
    T2_mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2_mask.npz')
    GD_mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_GD_mask.npz')
    padding_param = [(0,0), ((paded_size - 176)//2, (paded_size-176)//2),
                 ((paded_size - 192)//2, (paded_size-192)//2), (0,0)]
    if padding:
        FLAIR = np.pad(FLAIR, padding_param)
        T1 = np.pad(T1, padding_param)
        T1CE = np.pad(T1CE, padding_param)
        T2 = np.pad(T2, padding_param)
        T2_mask = np.pad(T2_mask, padding_param)
        GD_mask = np.pad(GD_mask, padding_param)
    if Z_score == True:
        FLAIR, T1, T1CE, T2 = institute_zscore(FLAIR, T1, T1CE, T2, inst)
    if SZ ==True:
        FLAIR = sample_z(FLAIR)
        T1 = sample_z(T1)
        T1CE = sample_z(T1CE)
        T2= sample_z(T2)
    if double == True:
        FLAIR = FLAIR *2 -1
        T1 = T1 *2 -1
        T1CE = T1CE*2 -1
        T2 = T2 *2 -1
    return FLAIR, T1, T1CE, T2, T2_mask,GD_mask

def load_oneSlice_JC_arrays(data_path, sample_ID, sliceN,Z_score = False, SZ=False, inst = None, double = False):
    
    FLAIR= load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
    T1 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
    T1CE = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
    T2 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
    T2_mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2_mask.npz')
    GD_mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_GD_mask.npz')
    if Z_score == True:
        FLAIR, T1, T1CE, T2 = institute_zscore(FLAIR, T1, T1CE, T2, inst)
    if SZ ==True:
        FLAIR = sample_z(FLAIR)
        T1 = sample_z(T1)
        T1CE = sample_z(T1CE)
        T2= sample_z(T2)
    if double == True:
        FLAIR = FLAIR *2 -1
        T1 = T1 *2 -1
        T1CE = T1CE*2 -1
        T2 = T2 *2 -1
    FLAIR = FLAIR[:,:,:,sliceN]
    T1 = T1[:,:,:,sliceN]
    T1CE = T1CE[:,:,:,sliceN]
    T2 = T2[:,:,:,sliceN]
    T2_mask = T2_mask[:,:,:,sliceN]
    GD_mask = GD_mask[:,:,:,sliceN]

    return FLAIR, T1, T1CE, T2, T2_mask,GD_mask
    
def load_TCGA_arrays(data_path, sample_ID, Z_score = False, SZ=False,
                        inst =None, double = False, padding=False, paded_size =224):

    FLAIR= load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
    T1 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
    T1CE = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
    T2 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
    mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_mask.npz')
    padding_param = [(0,0), ((paded_size - 176)//2, (paded_size-176)//2),
                 ((paded_size - 192)//2, (paded_size-192)//2), (0,0)]
    if padding:
        FLAIR = np.pad(FLAIR, padding_param)
        T1 = np.pad(T1, padding_param)
        T1CE = np.pad(T1CE, padding_param)
        T2 = np.pad(T2, padding_param)
        mask = np.pad(mask, padding_param)
    if Z_score == True:
        FLAIR, T1, T1CE, T2 = institute_zscore(FLAIR, T1, T1CE, T2, inst)
    if SZ ==True:
        FLAIR = sample_z(FLAIR)
        T1 = sample_z(T1)
        T1CE = sample_z(T1CE)
        T2= sample_z(T2)
    if double == True:
        FLAIR = FLAIR *2 -1
        T1 = T1 *2 -1
        T1CE = T1CE*2 -1
        T2 = T2 *2 -1
    return FLAIR, T1, T1CE, T2, mask
    
def load_oneSlice_TCGA_arrays(data_path, sample_ID, sliceN,Z_score = False, SZ=False, inst = None, double = False):

    FLAIR= load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
    T1 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
    T1CE = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
    T2 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
    mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_mask.npz')
    if Z_score == True:
        FLAIR, T1, T1CE, T2 = institute_zscore(FLAIR, T1, T1CE, T2, inst)
    if SZ ==True:
        FLAIR = sample_z(FLAIR)
        T1 = sample_z(T1)
        T1CE = sample_z(T1CE)
        T2= sample_z(T2)
    if double == True:
        FLAIR = FLAIR *2 -1
        T1 = T1 *2 -1
        T1CE = T1CE*2 -1
        T2 = T2 *2 -1
    FLAIR = FLAIR[:,:,:,sliceN]
    T1 = T1[:,:,:,sliceN]
    T1CE = T1CE[:,:,:,sliceN]
    T2 = T2[:,:,:,sliceN]
    mask = mask[:,:,:,sliceN]
    
    return FLAIR, T1, T1CE, T2, mask


def translate_mask(mask, translate_type='T2'):
    base_1 = np.zeros_like(mask)
    base_1[mask.astype(int) == 1] = 1
    base_2 = np.zeros_like(mask)
    base_2[mask.astype(int) == 2] = 1
    base_4 = np.zeros_like(mask)
    base_4[mask.astype(int) == 4] = 1
    if translate_type == 'T2':
        translated_mask = (base_1 + base_2 + base_4).astype(int)
    if translate_type == 'GD':
        translated_mask = (base_1 + base_4).astype(int)
    return np.where(translated_mask>0.9, 1,0)

def convert_TCGA_style(baseimg,imgType,base_dir,instName):
    gen = Generator()
    gen.load_state_dict(torch.load(os.path.join(base_dir,'log_' + instName + imgType, net_name)))
    tensor = torch.from_numpy(baseimg.reshape(3,1,176,192).astype(np.float32))
    con_img = gen(tensor)
    con_img = con_img.view(3,176,192)
    return con_img

def get_converted_arrys(sampled_image,base_dir, instName):
    ''''
    input 0-1 ranged image
    output 0-1 rqnaged image
    '''
    sampled_image = sampled_image *2 -1 #0to1 -> -1to1 
    sFLAIRs = sampled_image[0,:,:,:]
    sT1s = sampled_image[1,:,:,:]
    sT1CEs = sampled_image[2,:,:,:]
    sT2s= sampled_image[3,:,:,:]
    images = [sFLAIRs, sT1s,sT1CEs,sT2s]
    tnames =['FLAIR', 'T1', 'T1CE','T2']
    empty= []
    for i, image in enumerate(images):
        con_images = convert_TCGA_style(image,tnames[i],base_dir, instName)
        empty.append(con_images)
    stacked = torch.stack([empty[0],empty[1],empty[2],empty[3]])
    stacked = (stacked +1)/2 #-1 to1 -> 0to1
    return stacked.to('cpu').detach().numpy().copy()

def imshow(img):
    if type(img) is np.ndarry:
        npimg=img
    else:
        npimg = img.numpy()
    npimg=img
    npimg = 0.5 * (npimg + 1) # [-1,1] => [0,1]
    npimg = np.squeeze(npimg)
    plt.imshow(npimg,cmap='gray')
def show_array_90(nii_array, number=80):
    img = np.rot90(np.squeeze(nii_array),1)
    #img = np.squeeze(nii_array)[:, :, number]
    plt.imshow(img, cmap='gray')
def show_array_270(nii_array, number=80):
    img = np.rot90(np.squeeze(nii_array),3)
    #img = np.squeeze(nii_array)[:, :, number]
    plt.imshow(img, cmap='gray')

'''def get_JC_axialSampling_list(data_path, ID, need_number=3):
    _, _, _, _, emask, _, = load_JC_arrays(data_path,ID)
    emask = np.squeeze(emask)
    a=np.sum(emask, axis=0)
    a= np.sum(a,axis=0)
    indexes = np.argsort(a)[-need_number:]
    repeat_ID = np.repeat(ID, len(indexes))
    return repeat_ID, indexes

def get_TCGA_axialSampling_list(data_path, ID, need_number=3):
    _, _, _, _, emask = load_TCGA_arrays(data_path,ID)
    emask = np.squeeze(emask)
    a=np.sum(emask, axis=0)
    a= np.sum(a,axis=0)
    indexes = np.argsort(a)[-need_number:]
    repeat_ID = np.repeat(ID, len(indexes))
    return repeat_ID, indexes'''

def get_JC_axialSampling_list(data_path, ID, need_number=3):
    _, _, _, _, emask, _, = load_JC_arrays(data_path,ID)
    emask = np.squeeze(emask)
    a=np.sum(emask, axis=0)
    a= np.sum(a,axis=0)
    center= np.argsort(a)[-1]
    slice_width = need_number//2
    start = center -(slice_width +1 )
    end  = center +slice_width
    indexes =np.array(range(start,end))
    repeat_ID = np.repeat(ID, len(indexes))
    return repeat_ID, indexes

def get_TCGA_axialSampling_list(data_path, ID, need_number=3):
    _, _, _, _, emask = load_TCGA_arrays(data_path,ID)
    emask = translate_mask(emask)
    emask = np.squeeze(emask)
    a=np.sum(emask, axis=0)
    a= np.sum(a,axis=0)
    center= np.argsort(a)[-1]
    slice_width = need_number//2
    start = center -(slice_width +1 )
    end  = center +slice_width
    indexes =np.array(range(start,end))
    repeat_ID = np.repeat(ID, len(indexes))
    return repeat_ID, indexes

def get_sampling_list(data_path, dataframe,need_number=3, is_TCGA = False):
    if is_TCGA:
        IDs = dataframe['BraTS_2019_subject_ID']
        sample_method = get_TCGA_axialSampling_list
    else:
        IDs = dataframe['ID']
        sample_method = get_JC_axialSampling_list
    repeat_names = []
    slices = [] 
    for name in IDs:
        rname, slice_number = sample_method(data_path, name, need_number)
        repeat_names.append(rname)
        slices.append(slice_number)
        name_array = np.array(repeat_names).flatten()
        slice_array =np.array(slices).flatten()
    sampled_list = list(zip(name_array, slice_array))
    return sampled_list


class TCGA_Dataset_for_view(Dataset):
    """TCGA_Dataset"""
    
    def __init__(self, df, image_dir, sampling_tech,imtype, transform = None):
        """
        Args:
           csv_file(string):Path to the csv files with annotations
           image_dir(string):Directory with all images.
           mask_dir(string):Directory with mask
           sampling_tech:Type of Sampling techniques, 
                         axial_sampling or coronal_sampling or sagital_sampling
           transform(callable, optional):Optional transform to be applied on a sample
        """
        self.imtype = imtype
        self.Training_frames = df
        self.image_dir = image_dir
        self.sampling_tech = sampling_tech
        self.transform = transform
        
    def __len__(self):
        return len(self.Training_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        target = self.Training_frames.iloc[idx,:]
        img_name =target['BraTS_2019_subject_ID']
        FLAIR, T1, T1CE, T2, mask = load_TCGA_arrays(self.image_dir, img_name,SZ=False)
 
        
        con_image = np.concatenate([FLAIR, T1, T1CE,T2])
        
        T2_mask = translate_mask(mask,translate_type='T2')
        T2_mask_r = T2_mask.reshape(T2_mask.shape[1],T2_mask.shape[2],T2_mask.shape[3])
        
        sampled_image = self.sampling_tech(con_image, T2_mask_r)
        sampled_image = sampled_image.transpose((0,3,1,2))
        if self.imtype == 'FLAIR':
            sampled_image =sampled_image[0]
        elif self.imtype =='T1':
            sampled_image = sampled_image[1]
        elif self.imtype =='T1CE':
            sampled_image = sampled_image[2]
        elif self.imtype =='T2':
            sampled_image = sampled_image[3]
        
        sample = {'Sampled_image':sampled_image,'IDH_status':target['IDH1_2'],
                 'Age':target['age_at_initial_pathologic_diagnosis'],
                 'Gender':target['gender'],'Race':target['race'],
                 'Patho':target['histological_type'],
                 'Name':img_name}
        if self.transform:
            sample = self.transform(sample)
            
        return sample 

class ToTensor(object):
    def __init__(self):
        self.norm = transforms.Normalize([0.154518, 0.161404, 0.131060, 0.110935], [0.252240, 0.257210, 0.210441, 0.183151])
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        dict_g = {'FEMALE': 0, 'MALE': 1}
        dict_h = {'Astrocytoma': 0,'Glioblastoma': 1,'Oligoastrocytoma': 2,'Oligodendroglioma': 3}
        dict_r = {'BLACK OR AFRICAN AMERICAN': 0, 'WHITE': 1, '[Not Available]': 2}
        sampled_image, IDH_status,age= sample['Sampled_image'],sample['IDH_status'],sample['Age']
        Gender, Race, Patho,Name=dict_g[sample['Gender']], dict_r[sample['Race']],dict_h[sample['Patho']],sample['Name'] 
        
        images = torch.from_numpy(sampled_image)
        #images = self.norm(images)
        imag_size =images.size()
        images = images.float()
        #images = torch.rot90(images, 2,(1,2)) # for vizualization rotate 180 degree
        #images = torch.rot90(images, 3,(1,2))
        return images,(torch.tensor(float(age)).log_()/4.75,torch.tensor(Gender),torch.tensor(Patho),torch.tensor(IDH_status)) 




class JC_Dataset_for_view(Dataset):
    """JC_Dataset"""
    
    def __init__(self, df, image_dir, sampling_tech,imtype,transform = None):
        """
        Args:
           csv_file(string):Path to the csv files with annotations
           image_dir(string):Directory with all images.
           mask_dir(string):Directory with mask
           sampling_tech:Type of Sampling techniques, 
                         axial_sampling or coronal_sampling or sagital_sampling
           transform(callable, optional):Optional transform to be applied on a sample
        """
        self.imtype = imtype
        self.Training_frames = df
        self.image_dir = image_dir
        self.sampling_tech = sampling_tech
        self.transform = transform
        
    def __len__(self):
        return len(self.Training_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        target = self.Training_frames.iloc[idx,:]
        img_name =target['ID']
        #img_name = img_name.astype('str')
        FLAIR, T1, T1CE, T2, T2_mask,GD_mask = load_JC_arrays(self.image_dir, img_name,Z_score=False)
        
        con_image = np.concatenate([FLAIR, T1, T1CE,T2])
        
        T2_mask_r = T2_mask.reshape(T2_mask.shape[1],T2_mask.shape[2],T2_mask.shape[3])
        
        sampled_image = self.sampling_tech(con_image, T2_mask_r)
        sampled_image = sampled_image.transpose((0,3,1,2))
        if self.imtype == 'FLAIR':
            sampled_image =sampled_image[0]
        elif self.imtype =='T1':
            sampled_image = sampled_image[1]
        elif self.imtype =='T1CE':
            sampled_image = sampled_image[2]
        elif self.imtype =='T2':
            sampled_image = sampled_image[3]
        
        sample = {'Sampled_image':sampled_image,'IDH_status':target['IDH1_2'],
                 'Age':target['age_at_initial_pathologic_diagnosis'],
                 'Gender':target['gender'],
                 'Patho':target['histological_type'],
                 'Name':img_name}
        if self.transform:
            sample = self.transform(sample)
        
        return sample 

class ToTensor_n(object):
    def __init__(self):
        self.norm = transforms.Normalize([0.154518, 0.161404, 0.131060, 0.110935], [0.252240, 0.257210, 0.210441, 0.183151])
    
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        dict_gn = {'F': 0, 'M': 1}
        dict_hn = {'DA': 0,'GBM': 1,'OL': 3,'AA': 4,'AO': 5, 'FALSE':6}
        dict_rn = {'BLACK OR AFRICAN AMERICAN': 0, 'WHITE': 1, '[Not Available]': 2}
        sampled_image, IDH_status,age= sample['Sampled_image'],sample['IDH_status'],sample['Age']
        Gender, Race, Patho,Name=dict_gn[sample['Gender']], 2,dict_hn[sample['Patho']],sample['Name'] 
        
        images = torch.from_numpy(sampled_image)
        #images = self.norm(images)
        imag_size =images.size()
        #images = images.reshape(imag_size[0]*imag_size[1], imag_size[2],imag_size[3]).float()
        images = images.float()
        
        return images,(torch.tensor(float(age)).log_()/4.75,torch.tensor(Gender),torch.tensor(Patho),torch.tensor(IDH_status))


def r_number_list (pd,id_names='BraTS_2019_subject_ID', end_sliec=130, need_number = 1500):
    per_n = need_number // len(pd) +1
    r_numbers = np.random.randint(0, end_sliec +1, (len(pd),per_n))
    name_list = list(pd[id_names])
    name_list_flat = np.repeat(name_list,per_n)
    target_list = list(zip(name_list_flat,r_numbers.flatten() ))
    selected_list = random.sample(target_list, need_number)
    return selected_list

def array_to_itk(array):
    if array.ndim == 4:
        array = np.squeeze(array)
        array = array.transpose(2,0,1)
    else:
        pass
    itk = sitk.GetImageFromArray(array)
    return itk

def concat_JC_and_TCGA(cdf,JCdf, TCGAdf):
    radiomics_features = pd.concat([JCdf, TCGAdf])
    radiomics_features = radiomics_features.rename(columns={'Unnamed: 0': "ID"})
    radiomics_features['ID'] = radiomics_features['ID'].astype(str)
    radiomics_features.index = np.arange(0,len(radiomics_features))
    target_df = pd.merge(radiomics_features, cdf, on='ID')
    return target_df

def values_and_label(df,forEnsemble=False):
    data = df.iloc[:,6:].values
    label = df['IDH1_2'].values
    Ensemblelabel =label[::3]
    if forEnsemble:
        return data, label,Ensemblelabel
    else:
        return data, label