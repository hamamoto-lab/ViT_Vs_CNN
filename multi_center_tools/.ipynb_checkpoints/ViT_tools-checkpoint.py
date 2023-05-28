import numpy  as np
import pandas as pd
from multi_center_tools.multi_tools import translate_mask, load_npz
from multi_center_tools.IDH_tools import dict_inst,dict_g, dict_gn, dict_h, dict_hn, dict_r, dict_rn
import torch
from torch.utils.data import Dataset, DataLoader
from multi_center_tools.multi_tools import load_TCGA_arrays,translate_mask, load_JC_arrays


def load_TCGA_mask(data_path, sample_ID):
    mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_mask.npz')
    return mask

def load_TCGA_FLAIR(data_path, sample_ID):
    FLAIR = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
    return FLAIR

def load_TCGA_T1(data_path, sample_ID):
    T1 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
    return T1

def load_TCGA_T1CE(data_path, sample_ID):
    T1CE = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
    return T1CE

def load_TCGA_T2(data_path, sample_ID):
    T2 = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
    return T2

def load_JCT2_mask(data_path, sample_ID):
    mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2_mask.npz')
    return mask

def load_JCGD_mask(data_path, sample_ID):
    mask = load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_GD_mask.npz')
    return mask


def get_ViT_series(pd, _width =2):
    #input dataframe with tumor center index
    #The last column is 'tumor_index'  
    _serieses = []
    if not _width == 0:
        for i in range(len(pd)):
            #lower loop
            for w in range(1, _width+1):
                _serise = pd.iloc[i, :]
                _serise [-1] = _serise [-1] - w
                _serieses.append(_serise)
            
            #center
            _serieses.append(pd.iloc[i, :])
            
            #lower loop
            for w in range(1, _width+1):
                _serise = pd.iloc[i, :]
                _serise [-1] = _serise [-1] + w
                _serieses.append(_serise)
        #pandas error? I can't output pandas dataframe
        #result = pd.DataFrame(_serieses)
    else:
         #center
         for i in range(len(pd)):
            _serieses.append(pd.iloc[i, :])
    return _serieses

def axial_samplingIndex(mask):
    """
    sampling axial cutting image that contatin top 1
    retun: axial index(max tumor volume)
    mask numpy array (W,D,H)
    """
    a=np.sum(mask, axis=0)
    a= np.sum(a,axis=0)
    idexs=np.argsort(a)
    idex = idexs[-1]
    return idex

def get_center_indexes(pd,img_loc, ID_name ='BraTS_2019_subject_ID', JCT2 = False, JCGD = False):
    #get tumor center_indexs from mask
    tumor_indexes = []
    for i in range(len(pd)):
        tumorid = pd[ID_name][i]
        if JCT2:
            _mask =load_JCT2_mask(img_loc, tumorid)
        elif JCGD:
            _mask =load_JCGD_mask(img_loc, tumorid)
        else:
            _mask =load_TCGA_mask(img_loc, tumorid)
            _mask = translate_mask(_mask)
        _center_idx = axial_samplingIndex(_mask[0])
        tumor_indexes.append(_center_idx)
    return tumor_indexes

###pamds bug??? not working now
def get_ViT_pd(pd,img_loc, ID_name ='BraTS_2019_subject_ID', width=2):
    pd['tumor_index'] = get_center_indexes(train_pd, img_loc, ID_name)
    v_train_pd = pd.DataFrame(get_ViT_series(pd, width))
    v_train_pd.reset_index()
    return v_train_pd


class Vit_TCGA_Dataset(Dataset):
    """TCGA_Dataset"""
    
    def __init__(self, df, image_dir,transform = None, z_score = False, sz=False,
                 inst=None, double = False, padding = True, paded_size = 224, inclued_mask = False):
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
        self.transform = transform
        self.z_score =z_score
        self.sz = sz
        self.inst = inst
        self.double = double
        self.padding = padding
        self.paded_size = paded_size
        self.inclued_mask = inclued_mask

    def __len__(self):
        return len(self.Training_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        target = self.Training_frames.iloc[idx,:]
        img_name =target['BraTS_2019_subject_ID']
        tumor_index = target['tumor_index']
        FLAIR, T1, T1CE, T2, mask = load_TCGA_arrays(self.image_dir, img_name,Z_score=self.z_score, 
                                                     SZ=self.sz,inst=self.inst, double = self.double,
                                                    padding=self.padding, paded_size=self.paded_size)
        
        T2_mask = translate_mask(mask,translate_type='T2')
        if self.inclued_mask :
            con_image = np.concatenate([FLAIR, T1, T1CE,T2,T2_mask])
        else:
            con_image = np.concatenate([FLAIR, T1, T1CE,T2])
        
        sampled_image = con_image[:,:,:,tumor_index]
        
        
        sample = {'Sampled_image':sampled_image,'IDH_status':target['IDH1_2'],
                 'Age':target['age_at_initial_pathologic_diagnosis'],
                 'Gender':target['gender'],'Race':target['race'],
                 'Patho':target['histological_type'],
                 'Name':img_name}
        if self.transform:
            sample = self.transform(sample)
        
        return sample 


class Vit_JC_Dataset(Dataset):
    """JC_Dataset"""
    
    def __init__(self, df, image_dir,transform = None, z_score = False, sz=False,
                 inst=None, double = False, padding = True, paded_size = 224, inclued_mask = False):
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
        self.transform = transform
        self.z_score =z_score
        self.sz = sz
        self.inst = inst
        self.double = double
        self.padding = padding
        self.paded_size = paded_size
        self.inclued_mask = inclued_mask
        
    def __len__(self):
        return len(self.Training_frames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        target = self.Training_frames.iloc[idx,:]
        img_name =target['ID']
        tumor_index = target['tumor_index']
        #img_name = img_name.astype('str')
        FLAIR, T1, T1CE, T2, T2_mask,GD_mask = load_JC_arrays(self.image_dir, img_name,Z_score=self.z_score, SZ=self.sz,inst=self.inst, double = self.double,
                                                    padding=self.padding, paded_size=self.paded_size)
        
        if self.inclued_mask :
            con_image = np.concatenate([FLAIR, T1, T1CE,T2,T2_mask])
        else:
            con_image = np.concatenate([FLAIR, T1, T1CE,T2])
        
        sampled_image = con_image[:,:,:,tumor_index]
        
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


class VitTCGAToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sampled_image, IDH_status,age= sample['Sampled_image'],sample['IDH_status'],sample['Age']
        Gender, Race, Patho,Name=dict_g[sample['Gender']], dict_r[sample['Race']],dict_h[sample['Patho']],sample['Name'] 
        
        images = torch.from_numpy(sampled_image)
        # timm model float, not double
        images = images.float()
        return images,(torch.tensor(float(age)).log_()/4.75,torch.tensor(Gender),torch.tensor(Patho),torch.tensor(IDH_status))

class VitJCToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sampled_image, IDH_status,age= sample['Sampled_image'],sample['IDH_status'],sample['Age']
        Gender, Race, Patho,inst, Name = dict_gn[sample['Gender']], 2, dict_hn[sample['Patho']], dict_inst[sample['Institution']], sample['Name'] 
        
        images = torch.from_numpy(sampled_image)
        # timm model float, not double
        images = images.float()
        return images,(torch.tensor(float(age)).log_()/4.75,torch.tensor(Gender),torch.tensor(Patho),torch.tensor(IDH_status),
        torch.tensor(inst))