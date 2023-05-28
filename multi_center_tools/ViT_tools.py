import numpy  as np
import pandas as pd
from multi_center_tools.multi_tools import translate_mask, load_npz
from multi_center_tools.IDH_tools import dict_inst,dict_g, dict_gn, dict_h, dict_hn, dict_r, dict_rn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
from multi_center_tools.multi_tools import load_TCGA_arrays,translate_mask, load_JC_arrays
import matplotlib.pyplot as plt


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

def imshow_vit(img, cmap='gray'):
    npimg = img.numpy()
    npimg=img
    npimg = 0.5 * (npimg + 1) # [-1,1] => [0,1]
    npimg = np.squeeze(npimg)
    plt.imshow(npimg,cmap=cmap)

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
                 inst=None, double = False, padding = True, paded_size = 224, inclued_mask = False,
                FLAIR=True, T1=True, T1CE=True, T2=True):
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
        self.FLAIR = FLAIR
        self.T1 = T1
        self.T1CE = T1CE
        self.T2 = T2

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
        if self.inclued_mask and self.FLAIR and self.T1 and self.T1CE and self.T2:
            con_image = np.concatenate([FLAIR, T1, T1CE,T2,T2_mask])
        
        #all images
        elif ~self.inclued_mask and self.FLAIR and self.T1 and self.T1CE and self.T2:
            con_image = np.concatenate([FLAIR, T1, T1CE,T2])

        # FLAIR, T1,T1CE
        elif ~self.inclued_mask and self.FLAIR and self.T1 and self.T1CE and ~self.T2:
            con_image = np.concatenate([FLAIR, T1, T1CE])

        # FLAIR, T1,T2
        elif ~self.inclued_mask and self.FLAIR and self.T1 and ~self.T1CE and self.T2:
            con_image = np.concatenate([FLAIR, T1, T2])

        # FLAIR,T1CE,T2
        elif ~self.inclued_mask and self.FLAIR and ~self.T1 and self.T1CE and self.T2:
            con_image = np.concatenate([FLAIR, T1CE,T2])
        # T1,T1CE,T2
        elif ~self.inclued_mask and ~self.FLAIR and self.T1 and self.T1CE and self.T2:
            con_image = np.concatenate([T1, T1CE,T2])
       
        
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
                 inst=None, double = False, padding = True, paded_size = 224, inclued_mask = False,
                 FLAIR=True, T1=True, T1CE=True, T2=True):
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
        self.FLAIR = FLAIR
        self.T1 = T1
        self.T1CE = T1CE
        self.T2 = T2
        
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
        
        if self.inclued_mask and self.FLAIR and self.T1 and self.T1CE and self.T2:
            con_image = np.concatenate([FLAIR, T1, T1CE,T2,T2_mask])
        
        #all images
        elif ~self.inclued_mask and self.FLAIR and self.T1 and self.T1CE and self.T2:
            con_image = np.concatenate([FLAIR, T1, T1CE,T2])

        # FLAIR, T1,T1CE
        elif ~self.inclued_mask and self.FLAIR and self.T1 and self.T1CE and ~self.T2:
            con_image = np.concatenate([FLAIR, T1, T1CE])

        # FLAIR, T1,T2
        elif ~self.inclued_mask and self.FLAIR and self.T1 and ~self.T1CE and self.T2:
            con_image = np.concatenate([FLAIR, T1, T2])

        # FLAIR,T1CE,T2
        elif ~self.inclued_mask and self.FLAIR and ~self.T1 and self.T1CE and self.T2:
            con_image = np.concatenate([FLAIR, T1CE,T2])
        # T1,T1CE,T2
        elif ~self.inclued_mask and ~self.FLAIR and self.T1 and self.T1CE and self.T2:
            con_image = np.concatenate([T1, T1CE,T2])
        
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
    

def extract(model, target, inputs):
    feature = None
    
    def forward_hook(module, inputs, outputs):
        global features
        features = outputs.detach().clone()
    
    # Register callback func
    handle = target.register_forward_hook(forward_hook)
    
    # predict
    model.eval()
    model(inputs)
    
    # Release callback func
    handle.remove()
    
    return features

class GetAttentionWeights(nn.Module):
    def __init__(self, dim, qkv, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = qkv
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return attn

def get_joint_aug_attentions(attentions):
    att_mat = torch.stack(attentions).squeeze(1)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]
    #joint_attentions[0] = att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    return joint_attentions, aug_att_mat

def get_vit_small_patch16_22_attentions(timm_model, target_tensor):
    attentions = []
    for i in range(len(timm_model.blocks)):
        target_module = timm_model.blocks[i].norm1
        qkv = timm_model.blocks[i].attn.qkv
        features = extract(timm_model, target_module, target_tensor.unsqueeze(0))
        #Define  attention extractor
        Get_attention = GetAttentionWeights(dim=384, qkv=qkv,num_heads=6)
        attention = Get_attention(features)
        attentions.append(attention)
    return attentions

def get_vit_small_patch16_22_attention_mask(timm_model, target_tensor):
    attentions = get_vit_small_patch16_22_attentions(timm_model, target_tensor)
    joint_attentions, aug_att_mat = get_joint_aug_attentions(attentions)
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    mask = cv2.resize(mask / mask.max(), dsize=(224,224))
    return mask
