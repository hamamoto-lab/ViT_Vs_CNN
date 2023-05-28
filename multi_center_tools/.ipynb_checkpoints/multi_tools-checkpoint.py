import numpy as np
import pickle
import matplotlib.pyplot as plt

def ch3_to_GD(label_3ch):
    GD=label_3ch[0,:,:,:]+label_3ch[2,:,:,:]
    GD=np.where(GD>0.9, 1, 0)
    return GD

def ch3_to_t2(label_3ch):
    t2=label_3ch[0,:,:,:]+label_3ch[1,:,:,:]+label_3ch[2,:,:,:]
    t2=np.where(t2>0.9, 1, 0)
    return t2

def axial_sampling(img,mask,n_sample=3):
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
    return np.stack(base, -1)


def coronal_sampling(img,mask,n_sample=3):
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
    return np.stack(base, -1)


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

#画像のzスコア化
def zscore(img, mean,std):
    return (img - mean) / std

def sample_z(img):
    mean = img.mean()
    std = img.std()
    return (img - mean) / std

def load_JC_arrays(data_path, sample_ID, Z_score = False, sample_z=False):
    
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
    
    if sample_z ==True:
        FLAIR = sample_z(FLAIR)
        T1 = sample_z(T1)
        T1CE = sample_z(T1CE)
        T2= sample_z(T2)
    
    return FLAIR, T1, T1CE, T2, T2_mask,GD_mas
    
        

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