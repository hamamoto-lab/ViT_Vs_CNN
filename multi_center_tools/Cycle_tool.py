import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torchvision import transforms, utils
import random
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

def load_npz(dirc):
    return np.load(dirc)['arr_0']

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

class get_converted_arrys():
    def __init__(self,base_dir, instName,suffix,net_name,separation=False):
        self.base_dir = base_dir
        self.instName = instName
        self.suffix = suffix
        self.net_name = net_name
        self.separation = separation

    def convert_TCGA_style(self,baseimg,imgType):
        gen = Generator()
        #gen.load_state_dict(torch.load(os.path.join(self.base_dir,'log_' + self.instName + imgType + self.suffix, self.net_name)))
        gen.load_state_dict(torch.load(os.path.join(self.base_dir,imgType,'log_' + self.instName + imgType + self.suffix, self.net_name)))
        gen.eval()
        baseimg = baseimg*2 -1 #0to1 -> -1to1
        tensor = torch.from_numpy(baseimg.reshape(3,1,176,192).astype(np.float32))
        conv_img = gen(tensor)
        conv_img = conv_img.view(3,176,192)
        conv_img = (conv_img +1)/2 #-1 to1 -> 0to1
        return conv_img

    def __call__(self, sampled_image):
        ''''
        input 0-1 ranged image
        imputimage(imagetype, slice, h, w) example(4,3,176,192)
        output 0-1 rqnaged image
        '''
        #sampled_image = sampled_image *2 -1 #0to1 -> -1to1 
        sFLAIRs = sampled_image[0,:,:,:]
        sT1s = sampled_image[1,:,:,:]
        sT1CEs = sampled_image[2,:,:,:]
        sT2s= sampled_image[3,:,:,:]
        images = [sFLAIRs, sT1s,sT1CEs,sT2s]
        tnames =['FLAIR', 'T1', 'T1CE','T2']
        empty= []
        for i, image in enumerate(images):
            conv_images = self.convert_TCGA_style(image,tnames[i])
            empty.append(conv_images)
        if self.separation:
            #FLAIR = (empty[0] +1)/2 #-1 to1 -> 0to1
            #T1 = (empty[1] +1)/2 #-1 to1 -> 0to1
            #T1CE = (empty[2] +1)/2 #-1 to1 -> 0to1
            #T2 = (empty[3] +1)/2 #-1 to1 -> 0to1

            FLAIR = empty[0]
            T1 = empty[1]
            T1CE = empty[2]
            T2 = empty[3]
            FLAIR = FLAIR.to('cpu').detach().numpy().copy()
            T1 = T1.to('cpu').detach().numpy().copy()
            T1CE = T1CE.to('cpu').detach().numpy().copy()
            T2 = T2.to('cpu').detach().numpy().copy()
            return FLAIR, T1, T1CE, T2
        else:
            stacked = torch.stack([empty[0],empty[1],empty[2],empty[3]])
            #stacked = (stacked +1)/2 #-1 to1 -> 0to1
            return stacked.to('cpu').detach().numpy().copy()


class TCGA_To_NCC_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir,A_list, B_list,itype, is_train = True):
        super(torch.utils.data.Dataset,self).__init__()
        
        self.image_type = itype
        if is_train:
            dir_A = os.path.join(root_dir, 'JC_array')
            dir_B = os.path.join(root_dir, 'FR_BraTS2019_array')
        else:
            dir_A = os.path.join(root_dir, 'JC_array')
            dir_B = os.path.join(root_dir, 'FR_BraTS2019_array')
            
        self.image_paths_A = dir_A
        self.image_paths_B = dir_B
        
        self.A_list = A_list
        self.B_list = B_list
        
        self.size_A = len(self.A_list)
        self.size_B = len(self.B_list)
        
        self.transform =self._make_transform(is_train)
        
    def __getitem__(self, index):
        index_A = index
        name_A = self.A_list[index_A]
        array_A = self.getJCArray(self.image_paths_A, name_A[0], name_A[1],self.image_type)
        array_A = np.expand_dims(array_A,0)
        #May be loss some information.
        #img_A= Image.fromarray(np.uint8(array_A * 255), 'L')
        
        
        # Images B belongins class B is selected randomly
        index_B = random.randint(0, self.size_B -1)
        name_B = self.B_list[index_B]
        array_B = self.getTCGAArray(self.image_paths_B, str(name_B[0]), name_B[1], self.image_type)
        array_B = np.expand_dims(array_B,0)
        #May be loss some information.
        #img_B = Image.fromarray(np.uint8(array_B * 255), 'L')
        
        # Data expansion
        A = torch.from_numpy(array_A.astype(np.float32)).clone()
        B = torch.from_numpy(array_B.astype(np.float32)).clone()
        A = self.transform(A)
        B = self.transform(B)
        
        return{'A':A, 'B':B, 'name_A':name_A, 'name_B':name_B}
    
    def __len__(self):
        return max(self.size_A, self.size_B)
    
    def loadArray(self,data_path, sample_ID, slice_n, itype):
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
   
    def getJCArray(self,data_path, sample_ID, slice_n, itype):
        array = self.loadArray(data_path, sample_ID, slice_n,itype)
        #array = np.rot90(np.squeeze(array),1)
        array = np.squeeze(array)
        return array
   
    def getTCGAArray(self,data_path, sample_ID, slice_n, itype):
        array = self.loadArray(data_path, sample_ID,slice_n,itype)
        #array = np.rot90(np.squeeze(array),3)
        array = np.squeeze(array)
        return array

    
    def _make_transform(self, is_train):
        transform_list = []
        #transform_list.append(transforms.Resize((load_size,load_size), Image.BICUBIC))
        #transform_list.append(transforms.RandomCrop(fine_size))
        if is_train:
            transform_list.append(transforms.RandomHorizontalFlip())
        #transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(( 0.5),(0.5))) # [0, 1] => [-1, 1]
        return transforms.Compose(transform_list)


'''class get_converted_arrys():
    def __init__(self,base_dir, instName,suffix,net_name,separation=False):
        self.base_dir = base_dir
        self.instName = instName
        self.suffix = suffix
        self.net_name = net_name
        self.separation = separation

    def convert_TCGA_style(self,baseimg,imgType):
        gen = Generator()
        gen.load_state_dict(torch.load(os.path.join(self.base_dir,'log_' + self.instName + imgType + self.suffix, self.net_name)))
        gen.eval()
        tensor = torch.from_numpy(baseimg.reshape(3,1,176,192).astype(np.float32))
        conv_img = gen(tensor)
        conv_img = conv_img.view(3,176,192)
        return conv_img

    def __call__(self, sampled_image):

        sampled_image = sampled_image *2 -1 #0to1 -> -1to1 
        sFLAIRs = sampled_image[0,:,:,:]
        sT1s = sampled_image[1,:,:,:]
        sT1CEs = sampled_image[2,:,:,:]
        sT2s= sampled_image[3,:,:,:]
        images = [sFLAIRs, sT1s,sT1CEs,sT2s]
        tnames =['FLAIR', 'T1', 'T1CE','T2']
        empty= []
        for i, image in enumerate(images):
            conv_images = self.convert_TCGA_style(image,tnames[i])
            empty.append(conv_images)
        if self.separation:
            FLAIR = (empty[0] +1)/2 #-1 to1 -> 0to1
            T1 = (empty[1] +1)/2 #-1 to1 -> 0to1
            T1CE = (empty[2] +1)/2 #-1 to1 -> 0to1
            T2 = (empty[3] +1)/2 #-1 to1 -> 0to1
            FLAIR = FLAIR.to('cpu').detach().numpy().copy()
            T1 = T1.to('cpu').detach().numpy().copy()
            T1CE = T1CE.to('cpu').detach().numpy().copy()
            T2 = T2.to('cpu').detach().numpy().copy()
            return FLAIR, T1, T1CE, T2
        else:
            stacked = torch.stack([empty[0],empty[1],empty[2],empty[3]])
            stacked = (stacked +1)/2 #-1 to1 -> 0to1
            return stacked.to('cpu').detach().numpy().copy()'''