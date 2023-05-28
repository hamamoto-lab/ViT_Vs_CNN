import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch.optim as optim
from multi_center_tools.multi_tools import load_npz, loadArray
from multi_center_tools.Cycle_tool import Generator


act_fn_by_name = {
    "tann":nn.Tanh,
    "relu":nn.ReLU,
    "leakyrelu":nn.LeakyReLU,
    "gelu":nn.GELU
}

class TCGA_dis_NCC_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir,A_list, B_list,itype, is_train = True):
        super(torch.utils.data.Dataset,self).__init__()
        
        self.image_type = itype
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
        ind= index//2
        label = index %2
        if label ==0:
            name_A = self.A_list[ind]
            array_A = self.getJCArray(self.image_paths_A, name_A[0], name_A[1],self.image_type)
            array= np.expand_dims(array_A,0)
        #May be loss some information.
        #img_A= Image.fromarray(np.uint8(array_A * 255), 'L')
        
        if label ==1:
            name_B = self.B_list[ind]
            array_B = self.getTCGAArray(self.image_paths_B, str(name_B[0]), name_B[1], self.image_type)
            array = np.expand_dims(array_B,0)
        #May be loss some information.
        #img_B = Image.fromarray(np.uint8(array_B * 255), 'L')
        
        # Data expansion
        tensor = torch.from_numpy(array.astype(np.float32)).clone()
        tensor = self.transform(tensor)
        
        return tensor, label
    
    def __len__(self):
        return min(self.size_A, self.size_B)*2
    
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
        #transform_list.append(transforms.Normalize(( 0.5),(0.5))) # [0, 1] => [-1, 1]
        return transforms.Compose(transform_list)


class OriginDisModule(pl.LightningModule):
    
    def __init__(self, model,  optimizer_name, optimizer_hparams):
        """
        Inputs:
              optimizer_name - Name of the optimzer to use. Currently supported: Adam, SGD
              optimizer_hparams - Hyperparameters for the optimizer, as dictiornay. This includes learning rate weight decay, etc.
        """
        super().__init__()
        #Exports the hyperparameters to YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        #Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1,1,176,192), dtype=torch.float32)
        
    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)
    
    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            #AdamW is Adam with a correct implementation of weight decay
            optimizer = optim.AdamW(
            self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(),**self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: \"{self.hparams.optimizer_name}\""
        # We will redudce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3,total_steps=600)
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, label = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, label)
        acc = (preds.argmax(dim =-1) == label).float().mean()
        
        #Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_acc',acc,on_step=False, on_epoch = True)
        self.log('train_loss', loss)
        return loss # Return tensor to call ".backward" on
    
    def validation_step(self, batch, batch_idx):
        imgs, label = batch
        preds = self.model(imgs).argmax(dim =-1)
        acc = (label == preds).float().mean()
        # By default logs it per epoch (weighted average over batches)
        self.log('val_acc', acc)
        
    def test_step(self, batch, batch_idx):
        imgs, label = batch
        preds = self.model(imgs).argmax(dim =-1)
        acc = (label == preds).float().mean()
        # By the default logs it per epoch(weight average over batches), and return it afterwards
        self.log('test_acc', acc)

class get_origin_result:
    def __init__(self,data_path,lists, itype, model, label):
        self.data_path = data_path
        self.lists = lists
        self.itype = itype
        self.model = model
        self.label = label
        
    def load_npz(self,dirc):
        return np.load(dirc)['arr_0']

    def loadArray(self,data_path, sample_ID, slice_n, itype):
        if itype  == "FLAIR":
            img= self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
        elif itype == 'T1':
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
        elif itype == 'T1CE':
            img= self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
        elif itype == 'T2':
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
        else:
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_mask.npz')
        return img[:,:,:,slice_n]
    
    def getJCArray(self,data_path, sample_ID, slice_n, itype):
        array = self.loadArray(data_path, sample_ID, slice_n,itype)
        array = np.squeeze(array)
        array = torch.from_numpy(array.astype(np.float32)).clone()
        array = torch.unsqueeze(array,0)
        return array
    def __call__(self, *args,**kwargs):
        result = []
        softM = nn.Softmax()
        for exs in self.lists:
            image_tensor=self.getJCArray(data_path=self.data_path, sample_ID=exs[0], slice_n=exs[1], itype=self.itype)
            result_tensor = self.model(torch.unsqueeze(image_tensor,0))
            result_tensor = softM(result_tensor)
            result_array = result_tensor.to('cpu').detach().numpy().copy()
            result.append(result_array[0])
        result = np.array(result)
        preds = np.argmax(result, axis=1)
        labels =  np.repeat(self.label, len(result))
        accuracy = np.mean(preds == labels)
        uncertainty = np.mean(result[:,self.label])
        return accuracy, uncertainty

"""class get_origin_result_with_converter:
    def __init__(self,data_path,lists, itype, model, net_name,label):
        self.data_path = data_path
        self.lists = lists
        self.itype = itype
        self.model = model
        self.label = label
        self.net_name = net_name
        
    def load_npz(self,dirc):
        return np.load(dirc)['arr_0']

    def loadArray(self,data_path, sample_ID, slice_n, itype):
        if itype  == "FLAIR":
            img= self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
        elif itype == 'T1':
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
        elif itype == 'T1CE':
            img= self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
        elif itype == 'T2':
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
        else:
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_mask.npz')
        return img[:,:,:,slice_n]
    
    def getJCArray(self,data_path, sample_ID, slice_n, itype):
        array = self.loadArray(data_path, sample_ID, slice_n,itype)
        array = np.squeeze(array)
        array = torch.from_numpy(array.astype(np.float32)).clone()
        array = torch.unsqueeze(array,0)
        return array
    def convert_TCGA_style(self,baseimg):
        '''
        input tensor (1,176,192)
        output tensor(1,176,192)
        '''
        gen = Generator()
        gen.load_state_dict(torch.load(self.net_name))
        gen.eval()
        tensor = baseimg.view(1,1,176,192)
        #tensor = torch.from_numpy(baseimg.reshape(1,1,176,192).astype(np.float32))
        conv_img = gen(tensor)
        conv_img = conv_img.view(1,176,192)
        return conv_img

    def __call__(self, *args,**kwargs):
        result = []
        softM = nn.Softmax()
        for exs in self.lists:
            image_tensor=self.getJCArray(data_path=self.data_path, sample_ID=exs[0], slice_n=exs[1], itype=self.itype)
            image_tensor = self.convert_TCGA_style(image_tensor) #convert image style
            result_tensor = self.model(torch.unsqueeze(image_tensor,0))
            result_tensor = softM(result_tensor)
            result_array = result_tensor.to('cpu').detach().numpy().copy()
            result.append(result_array[0])
        result = np.array(result)
        preds = np.argmax(result, axis=1)
        labels =  np.repeat(self.label, len(result))
        accuracy = np.mean(preds == labels)
        uncertainty = np.mean(result[:,self.label])
        return accuracy, uncertainty"""

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


class get_converted_arrys():
    def __init__(self,base_dir, instName,suffix,net_name):
        self.base_dir = base_dir
        self.instName = instName
        self.suffix = suffix
        self.net_name = net_name

    def convert_TCGA_style(self,baseimg,imgType):
        gen = Generator()
        gen.load_state_dict(torch.load(os.path.join(self.base_dir,'log_' + self.instName + imgType + self.suffix, self.net_name)))
        gen.eval()
        tensor = torch.from_numpy(baseimg.reshape(3,1,176,192).astype(np.float32))
        conv_img = gen(tensor)
        conv_img = conv_img.view(3,176,192)
        return conv_img

    def __call__(self, sampled_image):
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
            conv_images = self.convert_TCGA_style(image,tnames[i])
            empty.append(conv_images)
        stacked = torch.stack([empty[0],empty[1],empty[2],empty[3]])
        stacked = (stacked +1)/2 #-1 to1 -> 0to1
        return stacked.to('cpu').detach().numpy().copy()


class TCGA_dis_NCC_R_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir,A_list, B_list,itype, is_train = True):
        super(torch.utils.data.Dataset,self).__init__()
        
        self.image_type = itype
        dir_A = os.path.join(root_dir, 'FR_BraTS2019_array')
        dir_B = os.path.join(root_dir, 'JC_array')
            
        self.image_paths_A = dir_A
        self.image_paths_B = dir_B
        
        self.A_list = A_list
        self.B_list = B_list
        
        self.size_A = len(self.A_list)
        self.size_B = len(self.B_list)
        
        self.transform =self._make_transform(is_train)
        
    def __getitem__(self, index):
        ind= index//2
        label = index %2
        if label ==0:
            name_A = self.A_list[ind]
            array_A = self.getJCArray(self.image_paths_A, name_A[0], name_A[1],self.image_type)
            array= np.expand_dims(array_A,0)
        #May be loss some information.
        #img_A= Image.fromarray(np.uint8(array_A * 255), 'L')
        
        if label ==1:
            name_B = self.B_list[ind]
            array_B = self.getTCGAArray(self.image_paths_B, str(name_B[0]), name_B[1], self.image_type)
            array = np.expand_dims(array_B,0)
        #May be loss some information.
        #img_B = Image.fromarray(np.uint8(array_B * 255), 'L')
        
        # Data expansion
        tensor = torch.from_numpy(array.astype(np.float32)).clone()
        tensor = self.transform(tensor)
        
        return tensor, label
    
    def __len__(self):
        return min(self.size_A, self.size_B)*2
    
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
        #transform_list.append(transforms.Normalize(( 0.5),(0.5))) # [0, 1] => [-1, 1]
        return transforms.Compose(transform_list)


def get_origin_result_use_converter(dataloader, predicitor,label,converter=None):
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    result = []
    softM = nn.Softmax()
    for img in dataloader:
        img = img.to(DEVICE)
        predicitor = predicitor.to(DEVICE)

        if converter is None:
            conv_img =img.view(-1,1,176,192)
        else :
            converter = converter.to(DEVICE)
            img = img.view(-1,1,176,192)
            img =img *2 -1 #0to1 -> -1to1 
            conv_img =converter(img)
            conv_img = (conv_img +1)/2  # [-1,1] => [0,1]
        
        result_tensor = predicitor(conv_img)
        result_tensor = softM(result_tensor)
        result_array = result_tensor.to('cpu').detach().numpy().copy()
        result.append(result_array[0])
    
    result = np.array(result)
    preds = np.argmax(result, axis=1)
    labels =  np.repeat(label, len(result))
    accuracy = np.mean(preds == labels)
    uncertainty = np.mean(result[:,label])
    return accuracy, uncertainty

class img_from_list(Dataset):
    def __init__(self, data_path, image_list, itype, transfrom=None):
        self.data_path = data_path
        self.image_list = image_list
        self.itype = itype
        self.transfrom = transforms
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self,idx):
        exs = self.image_list[idx]
        image_tensor=self.getJCArray(data_path=self.data_path, sample_ID=exs[0], slice_n=exs[1], itype=self.itype)
        return image_tensor
    
    def load_npz(self,dirc):
        return np.load(dirc)['arr_0']
    
    def loadArray(self,data_path, sample_ID, slice_n, itype):
        if itype  == "FLAIR":
            img= self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_FLAIR.npz')
        elif itype == 'T1':
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1.npz')
        elif itype == 'T1CE':
            img= self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T1CE.npz')
        elif itype == 'T2':
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_T2.npz')
        else:
            img = self.load_npz(data_path + '/' + sample_ID  + '/' + sample_ID + '_mask.npz')
        return img[:,:,:,slice_n]
    
    def getJCArray(self,data_path, sample_ID, slice_n, itype):
        array = self.loadArray(data_path, sample_ID, slice_n,itype)
        array = np.squeeze(array)
        array = torch.from_numpy(array.astype(np.float32)).clone()
        array = torch.unsqueeze(array,0)
        
        return array
