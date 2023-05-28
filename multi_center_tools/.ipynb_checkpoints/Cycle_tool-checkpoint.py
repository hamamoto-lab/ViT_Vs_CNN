import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


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

class get_converted_arrys():
    def __init__(self,base_dir, instName,net_name):
        self.base_dir = base_dir
        self.instName = instName
        self.net_name = net_name

    def convert_TCGA_style(self,baseimg,imgType):
        gen = Generator()
        gen.load_state_dict(torch.load(os.path.join(self.base_dir,'log_' + self.instName + imgType, self.net_name)))
        tensor = torch.from_numpy(baseimg.reshape(3,1,176,192).astype(np.float32))
        conv_img = gen(tensor)
        conv_img = conv_img.view(3,176,192)
        return conv_img

    def __call__(self, sampled_image):
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
        return stacked.to('cpu').detach().numpy().copy()