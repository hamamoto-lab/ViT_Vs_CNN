import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import itertools
from torch.autograd import Variable
import os
import random

seed =1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

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

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
        nn.Conv2d(1,64,kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(64,128,kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(128,256,kernel_size=4, stride=2, padding=1),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(256,512,kernel_size=4, stride=1, padding=1),
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(0.2,True),
        nn.Conv2d(512,1,kernel_size=4,stride=1, padding=1)
            
        )
        
        #initial weights
        self.model.apply(self._init_weights)
    
    def forward(self,input):
        return self.model(input)
    
    def _init_weights(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class ImagePool():
    
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
            
    def query(self, images):
        # if a object don't use, return original images
        if self.pool_size == 0:
            return Variable(images)
        return_images = []
        for image in images:
            # Delete batch demention
            image = torch.unsqueeze(image,0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs +1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0,1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size -1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images,0))
        return return_images

class GANLoss(nn.Module):

    def __init__(self):
        super(GANLoss, self).__init__()
        self.real_label_var = None
        self.fake_label_var = None
        self.loss = nn.MSELoss()
        self.cuda= torch.cuda.is_available()

    def get_target_tensor(self, inputs, target_is_real):
        target_tensor = None
        if target_is_real:
            real_tensor = torch.ones(inputs.size())
            if self.cuda:
                real_tensor = real_tensor.cuda()
            self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            fake_tensor = torch.zeros(inputs.size())
            if self.cuda:
                fake_tensor = fake_tensor.cuda()
            self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor
    
    def __call__(self, inputs, target_is_real):
        target_tensor = self.get_target_tensor(inputs, target_is_real)
        return self.loss(inputs, target_tensor)

class CycleGAN(object):
    
    def __init__(self, lr = 0.0002,beta1 = 0.5,lambda_A=10,lambda_B =10, lambda_idt =0.5,log_dir = 'logs'):
        self.netG_A = Generator()
        self.netG_B = Generator()
        self.netD_A = Discriminator()
        self.netD_B = Discriminator()
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_idt =lambda_idt
        self.lr =lr
        self.beta1 =beta1
        self.cuda= torch.cuda.is_available()
        if self.cuda:
            self.netG_A.cuda()
            self.netG_B.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()
        
        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)
        
        # Define loss 
        self.criterionGAN = GANLoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        
        # Generators 2 parameters updated same time
        self.optimizer_G = torch.optim.Adam(
        itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
        lr=self.lr,
        betas=(self.beta1,0.999))
        
        
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D_A)
        self.optimizers.append(self.optimizer_D_B)
        
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def set_input(self,input):
        input_A = input['A']
        input_B = input['B']
        if self.cuda:
            input_A = input_A.cuda()
            input_B = input_B.cuda()
        self.input_A = input_A
        self.input_B = input_B
        #self.image_paths = input['path_A']
        
    def backward_G(self, real_A, real_B):
        # Calculate loss and backward about Generator
        
        idt_A = self.netG_A(real_B)
        loss_idt_A = self.criterionIdt(idt_A, real_B) * self.lambda_B * self.lambda_idt 
        
        idt_B = self.netG_B(real_A)
        loss_idt_B = self.criterionIdt(idt_B, real_A) * self.lambda_A * self.lambda_idt 
        
        #GAN loss D_A(G_A(real_A))
        #Gnerater make imagest that looks like real images (label 1)
        fake_B = self.netG_A(real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake,True)
        
        #GAN loss D_B(G_B(real_B))
        fake_A = self.netG_B(real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)
        
        #forward cycle loss
        #real A => fake_B => rec_A 
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.lambda_A
        
        #backward cycle loss
        #real B => fake_A => rec_B
        rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.lambda_B
        
        #combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        
        return loss_G_A.item(), loss_G_B.item(), loss_cycle_A.item(), loss_cycle_B.item(), loss_idt_A.item(), loss_idt_B.item(),fake_A.data, fake_B.data 
    
    def backward_D_A(self,real_B,fake_B):
        #Sampling from fakeB image pool that conist of 50 fake B images mad by netG_A
        fake_B = self.fake_B_pool.query(fake_B)
        
        #When netD_A judge real image, netD_A is prefered to return 1
        pred_real = self.netD_A(real_B)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        #When netD_A judge fake image made by netG_A, netD_A is prefered to return 0
        #Detach is necessary, because the grad made by netD_A woud not backward netG_A
        pred_fake = self.netD_A(fake_B.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        #combined loss
        loss_D_A =(loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        
        return loss_D_A.item()
    
    def backward_D_B(self, real_A, fake_A):
        #sampling from fakeA image pool thaat consist of 50 fake A images made by netG_B
        fake_A = self.fake_A_pool.query(fake_A)
        
        #When netD_B judge real image, netD_A is prefered to return 0
        pred_real = self.netD_B(real_A)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        #When netD_A judge fake image maby netG_B, netD_B is preferd to return 0
        #Detach is necessary, because the grad made by netD_A would not backward netG_B
        pred_fake = self.netD_B(fake_A.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        #combined loss 
        loss_D_B = (loss_D_real + loss_D_fake) *0.5
        loss_D_B.backward()
        
        return loss_D_B.item()
    
    def optimize(self):
        real_A = Variable(self.input_A)
        real_B = Variable(self.input_B)
        
        #update Generator(G_A and G_B)
        self.optimizer_G.zero_grad()
        loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_idt_A, loss_idt_B, fake_A, fake_B = self.backward_G(real_A, real_B)
        self.optimizer_G.step()
        
        # update D_A
        self.optimizer_D_A.zero_grad()
        loss_D_A = self.backward_D_A(real_B, fake_B)
        self.optimizer_D_A.step()
        
        # update D_B
        self.optimizer_D_B.zero_grad()
        loss_D_B = self.backward_D_B(real_A,fake_A)
        self.optimizer_D_B.step()
        
        
        ret_loss = [loss_G_A, loss_D_A,
                   loss_G_B, loss_D_B,
                   loss_cycle_A, loss_cycle_B,
                   loss_idt_A, loss_idt_B]
        
        return np.array(ret_loss)
    
    def train(self, data_loader):
        running_loss = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for batch_idx, data in enumerate(data_loader):
            self.set_input(data)
            losses = self.optimize()
            running_loss += losses
        running_loss /= len(data_loader)
        return running_loss
    
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.log_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        # Back to gpu model
        if self.cuda:
            network.cuda()
            
    def load_network(self, network, network_label, epoch_label):
        load_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self.log_dir, load_filename)
        network.load_state_dict(torch.load(load_path))
        
    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label)
        self.save_network(self.netG_B, 'G_B', label)
        self.save_network(self.netD_A, 'D_A', label)
        self.save_network(self.netD_B, 'D_B', label)
        
    def load(self, label):
        self.load_network(self.netG_A, 'G_A', label)
        self.load_network(self.netG_B, 'G_B', label)
        self.load_network(self.netD_A, 'D_A', label)
        self.load_network(self.netD_B, 'D_B', label)