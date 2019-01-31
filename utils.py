from __future__ import print_function
import torch

import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt


import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch.nn as nn
import random

unloader = transforms.ToPILImage()  # reconvert into PIL image

def map01(x):
    imin = torch.min(x.view(-1))
    imax = torch.max(x.view(-1))
    return (x - imin)/(imax-imin)



def get_blurred_noise(M,N,n_iter=5):
    x = torch.randn(1,3,M,N).cuda()
    f = get_gaussian_filt(1,15).cuda()
    for k in range(n_iter):
        x = channel_conv_layer(x,f)
    return x

def save_grid_borders(imgs,filename,padding=16,border_sz=4,nrow=1):
    imgs = torch.nn.functional.pad(imgs,(border_sz,border_sz,border_sz,border_sz),value=0)
    torchvision.utils.save_image(imgs,filename,padding=padding,nrow=nrow,pad_value=1)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = map01(image)
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)

    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()],max_iter=1)
    return optimizer

def channel_conv_layer(x,w):
    s = x.size()
    x = x.view(-1,1,s[2],s[3])
    x = torch.nn.functional.conv2d(x,w,padding=int((w.size(2)-1)/2) )
    x = x.view(s)
    return x
    
def get_gaussian_filt(sigma,N):
    a = np.linspace(-1,1,N)
    x = 1/(sigma*np.sqrt(2*np.pi))* np.exp(-.5*(a**2)/sigma**2)
    x = torch.tensor(x,dtype=torch.float32)
    x = x/x.sum()
    x = x.view(1,-1)
    
    x = torch.mm(x.t(), x).view(1,1,N,N)
    
    return x

def diffeo_warp(y,sigma=1,conv_iter=3,filt_size=11,alpha=100,pool_sz=128):
    u = torch.randn(1,1,y.size(2),y.size(3)).to(y.device)
    v = torch.randn(1,1,y.size(2),y.size(3)).to(y.device)
    
    w = get_gaussian_filt(sigma,filt_size).to(y.device)
    
    k = torch.cat([u,v],dim=1)
    for i in range(conv_iter):
        k = channel_conv_layer(k,w)
    
    M = y.size(2)
    
#    pl0 = torch.nn.AdaptiveAvgPool2d(128)
#    pl1 = torch.nn.AdaptiveAvgPool2d(y.size(2))
#    k = pl1(pl0(k))
    k = 1/M*alpha*k
    
    
    vf = np.meshgrid(np.linspace(-1,1,M), np.linspace(-1,1,M))
    vf = torch.tensor(vf,device=y.device,dtype=torch.float32)
    vf = vf.view(1,2,M,M)
    
    vf = vf + k
    vf = vf.permute([0,2,3,1])
    vf = vf.repeat(y.size(0),1,1,1)
    
    y = torch.nn.functional.grid_sample(y, vf, mode='bilinear', padding_mode='zeros')
    return y

def get_lap_pyr_loss(x,y,N_levels):
    imsize = x.size(2)
    pyr_layers = []
    for layer in range(N_levels):
        pyr_layers += [nn.AdaptiveAvgPool2d(imsize)]
        imsize = int(imsize/2)
    
    loss = 0
    
    for layer in range(N_levels-1):
        cur_layer = pyr_layers[layer]
        next_layer = pyr_layers[layer+1]
        x_d = cur_layer(x)-cur_layer(next_layer(x))
        y_d = cur_layer(y)-cur_layer(next_layer(y))
        loss+=torch.mean( torch.abs((x_d-y_d)))
    
    x_d = next_layer(x)
    y_d = next_layer(y)
    loss+=torch.mean( torch.abs((x_d-y_d)))
    
    return loss
    

def blur_image(y,sigma=1,filt_size=11):
    w = get_gaussian_filt(sigma,filt_size).to(y.device)
    y0 = channel_conv_layer(y,w)
    return y0

def corrupt_black_hole(y,alpha):
    ps = int(y.size(2)*alpha)
#    i = random.randint(0,y.size(2)-ps-1)
#    j = random.randint(0,y.size(2)-ps-1)
    i = random.randint(int(.25*y.size(2)),int(.5*y.size(2)))
    j = random.randint(int(.25*y.size(2)),int(.5*y.size(2)))
    z = y.clone()
    z[:,:,i:i+ps,j:j+ps] = .5*torch.randn(z.size(0),3,ps,ps).to(y.device).clamp(-1,1)
    return z


def sample_patches(x,patch_size,stride=1):
    P_exmpl = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride) # 1 x 3 x P x P x stride x stride
    return P_exmpl
    
def sw_loss(x,y,patch_size,Theta):
    
    Theta = torch.randn(patch_size*patch_size*3,patch_size*patch_size*3).cuda()
    Theta = Theta / torch.sqrt(torch.sum(Theta**2,dim=1))
    P_exmpl = sample_patches(x,patch_size=patch_size,stride=int(patch_size/2))
    P_synth =  sample_patches(y,patch_size=patch_size,stride=int(patch_size/2))
    
    P_exmpl = P_exmpl.squeeze(0).permute(1,2,3,4,0) 
    P_exmpl = P_exmpl.contiguous().view(P_exmpl.size(0)*P_exmpl.size(1),-1) # N x d

    P_synth  = P_synth.squeeze(0).permute(1,2,3,4,0) 
    P_synth = P_synth.contiguous().view(P_synth.size(0)*P_synth.size(1),-1) # N x d

    
    P_exmpl_theta = torch.mm(P_exmpl,Theta) # N x K
    P_exmpl_theta,_ = torch.sort(P_exmpl_theta, 0)
    P_synth_theta = torch.mm(P_synth,Theta) # N x K
    P_synth_theta,_ = torch.sort(P_synth_theta, 0)
    
    loss = torch.mean((P_synth_theta.view(-1) - P_exmpl_theta.view(-1))**2) 
    
    return loss



class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(1,-1, 1, 1)
        self.std = torch.tensor(std).view(1,-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
    
def get_vgg19_perceptual_loss_net(device):
    use_normalization = True
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    # if you want to use normalization, append it to first network
    vgg_relu1_1 = vgg[:7]
    if use_normalization:
        vgg_relu1_1 = nn.Sequential(Normalization(cnn_normalization_mean,cnn_normalization_std),vgg_relu1_1)
    vgg_relu2_1 = vgg[7:12]
    vgg_relu3_1 = vgg[12:21]
    vgg_relu4_1 = vgg[21:30]
    vgg = nn.ModuleList([vgg_relu1_1,vgg_relu2_1,vgg_relu3_1,vgg_relu4_1]).to(device)

    return vgg


# loss over every module in net_list
def multi_layer_loss(net_list,loss_layer,layer_lambda,x,y_feat):
    loss = 0
    for ind,net in enumerate(net_list):
        x = net(x)
        layer_loss = layer_lambda[ind]*loss_layer(x,y_feat[ind])
        loss += layer_loss
    return loss
        
# gets (detached) network outputs in a module list
def get_net_outputs(net_list,x):
    out = []
    for net in net_list:
        x = net(x)
        out.append(x.clone().detach())
    return out

