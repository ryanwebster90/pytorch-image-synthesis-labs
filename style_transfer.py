import torch
import utils
import torchvision
from vgg import VGG
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# sizes largest side of images
imsize0 = 450
imsize_style = 450

def get_blurred_noise(N,M,n_iter=3):
    x = torch.randn(1,3,N,M).cuda()
    f = utils.get_gaussian_filt(1,9).cuda()
    for k in range(n_iter):
        x = utils.channel_conv_layer(x,f)
    return x
    
loader = transforms.Compose([
    transforms.ToTensor()]) 
    
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image
    
# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G
    
style_name = 'starrynight_2'
target_name = 'city'
out_folder = f'./my_style_transfer/'
import os
if os.path.exists(out_folder)==False:
    os.mkdir(out_folder)
out_folder = out_folder + f'target_name_{target_name}_style_name_{style_name}/'
if os.path.exists(out_folder)==False:
    os.mkdir(out_folder)

#x = torch.randn(1,3,imsize,imsize).cuda()

data_file = 'Images/vangogh_starry_night.jpg'
y0 = 2*image_loader(data_file).cuda()-1
if y0.size(2) > y0.size(3):
    imsize0_style = imsize_style
    imsize1_style = int(y0.size(3)/y0.size(2)*imsize0_style)
else:
    imsize1_style = imsize_style
    imsize0_style = int(y0.size(2)/y0.size(3)*imsize1_style)

y0 =  nn.AdaptiveAvgPool2d((imsize0_style,imsize1_style))(y0)

data_file_target = 'Images/Tuebingen_Neckarfront.jpg'
y0_content = 2*image_loader(data_file_target).cuda()-1
aspect_ratio = y0_content.size(3)/y0_content.size(2)
imsize1 = int(imsize0*aspect_ratio)
nn.AdaptiveAvgPool2d((imsize0,imsize1))(y0_content)

loss_layer = nn.MSELoss()
gram_layer = GramMatrix()
#loss_layer = GramMSELoss()
out_keys = ['r11','r21','r31','r41','r51']
content_ind = 4

loss_lambda = 1e8*torch.tensor([1,1,1,1,1.0]).cuda()
content_lambda = 1e8
N_optim_iter = 600
save_every = 50

torchvision.utils.save_image(y0,out_folder+f'00_x_input.jpg',normalize=True)

vgg_net = VGG(pool='avg',out_keys=out_keys)
vgg_net.load_state_dict(torch.load('Models/vgg_conv.pth'))
vgg_net.cuda()

N_scales = 1
imsizes0_style = [int(imsize0_style/2**scale) for scale in range(N_scales-1,-1,-1)]
imsizes1_style = [int(imsize1_style/2**scale) for scale in range(N_scales-1,-1,-1)]
imsizes0 = [int(imsize0/2**scale) for scale in range(N_scales-1,-1,-1)]
imsizes1 = [int(imsize1/2**scale) for scale in range(N_scales-1,-1,-1)]
#imsizes = [256,512]

for s,cur_imsize in enumerate(zip(imsizes0,imsizes1)):
    
    # resize style
    y = nn.AdaptiveAvgPool2d((imsizes0_style[s],imsizes1_style[s]))(y0.detach())
    
    y = y[:,[2,1,0],:,:]
    y_feat = [gram_layer(out.detach()) for out in vgg_net(y)]
    
    
    scale_layer = nn.AdaptiveAvgPool2d((cur_imsize[0],cur_imsize[1]))
    y_content = scale_layer(y0_content.detach())
    y_content = vgg_net(y_content)[content_ind]
    
    # scale and send to cpu (to save memory with LBFGS)
    if s==0:
        x = .5*get_blurred_noise(cur_imsize[0],cur_imsize[1]).cpu() + .5*scale_layer(y0_content).cpu()
#        x = y0_content
#        x.requires_grad=True
    else:
        alpha = .4
        # add alpha to mitigate artifacts that arise at lower scales
        x = alpha*scale_layer(x.detach().cpu()) + (1-alpha)*get_blurred_noise(cur_imsize[0],cur_imsize[1]).cpu()
    
    #    x = torch.randn(1,3,imsize,imsize).cuda()
    optimizer = torch.optim.LBFGS([x.requires_grad_()],max_iter=1)
    
    im_folder = out_folder + f'synth_iterations_{cur_imsize}/'
    if os.path.exists(im_folder)==False:
        os.mkdir(im_folder)
    
    for i in range(N_optim_iter):
        def closure():
    
            optimizer.zero_grad()
            
            out_feat = vgg_net(x[:,[2,1,0],:,:].cuda().clamp(-1,1))
                
            out_content = out_feat[content_ind]
            
            style_loss = 0
            for ind in range(len(out_feat)):
                style_loss += loss_lambda[ind]*loss_layer(gram_layer(out_feat[ind]),y_feat[ind])
                
            content_loss = content_lambda*loss_layer(out_content,y_content.detach())
#            content_loss = content_lambda*loss_layer(x.cuda(),scale_layer(y0_content).detach())
            
            loss = style_loss+content_loss
            loss.backward()
#            print(f'iter={i:04d}, loss={loss.item():.03e}')
            print(f'iter={i:04d}, loss={loss.item():.03e}, style_loss={style_loss.item():.03e}, content_loss={content_loss.item():.03e}')
            return loss
        optimizer.step(closure)
        if i%save_every==0:
            torchvision.utils.save_image(utils.map01(x.cpu().detach().clamp(-1,1)),im_folder + f'{i:03d}.jpg',normalize=True)
        
    torchvision.utils.save_image(utils.map01(x.cpu().detach().clamp(-1,1)),out_folder + f'synth_{cur_imsize}.jpg')













