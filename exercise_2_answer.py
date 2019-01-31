import torch
import my_utils
import torchvision
import torch.nn as nn


# EXERCISE 2.1: Implement the Gram matrix computation
# G = transpose(X)*X
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(c, h * w)
        G = torch.mm(F, F.t())
        return G

#input('Completed Ex. 2.1')

# Choose image size and image file 
imsize0 = 512
data_file = f'Images/bones.jpg'
y0 = 2*my_utils.image_loader(data_file).cuda()-1


# Load image and VGG-19 network
out_folder = f'my_synthesis/'
import os
if os.path.exists(out_folder)==False:
    os.mkdir(out_folder)
aspect_ratio = y0.size(3)/y0.size(2)
imsize1 = int(imsize0*aspect_ratio)
y0 = nn.AdaptiveAvgPool2d((imsize0,imsize1))(y0)
# Load network
L2_loss = nn.MSELoss()
gram_matrix = GramMatrix()
vgg_net,loss_lambda = my_utils.get_vgg_net()

N_optim_iter = 500
save_every = 50
N_scales = 1
x = my_utils.get_input_noise(imsize0,imsize1)
optimizer = torch.optim.LBFGS([x.requires_grad_()],max_iter=1)
im_folder = out_folder + data_file[7:-4] + '/'
if os.path.exists(im_folder)==False:
    os.mkdir(im_folder)
torchvision.utils.save_image(y0,im_folder + 'input_image.jpg',normalize=True)

y_activ = [out.detach() for out in vgg_net(y0)]
# EXERCISE 2.2: Display sizes of entwork activations. 
# What does this mean in terms of image representation?
# Hint: Images are 1x (Channels) x (Height) x (Width)
for feat in y_activ:
     print(feat.size())

input('Viewing activation sizes. PRESS ENTER')

y_feats = [gram_matrix(activ) for activ in y_activ]


for i in range(N_optim_iter):
    def closure():

        optimizer.zero_grad()
        x_feats = vgg_net(x.cuda().clamp(-1,1))
        
        loss = 0
        for ll,x_feat,y_feat in zip(loss_lambda,x_feats,y_feats):
            # EXERCISE 2.3: Complete this loss function
            loss += ll*L2_loss(gram_matrix(x_feat),y_feat)

        loss.backward()
        print(f'iter={i:04d}, loss={loss.item():.03e}')
        return loss
    optimizer.step(closure)
    if i%save_every==0:
        torchvision.utils.save_image(x.cpu().detach().clamp(-1,1),im_folder + f'iteration_{i:03d}.jpg',normalize=True)
    
torchvision.utils.save_image(x.cpu().detach().clamp(-1,1),im_folder + f'final_synthesis.jpg',normalize=True)













