import torch
# EXERCISE 1.1: Implement ReLU forward and backward 
# ReLU is defined in slides
# ReLU(X) = X if X>0 : 0 if X<=0
# This operation must work on each element
def my_ReLU(x):
    x_c = x.clone()
    x_c[x_c<=0] = 0
    return x_c

# EXEERCISE 1.2: Backward of ReLU
def ReLU_backward(x,dzdy):
    # dydx is 1 x>0 and 0 x<=0
    dzdy_c = dzdy.clone()
    dzdy_c[x<=0] = 0
    dzdx = dzdy_c
    return dzdx
    
# EXERCISE 1.3: Implement backward of matrix multiplication w.r.t. x
def mm_backward(A,x,dzdy):
    dydx = A.t()
    dzdx = torch.mm(dydx,dzdy)
    return dzdx


# Exercise 1.4: Compute derivative of function composition ReLU(Ax) w.r.t. x
x = torch.randn(5,1)
A = torch.randn(5,5)

with torch.no_grad():
    y = A.mm(x)
    z = my_ReLU(x)
    dz = torch.randn(5,1)
    dzdy = ReLU_backward(y,dz)
    my_dzdx = mm_backward(A,x,dzdy)

import torch.nn as nn
relu = nn.ReLU()
x.requires_grad = True
y = A.mm(x)
z = relu(y)

z.backward(dz)
print(x.grad)
print(my_dzdx)

print('These two vectors should be the same!')

# Exercise 1.5: Optimize a toy function with .backward() and torch.optim
# Uncomment and do this exercise if you have time
#import torch.optim
#
#L2_loss = nn.MSELoss()
#
#x = torch.randn(512,1).cuda()
#x.requires_grad=True
#
#A = torch.randn(512,512).cuda()
#
#w = relu(torch.randn(512,1).cuda()).detach()
#optimizer = torch.optim.Adam([x])
#N_iterations= 1000
#
#for it in range(N_iterations):
#    y = A.mm(x)
#    z = relu(y)
#    loss = L2_loss(z,w)
#    
#    loss.backward()
#    optimizer.step()
#    if it%50==0:
#        print(f'iteration = {it}, loss = {loss.item():.04f}')
    
    



    
    
    


