import torch
# EXERCISE 1.1: Implement ReLU forward and backward 
# ReLU is defined in slides
def my_ReLU(x):

    # Hint: x<=0 is a logical array, with 1 where x>0, and 0 where x<0
    # answer here
    return x


# EXERCISE 1.2: Backward of ReLU
def ReLU_backward(x,dzdy):
    # Hint: dzdy is multiplied by the derivative of ReLU(x), with is 1 when w>0 and 0 when x<0
    # answer here
    return dzdx
    
# EXERCISE 1.3: Implement backward of matrix multiplication w.r.t. x
def mm_backward(A,x,dzdy):
    # answer here
    return dzdx


# Exercise 1.4: Compute derivative of function composition ReLU(Ax) w.r.t. x
x = torch.randn(5,1)
A = torch.randn(5,5)

with torch.no_grad():
    y = A.mm(x)
    z = my_ReLU(x)
    dz = torch.randn(5,1)
    dzdy = ReLU_backward(y,dz)
    my_dzdx = 0# answer here


# Import actual relu function, which has a derivative implemented in pytorch
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
# Minimize the function |ReLU(Ax) - w |^2, w.r.t. x
# computes the summed squared error (x-y)**2.sum()
L2_loss = nn.MSELoss()

x = torch.randn(512,1).cuda()
x.requires_grad=True
A = torch.randn(512,512).cuda()
# randomly initialize solution w
w = relu(torch.randn(512,1).cuda()).detach()

# torch.optim has many optimizers
optimizer = torch.optim.Adam([x])
N_iterations= 1000

for it in range(N_iterations):
    y = A.mm(x)
    z = relu(y)
    loss = L2_loss(z,w)
    
    loss.backward()
    
    # step updates optim variable (x), with derivatives computed by backprop
    # for example SGD is x(t) = x(t-1) - alpha*x.grad
    optimizer.step()
    if it%50==0:
        print(f'iteration = %d, loss = %.4f'%(it,loss.item()))
    
    



    
    
    


