import torch
import torchvision
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

loss_fn = torch.nn.MSELoss(reduction='sum')
tanh = torch.nn.Tanh()
learning_rate = 1e-4


def plot_nn(pltt,figg, w1, w2):
    a = np.linspace(-5, 5, 10)
    b = np.linspace(-5, 5, 10)
    z = np.linspace(-5, 5, 10)

    a_, b_, z_ = np.meshgrid(a, b, z)

    a_ = a_.flatten()
    b_ = b_.flatten()
    z_ = z_.flatten()

    samples = np.stack([a_, b_, z_], axis=1) # shape = (1000,3)

    c_ = []
    for i in range(samples.shape[0]):
        x_ = torch.from_numpy(np.array(samples[i, :]))
        x_ = x_.float()
        x_.resize_(1,3)
        val = tanh(x_.mm(w1)).mm(w2)
        c_.append(val)

    figg.clear()
    ax = pltt.axes(projection='3d')
    ax.scatter(a_, b_, z_, c=c_, cmap='Blues', linewidth=0.5)
    pltt.show()



def train(x_, y_, w1_, w2_):
    y_hat = tanh(x_.mm(w1_)).mm(w2_)
    loss = loss_fn(y_hat, y_)
    loss.backward()
    with torch.no_grad():
        w1_ -= learning_rate * w1_.grad
        w2_ -= learning_rate * w2_.grad
        w1_.grad.zero_()
        w2_.grad.zero_()


N, D_in, H, D_out = 1, 3 , 20, 1

x = torch.full((N, D_in), 0.1)
y = torch.full((N, D_out), 1.0)
#y = torch.rand(N, D_out)

w1 = torch.full((D_in, H), 1.0, requires_grad = True)
w2 = torch.full((H, D_out), 1.0, requires_grad = True)

# plot
fig = plt.figure()
ax = plt.axes(projection='3d')

for t in range(1000):
    if t % 100 == 0:
        plot_nn(plt,fig, w1, w2)
    train(x, y, w1, w2)




# plot

def f(a, b, z):
    return 1 / (a + b + z)


def f_sun(a, b, z):
    return abs(1 / a) + abs(1/b) + abs(1/z)


fig.clear()
ax = plt.axes(projection='3d')


a = np.linspace(-5, 5, 10)
b = np.linspace(-5, 5, 10)
z = np.linspace(-5, 5, 10)

a_, b_, z_ = np.meshgrid(a,b,z)

c_ = f_sun(a_, b_, z_)

print("c_[9,9,9]" + str(c_[9,9,9]))
print("c_[5,5,5]" + str(c_[5,5,5]))
print("c_[4,4,4]" + str(c_[4,4,4]))
print("c_[0,0,0]" + str(c_[0,0,0]))



ax.scatter(a_.flatten(), b_.flatten(), z_.flatten(), c=c_.flatten(), cmap='Blues', linewidth=0.5)

plt.show()

