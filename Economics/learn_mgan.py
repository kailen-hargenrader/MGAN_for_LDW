import torch
import torch.utils.data as data
import torch.nn as nn
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pyarrow.feather as feather
import pandas as pd
import kde

#Fully Connected Feed Forward Network
class FCFFNet(nn.Module):
    def __init__(self, layers, nonlinearity, nonlinearity_params=None, 
                 out_nonlinearity=None, out_nonlinearity_params=None, normalize=False):
        super(FCFFNet, self).__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))
            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))
                if nonlinearity_params is not None:
                    self.layers.append(nonlinearity(*nonlinearity_params))
                else:
                    self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            if out_nonlinearity_params is not None:
                self.layers.append(out_nonlinearity(*out_nonlinearity_params))
            else:
                self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


parser = argparse.ArgumentParser()
parser.add_argument("--monotone_param", type=float, default=.01, help="monotone penalty constant")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--n_layers", type=int, default=3, help="number of layers in network")
parser.add_argument("--n_units", type=int, default=128, help="number of hidden units in each layer")
parser.add_argument("--batch_size", type=int, default=1000, help="batch size (Should divide Ntest)")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
args = parser.parse_args()

torch.manual_seed(0)

#Pick device: cuda, cpu
device = torch.device('cpu')

#import data
path = "C:\\Users\Kailen\MGAN\Economics\cps.feather"
imported_df = feather.read_feather(path)
imported_df.head()

#get relevent columns
y_tensor = torch.tensor(imported_df[["age"]].values.astype(np.float32))
x_tensor = torch.tensor(imported_df[["re78"]].values.astype(np.float32))

#normalize
y_mean = torch.mean(y_tensor, dim=0)
y_std = torch.std(y_tensor, dim=0)
y_tensor = (y_tensor-y_mean)/y_std

x_mean = torch.mean(x_tensor, dim=0)
x_std = torch.std(x_tensor, dim=0)
x_tensor = (x_tensor-x_mean)/x_std
dataset = data.TensorDataset(x_tensor, y_tensor)

#train-test split
num_train = round(int(.9*len(dataset)),-3)
num_test = len(dataset) - num_train
train_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_test])

#define x and y
x_train, y_train = train_set[:]
y_train_np = y_train[:, 0].numpy()
x_train_np = x_train[:, 0].numpy()

x_test, y_test = test_set[:]
y_test_np = y_test[:, 0].numpy()
x_test_np = x_test[:, 0].numpy()
# plt.scatter(y_test_np, x_test_np)
# plt.xlabel("y")
# plt.ylabel("x")
# plt.show()
dx = x_train.shape[1]
dy = y_train.shape[1]

#define real data for histogram comparison
real20 = imported_df.query('18 <= age <= 22')[["re78"]].values.astype(np.float32)
real30 = imported_df.query('28 <= age <= 32')[["re78"]].values.astype(np.float32)
real40 = imported_df.query('38 <= age <= 42')[["re78"]].values.astype(np.float32)

real = [real20, real30, real40]
# plt.hist(real20, bins=200, density=True, label='$real x^* = '+str(20)+'$')
# plt.hist(real30, bins=200, density=True, label='$real x^* = '+str(30)+'$')
# plt.hist(real40, bins=200, density=True, label='$real x^* = '+str(40)+'$')
# plt.xlim(0, 30000)
# plt.legend()
# plt.show()

#Data loaders for training
bsize = args.batch_size
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train, x_train), batch_size=bsize, shuffle=True)
ydata_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(y_train, ), batch_size=bsize, shuffle=True)

#Define loss
mse_loss = torch.nn.MSELoss()

#Define softplus
softplus = torch.nn.Softplus()

#Transport map and discriminator
network_params = [dx+dy] + args.n_layers * [args.n_units]
F = FCFFNet(network_params + [dx], nn.LeakyReLU, nonlinearity_params=[0.2, True]).to(device)
D = FCFFNet(network_params + [1], nn.LeakyReLU, nonlinearity_params=[0.2, True], out_nonlinearity=nn.Sigmoid).to(device)

#Optimizers
optimizer_F = torch.optim.Adam(F.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

# Schedulers
sch_F = torch.optim.lr_scheduler.StepLR(optimizer_F, step_size = len(train_loader), gamma=0.995)
sch_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size = len(train_loader), gamma=0.995)

# define arrays to store results
monotonicity    = torch.zeros(args.n_epochs,)
D_train         = torch.zeros(args.n_epochs,)
F_train         = torch.zeros(args.n_epochs,)
loss_train      = torch.zeros(args.n_epochs,)
kde_loss        = torch.zeros(args.n_epochs,)
loss_difference = None


for ep in range(args.n_epochs):

    F.train()
    D.train()

    # define counters for inner epoch losses
    D_train_inner = 0.0
    F_train_inner = 0.0
    loss_train_inner = 0.0
    kde_loss_inner = 0.0
    mon_percent = 0.0

    for y, x in train_loader:
        #Data batch
        y, x = y.to(device), x.to(device)

        ones = torch.ones(bsize, 1, device=device)
        zeros = torch.zeros(bsize, 1, device=device)

        ###Loss for transport map###

        optimizer_F.zero_grad()

        #Draw from reference
        z1 = next(iter(ydata_loader))[0].to(device)
        z2 = torch.randn(bsize, dx, device=device)
        z = torch.cat((z1, z2), 1)

        #Transport reference to conditional x|y
        Fz = F(z)

        #Transport of reference z1 to y marginal is by identity map
        #Compute loss for generator
        F_loss = mse_loss(D(torch.cat((z1, Fz), 1)), ones)
        F_train_inner += F_loss.item()

        #Draw new reference sample
        z1_prime = next(iter(ydata_loader))[0].to(device)
        z2_prime = torch.randn(bsize, dx, device=device)
        z_prime = torch.cat((z1_prime, z2_prime), 1)

        #Monotonicity constraint
        mon_penalty = torch.sum(((Fz - F(z_prime)).view(bsize,-1))*((z2 - z2_prime).view(bsize,-1)), 1)
        if args.monotone_param > 0.0:
            F_loss = F_loss + args.monotone_param*torch.mean(softplus(-mon_penalty))# * (loss_difference))
            # F_loss = F_loss + args.monotone_param*torch.mean(-mon_penalty)
        # take step for F
        F_loss.backward()
        optimizer_F.step()
        sch_F.step()

        #Percent of examples in batch with monotonicity satisfied
        mon_penalty = mon_penalty.detach() + torch.sum((z1.view(bsize,-1) - z1_prime.view(bsize,-1))**2, 1).detach()
        mon_percent += float((mon_penalty>=0).sum().item())/bsize

        ###Loss for discriminator###

        optimizer_D.zero_grad()

        #Compute loss for discriminator
        D_loss = 0.5*(mse_loss(D(torch.cat((y,x),1)), ones) + mse_loss(D(torch.cat((z1, Fz.detach()), 1)), zeros))
        D_train_inner += D_loss.item()

        # take step for D
        D_loss.backward()
        optimizer_D.step()
        sch_D.step()
    F.eval()
    D.eval()

    #define test number and normalize
    test_num = 20
    test_num_scaled = (test_num - y_mean) / y_std

    if ep % 10 == 0:
        z = torch.randn(1000, dx, device=device)
        xs = torch.tensor([test_num_scaled]).view(1,1).repeat(1000,1).to(device)
        with torch.no_grad():
            Fz = F(torch.cat((xs, z), 1))
        Fz_numpy = Fz.cpu().numpy()
        # plt.hist(Fz_numpy * x_std.item() + x_mean.item(), bins=200, density=True, label='$generated x^* = '+str(test_num)+'$')

        # #real data
        # plt.hist(real20, bins=200, density=True, label='$real x^* = '+str(test_num)+'$')
        # plt.legend()
        # plt.show()
    #Average monotonicity percent over batches
    mon_percent = mon_percent/math.ceil(float(len(x_train))/bsize)
    monotonicity[ep] = mon_percent

    #Average generator and discriminator losses
    F_train[ep] = F_train_inner/math.ceil(float(len(x_train))/bsize)
    D_train[ep] = D_train_inner/math.ceil(float(len(x_train))/bsize)
    if ep != 0:
        loss_difference = F_train[ep]-F_train[ep-1]

    print('Epoch %3d, Monotonicity: %f, Generator loss: %f, Critic loss: %f' % \
         (ep, monotonicity[ep], F_train[ep], D_train[ep]))

# Plot losses
plt.figure()
plt.subplot(1,4,1)
plt.plot(np.arange(args.n_epochs), monotonicity.numpy())
plt.xlabel('Number of epochs')
plt.ylabel('Monotonicity')
plt.ylim(0,1)
plt.subplot(1,4,2)
plt.plot(np.arange(args.n_epochs), D_train.numpy(), label='Critic loss')
plt.plot(np.arange(args.n_epochs), F_train.numpy(), label='Generator loss')
plt.xlabel('Number of epochs')
plt.legend()
plt.show()

# Plot densities
# define conditionals
xst_unscaled = np.array([20.0, 30.0, 40.0]).astype(np.float32)
Ntest = 1000

# define domain
y_dom = [-1000,30000]
yy = np.linspace(y_dom[0], y_dom[1], 1000)
yy = np.reshape(yy, (1000, 1))

#scale
xst = (xst_unscaled - y_mean.item()) / y_std.item()
yy = (yy - x_mean.item()) / x_std.item()


#Sample each conditional
plt.figure()
for i,xi in enumerate(xst):
    plt.subplot(1,len(xst),i+1)
    # sample from conditional
    xit = torch.tensor([xi]).view(1,1)
    xit = xit.repeat(Ntest,1).to(device)
    z = torch.randn(Ntest, dx, device=device)
    with torch.no_grad():
        Fz = F(torch.cat((xit, z), 1))
    Fz_numpy = Fz.cpu().numpy()

    #Kernel Density Estimation on truth
    kde_estimator = kde.KernelDensityEstimator(torch.cat(((torch.tensor(real[i]) - x_mean.item()) / x_std.item(), torch.zeros(len(real[i]), 1, device=device)), 1), kernel = kde.GaussianKernel(bandwidth = .01))
    est_i = kde_estimator.forward(torch.cat((torch.from_numpy(yy), torch.zeros(1000, 1, device=device)), 1))

    est_i[est_i < 0] = 0
    est_i[est_i > 0.0] = np.exp(est_i[est_i > 0.0])

    area = torch.trapezoid(est_i, dx = (y_dom[1]-y_dom[0])/1000)
    est_i = est_i/area
    plt.plot(yy * x_std.item() + x_mean.item(), est_i, label="est_i")
 
    # plot density and samples
    plt.hist(Fz_numpy * x_std.item() + x_mean.item(), bins=200, density=True, label='$x^* = '+str(xst_unscaled[i])+'$')
    #plt.hist(real[i], bins=200, density=True, label='$x^* = '+str(xst_unscaled[i])+'$')
    plt.legend()
    plt.xlabel('$y$')
    plt.ylabel('$\pi(y|x^*)$')
plt.show()
