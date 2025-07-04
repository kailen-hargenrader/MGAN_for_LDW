import torch
import torch.nn as nn
import math
from tanh_models import tanh_v1, tanh_v2, tanh_v3
import numpy as np
import argparse
import kde
import matplotlib.pyplot as plt

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
parser.add_argument("--dataset", type=str, default='tanh_v2', help="one of: tanh_v1,tanh_v2,tanh_v3")
parser.add_argument("--n_train", type=int, default=10000, help="number of training samples")
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--n_layers", type=int, default=3, help="number of layers in network")
parser.add_argument("--n_units", type=int, default=128, help="number of hidden units in each layer")
parser.add_argument("--batch_size", type=int, default=1000, help="batch size (Should divide Ntest)")
parser.add_argument("--learning_rate", type=float, default=0.002, help="learning rate")
args = parser.parse_args()

torch.manual_seed(0)

#Pick device: cuda, cpu
device = torch.device('cpu')

# define density
dataset = args.dataset
if dataset == 'tanh_v1':
    pi = tanh_v1()
elif dataset == 'tanh_v2':
    pi = tanh_v2()
elif dataset == 'tanh_v3':
    pi = tanh_v3()
else:
    raise ValueError('Dataset is not recognized')

# load data (flipping x and y for supervised learning)
y_train = pi.sample_prior(args.n_train)
x_train = pi.sample_data(y_train)
plt.scatter(y_train, x_train)
plt.xlabel("y")
plt.ylabel("x")
plt.show()
y_train = torch.from_numpy(y_train.astype(np.float32))
x_train = torch.from_numpy(x_train.astype(np.float32))

dx = x_train.shape[1]
dy = y_train.shape[1]

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
            #F_loss = F_loss + args.monotone_param*torch.mean(softplus(-mon_penalty))# * (loss_difference))
            F_loss = F_loss# + args.monotone_param*torch.mean(-mon_penalty)
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

        #Compute training loss
        x_true = pi.sample_data(z1)
        fz_store = Fz
        loss_train_inner += mse_loss(fz_store, x_true).item()

        ###Compute KDE loss###
        Nsamples = 1000

        #Choose x
        x_float = -1.2 #torch.randn(1, dx, device = device).item()

        # define domain
        y_dom = [-10,10]
        dy = (y_dom[1]-y_dom[0])/1000
        yy = np.linspace(y_dom[0], y_dom[1], 1000)
        yy = np.reshape(yy, (1000, 1))

        # sample from conditional
        xit = torch.tensor([x_float]).view(1,1)
        xit = xit.repeat(Nsamples,1).to(device)
        z = torch.randn(Nsamples, dx, device=device)
        with torch.no_grad():
            Fz = F(torch.cat((xit, z), 1))
        Fz_numpy = Fz.cpu().numpy()

        # define true joint and normalize to get posterior
        post_i = pi.joint_pdf(np.array([[x_float]]), yy)
        post_i_norm_const = np.trapz(post_i[:,0], x=yy[:,0])
        post_i /= post_i_norm_const

        #Kernel Density Estimation
        kde_estimator = kde.KernelDensityEstimator(torch.cat((Fz, torch.zeros(Nsamples, 1, device=device)), 1), kernel = kde.GaussianKernel(bandwidth = .07))
        
        est_i = kde_estimator.forward(torch.cat((torch.from_numpy(yy), torch.zeros(Nsamples, 1, device=device)), 1))
        est_i[est_i < 0.0] = 0
        est_i[est_i > 0.0] = np.exp(est_i[est_i > 0.0])
        area = torch.trapezoid(est_i, dx = dy)
        est_i = est_i/area

        #take MSE loss
        kde_loss_inner += mse_loss(torch.from_numpy(post_i), est_i.resize(Nsamples, 1))
    F.eval()
    D.eval()


    #Average monotonicity percent over batches
    mon_percent = mon_percent/math.ceil(float(args.n_train)/bsize)
    monotonicity[ep] = mon_percent

    #Average generator and discriminator losses
    F_train[ep] = F_train_inner/math.ceil(float(args.n_train)/bsize)
    D_train[ep] = D_train_inner/math.ceil(float(args.n_train)/bsize)
    if ep != 0:
        loss_difference = F_train[ep]-F_train[ep-1]
    
    #Average training data loss
    loss_train[ep] = loss_train_inner/math.ceil(float(args.n_train)/bsize)

    #Average kde loss
    kde_loss[ep] = kde_loss_inner/math.ceil(float(args.n_train)/bsize)

    if ep % 30 == 0:
        # plt.plot(yy, post_i, label="post_i")
        # plt.plot(yy, est_i, label="est_i")
        # kde_samples = kde_estimator.sample(1000)
        # plt.hist(kde_samples[:,0], bins=200, density=True, label='KDE_Hist')
        # plt.show()

        plt.scatter(z1, x_true, label="real")
        plt.scatter(z1, fz_store.detach().numpy(), label="predict")
        plt.xlabel("z1")
        plt.ylabel("x_true/fz")
        plt.legend()
        plt.show()

    print('Epoch %3d, Monotonicity: %f, Generator loss: %f, Critic loss: %f' % \
         (ep, monotonicity[ep], F_train[ep], D_train[ep]))




#plot post_i vs est_i
plt.figure()
plt.plot(yy, post_i, label="post_i")
est_i = kde_estimator.forward(torch.cat((torch.from_numpy(yy), torch.zeros(Nsamples, 1, device=device)), 1))
est_i[est_i < 0] = 0
#est_i[est_i > 0.0] = np.exp(est_i[est_i > 0.0])
area = torch.trapezoid(est_i, dx = dy)
#est_i = est_i/area
plt.plot(yy, est_i, label="est_i")
kde_samples = kde_estimator.sample(1000)
plt.hist(kde_samples[:,0], bins=200, density=True, label='KDE_Hist')
plt.legend()
plt.show()

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
plt.subplot(1,4,3)
plt.plot(np.arange(args.n_epochs), loss_train.numpy(), label='Training data loss')
plt.xlabel('Number of epochs')
plt.legend()
plt.subplot(1,4,4)
plt.plot(np.arange(args.n_epochs), kde_loss.numpy(), label='kde_loss')
plt.xlabel('Number of epochs')
plt.legend()
plt.show()

# Plot densities
# define conditionals
xst = [-1.2, 0, 1.2]
Ntest = 1000

# define domain
y_dom = [-10,10]
yy = np.linspace(y_dom[0], y_dom[1], 1000)
yy = np.reshape(yy, (1000, 1))


#Sample each conditional
plt.figure()
for i,xi in enumerate(xst):

    # sample from conditional
    xit = torch.tensor([xi]).view(1,1)
    xit = xit.repeat(Ntest,1).to(device)
    z = torch.randn(Ntest, dx, device=device)
    with torch.no_grad():
        Fz = F(torch.cat((xit, z), 1))
    Fz_numpy = Fz.cpu().numpy()

    # define true joint and normalize to get posterior
    post_i = pi.joint_pdf(np.array([[xi]]), yy)
    post_i_norm_const = np.trapz(post_i[:,0], x=yy[:,0])
    post_i /= post_i_norm_const

    #Kernel Density Estimation
    kde_estimator = kde.KernelDensityEstimator(torch.cat((Fz, torch.zeros(Ntest, 1, device=device)), 1), kernel = kde.GaussianKernel(bandwidth = .07))
    kde_samples = kde_estimator.sample(1000)

        
    # plot density and samples
    plt.subplot(1,2,1)
    plt.plot(yy, post_i)
    plt.hist(Fz_numpy, bins=200, density=True, label='$x^* = '+str(xi)+'$')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(yy, post_i)
    plt.hist(kde_samples[:,0], bins=200, density=True, label='KDE | $x^* = '+str(xi)+'$')
    plt.legend()
plt.xlabel('$y$')
plt.ylabel('$\pi(y|x^*)$')
plt.subplot(1,2,1)
plt.xlabel('$y$')
plt.ylabel('$\pi(y|x^*)$')
plt.show()
