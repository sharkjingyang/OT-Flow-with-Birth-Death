# OTFlowProblem.py
import math
import torch
import numpy
from torch.nn.functional import pad
from src.Phi import *

import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg') # for linux server with no tkinter
# matplotlib.use('Agg') # assume no tkinter
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'inferno'
import os
import h5py
import datasets
from torch.nn.functional import pad
from matplotlib import colors # for evaluateLarge


def vec(x):
    """vectorize torch tensor x"""
    return x.view(-1,1)

def OTFlowProblem(x, Phi, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0,1.0] ):
    """

    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.

    :param x:       input data tensor nex-by-d
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    BD_alpha=0.1
    h = (tspan[1]-tspan[0]) / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 5, 0, 0), value=0)

    tk = tspan[0]
    g=torch.zeros(z.size(0),1).cuda()

    if stepper=='rk4':
        Phi_store=torch.zeros(z.size(0),1).cuda()
        d=z.size(1)-5
        # z    0 to d-1 is space , d is log
        for k in range(nt):

            

            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            nsample=z.size(0)
            one_tk=tk*torch.ones(z.size(0),1).cuda()
            x=z[:,:d]
            Phi_store=Phi(torch.cat((x,one_tk),dim=1))

        
            Phi_judge = (Phi_store-torch.mean(Phi_store))*(1/nt)*BD_alpha

            r=torch.rand(z.size(0),1).cuda()
            kill_inds=(-Phi_judge>r).nonzero()[:,0]
            dup_inds=(Phi_judge>r).nonzero()[:,0]
            # delta_n = len(dup_inds)-len(kill_inds)

            
            z = torch.cat((z,z[dup_inds,:].reshape(-1,d+5)),dim=0)
            

            survivor_inds = torch.tensor(list(set(range(z.shape[0]))-set(kill_inds.cpu().numpy())), dtype=torch.long).cuda()
            z = torch.index_select(z,0,survivor_inds)
            

            
            delta_n=z.size(0)-nsample
            if delta_n<0:
                clone_inds = torch.randint(nsample+delta_n, (-delta_n,), dtype=torch.long).cuda()
                z = torch.cat((z,torch.index_select(z,0,clone_inds)))
                
            
               
               
            if delta_n>0:
                kill_inds_b = (torch.randperm(z.shape[0], dtype=torch.int)[:delta_n]).numpy()
                survivor_inds_b = torch.tensor(list(set(range(z.shape[0]))-set(kill_inds_b)), dtype=torch.long).cuda()
                z = torch.index_select(z,0,survivor_inds_b)
                
           

            tk += h

    elif stepper=='rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    # ASSUME all examples are equally weighted
    costL  = torch.mean(z[:,-4])   ## 1/2 ||v^2||
    costC  = torch.mean(C(z))   
    costR  = torch.mean(z[:,-3])   ## regulization 
    cost_g=torch.mean(z[:,-2])    ##  1/2 g^2
    cost_alpha_tuta=torch.mean(z[:,-1])
    cs = [costL, costC+cost_alpha_tuta, costR, cost_g]
    # cs = [costL, costC, costR]
    return sum(i[0] * i[1] for i in zip(cs, alph)) , cs



def stepRK4(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, alph=alph)
    z += (1.0/6.0) * K

    return z

def stepRK1(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme.  Added for comparison, but we recommend stepRK4.
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z += (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    return z


def integrate(x, net, tspan , nt, stepper="rk4", alph =[1.0,1.0,1.0,1.0] ):
    """
        perform the time integration in the d-dimensional space
    :param x:       input data tensor nex-by-d
    :param net:     neural network Phi
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :param intermediates: bool, True means save all intermediate time points along trajectories
    :return:
        z - tensor nex-by-d+4, features at time t1
        OR zFull - tensor nex-by-d+3-by-nt+1 , trajectories from time t0 to t1 (when intermediates=True)
    """

    h = (tspan[1]-tspan[0]) / nt
    BD_alpha=0.1
    # initialize "hidden" vector to propagate with all the additional dimensions for all the ODEs
    d=x.size(1)
    z = pad(x, (0, 5, 0, 0), value=tspan[0])
    
    tk = tspan[0]

    
    zFull = torch.zeros( *z.shape , nt+1, device=x.device, dtype=x.dtype) # make tensor of size z.shape[0], z.shape[1], nt
    # x_ini = torch.zeros( x.shape , device=x.device, dtype=x.dtype)
    zFull[:,:,0] = z
    # x_ini=x
    
    rho_int=torch.zeros(z.size(0),1).cuda()
    rho_int=double_Gauss(z[:,0]).reshape(z.size(0),1)
    rho_all=torch.zeros(z.size(0),nt+1)
    # print(rho_all[:,0].size())
    rho_all[:,0]=(rho_int/(torch.exp(z[:,d]).reshape(z.size(0),1))).squeeze()
    if stepper == 'rk4':
        for k in range(nt):
            

            zFull[:,:,k+1] = stepRK4(odefun, zFull[:,:,k] , net, alph, tk, tk+h)
            
            z_temp=zFull[:,:,k+1]
            

            nsample=z_temp.size(0)
            one_tk=tk*torch.ones(z_temp.size(0),1).cuda()
            x=z_temp[:,:d]
            Phi_store=net(torch.cat((x,one_tk),dim=1))


            Phi_judge = (Phi_store-torch.mean(Phi_store))*(1/nt)*BD_alpha

            r=torch.rand(z_temp.size(0),1).cuda()
            kill_inds=(-Phi_judge>r).nonzero()[:,0]
            dup_inds=(Phi_judge>r).nonzero()[:,0]
            
            delta_n = len(dup_inds)-len(kill_inds)
          

            z_temp = torch.cat((z_temp,z_temp[dup_inds,:].reshape(-1,d+5)),dim=0)
            rho_int = torch.cat((rho_int,rho_int[dup_inds,:]),dim=0)
            

            

            survivor_inds = torch.tensor(list(set(range(z_temp.shape[0]))-set(kill_inds.cpu().numpy())), dtype=torch.long).cuda()
            z_temp = torch.index_select(z_temp,0,survivor_inds)
            rho_int = torch.index_select(rho_int,0,survivor_inds)
            
    
        
            delta_n=z_temp.size(0)-nsample
            if delta_n<0:
                clone_inds = torch.randint(nsample+delta_n, (-delta_n,), dtype=torch.long).cuda()
                z_temp = torch.cat((z_temp,torch.index_select(z_temp,0,clone_inds)))
                rho_int = torch.cat((rho_int,torch.index_select(rho_int,0,clone_inds)))
                
                
            
            if delta_n>0:
                kill_inds_b = (torch.randperm(z_temp.shape[0], dtype=torch.int)[:delta_n]).numpy()
                survivor_inds_b = torch.tensor(list(set(range(z_temp.shape[0]))-set(kill_inds_b)), dtype=torch.long).cuda()
                z_temp = torch.index_select(z_temp,0,survivor_inds_b)
                rho_int = torch.index_select(rho_int,0,survivor_inds_b)
            
            # print(rho_int.size())
            zFull[:,:,k+1]=z_temp
            rho_all[:,k+1]=(rho_int/(torch.exp(z_temp[:,d]).reshape(z.size(0),1))).squeeze()

            tk += h
        

            
    return zFull[:,0:d,:].squeeze(),zFull[:,d,:], rho_all  #output x and log(det) and rho_ini




def C(z):
    """Expected negative log-likelihood; see Eq.(3) in the paper"""
    d = z.shape[1]-5
    l = z[:,d] # log-det

    return -( torch.sum(  -0.5 * math.log(2*math.pi) - torch.pow(z[:,0:d],2) / 2  , 1 , keepdims=True ) + l.unsqueeze(1) )


def odefun(x, t, net, alph=[1.0,1.0,1.0,1.0]):
    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 5
    BD_alpha=0.1
    z = pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    gradPhi, trH = net.trHess(z)

    Phi_store=torch.zeros(x.size(0),1).cuda()
    Phi_store=net(z)

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
    dr = torch.abs(  -gradPhi[:,-1].unsqueeze(1) + alph[0] * dv  ) 
    d_g=0.5 * torch.pow(Phi_store-torch.mean(Phi_store), 2)*BD_alpha*BD_alpha
    d_alpha_tuta=-BD_alpha*(Phi_store-torch.mean(Phi_store))

    return torch.cat( (dx,dl,dv,dr,d_g,d_alpha_tuta) , 1  )



def odefun_3extra(x, t, net, alph=[1.0,1.0,1.0,1.0]):
    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    nex, d_extra = x.shape
    d = d_extra - 3
    
    z = pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    gradPhi, trH = net.trHess(z)

   

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
    dr = torch.abs(  -gradPhi[:,-1].unsqueeze(1) + alph[0] * dv  ) 
  

    return torch.cat( (dx,dl,dv,dr) , 1  )


def double_Gauss(x):
    a_1=1/(math.sqrt(2*math.pi*1)) *torch.exp(-torch.pow(x-3,2)/(2*1*1))
    a_2=1/(math.sqrt(2*math.pi*1)) *torch.exp(-torch.pow(x+3,2)/(2*1*1))
    
    return (a_1+a_2)/2





def plot_1d(net, x, nt_val, sPath, sTitle=""):
    """
    x - samples from rho_0
    y - samples from rho_1
    nt_val - number of time steps
    """

    d = net.d
    nSamples = x.shape[0]
    z_full,l_det,rho_full= integrate(x[:, 0:d], net, [0.0, 1.0], nt_val, stepper="rk4", alph=net.alph)
    # print(z_full.size())
    # z_full=torch.squeeze(z_full)  #n_sample * nt+1
    # l_det=torch.exp(l_det)      #n_sample * nt+1
    # rho_full=rho_int/l_det

    fig, ax = plt.subplots(5,5,figsize=(25,25))
    for i in range(5):
        for j in range(5):
            k=i*5+j+1
            # print(z_full[:,k-1].size())
            # print(rho_full[:,k-1].size())
            ax[i,j].scatter(z_full[:,k-1].cpu().numpy(),rho_full[:,k-1].cpu().numpy(),s=4)
            ax[i,j].set_ylim(0,1)
            ax[i,j].set_ylabel("rho (z) ")
            ax[i,j].set_xlabel("z ")

            # plt.subplot(3,3,k)
            # plt.scatter(z_full[:,k-1].cpu().numpy(),rho_full[:,k-1].cpu().numpy(),s=4)
            # plt.ylim((0,0.7))
            # plt.xlabel("z")
            # plt.xlabel("rho (z) ")
            # plt.title

    
    
    # plt.show()
    plt.savefig("image/density/"+sTitle+".jpg")
    plt.close()


    

    # z_full=torch.randn(4096,9)
    fig, ax = plt.subplots()
    for i in range(z_full.size(0)):
        ax.plot(z_full[i,:].cpu().numpy(),1/nt_val*torch.arange(nt_val+1).cpu().numpy(),c='b')
    # plt.show()
 
    # sPath = os.path.join(args.save, 'figs', sStartTime + '_{:04d}.png'.format(itr))
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))
    # plt.savefig(sPath, dpi=300)
    # plt.show()
    plt.savefig("image/trace/"+sTitle+".jpg")
    plt.close()