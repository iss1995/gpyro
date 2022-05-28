 #!/usr/bin/env python3
# %%
import numpy as np

import torch
import gpytorch

from copy import deepcopy 

# %%
def findK(system_dynamics_size):
    """
    @params size of optimization vector
    @returns number of convolutional filters
    """
    return int(system_dynamics_size/5-1)

def fromDynVecToDynamics(dyn_vec, k):
    '''
    dyn_vec = [M_N,M_B_conv,INfeat]. This is a function transforming the optimization vector to a dyb^namical system. 
    @params k : int - > convolutional filters used
    '''
    # dyn_vec = np.atleast_2d(dyn_vec).reshape(2*k+2,-1) 
    # dyn_vec = np.atleast_2d(dyn_vec).reshape(3*(k+1),-1) 

    dyn_vec = np.atleast_2d(dyn_vec).reshape(5*(k+1),-1) # just commented out, but why was it here?
    
    # if dyn_vec.shape[1] == 2*k+2:
    #     dyn_vec = dyn_vec.T
    # k = int(dyn_vec.shape[1]/2 - 1) #
    # dyn_vec_Mconv = dyn_vec[:k,:]
    # dyn_vec_Mel = np.atleast_2d(dyn_vec[k,:]).reshape(1,-1)
    # dyn_vec_Bconv = dyn_vec[k+1:-1,:]
    # dyn_vec_Bel = np.atleast_2d(dyn_vec[-1,:]).reshape(1,-1)
    # dyn_vec_Bconv1 = dyn_vec[ k+1 : 2*k,:]
    # dyn_vec_Bel1 = np.atleast_2d( dyn_vec[2*k,:]).reshape(1,-1)
    # dyn_vec_Bconv2 = dyn_vec[2*k+1:-1,:]
    # dyn_vec_Bel2 = np.atleast_2d(dyn_vec[-1,:]).reshape(1,-1)
    T_i_conv = dyn_vec[ :k ,: ]
    T_i = dyn_vec[ k,:]
    T_j_conv =  dyn_vec[  k+1 : 2*k+1 ,:]
    T_j =  dyn_vec[ 2*k+1 ,:]
    DT_amb_conv = dyn_vec[ 2*k+2 : 3*k+2 ,:]
    DT_amb = dyn_vec[ 3*k+2 ,:]
    IF_1_conv = dyn_vec[ 3*k+3 : 4*k+3 ,:]
    IF_1 = dyn_vec[ 4*k+3 ,:]
    IF_2_conv = dyn_vec[ 4*k+4 : 5*k+4 ,:]
    IF_2 = dyn_vec[ 5*k+4 ,:]


    # print("dyn_vec_conv ",dyn_vec_conv.shape)
    return np.vstack((
                    T_i_conv, T_j_conv, T_j_conv, T_j_conv, T_j_conv,
                    T_i, T_j, T_j, T_j, T_j,
                    DT_amb_conv, DT_amb_conv,
                    DT_amb, DT_amb,
                    IF_1_conv,IF_2_conv,
                    IF_1,IF_2
                    # dyn_vec_Mconv, dyn_vec_Mconv, dyn_vec_Mconv, dyn_vec_Mconv,
                    # dyn_vec_Mel, dyn_vec_Mel, dyn_vec_Mel, dyn_vec_Mel,
                    # dyn_vec_Bconv, dyn_vec_Bconv,
                    # dyn_vec_Bel, dyn_vec_Bel,
                    # dyn_vec_Bconv1, dyn_vec_Bconv1, dyn_vec_Bconv2, dyn_vec_Bconv2,
                    # dyn_vec_Bel1, dyn_vec_Bel1, dyn_vec_Bel2, dyn_vec_Bel2 
                    ))

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    # pad = np.full((len(a.shape),window),a[0]) # iss
    pad[-1] = window-1
    pad = list(zip(pad, np.ones(len(a.shape), dtype=np.int32)))
    # pad = list(zip(pad, np.ones(len(a.shape)))) # iss
    # a = np.pad(a, pad,mode='constant')
    a = np.pad(a, pad,mode='edge')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    try:
        strides = a.strides + (a.strides[1],) # not sure why this fails sometimes
    except:
        strides = a.strides + (a.strides[0],) # but when the other fails this works
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides) 

def linearModelCoef(temperatures,normalizer,k,models,likelihoods):
    """
    Infer coeficients for the linear models.
    """
    n = int(9*(k + 1)) # number of models
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_cuda = torch.from_numpy(temperatures/normalizer).float().cuda()
        feature_vec_cuda = likelihoods(*models(*[y_cuda for i in range(n)]))
        feature_vec = list(map(lambda x: x.mean.squeeze().cpu().detach().numpy(),feature_vec_cuda))
        return  fromDynVecToDynamics(feature_vec, k = k)

def moving_average(x, w):
    extended = np.pad(x,(int(w/2),int(w/2)-1),'edge')
    # print(f"extended.shape : {extended.shape}")
    # print(extended)
    return np.convolve(extended, np.ones(w), 'valid') / w

def movingAvg(x, w):
    """
    only pad towards one side.
    """
    if max(x.shape)>w:
        out = np.convolve(x, np.ones(w), 'valid') / w
    else:    
        extended = np.pad(x,(int(w)-1,0),'edge')
        out = np.convolve(extended, np.ones(w), 'valid') / w
    # print(f"extended.shape : {extended.shape}")
    # print(extended)
    return out

def setDevice(choice = None):
    '''
    set device for torch. Default is cpu (if available)
    @param choice: string with choice of device
    @return torch device
    '''

    import gc

    if choice is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gc.collect()
            torch.cuda.empty_cache()
            
        else:
            device = torch.device('cpu')
    
    else:
        device = torch.device(choice)

    return deepcopy(device)

def piecewiseLinear(x,y0,k1,k2):
    x0 = 0.5
    return np.piecewise( x, [x<x0], [lambda x : k1*x + y0-k1*x0, lambda x : k2*x + y0-k2*x0])

def checkConcave_(idx,vector):
    "Only locally"
    try:
        if vector[idx]>vector[idx+1] and vector[idx]>vector[idx-1]:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return None
