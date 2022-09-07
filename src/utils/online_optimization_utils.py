import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import torch
import gpytorch
import gc

from math import ceil
from gpytorch.mlls import SumMarginalLogLikelihood
from scipy.interpolate import interp1d
from copy import deepcopy as copy
from utils.generic_utils import setDevice, rolling_window_2D
from scipy.optimize import minimize,least_squares, Bounds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm

def objective(e : np.ndarray,y : np.ndarray) -> np.ndarray:
    """
    MAPE objective function
    @param e: error 
    @param y: target 
    @return: scalar MAPE
    """
    return np.mean( np.abs( (e/np.mean(np.abs(y))))  )

def regFun(a : np.ndarray, regularizer : float ) -> np.ndarray:
    """ 
    l2 regularizer. Leave coupling parameter out. 
    @param a: parameters
    @param regularizer: regularization strength
    @return: scalar regularization cost
    """
    return regularizer*np.linalg.norm(a[:-1])

class gTerm:
    def __init__(self,params : np.ndarray or None = None) -> None:
        """
        g term 
        @param params : array with G model parameters
        """
        if params is None:
            self.params = np.ones((3,1))
        else:
            self.params = params
        self.update(self.params)
    
    def __call__(self,h : np.ndarray,T : np.ndarray) -> np.ndarray:
        """
        @param h : array with heights, dim (n,)
        @param T : array with Temperatures, dim (n,)
        @return : array with G model output, dim (n,)
        """
        regressor = np.vstack(( np.ones_like(h), 1-h, 1-T)).T #dim (n,3)
        if len(self.params)>3:
            out = np.sum(regressor*self.params, axis = -1).squeeze() # dim (n,)
        else:
            out = np.matmul(regressor,self.params).squeeze()
        return out
    
    def update(self,params):
        """
        Update parameters
        @param params : array with G model parameters
        """
        self.params = np.atleast_2d(params).T

class mTerm:
    def __init__(self, neighbors : list, params : np.ndarray or None = None, d_grid : int = 27, d_grid_norm : int = 100, delta : np.ndarray or None = None, ambient_T : float= 0) -> None:
        """
        m term
        @param neighbors : list of lists containing idxs to neighboring nodes, e.g. neighbors_i = [j,k,l,m] with j,k,l,m being the idxs of i's neighbors
        @param params : array with M model parameters
        @param d_grid : grid spacing in mm
        @param d_grid_norm : normalization factor for grid spacing
        @param delta : array with boolean values indicating if node needs delta node or not
        @param ambient_T : ambient temperature
        """
        if params is None:
            self.params = np.ones((3,1))
        else:
            self.params = params
        self.update(self.params)

        self.neighbors = neighbors
        self.number_of_neighbors = np.asarray([len(neigh) for neigh in neighbors])
        self.d_grid = d_grid / d_grid_norm
        
        if delta is None:
            self.delta = np.zeros_like(self.number_of_neighbors).astype(np.bool8)
            self.delta[self.number_of_neighbors<4] = True
        else:
            self.delta = delta
        
        self.ambient_T = ambient_T
        
        # build map between neighbors
        self.neighbor_Ts = np.zeros((len(neighbors),4)) 
        self.idx_map = np.full_like(self.neighbor_Ts,-1).astype(np.int64)
        
        # create map
        for i,node_neighbors in enumerate(neighbors):
            for j,node_neigbor in enumerate(node_neighbors):
                self.idx_map[i,j] = node_neigbor
        
        # which elements are still nan?
        self.no_connections = self.idx_map < 0
        self.idx_map[self.no_connections] = 0 
        

    def __call__(self,T : np.ndarray) -> np.ndarray:
        """
        @param T : array with Temperatures, dim (n,)
        @return : array with M model output, dim (n,)
        """
        self.neighbor_Ts = T[self.idx_map]
        self.neighbor_Ts[self.no_connections] = 0 # zero out the non existing connections
        laplacians = (self.number_of_neighbors*T - np.sum(self.neighbor_Ts, axis = -1))/(self.d_grid) # dim (n,)
        regressor = np.vstack(( T, laplacians, (T - self.ambient_T)*self.delta)).T #dim (n,3)
        if len(self.params)>3:
            out = np.sum(regressor*self.params, axis = -1).squeeze() # dim (n,)
        else:
            out = np.matmul(regressor,self.params).squeeze()
        
        return out
    
    def update(self,params : np.ndarray) -> None:
        """
        Update parameters
        @param params : array with M model parameters
        """
        self.params = np.atleast_2d(params).T
    
    def updateTambient(self,ambient_T : float) -> None:
        """
        Update ambeinet temperature
        @param ambient_T : ambient temperature
        """
        self.ambient_T = ambient_T
    
    def calculateLaplacians(self,T: np.ndarray) -> np.ndarray:
        """
        for training
        @param T : array with Temperatures
        """
        T = np.atleast_2d(T)
        neighbor_Ts = T[self.idx_map,:]
        neighbor_Ts[self.no_connections,:] = 0 # zero out the non existing connections
        laplacians = (np.multiply(self.number_of_neighbors,T.T).T - np.sum(neighbor_Ts, axis = 1))/(self.d_grid) # dim (n,N)
        return laplacians
    
    def oneStepPropagation(self,T,laplacians,deltas):
        """
        for training
        @param T : array with Temperatures
        @param laplacians : array with laplacians
        @param deltas : boolean array indicating the need of deltas
        """
        regressor = np.vstack(( T, laplacians, (T - self.ambient_T)*deltas)).T #dim (N,3)
        return np.matmul(regressor,self.params).squeeze() # dim (N,)

class fTerm:
    def __init__(self, bell_resolution : int = 101, params : np.ndarray or None  = None) -> None:
        """
        f term
        @param bell_resolution : int with length of peak intervals' used to create the excitation signal. If too short, then the peaks won't fit 
        @param params : array with 1 model parameters ( lengthscale)
        """
        if params is None:
            self.params = np.ones((1,))
        else:
            self.params = params
        self.update(self.params)

        self.bell_resolution = bell_resolution
    
    def __call__(self, peak_idxs : list, array_length : int, multiple_params = False):
        """
        @param peak_idxs : list of peaks at each node
        @param T : array with Temperatures, dim (n,)
        @param multiple_params : boolean indicating if multiple lengthscale parameters are used. If true, then params should be provided
        @return : array with F model output, dim (n,)
        """
        out = []
        if not multiple_params:
            params = np.full(len(peak_idxs), self.params)
        else:
            params = self.params

        for (node_peaks,param) in zip(peak_idxs,params):
            out.append( Fsequence( node_peaks, param, array_length, bell_resolution = self.bell_resolution) )
        return out
    
    def update(self,params : np.ndarray) -> None:
        """
        Update parameters
        @param params : array with 1 model parameters ( lengthscale)
        """
        self.params = params

class totalModel:
    def __init__(self, f_seq :np.ndarray , h : np.ndarray , g: gTerm, m : mTerm, N : int = -1, theta_g :np.ndarray or None = None,\
                                 theta_m :np.ndarray or None = None, theta_p :np.ndarray or None = None,Dt : float = 0.5) -> None:
        """
        @param f_seq : array with precalculated attention values, dim (n,N)
        @param h : array with torch heights, dim (n,N)
        @param g : gTerm object
        @param m : mTerm object
        @param N : int with extrapolation horizon
        @param theta_g : array with g model parameters, dim (n,N)
        @param theta_m : array with m model parameters, dim (n,N)
        @param theta_p : array with p model parameters, dim (n,N)
        @param Dt : float with time step
        """

        if N == -1 or N>len(f_seq):
            self.N = len(f_seq[0])
        else:
            self.N = N
        
        self.g = g
        if theta_g is None:
            self.theta_g = np.ones((N,3))
        else:
            self.theta_g = theta_g
        self.g.update(theta_g)
        
        self.m = m
        if theta_m is None:
            self.theta_m = np.ones((N,3))
        else:
            self.theta_m = theta_m
        self.m.update(theta_m)
        
        if theta_p is None:
            self.theta_p = np.ones((N,1))
        else:
            self.theta_p = theta_p

        self.f_seq = f_seq
        self.h = h
        
        self._instantiate_h_int(h,self.N,Dt)
        self._instantiate_f_int(f_seq,self.N,Dt)
        self._instantiate_theta_m_int(theta_m,self.N,Dt)
        self._instantiate_theta_g_int(theta_g,self.N,Dt)
        self._instantiate_theta_p_int(theta_p,self.N,Dt)
    
    def __call__(self,t : float,T : np.ndarray) -> np.ndarray:
        """
        one step propagation
        @param t : time
        @param T : array with Temperatures, dim (n,)
        @return : temperatures at next timestep, dim (n,)
        """
        h_t = self.h_int(t)
        f_t = self.f_int(t)
        
        theta_M = self.theta_m_int(t)
        theta_G = self.theta_g_int(t)
        theta_p = self.theta_p_int(t)

        self.updateM(theta_M)
        self.updateG(theta_G)

        DT = (1-f_t) * self.m(T) + theta_p * f_t * self.g(h_t,T)
        return DT + T
    
    def updateM(self,theta_m : np.ndarray) -> None:
        """
        update m model parameters
        @param theta_m : array with m model parameters
        """
        self.theta_m = theta_m
        self.m.update(theta_m)

    def updateG(self,theta_g : np.ndarray) -> None:
        """
        update g model parameters
        @param theta_g : array with g model parameters
        """
        self.theta_g = theta_g
        self.g.update(theta_g)

    def _instantiate_h_int(self,h,N,Dt) -> None:
        self.h_int = interp1d(np.linspace(0,N*Dt-Dt,N),h)
        return None        

    def _instantiate_f_int(self,f,N,Dt) -> None:
        self.f_int = interp1d(np.linspace(0,N*Dt-Dt,N),f)
        return None

    def _instantiate_theta_m_int(self,theta_m,N,Dt) -> None:
        self.theta_m_int = interp1d(np.linspace(0,N*Dt-Dt,N),theta_m, axis = 1)
        return None

    def _instantiate_theta_g_int(self,theta_g,N,Dt) -> None:
        self.theta_g_int = interp1d(np.linspace(0,N*Dt-Dt,N),theta_g, axis = 1)
        return None

    def _instantiate_theta_p_int(self,theta_p,N,Dt) -> None:
        self.theta_p_int = interp1d(np.linspace(0,N*Dt-Dt,N),theta_p)
        return None       

class loss_F:
    def __init__(self, state_excitation_idxs : list, excited_points : list, regularizer : float, f : fTerm, parameter_scale : float = 10) -> None:
        """
        class for optimization of f model parameters
        @param state_excitation_idxs : list of lists with indices of peaks excited nodes 
        @param excited_points : list of point objects that are excited
        @param regularizer : float with regularizer
        @param f : fTerm object
        @param parameter_scale : float with parameter scale
        """
        self.state_excitation_idxs = state_excitation_idxs 
        self.excited_points = excited_points 
        self.f = f
        self.regularizer = regularizer
        self.parameter_scale = parameter_scale

    
    def __call__(self, params : np.ndarray) -> float:
        """
        Callable for optimization
        @param params : array with f model parameters, dim (n,)
        @return : loss value
        """
        # update lengthscale
        self.f.update(params*self.parameter_scale)
        
        # check if bell resolution is too little
        f_span = 3*ceil(params*self.parameter_scale)
        if f_span>self.f.bell_resolution/2:
            self.f.bell_resolution = 2*f_span

        # calculate attention
        half_resolution = int(self.f.bell_resolution/2)
        f_segment = self.f([[half_resolution]], self.f.bell_resolution)[0]

        # compare attention to data
        residuals = 0.
        for (idxs,p) in zip(self.state_excitation_idxs,self.excited_points):
            array_len = len(p.T_t_use)
            if len(idxs) == 0:
                continue
            new_residuals = 0.
            for idx in idxs:
                starting_idx = np.max([ 0, half_resolution - idx ] )
                ending_idx = np.min([ array_len - idx, half_resolution ] )
                f_segment_eval = f_segment[starting_idx : half_resolution + ending_idx ] # +1 is correct?
                
                T_segment_eval = p.T_t_use[idx - half_resolution + starting_idx : idx + ending_idx]
                coef = np.max(T_segment_eval) - T_segment_eval[0]
                error = (coef*f_segment_eval + np.full_like(f_segment_eval,T_segment_eval[0])) - T_segment_eval
                new_residuals += np.mean(error**2)
            
            new_residuals /= len(idxs)
            residuals += new_residuals + self.regularizer * np.linalg.norm(params)
    
        return residuals

class loss_G:
    def __init__(self, state_excitation_idxs : list, excited_points : list, g :gTerm, f:fTerm, lengthscale : float = 10, \
                                    regularizer : float = 1e-2, parameter_scale : float = 1, test_ratio : float = 0.) -> None:
        """
        class for optimization of g model parameters
        @param state_excitation_idxs : list of lists with indices of peaks excited nodes 
        @param excited_points : list of point objects that are excited
        @param g : gTerm object
        @param f : fTerm object
        @param lengthscale : float with lengthscale
        @param regularizer : float with regularizer
        @param parameter_scale : float with parameter scale
        @param test_ratio : float with ratio of test data
        """
       
        self.state_excitation_idxs = state_excitation_idxs
        self.g = g 
        self.f = f
        f.update(lengthscale)
        self.lengthscale = lengthscale
        self.parameter_scale = parameter_scale
        self.excited_points = excited_points
        self.regularizer = regularizer
        self.test_ratio = test_ratio
        self.best_test_score = 1e10
    
    def __call__(self, params : np.ndarray) -> float:
        """
        Callable for optimization
        @param params : array with g model parameters, dim (n,)
        @return : loss value
        """
       
        # update lengthscale
        self.g.update(params*self.parameter_scale)
        # calculate attention
        half_resolution = int(self.f.bell_resolution/2)
        f_segment = self.f([[half_resolution]], self.f.bell_resolution)[0] # TODO: why different [] from loss_f?
        
        # learn g with attention
        residuals = 0.
        all_f = []  
        all_h = []  
        all_T = []  
        max_length = 0
        for (idxs,p) in zip(self.state_excitation_idxs,self.excited_points):
            if len(idxs) == 0:
                continue

            for idx in idxs:
                # align and gather features
                activated_band = 3 * np.ceil(self.lengthscale).astype(np.int64)

                starting_idx_local =  half_resolution - activated_band + np.max([0,  activated_band - idx ] )
                ending_idx_local = half_resolution

                span = ending_idx_local - starting_idx_local

                starting_idx_global = idx - span
                ending_idx_global = idx

                all_f.append( f_segment[starting_idx_local : ending_idx_local ] ) # +1 is correct?
                all_T.append( p.T_t_use[starting_idx_global : ending_idx_global] )
                all_h.append( p.excitation_delay_torch_height_trajectory[starting_idx_global : ending_idx_global] )

                max_length = np.max([ max_length, ending_idx_global - starting_idx_global])
        

        # create feature matrix
        if max_length>0:
            if self.test_ratio>0:
                idx_train,idx_test = train_test_split( np.arange(len(all_f)), test_size=self.test_ratio, shuffle=True )
            else:
                rng = np.random.default_rng()
                idx_train = rng.choice(len(all_f), int(len(all_f)*(1-self.test_ratio)), replace=False)
                idx_test = []
            # zero padding for training sequences with less than max_length elements 
            Ts = np.zeros(( max_length, len(all_f)))
            Hs = np.zeros_like(Ts)
            Fs = np.zeros_like(Ts)

            for i,( T_node_idx,H_node_idx,F_node_idx)  in enumerate(zip(all_T,all_h,all_f)):
                Ts[:len(T_node_idx),i] = T_node_idx
                Hs[:len(T_node_idx),i] = H_node_idx
                Fs[:len(T_node_idx),i] = F_node_idx

            # unroll g
            residuals = self._unrollResponse(Ts,Fs,Hs,idx_train) + self.regularizer * np.linalg.norm(params)
            if len(idx_test)>0:
                test_residuals = self._unrollResponse(Ts,Fs,Hs,idx_test)
            else:
                test_residuals = self._unrollResponse(Ts,Fs,Hs,idx_train) 
            
            if test_residuals < self.best_test_score:
                self.best_test_score = test_residuals
                self.best_params = params 
        
        else:
            self.best_params = params*0. 
            residuals = 0.
        return residuals
    
    def _calculateFsequences(self) -> None:
        excitation_array = np.asarray([p.input_idxs for p in self.excited_points])
        self.f_seq = np.asarray(self.f(excitation_array,len(excitation_array)))[:,self.state_level_idxs]
        return None

    def _unrollResponse(self,Ts,Fs,Hs,node_idx_pairs) -> float:
        
        response = Ts[ 0, node_idx_pairs]
        history = [response.copy()]
        for (f_s, h_s) in zip( Fs[:,node_idx_pairs], Hs[:,node_idx_pairs]):
            response += f_s * self.g(h_s, response)
            history.append(response.copy())

        # if Ts is 0 (excitation interval finished), then history is 0 too
        history = np.vstack(history)[:-1,:]
        history[np.where(Ts[:,node_idx_pairs]==0)] = 0

        error = history - Ts[ :, node_idx_pairs]
        residuals = np.mean( (error / np.abs( Ts[ :, node_idx_pairs] + 1e-4))**2 )
        return residuals
    
    def _dbgResponse(self,range_start, range_end, Ts, history, node_idx_pairs) -> None:
        for idx_ in range(range_start,range_end,1):
            plt.figure()
            T_ = Ts[:,node_idx_pairs[idx_]]
            plt.plot(T_, label = "nominal")
            plt.plot(history[:,idx_], label = "predicted")
            plt.title(idx_)
        plt.legend()
        return None

class loss_M:
    def __init__(self, state_level_idxs : list, training_points : list, m :mTerm,  g :gTerm, f:fTerm, lengthscale = 10, theta_G = np.array([0,0,0]) , regularizer = 1e-2, \
                                N = 1, parameter_scale = 1,test_ratio = 0.25,random_state = 934, warp_loss_ = False, wrapping_fun = np.arctan, grad_relaxation = 0.1) -> None:

        """
        class for optimization of m model parameters

        @params state_level_idxs: list of lists with indices poinitng to each state
        @params training_points: list of points that are to be used for training
        @params m: mTerm object
        @params g: gTerm object
        @params f: fTerm object
        @params lengthscale: lengthscale of the RBF kernel
        @params theta_G: parameters of the gTerm
        @params regularizer: regularization parameter
        @params N: number of points to be used for training
        @params parameter_scale: scale of the parameters
        @params test_ratio: ratio of test data
        @params random_state: random state
        @params warp_loss_: if True, the loss is warped by the wrapping function
        @params wrapping_fun: function that is used to warp the loss
        @params grad_relaxation: relaxation of the gradient
        """

        self.m = m
        self.g = g 
        self.f = f

        self.f.update(lengthscale)
        self.g.update(theta_G)

        self.training_points = training_points
        self.state_level_idxs = state_level_idxs
        self.parameter_scale = parameter_scale
        self.regularizer = regularizer
        self.N = N
        self._createFeatureTargets(test_ratio,random_state)
        self.best_test_score = 1e10
        self.best_params = None
        self.grad_relaxation = grad_relaxation
        if warp_loss_:

            self.warp_loss = lambda x: wrapping_fun(grad_relaxation * x**2)
        else:
            self.warp_loss = lambda x: grad_relaxation*x 
    
    def __call__(self, params : np.ndarray) -> float:
        """
        @params params: parameters of the mTerm
        @return: loss
        """
        # update lengthscale
        self.m.update(params[:-1]*self.parameter_scale)

        if self.N == 1:
            # TODO : current implementation of M propagates system by one timestep and accepts a thermal field as input. Rethink of how to implement 
            m_val = self.m.oneStepPropagation( self.x, self.laplacians, self.deltas)
            preds = (1-self.f_seq[self.train_idxs])*m_val[self.train_idxs] + params[-1]*self.f_seq[self.train_idxs]*self.g(self.h[self.train_idxs],self.x[self.train_idxs])
            error = self.y[self.train_idxs] - preds

            test = (1-self.f_seq[self.test_idxs])*m_val[self.test_idxs] + params[-1]*self.f_seq[self.test_idxs]*self.g(self.h[self.test_idxs],self.x[self.test_idxs])

            test_error = self.y[self.test_idxs] - test

            loss = objective(error,self.y[self.train_idxs]) + regFun(params,self.regularizer)
            if len(self.test_idxs)>0:
                test_loss = objective(test_error,self.y[self.test_idxs])
            else:
                test_loss = objective(error,self.y[self.train_idxs])
        else:
            assert 1, "N>1 not implemented."

        if test_loss<self.best_test_score:
            self.best_params = params
            self.best_test_score = test_loss
    
        return self.warp_loss( loss ) 
    
    def _createFeatureTargets(self,test_size : float,random_state : int) -> None:
        all_temperatures = np.vstack([p.T_t_use for p in self.training_points])
        all_laplacians = self.m.calculateLaplacians(all_temperatures)

        temperatures = [p.T_t_use[idxs[:-1]] for (p,idxs) in zip(self.training_points,self.state_level_idxs)]
        delta_map = self.m.delta
        deltas = [np.ones_like(temp)*delta_val for (temp,delta_val) in zip(temperatures,delta_map)]
        laplacians = [lap[idxs[:-1]] for (lap,idxs) in zip( all_laplacians, self.state_level_idxs)]
        delta_temperatures = [np.diff(p.T_t_use[idxs]) for (p,idxs) in zip(self.training_points,self.state_level_idxs)]
        heights = [p.excitation_delay_torch_height_trajectory[idxs[:-1]] for (p,idxs) in zip(self.training_points,self.state_level_idxs)]
        excitation_idxs = [p.input_idxs for p in self.training_points]

        if self.N == 1:
            self.x = np.hstack(temperatures )
            self.laplacians = np.hstack(laplacians)
            self.y = np.hstack(delta_temperatures )
            self.h = np.hstack(heights)
            self.deltas = np.hstack(deltas)
            f_seq = []
            length = len(self.training_points[0].T_t_use)
            for (idxs, excite) in zip(self.state_level_idxs,excitation_idxs):
                f_seq.append( self.f( [excite], length )[0][idxs[0:-1]]  )
            
            self.f_seq = np.hstack(f_seq)

            
        else:
            self.x = rolling_window_2D(temperatures[:,:-1], self.N)[:,self.state_level_idxs]
            self.y = rolling_window_2D(delta_temperatures, self.N)[:,self.state_level_idxs]

            self.h = rolling_window_2D(heights,self.N)[:,self.state_level_idxs]
            self.f_sec = rolling_window_2D(f_seq)
            
            
        idxs = np.arange(len(self.x)).astype(np.int64)
        if test_size > 0:
            self.train_idxs,self.test_idxs = train_test_split(idxs, test_size = test_size, shuffle = True, random_state = random_state)
        else:
            rng = np.random.default_rng()
            self.train_idxs = rng.choice(len(idxs), int(len(idxs)*(1-test_size)), replace=False)
            self.test_idxs = []
        return None

class GPThetaModel(gpytorch.models.ExactGP):
    def __init__(self, train_x : torch.Tensor, train_y : torch.Tensor, likelihood :gpytorch.likelihoods, mean_len_prior : float= 0.025, var_len_prior : float = 0.1):
        """
        Model for each weight
        @param train_x: training inputs
        @param train_y: training targets
        @param likelihood: likelihood
        @param mean_len_prior: mean lengthscale prior
        @param var_len_prior: variance lengthscale prior
        """
        
        super(GPThetaModel, self).__init__(train_x, train_y, likelihood)
        
        lengthscale_prior = gpytorch.priors.NormalPrior(mean_len_prior, var_len_prior) # good results 4/8
        self.mean_module = bilinearMean(train_x,train_y)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5,lengthscale_prior=lengthscale_prior) )

    def forward(self, x : torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        @param x: inputs
        @return: MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class bilinearMean(gpytorch.means.Mean):                                                                                                                                                                        
    def __init__(self, xs : torch.Tensor, ys : torch.Tensor, batch_shape=torch.Size()):
        """
        You need at least 3 datapoints on each direction
        @param xs: training inputs
        @param ys: training targets
        @param batch_shape: batch shape
        """
        super().__init__()

        A_p = torch.stack( (xs[-3:], torch.ones((3,))) ).T
        sol_p,*_ = torch.linalg.lstsq(A_p,ys[-3:])
        self.w_p = sol_p[0]
        self.b_p = sol_p[1]

        # linear model for negative states
        A_n = torch.stack( (xs[:3], torch.ones((3,))) ).T
        sol_n,*_ = torch.linalg.lstsq(A_n,ys[:3])
        self.w_n = sol_n[0]
        self.b_n = sol_n[1]
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        @param x : inputs
        @return : bilinear mean
        """

        sigm_p = self.sigmoid(x)
        sigm_n = 1 - sigm_p

        ones = torch.ones_like(x)
        return ( self.pos(sigm_p*x,sigm_p*ones) + self.neg(sigm_n*x,sigm_n*ones) ).T
    
    def sigmoid(self,x,alpha = 20):
        return 1/(1+torch.exp(-alpha*x))
    
    def pos(self,x,o):
        return self.w_p*x + self.b_p*o
    
    def neg(self,x,o):
        return self.w_n*x + self.b_n*o
    
def optimizeF( states, excited_points : list,  param_f0 : np.ndarray, bounds : tuple or Bounds , \
                f : fTerm, F_reg : float = 1e-2, underestimate_lengthscales : float = 0)-> np.ndarray:
    """
    Optimize the f term
    @param states: iterable with states
    @param excited_points: excited points
    @param param_f0: initial parameters
    @param bounds: bounds for the parameters
    @param f: f term
    @param F_reg: regularization for the f term
    @param underestimate_lengthscales: underestimate the lengthscale by this factor
    @return: optimized parameters
    """
    #  learn F
    excitations = []
    for state_level in states:
        point_excitations_on_level_map = [] # 1 array for each point telling you which peaks to consider
        for p in excited_points:
            point_excitations_on_level_map.append(  p.input_idxs[ p.excitation_delay_torch_height == state_level ] )
    
        excitations.append( point_excitations_on_level_map )

    ## learn the lengthscale
    parameters_per_state_level = []
    for i,state_level_idxs in enumerate(excitations):
        optimizer = loss_F(state_level_idxs,excited_points, regularizer= F_reg, f = f)
        res = minimize( optimizer , param_f0, bounds= bounds, method= "L-BFGS-B", tol = 1e-12 )
        param_f0 = res.x
        parameters_per_state_level.append(res.x[0]) # omit the last one

        if i == 0:
            first_state_param_f0 = res.x

    ## scale lengthscales
    lengthscales = 10* np.abs(parameters_per_state_level ) * (1-underestimate_lengthscales)
    return lengthscales, excitations, param_f0

def optimizeG( f : fTerm, g : gTerm, states, excitations : list, excited_points : list, param_g0 : np.ndarray, \
                            lengthscaleModel : interp1d, bounds : tuple or Bounds , G_reg = 1e-2 ,test_ratio = 0.0):
    """
    Optimize the g term
    @param f: f term
    @param g: g term
    @param states: iterable with states
    @param excitations: list of lists with indices of excited points
    @param excited_points: list of point objects that are excited
    @param param_g0: initial parameters
    @param lengthscaleModel: lengthscale model
    @param bounds: bounds for the parameters
    @param G_reg: regularization for the g term
    @param test_ratio: ratio of test points
    @return: optimized parameters
    """
    #  learn G
    ## learn the lengthscale
    parameters_per_state_level = []
    fun_scores_per_state_level = []
    for i,(state_level_idxs,state) in enumerate(zip(excitations,states)):
        lengthscale = lengthscaleModel(state)
        optimizer = loss_G(state_level_idxs, excited_points, g, f, regularizer = G_reg, lengthscale = lengthscale, test_ratio=test_ratio)
        res = least_squares( optimizer , param_g0, bounds= bounds, ftol = 1e-12, method= "trf", loss = "arctan")
        if not np.all(optimizer.best_params==0):
            param_g0 = res.x
        parameters_per_state_level.append(optimizer.best_params) # omit the last one
        fun_scores_per_state_level.append(optimizer.best_test_score)
        if i == 0:
            first_state_param_g0 = res.x

    # param_g0 = first_state_param_g0
    coefficients = np.asarray(parameters_per_state_level)
    fun_scores = np.asarray(fun_scores_per_state_level)
    return coefficients, param_g0, fun_scores

def optimizeM( m : mTerm, f : fTerm, g : gTerm, states, training_points : list,  param_m0 : np.ndarray, lengthscaleModel : interp1d, gCoefModel : interp1d,\
     bounds: tuple or Bounds , M_reg : float = 1e-2,test_ratio : float = 0.0, wrapping_function = np.arctan, wrap_function_ : bool = False):
    """
    Optimize the m term
    @param m: m term
    @param f: f term
    @param g: g term
    @param states: iterable with unique state values
    @param training_points: list of point objects that are used for training
    @param param_m0: initial parameters
    @param lengthscaleModel: lengthscale model
    @param gCoefModel: g coefficients model
    @param bounds: bounds for the parameters
    @param M_reg: regularization for the m term
    @param test_ratio: ratio of test points
    @param wrapping_function: function to wrap the loss function
    @param wrap_function_: whether to wrap the loss function
    @return: optimized parameters
    """
    # find state levels for each point
    layer_idxs = []
    for state in states:
        point_layer_idxs = []
        for p in training_points:
            tmp = np.where(p.excitation_delay_torch_height_trajectory == state)[0]
            point_layer_idxs.append( tmp.astype(np.int64) )
        layer_idxs.append(point_layer_idxs)
    
    options = {'maxcor': 10, 'gtol': 1e-12, 'eps': 1e-10, 'maxfun': 15000, 'maxiter': 15000, 'iprint': 50, 'maxls': 20, 'finite_diff_rel_step': None}
    parameters_per_state_level = []
    fun_scores_per_state_level = []
    for i,(state_level_idxs,state) in enumerate(zip(layer_idxs,states)):
        lengthscale = lengthscaleModel(state)
        theta_G = gCoefModel(state)
        optimizer = loss_M(state_level_idxs, training_points, m, g, f,lengthscale = lengthscale, theta_G = theta_G, regularizer = M_reg, test_ratio=test_ratio, warp_loss_= wrap_function_, wrapping_fun = wrapping_function)
        res = least_squares( optimizer , param_m0, ftol = 1e-12, bounds= bounds, method= "trf", loss = "arctan")
        param_m0 = res.x
        parameters_per_state_level.append(optimizer.best_params) # omit the last one
        fun_scores_per_state_level.append(optimizer.best_test_score)
        if i == 0:
            first_state_param_m0 = res.x

    param_m0 = first_state_param_m0
    coefficients = np.asarray(parameters_per_state_level)
    fun_scores = np.asarray(fun_scores_per_state_level)
    return coefficients, param_m0, fun_scores

def pointsOnBoundaries_(experiment):
    """
    find the points on the boundaries of the domain
    @param experiment: experiment object
    @return: list with bools indicating if a point is on the boundaries
    """
    
    points_used_for_training = [p for p in experiment.Points if p._hasNeighbors_()]
    coordinate_list = [p.coordinates for p in points_used_for_training]
    coordinates = np.vstack(coordinate_list)
    boundaries = findDomainLimits(coordinates)
    on_boundary_ = [_isOnBoundary_(p,boundaries) for p in points_used_for_training]

    return on_boundary_

def findDomainLimits(coordinates):
    """
    For rectangular domain.
    """
    
    xMax = np.max(coordinates[:,0])
    xMin = np.min(coordinates[:,0])
    yMax = np.max(coordinates[:,1])
    yMin = np.min(coordinates[:,1])
    return (xMax,xMin,yMax,yMin)

def _isOnBoundary_(p,boundaries):
    
    [xMax,xMin,yMax,yMin] = boundaries
    coordinates = p.coordinates 
    if coordinates[0] == xMin or coordinates[0] == xMax:
        periphery_ = True
    elif coordinates[1] == yMin or coordinates[1] == yMax:
        periphery_ = True
    else:
        periphery_ = False

    return periphery_

def splitToLayers( points_used_for_training : list, tool_height_trajectory, debug = True):
    """
    @params points_used_for_training: list of points used for training
    @params tool_height_trajectory: tool height trajectory
    @returns list of indices that correspond to unique heights/states 
    """
    # find all deposition height levels -> you have to look at points that get ecited 
    excited_points = [ p for p in points_used_for_training if len(p.input_idxs)>0]
    height_levels = np.unique(excited_points[0].excitation_delay_torch_height) # consider that all points get excited on every layer 

    # find deposition idxs
    layer_idxs = []
    for height_level in height_levels:
        tmp = np.where(tool_height_trajectory == height_level)[0]
        start = tmp[0]
        end = tmp[-1]
        layer_idxs.append( np.arange(start,end+1,dtype = np.int64) )
    
    if debug:
        print(f"len(layer_idxs) : {len(layer_idxs)} and layer_idxs[0].shape : {layer_idxs[0].shape}")
        print(f"unique_height_values {height_levels}")
        print(f"tool_heights.shape : {tool_height_trajectory.shape}")
        print(f"layer_idxs[0].shape : {layer_idxs[0].shape}")

    return layer_idxs, height_levels

def Fsequence( peak_idxs : list, lengthscales : np.ndarray , array_length : int, bell_resolution : int = 101) -> np.ndarray:
    """
    Generate F sequence for each node.
    @param peak_idxs: list of peak idxs for each node
    @param lengthscales: array of lengthscale for each node
    @param array_length: length of the array to be generated
    @param bell_resolution: resolution of the bell curve
    @returns F: F sequence for each node
    """

    lengthscales = np.atleast_1d(lengthscales)
    out = np.zeros((array_length,))
    for (peak_idx,lengthscale) in zip(peak_idxs,lengthscales):
        assert peak_idx<array_length, "peak index larger than the array length"
        # form the right half-bell
        input_feature_span = 3*np.ceil(lengthscale).astype(np.int64) # how many non 0 elements to expect in the half bell 
        if input_feature_span>bell_resolution/2:
            bell_resolution = 2*input_feature_span
        bell_top_idx = int(bell_resolution/2) + 1 # where the bell top is
        bell = unitBellFunctionDerivative(lengthscale, resolution = bell_resolution) 
        bell[ bell<0 ] = 0  
        half_bell_shape = bell[ np.max( [bell_top_idx-input_feature_span,0] ) : bell_top_idx] # keep the non-0 bell elements

        # indexes to insert new half bell
        first_half_bell_idx_on_input_trajectory = np.max( [peak_idx - input_feature_span, 0] ) 
        first_half_bell_idx_on_bell = np.max([input_feature_span + first_half_bell_idx_on_input_trajectory - peak_idx, 0 ])

        # check if you are about to over-write a previous excitaion
        more_than_0 = out[first_half_bell_idx_on_input_trajectory : peak_idx] > 0
        if np.sum(more_than_0)>0:
            final_idx_with_non0_value = len(more_than_0) - np.argmax(more_than_0) - 1

            # move first idx to a non-written idx
            first_half_bell_idx_on_input_trajectory += final_idx_with_non0_value 
            first_half_bell_idx_on_bell = final_idx_with_non0_value

        
        # insert the half-bell to the input sequence
        out[ first_half_bell_idx_on_input_trajectory : peak_idx ] = half_bell_shape [ first_half_bell_idx_on_bell :  ]

    return out

def unitBellFunctionDerivative( lengthscale, resolution = 101, center = 0.0):

    bell = unitBellFunction(lengthscale, resolution, center)
    delta_bell = np.zeros_like(bell)
    delta_bell[1:] = np.diff(bell) 
    delta_bell[0] = delta_bell[1]
    scale = np.max(delta_bell)
    if scale<=1e-3:
        scale = 1

    return delta_bell/scale 

def unitBellFunction( lengthscale, resolution = 101, center = 0.0):
   
    x = np.linspace( -resolution/2, resolution/2,resolution)
    out = np.exp( -(( (x -  center) / lengthscale )**2) / (2**0.5) )

    return out 

def halfBellLegthscaleModel( lengthscale_models : np.ndarray, states : np.ndarray) -> interp1d:
    model = interp1d(states,lengthscale_models)
    return model

def modelsForInputCoefs(coefs :np.ndarray, heights : np.ndarray) -> interp1d:
    return interp1d(heights,coefs,axis = 0)

def GPOpt( models : gpytorch.models.IndependentModelList, likelihoods : gpytorch.likelihoods.LikelihoodList, learning_rate : float = 0.1, max_training_iter : int = 4000, \
                            loss_thress : float = 1e-3, no_improvement_thress : int = 200, gp_model_to_load : str = '', _device : str or None = None, messages_ : bool = False ):
    """
    Optimize lists of GPs models and likelihoods.
    @param models: list of GPyTorch models
    @param likelihoods: list of GPyTorch likelihoods
    @param learning_rate: learning rate for the optimizer
    @param max_training_iter: maximum number of iterations for the optimizer
    @param loss_thress: threshold for the loss to stop the optimization
    @param no_improvement_thress: number of iterations without improvement to stop the optimization
    @param gp_model_to_load: path to a saved model to load
    @param _device: device to use for the optimization
    @param messages_: print messages
    @returns models: list of optimized GPyTorch models
    @returns likelihoods: list of optimized GPyTorch likelihoods
    @returns gp_data: GP datapoints
    """
    device = setDevice(_device)

    gp_likelihhod_to_load = gp_model_to_load + "_like"
    gp_data_to_load = gp_model_to_load + "_data"
    gp_data = None

    models.train()
    likelihoods.train()

    best_models = copy(models)
    
    min_loss = 1e4
    min_loss_prev = min_loss
    no_improvement = 0
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = SumMarginalLogLikelihood(likelihoods, models)
    lrcurr = learning_rate
    use_Adam_ = True

    # load model 
    if gp_model_to_load:
        state_dict_model = torch.load(gp_model_to_load + ".pth")
        state_dict_likelihoods = torch.load(gp_likelihhod_to_load+ ".pth")
        
        # check if there any data to load
        try: 
            gp_data = pd.read_csv(gp_data_to_load + ".csv", header = None, index_col = None).to_numpy()[0]
            
            new_gp_data = []
            # extremly ugly way to convert the data to the right type
            for data in gp_data:
                data = data.replace('[ ','')
                data = data.replace(' ]','')
                data = data.replace('[','')
                data = data.replace(']','')
                data = data.replace('  ',' ')  
                data = data.replace('\n','')
                data = data.split(' ')
                new_gp_data.append( data )

            model_temperatures = torch.tensor(  [float(d) for d in new_gp_data[0]] ).to(device = device)            
            model_points = torch.tensor(  [[float(d) for d in feature if d !=""] for feature in new_gp_data[1:]] ).to(device = device)

            gp_data = { 'temperatures': [model_temperatures], 'weights': [model_points]}
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp_list = []

            for feature in model_points:
                # initialize likelihood and model
                gp_list.append(GPThetaModel(model_temperatures, feature, copy(likelihood)).to(device = device))

            models = gpytorch.models.IndependentModelList(*gp_list).float().to(device = device)
            likelihoods = gpytorch.likelihoods.LikelihoodList(*[gp.likelihood for gp in gp_list]).to(device = device)
        except Exception as e:
            print(e)
            
        models.load_state_dict(state_dict_model)
        likelihoods.load_state_dict(state_dict_likelihoods)
        
    # optimize
    else:
        for i in range(max_training_iter):
            # Zero gradients from previous iteration
            min_loss_prev = min_loss
            optimizer.zero_grad()

            # Output from model
            output = models(*models.train_inputs)
            loss = -mll(output, models.train_targets)
            
            #early_stopping
            if loss.item()<min_loss:
                best_models = copy(models)
                min_loss = loss.item()

            if np.abs(min_loss-min_loss_prev)<loss_thress:
                no_improvement += 1
                if no_improvement > no_improvement_thress:
                    if use_Adam_ :
                        if messages_:
                            print("\tOptimization restart")
                        lrcurr *= 0.1
                        optimizer = torch.optim.AdamW(models.parameters(), lr= lrcurr)
                        use_Adam_ = False
                        no_improvement = 0
                    else:
                        break
                
            else:
                no_improvement = 0

            loss.backward() 
            if not (i+1)%100:
                if messages_:
                    print('Iter %d/%d - Loss: %.3f  ' % (
                        i + 1, max_training_iter, min_loss
                    ))
            optimizer.step()

        models = best_models
    models.eval()
    likelihoods.eval()
    return copy(models), copy(likelihoods), copy(gp_data)

def initializePoints(points_updated_online):
    for p in points_updated_online:
        p.T_t_use = None
        p.excitation_delay_torch_height_trajectory = None
        p.excitation_delay_torch_height = None
        p.input_idxs = None
        p.Delta_T = None
        p.Delta_T_ambient = None    

    return points_updated_online

def updatePoints(points_used_for_training,points_updated_online,idxs):
    max_idx = idxs[-1]
    for (p,p_online) in zip(points_used_for_training,points_updated_online):
        
        p_online.T_t_use = p.T_t_use[:max_idx]
        p_online.excitation_delay_torch_height_trajectory = p.excitation_delay_torch_height_trajectory[:max_idx]
        p_online.Delta_T = p.Delta_T[:,:max_idx]
        p_online.Delta_T_ambient = p.Delta_T_ambient[:,:max_idx]

        p_online.excitation_delay_torch_height = p.excitation_delay_torch_height[p.input_idxs <= max_idx]
        p_online.input_idxs = p.input_idxs[ p.input_idxs <= max_idx]
    
    return points_updated_online

def learnDelay(excited_points):
    delays = []
    heights_at_excitation = []
    temperatures_at_excitation = []
    for p in excited_points:
        max_idx = np.min([ len(p.excitation_delay), len(p.excitation_delay_torch_height) ])
        delays.append(p.excitation_delay[:max_idx])
        heights_at_excitation.append(p.excitation_delay_torch_height[:max_idx])
        temperatures_at_excitation.append(p.excitation_temperatures[:max_idx])

    target_delays = np.hstack(delays).reshape(-1,1)
    feature_heights_at_excitation = np.hstack(heights_at_excitation).reshape(-1,1)
    feature_temperatures_at_excitation = np.hstack(temperatures_at_excitation).reshape(-1,1)

    all_features = feature_heights_at_excitation

    ## train linear model
    delay_model = LinearRegression(fit_intercept=True, normalize= True)
    train_x = all_features
    train_y = target_delays
    delay_model.fit( train_x, train_y.squeeze())
    return delay_model

def initializeGPModels( parameters : np.ndarray, unique_states : np.ndarray, GP_normalizers : torch.Tensor = None, device_to_optimize : str = 'cpu',\
                                                                     output_scale : float = 2, length_mean : float = 0.025, length_var : float = 0.1):

    """
    Initialize GP models for each parameter.
    @param parameters: parameters to be fitted (n_parameters, n_unique_states)
    @param unique_states: unique states of the system (n_unique_states)
    @param GP_normalizers: normalizers for the GP models (n_parameters)
    @param device_to_optimize: device to optimize on (cpu or cuda)
    @param output_scale: output scale for the GP models
    @param length_mean: mean of the lengthscale for the GP models
    @param length_var: variance of the lengthscale for the GP models
    @return: GP models for each parameter as an IndependentModelList and LikehoodList
    """
    device = setDevice(device_to_optimize)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    gp_list = []
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-10))

    if GP_normalizers is None:
        GP_weights_normalizers = []

    # Create GP models for each variable
    for i,feature in enumerate(parameters):

        feature_tensor = torch.from_numpy(feature.T).to(device = device)
        # normalize features
        if GP_normalizers is None:
            normalizer_idx = torch.argmax(torch.abs(feature_tensor))
            normalizer = feature_tensor[normalizer_idx]
        else:
            normalizer = GP_normalizers[i]
        normalized_feature_tensor = feature_tensor/normalizer

        band_nominal_height_tensor = torch.from_numpy(unique_states.T).to(device = device)

        current_model = GPThetaModel(band_nominal_height_tensor, normalized_feature_tensor, copy(likelihood), length_mean , length_var).to(device = device)

        mean_feature = torch.max(torch.abs(normalized_feature_tensor))
        var_feature = torch.var(normalized_feature_tensor)
        current_model.covar_module.outputscale_prior = gpytorch.priors.NormalPrior(mean_feature, mean_feature*2)
        
        if GP_normalizers is None:
            GP_weights_normalizers.append(normalizer)
        
        gp_list.append( current_model )

        if mean_feature>1e-7:
            # print(F"DEBUG {output_scale}")

            hypers = {
            'likelihood.noise_covar.noise': 1e-3 * mean_feature.clone().detach().requires_grad_(True),
            'covar_module.outputscale': output_scale*mean_feature.clone().detach().requires_grad_(True),
            }
            gp_list[-1].initialize(**hypers)

    # Put all models in a container
    models = gpytorch.models.IndependentModelList(*gp_list).float().to(device = device)
    likelihoods = gpytorch.likelihoods.LikelihoodList(*[gp.likelihood for gp in gp_list]).to(device = device)
    if GP_normalizers is None:
        GP_weights_normalizers = torch.tensor(GP_weights_normalizers)
    else:
        GP_weights_normalizers = GP_normalizers

    return models,likelihoods, GP_weights_normalizers, band_nominal_height_tensor, device

def statesForBoundaries( points_used_for_training : list, on_boundary_ : list ):
    """
    Define states for the nodes on the boundaries of the domain
    @param points_used_for_training: list of points used for training
    @param on_boundary_: list of bools indicating if the point is on the boundary
    @return: updated points_used_for_training
    """
    #  find the closest node on grid to boundary points
    nodes_on_boundaries = np.asarray( [p.node for (p,b) in zip(points_used_for_training,on_boundary_) if b] )
    points_on_boundaries = np.asarray( [p for (p,b) in zip(points_used_for_training,on_boundary_) if b] )

    internal_nodes = np.asarray( [p.node for (p,b) in zip(points_used_for_training,on_boundary_) if not b] )

    ## Distance matrix: distances between nodes
    nop = len(points_used_for_training)
    D = np.zeros((nop,nop))
    ### Build upper half
    for i,pi in enumerate(points_used_for_training):
        for j,pj in enumerate(points_used_for_training[i::]):
            D[i,i+j] = norm(pj.coordinates - pi.coordinates)

    ### Build lower symmetric half
    D = D + D.T - np.diag(D.diagonal())

    # find closest nodes
    nodes_closest_to_boundaries = []
    for d in D[nodes_on_boundaries,:]:
        arg = np.argmin(d[internal_nodes])
        closest_internal_node = internal_nodes[arg]
        nodes_closest_to_boundaries.append(closest_internal_node)

    # update state sequence of points on boundaries with the opposite state of their closest nodes
    for (p,closest_node) in zip( points_on_boundaries, nodes_closest_to_boundaries):
        closest_point =  points_used_for_training[closest_node]
        p.excitation_delay_torch_height_trajectory = -closest_point.excitation_delay_torch_height_trajectory
        p.excitation_delay_torch_height = -closest_point.excitation_delay_torch_height

    return points_used_for_training

def onlineOptimization(layer_idxs, unique_state_values, points_used_for_training, bounds_f, bounds_g, bounds_m, F_reg, G_reg, M_reg, param_f0, param_g0, param_m0, on_boundary_, d_grid ):
    """
    Simulate online optimization by only feeding the system with data streamed on each layer.
    """
    points_updated_online = copy(points_used_for_training)
    # initialize online Points
    points_updated_online = initializePoints(points_updated_online)
    
    g_parameters_per_building_layer_height = []
    f_parameters_per_building_layer_height = []
    m_parameters_per_building_layer_height = []
    all_training_times_per_state = []

    print("Starting optimization...")
    # simulating learning layer-by-layer
    for i,(idxs,building_layer_height) in enumerate(zip( layer_idxs, unique_state_values)): 

        # update data in points
        points_updated_online = updatePoints(points_used_for_training,points_updated_online,idxs)
        states = np.unique([p.excitation_delay_torch_height_trajectory for p in points_updated_online])
        
        ## find peaks in layer
        excited_points = [ p for p in points_updated_online if len(p.input_idxs)>0]

        t = time.time()

        theta_F, excitations, param_f0 = learnF( points_updated_online, states, bounds = bounds_f, regularizer =  F_reg, param_f0 = param_f0)
        # negative states are not excited. Overwrite the weights for a better fit in the gp.
        state_0_idx = np.argmin(np.abs(states))
        theta_F[:state_0_idx] = theta_F[state_0_idx]

        lengthscaleModel = halfBellLegthscaleModel( theta_F, states) # underestimate 

        # learn G
        theta_G, param_g0 = learnG( states, excitations, lengthscaleModel, excited_points, bounds = bounds_g, regularizer = G_reg, param_g0 = param_g0)
        # negative states are not excited. Overwrite the weights for a better fit in the gp.
        theta_G[:state_0_idx,:] = theta_G[state_0_idx,:]

        ipCoefModel = modelsForInputCoefs(theta_G, states)

        # learn m coupled with G and F
        xs_train, ys_train, band_state ,section_idxs = formFeatureTargetPairs(on_boundary_, lengthscaleModel,ipCoefModel,points_updated_online,d_grid=d_grid,HORIZON=1)

        theta_M, param_m0 = learnM(xs_train,ys_train,m_parameters_per_building_layer_height, bounds = bounds_m, regularizer = M_reg, param_m0 = param_m0)

        elapsed = time.time() - t
        all_training_times_per_state.append(elapsed/len(states))

        # negative states are not excited. Overwrite the theta_g for a better fit in the gp.
        theta_M[:state_0_idx,-1] = theta_M[state_0_idx,-1]
        # boundaries are only in negative states. Overwrite the theta_g for a better fit in the gp.
        theta_M[state_0_idx:,2:-1] = theta_M[state_0_idx,2:-1]

        f_parameters_per_building_layer_height.append(theta_F)
        g_parameters_per_building_layer_height.append(theta_G)
        m_parameters_per_building_layer_height.append(theta_M)

    return f_parameters_per_building_layer_height, g_parameters_per_building_layer_height, m_parameters_per_building_layer_height, all_training_times_per_state
 
def compareWithPast(prev_best_score : float, current_score : float, best_params : np.ndarray, current_params : np.ndarray):
    """
    Compare the current score with the previous best score. If the current score is better, update the best score and the best parameters.
    @param prev_best_score: previous best score
    @param current_score: current score
    @param best_params: best parameters
    @param current_params: current parameters
    @return: updated best score and best parameters
    """
    if len(prev_best_score)==0:
        prev_best_score = current_score.copy()
        best_params = current_params.copy()
    else:
        change_ = np.mean(prev_best_score) > np.mean(current_score)
        if change_:
            prev_best_score = current_score
            best_params = current_params 
    return best_params, prev_best_score 

def batchOptimization(states, points_used_for_training : list , m : mTerm, f : fTerm, g : gTerm, param_m0 : np.ndarray, 
    param_f0 : np.ndarray, param_g0 : np.ndarray, bounds_m : tuple, bounds_f : tuple, bounds_g : tuple, 
    M_reg : int, F_reg : int, G_reg : int, epochs = 2, verbose = True, perturbation_magnitude = 0.1, wrap_function_ = True, wrapping_function = np.arctan):
    """
    Optimize model with all the data gathered in an experiment.
    @param states: iterable of unique states
    @param points_used_for_training: list of Points objects that will be used for training
    @param m: mTerm object
    @param f: fTerm object
    @param g: gTerm object
    @param param_m0: initial parameters for m
    @param param_f0: initial parameters for f
    @param param_g0: initial parameters for g
    @param bounds_m: bounds for m
    @param bounds_f: bounds for f
    @param bounds_g: bounds for g
    @param M_reg: regularization for m
    @param F_reg: regularization for f
    @param G_reg: regularization for g
    @param epochs: number of epochs for training
    @param verbose: print training loss
    @param perturbation_magnitude: magnitude of perturbation for the initial parameters
    @param wrap_function_: wrap loss with a function
    @param wrapping_function: function to wrap the input to the gp
    """

    excited_points = [ p for p in points_used_for_training if len(p.input_idxs)>0]
    g_parameters_per_building_layer_height = []
    f_parameters_per_building_layer_height = []
    m_parameters_per_building_layer_height = []

    g_best_scores_per_building_layer_height = np.asarray([])
    m_best_scores_per_building_layer_height = np.asarray([])
    theta_G = np.asarray([])
    theta_M = np.asarray([])

    all_training_times_per_state = []
    for epoch in range(epochs):
        if verbose:
            print(f"epoch {epoch}")

        t = time.time()
        # optimize F
        theta_F, excitations, param_f0 = optimizeF( states, excited_points, f=f, param_f0=param_f0, bounds=bounds_f, F_reg = F_reg)

        # negative states are not excited. Overwrite the weights for a better fit in the gp.
        state_0_idx = np.argmin(np.abs(states))
        theta_F[:state_0_idx] = theta_F[state_0_idx]

        lengthscaleModel = halfBellLegthscaleModel( theta_F, states) # underestimate 
        f.update(theta_F)
        
        # optimize G
        theta_G_new, param_g0, fun_scores_g = optimizeG( f, g, states, excitations, excited_points, param_g0, lengthscaleModel, bounds_g, G_reg = G_reg)
        theta_G, g_best_scores_per_building_layer_height = compareWithPast(g_best_scores_per_building_layer_height, fun_scores_g, theta_G, theta_G_new)
        ipCoefModel = modelsForInputCoefs(theta_G, states)
        
        # optimize M
        theta_M_new, param_m0, fun_scores_m = optimizeM( m,f, g, states, points_used_for_training, param_m0 = param_m0,gCoefModel = ipCoefModel, lengthscaleModel = lengthscaleModel, bounds = bounds_m, M_reg = M_reg, wrap_function_=wrap_function_, wrapping_function=wrapping_function)
        theta_M, m_best_scores_per_building_layer_height = compareWithPast(m_best_scores_per_building_layer_height, fun_scores_m, theta_M, theta_M_new)

        # keep the best params
        if len(m_best_scores_per_building_layer_height)==0:
            m_best_scores_per_building_layer_height = fun_scores_m
            theta_M = theta_M_new.copy()
        else:
            change_ = m_best_scores_per_building_layer_height > fun_scores_m
            theta_M[change_,:] = theta_M_new[change_,:] 

        elapsed = time.time() - t
        
        # assign weight values found
        f_parameters_per_building_layer_height.append(theta_F)
        g_parameters_per_building_layer_height.append(theta_G)
        m_parameters_per_building_layer_height.append(theta_M)
        all_training_times_per_state.append(elapsed/len(states))

        # inject noise to initial parameters for next epoch
        param_f0 += perturbation_magnitude*param_f0*np.random.rand(len(param_f0))
        param_g0 += perturbation_magnitude*param_g0*np.random.rand(len(param_g0))
        param_m0 += perturbation_magnitude*param_m0*np.random.rand(len(param_m0))
    
    return f_parameters_per_building_layer_height, g_parameters_per_building_layer_height, m_parameters_per_building_layer_height, all_training_times_per_state

if __name__ == "__main__":
    
    dummy = 1