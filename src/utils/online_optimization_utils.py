from urllib import response
from cv2 import Laplacian, sepFilter2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils.generic_utils as gnu
import time

import torch
import gpytorch
import gc

from math import ceil
from gpytorch.mlls import SumMarginalLogLikelihood
from scipy.interpolate import interp1d
from copy import deepcopy as copy
from utils.generic_utils import rolling_window, setDevice, rolling_window_2D
from scipy.optimize import minimize,least_squares, Bounds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm
from scipy.interpolate import interp1d, interp2d

class gTerm:
    def __init__(self,params = None) -> None:
        """
        @param params : array with G model parameters
        """
        if params is None:
            self.params = np.ones((3,1))
        else:
            self.params = params
        self.update(self.params)
    
    def __call__(self,h,T):
        """
        @param h : array with heights, dim (n,)
        @param T : array with Temperatures, dim (n,)
        """
        regressor = np.vstack(( np.ones_like(h), 1-h, 1-T)).T #dim (n,3)
        if len(self.params)>3:
            out = np.sum(regressor*self.params, axis = -1).squeeze() # dim (n,)
        else:
            out = np.matmul(regressor,self.params).squeeze()
        return out
    
    def update(self,params):
        self.params = np.atleast_2d(params).T

class mTerm:
    def __init__(self, neighbors, params = None, d_grid = 27, d_grid_norm = 100, delta = None, ambient_T = 0) -> None:
        """
        @param neighbors : list of lists containing idxs to neighboring nodes, e.g. neighbors_i = [j,k,l,m] with j,k,l,m being the idxs of i's neighbors
        @param params : array with M model parameters
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
        

    def __call__(self,T):
        """
        @param h : array with heights, dim (n,)
        @param T : array with Temperatures, dim (n,)
        """
        # neighbor_Ts = [[T[i] for i in neighbor] for neighbor in self.neighbors]
        self.neighbor_Ts = T[self.idx_map]
        self.neighbor_Ts[self.no_connections] = 0 # zero out the non existing connections
        laplacians = (self.number_of_neighbors*T - np.sum(self.neighbor_Ts, axis = -1))/(self.d_grid) # dim (n,)
        # laplacians = self.calculateLaplacians(T)
        regressor = np.vstack(( T, laplacians, (T - self.ambient_T)*self.delta)).T #dim (n,3)
        if len(self.params)>3:
            out = np.sum(regressor*self.params, axis = -1).squeeze() # dim (n,)
        else:
            out = np.matmul(regressor,self.params).squeeze()
        
        return out
    
    def update(self,params):
        self.params = np.atleast_2d(params).T
    
    def updateTambient(self,ambient_T):
        self.ambient_T = ambient_T
    
    def calculateLaplacians(self,T):
        """
        for training
        """
        T = np.atleast_2d(T)
        neighbor_Ts = T[self.idx_map,:]
        neighbor_Ts[self.no_connections,:] = 0 # zero out the non existing connections
        laplacians = (np.multiply(self.number_of_neighbors,T.T).T - np.sum(neighbor_Ts, axis = 1))/(self.d_grid) # dim (n,N)
        return laplacians
    
    def oneStepPropagation(self,T,laplacians,deltas):
        """
        for training
        """
        regressor = np.vstack(( T, laplacians, (T - self.ambient_T)*deltas)).T #dim (N,3)
        return np.matmul(regressor,self.params).squeeze() # dim (N,)

class fTerm:
    def __init__(self, bell_resolution = 101, params = None) -> None:
        """
        @param bell_resolution : int with length of peak intervals. If too short, then the peaks won't fit 
        @param params : array with 1 model parameters
        """
        if params is None:
            self.params = np.ones((1,))
        else:
            self.params = params
        self.update(self.params)

        self.bell_resolution = bell_resolution
    
    def __call__(self,peak_idxs,array_length, multiple_params = False):
        """
        @param peak_idxs : list of peaks at each node
        @param T : array with Temperatures, dim (n,)
        """
        out = []
        if not multiple_params:
            params = np.full(len(peak_idxs), self.params)
        else:
            params = self.params

        for (node_peaks,param) in zip(peak_idxs,params):
            out.append( Fsequence( node_peaks, param, array_length, bell_resolution = self.bell_resolution) )
        return out
    
    def update(self,params):
        self.params = params

class totalModel:
    def __init__(self, f_seq, h, g: gTerm, m : mTerm, N = -1, theta_g = None, theta_m = None, theta_p = None,Dt = 0.5) -> None:
        """
        @param bell_resolution : int with length of peak intervals. If too short, then the peaks won't fit 
        @param theta : arrays with parameter sequences model parameters
        @param h : array with heights, dim (n,N)
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
    
    def __call__(self,t,T):
        """
        @param T0 : array with Temperatures, dim (n,)
        @param N : int with extrapolation Horizon. Cannot be longer than len(f_seq)
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
    
    def updateM(self,theta_m):
        self.theta_m = theta_m
        self.m.update(theta_m)

    def updateG(self,theta_g):
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
    def __init__(self, state_excitation_idxs, excited_points, regularizer, f : fTerm, parameter_scale = 10) -> None:
        self.state_excitation_idxs = state_excitation_idxs 
        self.excited_points = excited_points 
        self.f = f
        self.regularizer = regularizer
        self.parameter_scale = parameter_scale

    
    def __call__(self, params):
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
    def __init__(self, state_excitation_idxs, excited_points, g :gTerm, f:fTerm, lengthscale = 10, regularizer = 1e-2, parameter_scale = 1, test_ratio = 0.25) -> None:
       
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
        # self._calculateFsequences()
    
    def __call__(self, params):
       
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
            idx_train,idx_test = train_test_split( np.arange(len(all_f)), test_size=self.test_ratio )

            # zero padding for training sequences with less than max_length elements 
            Ts = np.zeros(( max_length, len(all_f)))
            Hs = np.zeros_like(Ts)
            Fs = np.zeros_like(Ts)

            for i,( T_node_idx,H_node_idx,F_node_idx)  in enumerate(zip(all_T,all_h,all_f)):
                Ts[:len(T_node_idx),i] = T_node_idx
                Hs[:len(T_node_idx),i] = H_node_idx
                Fs[:len(T_node_idx),i] = F_node_idx

            # unroll g
            residuals = self._unrollResponse(Ts,Fs,Hs,idx_train)
            test_residuals = self._unrollResponse(Ts,Fs,Hs,idx_test)
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
        history = [response]
        for (f_s, h_s) in zip( Fs[:,node_idx_pairs], Hs[:,node_idx_pairs]):
            response += f_s * self.g(h_s, response)
            history.append(response)

        error = np.asarray(history[:-1]) - Ts[ :, node_idx_pairs]
        residuals = np.mean( (error / np.abs( Ts[ :, node_idx_pairs] + 1e-4))**2 )
        # residuals = np.mean( (np.hstack(error)/np.mean(np.abs( np.hstack(Ts[ :, node_idx_pairs]))) )**2 )
        return residuals

class loss_M:
    def __init__(self, state_level_idxs, training_points, m :mTerm,  g :gTerm, f:fTerm, lengthscale = 10, theta_G = np.array([0,0,0]) , regularizer = 1e-2, N = 1, parameter_scale = 1,test_ratio = 0.25,random_state = 934) -> None:
       
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
        self.min_test_loss = 1e10
        self.best_params = None
    
    def __call__(self, params):
       
        # update lengthscale
        self.m.update(params[:-1]*self.parameter_scale)

        if self.N == 1:
            # TODO : current implementation of M propagates system by one timestep and accepts a thermal field as input. Rethink of how to implement 
            m_val = self.m.oneStepPropagation( self.x, self.laplacians, self.deltas)
            preds = (1-self.f_seq[self.train_idxs])*m_val[self.train_idxs] + params[-1]*self.f_seq[self.train_idxs]*self.g(self.h[self.train_idxs],self.x[self.train_idxs])
            error = self.y[self.train_idxs] - preds

            test = (1-self.f_seq[self.test_idxs])*m_val[self.test_idxs] + params[-1]*self.f_seq[self.test_idxs]*self.g(self.h[self.test_idxs],self.x[self.test_idxs])

            test_error = self.y[self.test_idxs] - test

            loss = objective(error,self.y[self.train_idxs]) + regFun(params[:-1],self.regularizer)
            test_loss = objective(test_error,self.y[self.test_idxs])
        else:
            assert 1, "N>1 not implemented."

        if test_loss<self.min_test_loss:
            self.best_params = params
            self.min_test_loss = test_loss
    
        return loss 
    
    def _createFeatureTargets(self,test_size,random_state) -> None:
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
        self.train_idxs,self.test_idxs = train_test_split(idxs, test_size = test_size, shuffle = True, random_state = random_state)

        return None


def optimizeF( states, excited_points,  param_f0, bounds , f : fTerm, F_reg = 1e-2, underestimate_lengthscales = 0):
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
        # res = minimize( optimizer , param_f0, bounds= bounds, method= "L-BFGS-B", tol = 1e-12 )
        res = least_squares( optimizer , param_f0, bounds= bounds, ftol = 1e-12, method= "trf", loss = "arctan" )
        param_f0 = res.x
        parameters_per_state_level.append(res.x[0]) # omit the last one

        if i == 0:
            first_state_param_f0 = res.x

    param_f0 = first_state_param_f0

    ## scale lengthscales
    lengthscales = 10* np.abs(parameters_per_state_level ) * (1-underestimate_lengthscales)
    # return lengthscales, excitations, param_f0 
    ###########################
    return lengthscales, excitations, param_f0

def optimizeG( f : fTerm, g : gTerm, states, excitations, excited_points, param_g0, lengthscaleModel : interp1d, bounds , G_reg = 1e-2 ,test_ratio = 0.25):
    
    #  learn G
    ## learn the lengthscale
    parameters_per_state_level = []
    for i,(state_level_idxs,state) in enumerate(zip(excitations,states)):
        lengthscale = lengthscaleModel(state)
        optimizer = loss_G(state_level_idxs, excited_points, g, f, regularizer = G_reg, lengthscale = lengthscale, test_ratio=test_ratio)
        # res = minimize( optimizer , param_g0, bounds= bounds, method= "L-BFGS-B", tol = 1e-12 )
        res = least_squares( optimizer , param_g0, bounds= bounds, ftol = 1e-12, method= "trf", loss = "arctan")
        param_g0 = res.x
        parameters_per_state_level.append(optimizer.best_params) # omit the last one

        if i == 0:
            first_state_param_g0 = res.x

    param_g0 = first_state_param_g0
    coefficients = np.asarray(parameters_per_state_level)
    return coefficients, param_g0

def optimizeM( m : mTerm, f : fTerm, g : gTerm, states, training_points,  param_m0, lengthscaleModel : interp1d, gCoefModel : interp1d, bounds , M_reg = 1e-2,test_ratio = 0.25):
    
    # find state levels for each point
    layer_idxs = []
    for state in states:
        point_layer_idxs = []
        for p in training_points:
            tmp = np.where(p.excitation_delay_torch_height_trajectory == state)[0]
            point_layer_idxs.append( tmp.astype(np.int64) )
        layer_idxs.append(point_layer_idxs)
    
    parameters_per_state_level = []
    for i,(state_level_idxs,state) in enumerate(zip(layer_idxs,states)):
        lengthscale = lengthscaleModel(state)
        theta_G = gCoefModel(state)
        optimizer = loss_M(state_level_idxs, training_points, m, g, f,lengthscale = lengthscale, theta_G = theta_G, regularizer = M_reg, test_ratio=test_ratio)
        # res = minimize( optimizer , param_m0, bounds= bounds, ftol = 1e-12, method= "trf", loss = "arctan" )
        res = least_squares( optimizer , param_m0, ftol = 1e-12, bounds= bounds, method= "trf", loss = "arctan")
        param_m0 = res.x
        parameters_per_state_level.append(optimizer.best_params) # omit the last one

        if i == 0:
            first_state_param_m0 = res.x

    param_m0 = first_state_param_m0
    coefficients = np.asarray(parameters_per_state_level)
    return coefficients, param_m0

def learnG( states, excitations, lengthscaleModel, excited_points, bounds, regularizer, param_g0):

    parameters_per_height_level = []
    # param_g0 = np.ones((7))*0.1
    for i,(height_lvl,height_level_idxs) in enumerate(zip(states,excitations)):

        height_lengthscale = lengthscaleModel(height_lvl)
        optimizer = optimizeFandG(height_level_idxs,excited_points,height_lengthscale,regularizer = regularizer)
        # optimizer = ifu.optimizeLayerInputParameters(height_level_idxs,excited_points,height_lengthscale,regularizer = G_reg)
        res = minimize( optimizer , param_g0, bounds= bounds, method= "L-BFGS-B", tol = 1e-12 )
        G_param = res.x[:-1] # omit the last one
        parameters_per_height_level.append(res.x[:-1])
        param_g0 = res.x
        if i == 0:
            first_state_param_g0 = res.x

    param_g0 = first_state_param_g0
    # if np.any(states<0):
    #     neg_states = states[states<0]
    #     for neg_state in neg_states:
    #         parameters_per_height_level.insert(0,parameters_per_height_level[0])
    coefficients = np.asarray(parameters_per_height_level)[:,:5]

    return coefficients, param_g0

def pointsOnBoundaries_(experiment):
    
    points_used_for_training = [p for p in experiment.Points if p._hasNeighbors_()]
    coordinate_list = [p.coordinates for p in points_used_for_training]
    coordinates = np.vstack(coordinate_list)
    boundaries = findDomainLimits(coordinates)
    on_boundary_ = [_isOnBoundary_(p,boundaries) for p in points_used_for_training]
    
    # find cornenrs as well
    # on_corner_ = [_isOnCorner_(p,boundaries) for p in points_used_for_training]
    # on_corner_ = [False for p in points_used_for_training]

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

def splitToLayers( points_used_for_training, tool_height_trajectory, debug = True):
    """
    @returns layer_idxs: list with arrays of idxs for different layers
    @returns unique_height_values: height of each layer
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

class optimizeFandG:
    
    def __init__(self,height_level_idxs,excited_points,lengthscale = None, regularizer = 0.001):
        self.height_level_idxs = height_level_idxs 
        self.excited_points = excited_points 
        self.lengthscale = lengthscale
        self.regularizer = regularizer

    def __call__(self,params):
        """
        return the sum over all points of the residuals that the given parameter set resulted in. When you want to define the lengthscale, fit the bell
        to the temperature peaks while for tuning the input model parameters fit the dTdt by extrapolation. 
        """
        (a,b,c,d,lengthscale,e,f) = params
        resolution = 100
        if self.lengthscale is not None:
            lengthscale = self.lengthscale
            input_feature_span = 3*int(lengthscale) # how many non 0 elements to expect in the half bell 
            if input_feature_span>resolution/2:
                resolution = 2*input_feature_span
            bellDer = unitBellFunctionDerivative( lengthscale, resolution = resolution)
            bell = unitBellFunctionDerivative( lengthscale, resolution = resolution)
            # keep only positive part for input, let negative be handled by the rest
            bell[bellDer<0] = 0

        else:
            lengthscale *=10
            input_feature_span = 3*int(lengthscale) # how many non 0 elements to expect in the half bell 
            if input_feature_span>resolution/2:
                resolution = 2*input_feature_span
            bell = unitBellFunction( lengthscale, resolution = resolution)
            bellDer = unitBellFunctionDerivative( lengthscale, resolution = resolution)
            bell[bellDer<0] = 0
            a,b,c,d,e = 1, 0, 0, 0, 0

        noe = int( len(bell)/2 )

        residuals = 0
        for (idxs,p) in zip(self.height_level_idxs,self.excited_points):
            # in this loop level you have all the idxs mapping to peaks in the thermal data for one node and height level
            if len(idxs) == 0:
                continue
            new_residuals = 0
            for idx in idxs:
                calculated_input,excitation_heights,T,T_start = calculateGFeatures(idx,noe,p,bell,lengthscale,resolution)
                
                # propagate T to see how does your input model work
                if self.lengthscale is not None:
                    input_model = []
                    state = T[0]
                    for (excitation,height) in zip(calculated_input,excitation_heights):    
                        state += G(a,b,c,d,e,F = excitation, h = height, T = state)
                        input_model.append(state)
                else:
                    input_model = G(a,b,c,d,e, F = calculated_input,h = excitation_heights,T = T)


                # evaluate residuals
                learned = np.ones_like(T)* T[0]   + f*np.asarray(input_model)
                error = learned - T
                new_residuals += np.mean(error**2)  
            
            new_residuals /= len(idxs)
            residuals += new_residuals + self.regularizer * np.linalg.norm(params)
        
        # regularize
        # residuals += self.regularizer*np.linalg.norm(params) 

        return residuals

    def plotResponse(self,learned,T):
        time = np.arange(len(learned))
        plt.plot(time,learned,time,T)
        plt.show()

def calculateGFeatures(idx,noe,p,bell,lengthscale,resolution):
    if idx>noe:
        final_idx = np.min([ idx + noe + 1 , len(p.T_t_use) - 1 ])
        bell_final_idx = resolution + np.min([ (len(p.T_t_use) - 1) - (idx + noe + 1) , 0  ])
        T = p.T_t_use[ idx-noe+1 : final_idx ] 
        dTdt = T - p.T_t_use[ idx - noe : final_idx - 1 ]
        excitation_heights = p.excitation_delay_torch_height_trajectory[ idx - noe + 1 : final_idx]
        calculated_input = bell [:bell_final_idx]
        T_start_idx = noe - int(lengthscale) * 3
        T_start = np.ones_like(calculated_input)*T[T_start_idx]
        dummy = 1
    else:
        T = p.T_t_use[ 1 : idx + noe +1 ]
        dTdt = T - p.T_t_use[ : idx + noe ]
        excitation_heights = p.excitation_delay_torch_height_trajectory[ 1 : idx + noe + 1 ]
        calculated_input = bell[- (noe+idx):]
        T_start_idx = 0
        T_start = np.ones_like(calculated_input)*T[T_start_idx]
        dummy = 1
    
    return calculated_input,excitation_heights,T,T_start
    
def G(a,b,c,d,e,F,h,T,T_start = None):
    """
    @params a,b,c,d for model the coefficient
    """

    coef =  a + b *(1 - h) + c * (1 - T) + d * h * (1 - T)* 0
    out = F * coef
    
    return out

def Fsequence(peak_idxs,lengthscales,array_length,bell_resolution = 101):

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
            # first_half_bell_idx_on_input_trajectory += final_idx_with_non0_value + 1
            first_half_bell_idx_on_input_trajectory += final_idx_with_non0_value 
            first_half_bell_idx_on_bell = final_idx_with_non0_value
        
        # check if your first peak in bell trajectory is 0
        # elif first_half_bell_idx_on_input_trajectory==0 :
        #     first_half_bell_idx_on_bell = input_feature_span - peak_idx    
        
        # insert the half-bell to the input sequence
        out[ first_half_bell_idx_on_input_trajectory : peak_idx ] = half_bell_shape [ first_half_bell_idx_on_bell :  ]

    return out

class optimzeM:
    def __init__(self,x_train,x_test,y_train,y_test,regularizer,min_test_loss = 1e10):

        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test = y_test

        self.min_test_loss = min_test_loss
        self.regularizer = regularizer
        self.objFun = None
        self.regFun = None

    def __call__(self, a):
        """
        Save the model achieving the min loss on the test set and return the train loss.
        """

        train_loss = objFun(a, self.x_train, self.y_train) + regFun(a,self.regularizer)
        test_loss = objFun(a, self.x_test, self.y_test)

        if test_loss<self.min_test_loss:
            self.a_star = a
            self.min_test_loss = test_loss

        return train_loss

    def test(self,a):
        return objFun(a, self.x_test, self.y_test)

    def setObjFun(self,fun):
        self.objFun = fun

    def setRegFun(self,fun):
        self.regFun = fun

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

def halfBellLegthscaleModel( lengthscale_models, torch_heights_at_excitation ):
    model = interp1d(torch_heights_at_excitation,lengthscale_models)
    return model

def modelsForInputCoefs(coefs,heights):
    return interp1d(heights,coefs,axis = 0)

class GPThetaModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_len_prior = 0.025, var_len_prior = 0.1):
        super(GPThetaModel, self).__init__(train_x, train_y, likelihood)
        
        lengthscale_prior = gpytorch.priors.NormalPrior(mean_len_prior, var_len_prior) # good results 4/8
        self.mean_module = bilinearMean(train_x,train_y)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5,lengthscale_prior=lengthscale_prior) )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def GPOpt( models, likelihoods, learning_rate = 0.1, training_iter = 4000, loss_thress = 1e-3, no_improvement_thress = 200, gp_model_to_load = '', _device = None, messages_ = False ):
    """
    Implementation for optimization algorithm for GPModels
    @params models : gpytorch.models.IndependentModelList
    @params likelihoods : gpytorch.likelihoods.LikelihoodList
    @params learning_rate : real
    @params training_iter : int - > maximum number of itterations
    @params loss_thress : real - > thresshold for no improvement in the marginal likelihood
    @params gp_model_to_load : string - > name of model to load. if empty then optimize.
    
    @returns  copy(models), copy(likelihoods) : copy of models and likelihoods
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
    # gpytorch.settings.cholesky_jitter(float = 1e-4)

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
        
    else:
        for i in range(training_iter):
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

            loss.backward() #comment for lbfgs
            if not (i+1)%100:
                if messages_:
                    print('Iter %d/%d - Loss: %.3f  ' % (
                        i + 1, training_iter, min_loss
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

def learnF( excited_points, states,  bounds, regularizer, param_f0, underestimate_lengthscales = 0):

    excitations = []
    for state_level in states:
        point_excitations_on_level_map = [] # 1 array for each point telling you which peaks to consider
        for p in excited_points:
            point_excitations_on_level_map.append(  p.input_idxs[ p.excitation_delay_torch_height == state_level ] )
    
        excitations.append( point_excitations_on_level_map )

    ## fit lengthscales that look like the bells you see
    parameters_per_state_level = []

    ## learn the lengthscale
    for i,state_level_idxs in enumerate(excitations):
        # optimizer = ifu.optimizeLayerInputParameters(state_level_idxs,excited_points, regularizer= F_reg)
        optimizer = optimizeFandG(state_level_idxs,excited_points, regularizer= regularizer)
        res = minimize( optimizer , param_f0, bounds= bounds, method= "L-BFGS-B", tol = 1e-12 )
        param_f0 = res.x
        parameters_per_state_level.append(res.x[:-1]) # omit the last one

        if i == 0:
            first_state_param_f0 = res.x

        # ## for testing
        # check =optimizeFandG(state_level_idxs,excited_points)
        # resp = check(param_f0)

    param_f0 = first_state_param_f0
    # incorporate negative state
    # if np.any(states<0):
    #     neg_states = states[states<0]
    #     for neg_state in neg_states:
    #         parameters_per_state_level.insert(0,parameters_per_state_level[0])
    ## scale lengthscales
    lengthscales = 10* np.abs( [x[4] for x in parameters_per_state_level] ) * (1-underestimate_lengthscales)
    
    return lengthscales, excitations, param_f0

def formFeatureTargetPairs(on_boundary_, lengthscaleModel,ipCoefModel,points_updated_online,d_grid=27,HORIZON=1):
    ## form features
    for p in points_updated_online:

        Delta_T = p.Delta_T[1:,:] 
        L = np.sum(Delta_T,axis=0) / (d_grid/100)
        overheating = p.Delta_T[0,:] 

        boundary_ = on_boundary_[p.node]

        Delta_T_amb = np.zeros_like(p.Delta_T_ambient[:,:])
        if boundary_:
            Delta_T_amb[0,:] = p.Delta_T_ambient[0,:] 

        ## calculate F for point
        observed_peak_indexes = p.input_idxs
        lengthscales_for_point_input = lengthscaleModel(p.excitation_delay_torch_height)
        array_length =  np.max(overheating.shape)
        F = Fsequence(observed_peak_indexes,lengthscales_for_point_input,array_length,bell_resolution = 100)
        
        ## calculate G for point
        excitation_heights = p.excitation_delay_torch_height_trajectory
        ip_coef_tr = ipCoefModel(p.excitation_delay_torch_height_trajectory).T
        G_val = G(*ip_coef_tr, F = F, h = excitation_heights, T = overheating)

        overheating *= (1-F)
        L *= (1-F)
        Delta_T_amb *= (1-F)

        point_features = np.vstack( (overheating, L, Delta_T_amb, G_val) ) # 1,1,2,1 features
        p._setFeatures(point_features)

    xs_train, ys_train, band_state ,section_idxs = gnu.splitDataByState( points_updated_online, HORIZON)  
    return    xs_train, ys_train, band_state ,section_idxs

def learnM(xs_train,ys_train,m_parameters_per_building_layer_height, bounds, regularizer, param_m0):
    theta_M = []
    nog = len(ys_train)
    # find previous solution
    a_star_prev = param_m0

    for i,( x_group_train, y_group_train) in enumerate(zip(xs_train,ys_train)):
        #print(f"Group {i+1}/{nog}:")

        # split to train and test set
        X_train, X_test, y_train, y_test = train_test_split(x_group_train.T, y_group_train, train_size= 0.75, shuffle = True, random_state = 934)

        # create an optimizer object and run the optimization
        opt = optimzeM(X_train, X_test, y_train, y_test, regularizer)
        res = least_squares( opt, param_m0, ftol = 1e-12, bounds= bounds, method= "trf", loss = "arctan")

        # check if current solution is better than the previous one
        if a_star_prev is not None:    
            if len(a_star_prev) > i:            
                new_params = opt.test( opt.a_star )
                old_params = opt.test(  a_star_prev )
                if old_params<new_params:
                    opt.a_star = a_star_prev
                    
        # keep solution and convergence metrics
        theta_M.append(opt.a_star)
        param_m0 = opt.a_star

        #print(f"\tGroup Optimization completed. Train loss: {res.fun} | Validation loss {opt.min_test_loss}\n ")
    theta_M = np.asarray(theta_M)
    return theta_M, param_m0

def objFun(a,x,y): 
    return np.mean( ((y-a.dot(x.T))/np.mean(np.abs(y)))**2 )

def objective(e,y): 
    return np.mean( np.abs( (e/np.mean(np.abs(y))))  )

def regFun(a, regularizer ):
    return regularizer*np.linalg.norm(a[:-1])

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

def initializeGPModels( parameters, unique_states, GP_normalizers = None, device_tp_optimize = 'cpu', output_scale = 2, length_mean = 0.025, length_var = 0.1):

    device = setDevice(device_tp_optimize)
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    gp_list = []
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-10))

    if GP_normalizers is None:
        GP_weights_normalizers = []

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
            'likelihood.noise_covar.noise': 5e-3 * mean_feature.clone().detach().requires_grad_(True),
            'covar_module.outputscale': output_scale*mean_feature.clone().detach().requires_grad_(True),
            }
            gp_list[-1].initialize(**hypers)

    models = gpytorch.models.IndependentModelList(*gp_list).float().to(device = device)
    likelihoods = gpytorch.likelihoods.LikelihoodList(*[gp.likelihood for gp in gp_list]).to(device = device)
    if GP_normalizers is None:
        GP_weights_normalizers = torch.tensor(GP_weights_normalizers)
    else:
        GP_weights_normalizers = GP_normalizers

    return models,likelihoods, GP_weights_normalizers, band_nominal_height_tensor, device

def statesForBoundaries( points_used_for_training ,on_boundary_ ):
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

class bilinearMean(gpytorch.means.Mean):                                                                                                                                                                        
    def __init__(self, xs, ys, batch_shape=torch.Size()):
        """
        You need at least 3 datapoints on each direction
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
        
    def forward(self, x):

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
 
def batchOptimization_(layer_idxs, unique_state_values, points_used_for_training, bounds_f, bounds_g, bounds_m, F_reg, G_reg, M_reg, param_f0, param_g0, param_m0, on_boundary_, d_grid, epochs = 3 ):
    """
    Feed data coming from all layers at once.
    """
    all_training_times_per_state = []

    print("Starting optimization...")
    points_used_for_training = updatePoints(points_used_for_training,points_used_for_training,layer_idxs[-1])

    # Learn all layers at once 
    for epoch in range(epochs):
        g_parameters_per_building_layer_height = []
        f_parameters_per_building_layer_height = []
        m_parameters_per_building_layer_height = []

        # states = unique_state_values
        states = np.unique([p.excitation_delay_torch_height_trajectory for p in points_used_for_training])
        
        ## find peaks in layer
        excited_points = [ p for p in points_used_for_training if len(p.input_idxs)>0]

        t = time.time()

        theta_F, excitations, param_f0 = learnF( points_used_for_training, states, bounds = bounds_f, regularizer =  F_reg, param_f0 = param_f0)
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
        xs_train, ys_train, band_state ,section_idxs = formFeatureTargetPairs(on_boundary_,lengthscaleModel,ipCoefModel,points_used_for_training,d_grid=d_grid,HORIZON=1)

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

def batchOptimization(states, points_used_for_training, m : mTerm, f : fTerm, g : gTerm, param_m0 : np.ndarray, param_f0 : np.ndarray, param_g0 : np.ndarray, bounds_m : tuple, bounds_f : tuple, bounds_g : tuple, M_reg : int, F_reg : int, G_reg : int, epochs = 2, verbose = True, noise_magnitude = 0.1):

    excited_points = [ p for p in points_used_for_training if len(p.input_idxs)>0]
    g_parameters_per_building_layer_height = []
    f_parameters_per_building_layer_height = []
    m_parameters_per_building_layer_height = []
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
        theta_G, param_g0 = optimizeG( f, g, states, excitations, excited_points, param_g0, lengthscaleModel, bounds_g, G_reg = G_reg)
        ipCoefModel = modelsForInputCoefs(theta_G, states)
        
        # optimize M
        theta_M, param_m0 = optimizeM( m,f, g, states, points_used_for_training, param_m0 = param_m0,gCoefModel = ipCoefModel, lengthscaleModel = lengthscaleModel, bounds = bounds_m, M_reg = M_reg)

        elapsed = time.time() - t
        
        # assign weight values found
        f_parameters_per_building_layer_height.append(theta_F)
        g_parameters_per_building_layer_height.append(theta_G)
        m_parameters_per_building_layer_height.append(theta_M)
        all_training_times_per_state.append(elapsed/len(states))

        # inject noise to initial parameters for next epoch
        param_f0 += noise_magnitude*param_f0*np.random.rand(len(param_f0))
        param_g0 += noise_magnitude*param_g0*np.random.rand(len(param_g0))
        param_m0 += noise_magnitude*param_m0*np.random.rand(len(param_m0))
    
    return f_parameters_per_building_layer_height, g_parameters_per_building_layer_height, m_parameters_per_building_layer_height, all_training_times_per_state

if __name__ == "__main__":
    
    dummy = 1