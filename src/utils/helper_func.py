import numpy as np
import matplotlib.pyplot as plt
import utils.input_feature_utils as ifu
import pandas as pd

import torch
import gpytorch
import os


from utils.generic_utils import movingAvg, rolling_window
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt


# %%¨

def calculateMeanNeighborTemperature(node_temperatures,list_of_neighbors,smoothing_window = 50):
    """
    Calculate mean temperature of neighbors for replacing ambient temp.
    """
    mean_T = []
    delta_mean_T = []
    for (T_t,neighb) in zip(node_temperatures,list_of_neighbors):
        tmp = np.zeros_like(T_t)
        for nei in neighb:
            # tmp += node_temperatures[nei,:]/4
            tmp += node_temperatures[nei,:]/len(neighb)
        
        tmp = movingAvg( tmp, smoothing_window)
        mean_T.append(tmp)
        delta_mean_T.append( T_t[-tmp.shape[0]:] - tmp )
    
    return np.vstack(delta_mean_T), np.vstack(mean_T)

# dynamic vector transformations
def dynVectorNoLags_old(a):
    """
    no lags considered in variables.
    @param a : weights for points. dim 0->point, dim 2->weights
    @return dynvector : array used for predictions. dim 0->point, dim 2->weights
    """
    a = np.atleast_2d(a)
    ones = 0.001*np.ones_like(a[:,0])

    return np.array( (
                ones,               # Tdot

                # a[:,0],               # Tdot
                a[:,1],               # T
                a[:,2], a[:,2], a[:,2], a[:,2], # Delta T
                a[:,3]*0,               # Mean T
                a[:,4],               # in 1
                        )).T

def dynVectorNoLags_sameWeights(a):
    """
    no lags considered in variables.
    @param a : weights for points. dim 0->point, dim 2->weights
    @return dynvector : array used for predictions. dim 0->point, dim 2->weights
    """
    a = np.atleast_2d(a)

    return np.array( (a[:,0],             # T - Tmean
                a[:,1], a[:,1], a[:,1], a[:,1], # Delta T
                a[:,2], a[:,2],           # Delta T ambient
                a[:,3],                   # DTdt
                a[:,4],                   # Height
                a[:,5],                   # input
                        )).T 

def dynVectorNoLags(a):
    """
    no lags considered in variables.
    @param a : weights for points. dim 0->point, dim 2->weights
    @return dynvector : array used for predictions. dim 0->point, dim 2->weights
    """
    a = np.atleast_2d(a)

    return np.array( (a[:,0],             # T - Tmean
                a[:,1], a[:,1], a[:,1], a[:,1], # Delta T
                a[:,2],                   # Delta T ambient 
                a[:,3],            # Delta T ambient corners
                a[:,4],                   # DTdt
                np.zeros_like(a[:,4]),    # Height
                a[:,5],                   # input
                        )).T 

def dynVectorNoLagsWithInput(a):
    """
    no lags considered in variables.
    @param a : weights for points. dim 0->point, dim 2->weights
    @return dynvector : array used for predictions. dim 0->point, dim 2->weights
    """
    a = np.atleast_2d(a)

    input_parameters = np.asarray([
                a[:,-4],                   # input parameter a
                a[:,-3],                   # input parameter b
                a[:,-2],                   # input parameter c
                a[:,-1],                   # input parameter d
                ]).T
    
    weights = np.array( (a[:,0],          # T - Tmean
                a[:,1],                   # Delta T
                a[:,2],                   # Delta T ambient 
                a[:,3]*a[:,2],            # Delta T ambient corners
                a[:,4],                   # DTdt
                np.zeros_like(a[:,4]),    # Height
                a[:,5],                   # input weight
                        )).T 

    return weights,input_parameters

def dynVectorNoLags_laplacians(a):
    """
    no lags considered in variables.
    @param a : weights for points. dim 0->point, dim 2->weights
    @return dynvector : array used for predictions. dim 0->point, dim 2->weights
    """
    a = np.atleast_2d(a)

    return np.array( (a[:,0],             # T - Tmean
                a[:,1],                   # laplacians
                a[:,2],                   # Delta T ambient 
                a[:,3]*a[:,2],            # Delta T ambient corners
                a[:,4],                   # DTdt
                np.zeros_like(a[:,4]),    # Height
                a[:,5],                   # input
                        )).T 
# code class for validation
class optimzer:
    def __init__(self,x_train,x_test,y_train,y_test,min_test_loss = 1e10):

        self.x_train = x_train
        # self.x_train = x_train[:,:-2]
        # self.dTdt_train = x_train[:,0]
        # self.T_train = x_train[:,1]
        # self.relevance_train = x_train[:,-2]
        # self.heights_train = x_train[:,-1]

        self.x_test = x_test
        # self.x_test = x_test[:,:-2]
        # self.dTdt_test = x_test[:,0]
        # self.T_test = x_test[:,1]
        # self.relevance_test = x_test[:,-2]
        # self.heights_test = x_test[:,-1]

        self.y_train = y_train
        self.y_test = y_test
        # print(f"self.x_train.shape {self.x_train.shape}")

        self.min_test_loss = min_test_loss
        self.objFun = None
        self.regFun = None

    def __call__(self, a):
        """
        Save the model achieving the min loss on the test set and return the train loss.
        """
        dynVector = dynVectorNoLags(a)

        # build feature
        # ip_train = trainIpFeature( a[-3], a[-2], a[-1], self.relevance_train,self.heights_train, self.T_train)[:,None]
        # ip_train = trainIpFeature( a[-3], a[-2], a[-1], self.relevance_train,self.heights_train, self.dTdt_train)[:,None]
        # ip_train = trainIpFeature( a[-2], a[-1], self.relevance_train,self.heights_train)[:,None]
        # features_train = np.hstack( ( self.x_train, ip_train ) )
        # ip_test = trainIpFeature( a[-3], a[-2], a[-1], self.relevance_test,self.heights_test, self.T_test)[:,None]
        # ip_test = trainIpFeature( a[-3], a[-2], a[-1], self.relevance_test,self.heights_test, self.dTdt_test)[:,None]
        # ip_test = trainIpFeature( a[-2], a[-1], self.relevance_test,self.heights_test)[:,None]
        # features_test = np.hstack( ( self.x_test, ip_test ) )

        # print(f"ip_feat.shape {ip_feat.shape}")
        # print(f"self.x_test[:,:-2].shape {self.x_test[:,:-2].shape}")

        # features = np.hstack( ( self.x_test[:,:-2], ip_feat ) )
        # test_loss = fun(dynVector[:,:-1],features,self.y_test)

        # train_loss = objFun(dynVector, features_train, self.y_train) + regFun(a)
        # test_loss = objFun(dynVector, features_test, self.y_test)

        train_loss = self.objFun(dynVector, self.x_train, self.y_train) + self.regFun(a)
        test_loss = self.objFun(dynVector, self.x_test, self.y_test)

        if test_loss<self.min_test_loss:
            self.a_star = a
            self.min_test_loss = test_loss

        # self.ip_feat = ip_feat
        return train_loss

    def setObjFun(self,fun):
        self.objFun = fun

    def setRegFun(self,fun):
        self.regFun = fun
class globalModelOptimzer:
    def __init__(self, x_train, x_test, y_train, y_test, min_test_loss = 1e10):

        # self.x_train = x_train
        self.x_train = x_train[:,:-2]
        self.T_train = x_train[:,0]
        self.input_sequence_train = x_train[:,-2]
        self.heights_train = x_train[:,-1]
        self.dTdt_train = x_train[:,4]

        # self.x_test = x_test
        self.x_test = x_test[:,:-2]
        self.T_test = x_test[:,0]
        self.input_sequence_test = x_test[:,-2]
        self.heights_test = x_test[:,-1]
        self.dTdt_test = x_test[:,4]

        self.y_train = y_train
        self.y_test = y_test
        # print(f"self.x_train.shape {self.x_train.shape}")

        self.min_test_loss = min_test_loss
        self.objFun = None
        self.regFun = None

    def __call__(self, a):
        """
        Save the model achieving the min loss on the test set and return the train loss.
        """
        # dynVector = dynVectorNoLags(a)
        dynVector, input_parameters = dynVectorNoLagsWithInput(a)
        (a_ip,b,c,d) = input_parameters[0,:]
        # build feature
        ip_train = ifu.trainIpFeature( a_ip,b,c,d,0, self.input_sequence_train, self.heights_train, self.T_train, self.dTdt_train)[:,None]
        regressor_train =np.hstack( (self.x_train, ip_train) )
        train_loss = self.objFun(dynVector,regressor_train, self.y_train) + self.regFun(a)
        
        ip_test = ifu.trainIpFeature( a_ip,b,c,d,0, self.input_sequence_test, self.heights_test, self.T_test, self.dTdt_test)[:,None]
        regressor_test =np.hstack( (self.x_test, ip_test) )
        test_loss = self.objFun(dynVector, regressor_test, self.y_test)

        if test_loss<self.min_test_loss:
            self.a_star = a
            self.min_test_loss = test_loss

        # self.ip_feat = ip_feat
        return train_loss

    def setObjFun(self,fun):
        self.objFun = fun

    def setRegFun(self,fun):
        self.regFun = fun

# code class for validation
class lsOptimzer:
    def __init__(self,x_train,x_test,y_train,y_test,min_test_loss = 1e10):

        self.x_train = x_train
        self.x_test = x_test

        self.y_train = y_train
        self.y_test = y_test

        self.min_test_loss = min_test_loss
        self.objFun = None
        self.regFun = None

    def __call__(self, a):
        """
        Save the model achieving the min loss on the test set and return the train loss.
        """
        dynVector = dynVectorNoLags(a)


        # train_loss = self.objFun(dynVector, self.x_train, self.y_train) + self.regFun(a)
        # test_loss = self.objFun(dynVector, self.x_test, self.y_test)
        residuals = np.sum( np.abs( self.y_train - dynVector.dot(self.x_train.T)) / np.mean(np.abs( self.y_train )) / self.y_train.shape[0] )

        if test_loss<self.min_test_loss:
            self.a_star = a
            self.min_test_loss = test_loss

        # self.ip_feat = ip_feat
        return residuals

    def setObjFun(self,fun):
        self.objFun = fun

    def setRegFun(self,fun):
        self.regFun = fun

def calculateTGrad(points):

    neighbor_list = [p.neighbor_nodes for p in validation_experiment.Points if p._hasNeighbors_()]
    temperature_gradients = []
    for (p,neighbors) in zip(points,neighbor_list):
        tmp = [p.T_t_use - points[n].T_t_use for n in neighbors]
        temperature_gradients.append(tmp)

    return np.stack( (temperature_gradients), axis = 0 )

def calculateTGrad2(temp_matrix,neighbor_list):
    
    # ensure correct dimensions
    temperature_gradients = []
    if np.argmax( temp_matrix.shape) == 1:
        temp_matrix = temp_matrix.T

    # calculate gradients
    for (T_t,neighbors) in zip(temp_matrix.T,neighbor_list):

        tmp = np.asarray([T_t - temp_matrix[:,n] for n in neighbors])
        temperature_gradients.append(tmp)

    return temperature_gradients

def plotContributionOfNeighborExcitation(p,experiment,complementary_relevance):
    
    fig = plt.figure(figsize=(16,18))
    ax = fig.add_subplot(211)
    ax.plot(p.T_t_use, label = "Temperature profile")
    ax.plot(p.input_raw_features[1,:] , label = f"Inputs node {p.node}", linestyle = "-.")
    ax.plot(complementary_relevance , label = f"Complementary relevance", linestyle = "-.")

    for neighb in p.neighbor_nodes:
        ax.plot(experiment.Points[neighb].input_raw_features[1,:] , label = f"Inputs node {neighb}", linestyle = "-.")

    ax.set_title(f"Excitation from neighboring nodes - node {p.node}")
    ax.legend(bbox_to_anchor = ( 1.1 , 1 ))

    # delta T
    dTdt = p.T_t_use[1:] - p.T_t_use[:-1]


    ax1 = fig.add_subplot(212)
    ax1.plot(dTdt * 100, label = "Delta T * 100")

    for i,neighb in enumerate(p.neighbor_nodes):
        ax1.plot( p.Delta_T[i+1,:-1] * complementary_relevance[:-1] , label = f"Delta Ts node {neighb},i {i}", linestyle = "-.")

    ax1.set_title(f"Temperature gradients - node {p.node}")
    ax1.legend(bbox_to_anchor = ( 1.1 , 1 ))
    ax1.set_ylim([-0.51, 0.51])

def optimizeInputFeature_oldTimesClassic(points_used_for_training, debug = False):

    # optimize input feature 
    all_inputs = []
    all_heights = []
    all_temperatures = []
    for p in points_used_for_training:
        all_inputs.append(p.input_raw_features[1,:])
        all_heights.append(p.excitation_delay_torch_height_trajectory)
        all_temperatures.append(p.T_t_use)
        
    calculated_inputs = np.hstack(all_inputs)
    excitation_heights = np.hstack(all_heights)
    temperatures = np.hstack(all_temperatures)


    inputFeatureObjective = lambda a: np.linalg.norm ((a[0]*ifu.trainIpFeature(a[1],calculated_inputs,excitation_heights) - temperatures)**2)
    res = minimize(inputFeatureObjective,(1,1), tol = 1e-10)
    ip_feature_coef = res.x[1]

    if debug:
        print(f"calculated_inputs.shape {calculated_inputs.shape}")
        print(f"excitation_heights.shape {excitation_heights.shape}")
        print(f"temperatures.shape {temperatures.shape}")
        print(f"best input featrue coefficient {res.x[1]}")

    return ip_feature_coef

def optimizeInputFeatureOnRelevance(points_used_for_training, debug = False):
    all_inputs = []
    all_heights = []
    all_temperatures = []
    all_delta_temperatures = []
    all_delta_temperatures_lag = []
    for p in points_used_for_training:
        all_inputs.append(p.input_raw_features)
        all_heights.append(p.excitation_delay_torch_height_trajectory)
        all_temperatures.append(p.T_t_use)
        dTdt = np.zeros_like(p.T_t_use)
        dTdt_lag = np.zeros_like(p.T_t_use)
        dTdt[:-1] = p.T_t_use[1:]-p.T_t_use[:-1] # will occur
        dTdt_lag[1:] = p.T_t_use[1:]-p.T_t_use[:-1] # previous

        b,a = butter(8,0.1)
        dTdt_lag_filt = filtfilt(b,a,dTdt_lag)

        all_delta_temperatures.append(dTdt)
        # all_delta_temperatures_lag.append(dTdt)
        all_delta_temperatures_lag.append(dTdt_lag_filt)

    calculated_inputs = np.hstack(all_inputs)
    excitation_heights = np.hstack(all_heights)
    temperatures = np.hstack(all_temperatures)
    delta_temperatures = np.hstack(all_delta_temperatures)
    delta_temperatures_lag = np.hstack(all_delta_temperatures_lag)

    delta_temperatures[ delta_temperatures<0 ] = 0 # Train ONLY on positive delta_temperatures

    # inputFeatureObjective = lambda a: np.linalg.norm ( (a[5]*trainIpFeature(a[0],a[1],a[2],a[3],a[4],calculated_inputs[-1,:],excitation_heights,temperatures,delta_temperatures_lag) - delta_temperatures)**2 ) #+ 1e-4*np.linalg.norm(a[:2]) + 1e-2*np.linalg.norm(a[2:])
    # inputFeatureObjective = lambda a: np.linalg.norm ( (a[5]*trainIpFeature(a[0],a[1],a[2],a[3],a[4],calculated_inputs[-1,:],excitation_heights,temperatures,delta_temperatures_lag) - temperatures)**2 ) # Temperatures help in keeping the excitation consistent!
    inputFeatureObjective = lambda a: np.linalg.norm ( (a[5]*trainIpFeatureDelayFit(a[0],a[1],a[2],a[3],a[4],calculated_inputs[-1,:],excitation_heights,temperatures,delta_temperatures_lag) - temperatures)**2 ) # Temperatures help in keeping the excitation consistent!
    res = minimize(inputFeatureObjective,(0.05,0.05,0.5,0.5,50,0.5), tol = 1e-10)
    ip_feature_coef_a = res.x[0]
    ip_feature_coef_b = res.x[1]
    ip_feature_coef_c = res.x[2]
    ip_feature_coef_d = res.x[3]
    ip_feature_coef_e = res.x[4]

    # debug input training plot
    if debug:
        print(f"calculated_inputs.shape {calculated_inputs.shape}")
        print(f"excitation_heights.shape {excitation_heights.shape}")
        print(f"temperatures.shape {temperatures.shape}")

        inps = ifu.trainIpFeature(res.x[0],res.x[1],res.x[2],res.x[3],res.x[4],calculated_inputs[-1,:],excitation_heights,temperatures,delta_temperatures_lag)
        resps = res.x[5]*inps[ : 5000]

        fig = plt.figure(figsize = (16,9))
        ax = fig.add_subplot(111)
        ax.plot(resps, label = "responses")
        ax.plot(delta_temperatures[ : 5000], label = "temperatures", alpha = 0.5)
        ax.plot(inps[ : 5000], label = "Inputs", linestyle = ":")

        ax.legend()
        ax.set_title("Debug input training ")

    return ip_feature_coef_a,ip_feature_coef_b,ip_feature_coef_c,ip_feature_coef_d,ip_feature_coef_e

def calculatePocketTemperature(temperatures,pocket_neighbors):
    """
    @pocket_neighbors : nodes sourrounding a pocket
    @temperatures : temperatures of nodes sourrounding a pocket
    """
    # temperatures = np.atleast_2d(temperatures).T
    pocket_temperatures = []
    for pocket in pocket_neighbors:
        tmp = np.zeros_like(temperatures[0,:])
        norm = len(pocket)
        for neighbor in pocket:
            tmp += temperatures[neighbor,:]

        pocket_temperatures.append( tmp / norm )
    
    pocket_temperatures = np.vstack(pocket_temperatures)
    return pocket_temperatures

def smoothPocketTemperatures(pocket_temperatures,window_size):
    smoothed_pocket_temperatures = []
    for pocket_temperature in pocket_temperatures:
        pocket_temperature_windows =  rolling_window(pocket_temperature,window_size)
        tmp = []
        for temperature_window in pocket_temperature_windows:
            tmp.append( movingAvg( temperature_window, window_size)[-1] )

        smoothed_pocket_temperatures.append(tmp)

    return np.vstack(smoothed_pocket_temperatures)[:,1:] # smoothed temperatures in pockets

def adjacentPocketsToNode(points_used_for_training, pocket_neighbors):
    pockets_for_each_node = []
    for p in points_used_for_training:
        node_id = p.node
        tmp = []
        for i,pocket in enumerate(pocket_neighbors):
            pocket = set(pocket)
            if node_id in pocket:
                tmp.append( i )

        pockets_for_each_node.append( tmp )
    
    return pockets_for_each_node

def calculateDeltaTwithTpocket(all_T,pockets_for_each_node,smoothed_pocket_temperatures):
    mean_delta_temps = []
    mean_temps = []
    for (T_t,pockets) in zip(all_T,pockets_for_each_node):
        delta_tmp = np.zeros_like(T_t)
        tmp = np.zeros_like(T_t)
        for pocket in pockets:
            tmp += (smoothed_pocket_temperatures[pocket,:])
            delta_tmp += (T_t - smoothed_pocket_temperatures[pocket,:])
        
        mean_delta_temps.append( delta_tmp / len(pockets) )
        mean_temps.append( tmp / len(pockets) )

    return np.vstack(mean_delta_temps), np.vstack(mean_temps)
     
def meanTempValuesPockets( experiment,window_size = 5, pocket_neighbors = None, debug = False):

    if pocket_neighbors is None:
        *_, pocket_neighbors = findEnclosedHallucinatedNodes(experiment)

    points_used_for_training = [p for p in experiment.Points if p._hasNeighbors_() ]

    # calculate pocket temperatures
    all_T = [p.T_t_use for p in points_used_for_training]
    all_T = np.vstack(all_T)

    pocket_temperatures = calculatePocketTemperature(all_T,pocket_neighbors)
    
    # smooth pocket temperatures
    smoothed_pocket_temperatures = smoothPocketTemperatures(pocket_temperatures,window_size)

    # find which pockets surround each nodes
    pockets_for_each_node = adjacentPocketsToNode(points_used_for_training,pocket_neighbors)

    # calculated T - Tm
    mean_delta_temps, mean_temps = calculateDeltaTwithTpocket(all_T,pockets_for_each_node,smoothed_pocket_temperatures)

    if debug:
        print(f"all_T_windows.shape : {smoothed_pocket_temperatures.shape}")
        print(f"points_used_for_training[0].node : {points_used_for_training[0].node} and points_used_for_training[1].node : {points_used_for_training[1].node}")
        print(f"all_T.shape : {all_T.shape}")
        # print(f"mean_temps.shape : {mean_temps.shape}")
        print(f"len(pockets_for_each_node) : {len(pockets_for_each_node)} and pockets_for_each_node[0] : {pockets_for_each_node[0]}")
        for node in [0,13]:

            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(111)
            ax.plot(all_T[node,:],label = "node")
            ax.plot(mean_delta_temps[node,:],label = "mean delta surround")

            ax.legend(title = "Temperature")
            ax.set_title(f"Node {node}")
        
        # plot some pockets
        pocket = 2
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        ax.plot(smoothed_pocket_temperatures[pocket,:],label = f"smooth pocket {pocket}")
        ax.plot(all_T[0,:],label = f" node {0}")

        ax.legend(title = "Temperatures")
        ax.set_title(f"Smoothed Pocket {pocket} Temperature")

    return mean_temps
    # return mean_delta_temps

def meanTempValuesInSouroundingNodes( points_used_for_training, experiment,window_size = 5, debug = False):

    all_T = [p.T_t_use for p in points_used_for_training]
    all_T = np.vstack(all_T)

    all_T_windows = []
    for T_t in all_T:
        all_T_windows.append( rolling_window(T_t,window_size) )
    all_T_windows = np.stack(all_T_windows,axis=2)

    neighbor_list = [p.neighbor_nodes for p in experiment.Points if p._hasNeighbors_()]
    
    mean_temps = []
    mean_delta_temps = []
    for T_windows in all_T_windows:
        # T_windows has all the point windows corresponding to 1 timestep
        delta_tmp, tmp = calculateMeanNeighborTemperature(T_windows.T,neighbor_list, window_size) 
        mean_temps.append(tmp[:,-1])
        mean_delta_temps.append(delta_tmp[:,-1])
    mean_temps = np.vstack(mean_temps).T[:,:-1] # drop last window
    mean_delta_temps = np.vstack(mean_delta_temps).T[:,:-1] # drop last window

    if debug:
        print(f"all_T_windows.shape : {all_T_windows.shape}")
        print(f"points_used_for_training[0].node : {points_used_for_training[0].node} and points_used_for_training[1].node : {points_used_for_training[1].node}")
        print(f"all_T.shape : {all_T.shape}")
        print(f"mean_temps.shape : {mean_temps.shape}")
        print(f"len(neighbor_list) : {len(neighbor_list)} and neighbor_list[0] : {neighbor_list[0]}")
        for node in [0,13]:

            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(111)
            ax.plot(all_T[node,:],label = "node")
            ax.plot(mean_temps[node,:],label = "mean surround")

            ax.legend(title = "Temperature")
            ax.set_title(f"Node {node}")

    return mean_temps
    # return mean_delta_temps

def onlineMeanTempsCalculation(T_state, pocket_neighbors,pockets_for_each_node, window_size):
    
    T_state = np.atleast_2d(T_state).T
    if T_state.shape[1]>window_size:
        T_state = T_state[:,-window_size:]

    # update pocket temperatures
    pocket_temperatures = calculatePocketTemperature(T_state,pocket_neighbors)
    
    # smooth pocket temperatures
    smoothed_pocket_temperatures = smoothPocketTemperatures(pocket_temperatures,window_size)

    # calculated T - Tm
    Delta_T_mean, T_mean = calculateDeltaTwithTpocket(T_state,pockets_for_each_node,smoothed_pocket_temperatures)

    return Delta_T_mean, T_mean

def onlineMeanTempValuesInSouroundingNodes( T_state, neighbor_list, window_size = 5):

    T_state = np.atleast_2d(T_state).T
    if T_state.shape[1]>window_size:
        T_state = T_state[:,-window_size:]

    all_T_windows = []
    for T_t in T_state:
        all_T_windows.append( rolling_window(T_t,window_size) )
    all_T_windows = np.stack(all_T_windows,axis=2)
    
    mean_temps = []
    mean_delta_temps = []
    for T_windows in all_T_windows:
        # T_windows has all the point windows corresponding to 1 timestep
        delta_tmp, tmp = calculateMeanNeighborTemperature(T_windows.T,neighbor_list, window_size) 
        mean_temps.append(tmp[:,-1])
        mean_delta_temps.append(delta_tmp[:,-1])
    mean_temps = np.vstack(mean_temps).T[:,:-1] # drop last window
    mean_delta_temps = np.vstack(mean_delta_temps).T[:,:-1] # drop last window

    return mean_temps, mean_delta_temps
    
def plotFeatureTargetDTdt(p,delta_T_container, axlim = [1950,2050]):
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.plot(p.features[0,:], label = "feature")
    ax.plot(delta_T_container[p.node], label = "target")

    ax.legend(title="Target should lead")
    ax.set_title(f"Temporal change in T | node {p.node}")
    ax.set_xlim(axlim)

def splitData(experiment, T_vector, features_vector, delta_T_vector, min_temperature = 0.0,
    max_temperature = 1.2, number_of_extra_models = 10, elem_thress = 1000, band_overlap = 1.5, debug = False):

    step = (max_temperature - min_temperature)/number_of_extra_models

    band_length = band_overlap*step
    # band_length = 1*step
    bounds = [(-10, min_temperature )]
    ubs = np.linspace(min_temperature + step,max_temperature,number_of_extra_models)
    for ub in ubs:
        bounds.append((ub-band_length,ub))

    bounds,section_idxs = experiment.disectTemperatureRegions(experiment.Points[-1], T_t_custom = T_vector, bounds = bounds, element_thresshold = elem_thress)

    xs_train = [features_vector[:,idxs] for idxs in section_idxs] # data split in groups for model training
    ys_train = [delta_T_vector[idxs] for idxs in section_idxs] # targets split in groups for model training
    Ts = [T_vector[idxs] for idxs in section_idxs]
    band_nominal_T = np.asarray( [np.mean(Ti) for Ti in Ts] )

    # plot the temperature groups
    if debug:
        figs = []
        figs.append(plt.figure(figsize = (16,9)))
        ax = figs[-1].add_subplot(111)
        for i in range(len(bounds)):
            y = Ts[i]
            ax.plot( y, alpha = 0.5, label = f"group {i+1}")
            ax.plot([0, len(y)-1], [band_nominal_T[i],band_nominal_T[i]], color = 'red')

            ax.legend(bbox_to_anchor=(1., 0.75),ncol= 2)

        ax.set_title(f"Training Groups")

    return xs_train, ys_train, band_nominal_T, section_idxs

def splitDataByState( training_points, steps_to_look_ahead = 1):

    # find unique deposition heights. -1 for not being activated
    all_heights = np.vstack([p.excitation_delay_torch_height_trajectory for p in training_points]).T
    unique_heights = np.unique(all_heights)

    # for each point find the idxs of the every height level
    height_level_idxs_container = [ ] # list of lists -> every list corresponds to a point and includes the idxs for every height level  
    for p in training_points:
        tmp = []

        for height in unique_heights:
            idxs_on_layer = np.asarray(np.where(p.excitation_delay_torch_height_trajectory == height))
            tmp.append(idxs_on_layer)
        
        height_level_idxs_container.append(tmp)

    # create the training segments
    xs_train = []
    ys_train = []
    band_heights = []
    for state,height in enumerate(unique_heights):
        xs_current_state = []
        ys_current_state = []
        heights_current_state = []
        
        # assign only data that were collected when printing this state 
        for (p,point_idxs) in zip(training_points,height_level_idxs_container):
            idxs_to_pick = np.atleast_1d(point_idxs[state].squeeze())
            if len(idxs_to_pick)>0:
                xs_current_state_point = p.features[:,idxs_to_pick]
                ys_current_state_point = p.T_t_use[idxs_to_pick] - p.T_t_use[idxs_to_pick-1]

                xs_current_state.append( xs_current_state_point[:,:-steps_to_look_ahead] )
                ys_current_state.append( ys_current_state_point[steps_to_look_ahead:] )

                heights_current_state.append ( np.ones( ( len(idxs_to_pick)-1, ) )*height )
        
        xs_train.append(np.hstack( xs_current_state ))
        ys_train.append( np.hstack(ys_current_state ))
        band_heights.append( np.hstack(heights_current_state ))
    
    # stack training segments to form feature-target array
    # xs_train = np.hstack(xs_train)
    # ys_train = np.hstack(ys_train)
    # band_heights = np.hstack(band_heights)
    
    return xs_train, ys_train, unique_heights, height_level_idxs_container

def plotDeltaT(xs_train,ys_train):
    for i,( x_group_train, y_group_train) in enumerate(zip(xs_train,ys_train)):
        print(i)
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        for j in range(4):
            ax.plot(x_group_train[2+j,:5000], label = f"DT {j}")
        ax.plot(y_group_train[:5000]*10,label = "target",alpha = 0.5)
        ax.legend()
        ax.set_title(f"{i+1} group")

def  plotGPWeights( likelihoods, models, RESULTS_FOLDER, device = None):
    # plot GPs
    n = len(likelihoods.likelihoods)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(-0.6, 0.5, 111).float().to(device = device)
        # This contains predictions for both outcomes as a list
        predictions = likelihoods(*models(*[test_x for i in range(n)]))

    path = RESULTS_FOLDER + 'GP_weights/'
    os.makedirs(path, exist_ok = True)

    for i,(submodel, prediction) in enumerate(zip(models.models, predictions)):
        f, ax = plt.subplots(1,1, figsize=(16,9))
        mean = prediction.mean.squeeze()
        lower, upper = prediction.confidence_region()

        tr_x = submodel.train_inputs[0].cpu().detach().numpy()
        tr_y = submodel.train_targets.cpu().detach().numpy()

        # Plot training data as black stars
        ax.plot(tr_x, tr_y, 'k*')
        # # Predictive mean as blue line
        ax.plot(test_x.cpu().numpy(), mean.cpu().numpy(), 'b')
        # Shade in confidence
        ax.fill_between(test_x.cpu().numpy(), lower.squeeze().cpu().detach().numpy(), upper.squeeze().cpu().detach().numpy(), alpha=0.5)
        # # Shade in confidence
        ax.legend(['Observed Data', 'Mean', 'Confidence'],loc = "upper left")
        ax.set_title('Feature {}'.format(i))
        ax.set_xlabel("Temperature [K ]")
        ax.set_ylabel("Feature weight")

        plt.savefig( path + "{}.jpg".format(i), bbox_inches='tight')
        plt.close(f)

def excitationHeightFeature(height):
    return 1/(height+1)

def formFeatures(ins, LAGS_AHEAD = 1):
    # unpack
    (p,mean_temps,ip_coefs,boundary_, corner_) = ins

    # calculate overheating with respect to the surounding area
    T_t = p.Delta_T[0,:]
    T_pow = T_t**4
    Delta_T = p.Delta_T[1:,:]
    overheating = T_t - mean_temps

    # calculate lag
    T_t_lag = np.zeros_like(T_t)
    T_t_lag[1:] = T_t[:-1]
    DTdt = T_t - T_t_lag

    # keep ambient temperature only if you are in the periphery
    Delta_T_amb = np.zeros_like(p.Delta_T_ambient)
    T_der = np.zeros_like(T_t)
    if boundary_:
        # nohn = 4 - len(p.neighbor_nodes)
        # tmp = [T_t] * nohn
        # tmp += [np.zeros_like(T_t)] * (2 - nohn)
        # Delta_T_amb = np.vstack(tmp)
        
        if corner_:
            Delta_T_amb[0,:] = p.Delta_T_ambient[0,:] 
            Delta_T_amb[1,:] = p.Delta_T_ambient[0,:] 
        else:
            Delta_T_amb[0,:] = p.Delta_T_ambient[0,:] 
    else:
        dummy = 0
        # T_der = DTdt

    # calculate inputs
    calculated_inputs = p.input_raw_features[1,:]
    excitation_heights = p.excitation_delay_torch_height_trajectory
    T_at_the_start_of_input = p.T_at_the_start_of_input

    excitation_height_feature = excitationHeightFeature(excitation_heights)

    # dump
    # overheating *= excitation_height_feature
    # Delta_T *= excitation_height_feature
    # Delta_T_amb *= excitation_height_feature
    # T_der *= excitation_height_feature

    new_ip_feature = ifu.trainIpFeature(*ip_coefs,calculated_inputs,excitation_heights,T_t,T_at_the_start_of_input)
    # new_ip_feature = trainIpFeature(ip_coefs,calculated_inputs,excitation_heights)
    # new_ip_feature = ipFeature( calculated_inputs, excitation_heights)
    # include the values that you will use to learn your feature
    # new_ip_feature = np.vstack( (calculated_inputs[1,:], excitation_heights))

    # feature_set = np.vstack( (T_t, Delta_T, Delta_T_amb, mean_temps, new_ip_feature) ) # 1,4,2,1,1 features
    # feature_set = np.vstack( (overheating, Delta_T, Delta_T_amb, new_ip_feature) ) # 1,4,2,1 features
    feature_set = np.vstack( (overheating, Delta_T, Delta_T_amb, T_der, np.zeros_like(overheating), new_ip_feature) ) # 1,4,2,1,1,1 features

    return feature_set

def formFeaturesWithActivation(ins, LAGS_AHEAD = 1):
    """"
    Form your feature vectors but zero out this time the contribution of all the features that do not contribute to the inputs when the inputs are activated.
    @param inputAtivationSequence : array like relevance or bells that replicates the thermal response of inputs and it's zero at all other times
    """
    # unpack
    (p,mean_temps,ip_coefs,boundary_, corner_) = ins

    # form activation coeffiecients
    input_activation_sequence = p.input_raw_features[0,:]
    max_activation_value = np.max(input_activation_sequence)
    # #print(f"DEBUG: Node {p.node} Max activation value : {max_activation_value}")

    # calculate overheating with respect to the surounding area
    T_t = p.Delta_T[0,:]
    T_pow = (T_t**4) 
    Delta_T = p.Delta_T[1:,:] 
    # overheating = (T_t - mean_temps) 
    overheating = T_t 

    # calculate lag
    T_t_lag = np.zeros_like(T_t)
    T_t_lag[1:] = T_t[:-1]
    DTdt = (T_t - T_t_lag) 

    # keep ambient temperature only if you are in the periphery
    Delta_T_amb = np.zeros_like(p.Delta_T_ambient)
    T_der = np.zeros_like(T_t)
    if boundary_:
        # nohn = 4 - len(p.neighbor_nodes)
        # tmp = [T_t] * nohn
        # tmp += [np.zeros_like(T_t)] * (2 - nohn)
        # Delta_T_amb = np.vstack(tmp)
        
        if corner_:
            Delta_T_amb[0,:] = p.Delta_T_ambient[0,:]  
            Delta_T_amb[1,:] = p.Delta_T_ambient[0,:] 
        else:
            Delta_T_amb[0,:] = p.Delta_T_ambient[0,:] 
    else:
        dummy = 0
    # T_der = DTdt 

    # calculate inputs
    calculated_inputs = p.input_raw_features[1,:]
    # feature_activation_coefficients = 1 - input_activation_sequence
    highest_base_val = np.max(np.abs(calculated_inputs))
    if highest_base_val == 0:
        highest_base_val = 1
    feature_activation_coefficients = (highest_base_val - calculated_inputs)/highest_base_val
    excitation_heights = p.excitation_delay_torch_height_trajectory
    if p.T_at_the_start_of_input is None:
        T_at_the_start_of_input = np.zeros_like(T_t)
    else:
        T_at_the_start_of_input = p.T_at_the_start_of_input
    # excitation_height_feature = excitationHeightFeature(excitation_heights) 

    # dump
    # overheating *= excitation_height_feature
    # Delta_T *= excitation_height_feature
    # Delta_T_amb *= excitation_ height_feature
    # T_der *= excitation_height_feature

    new_ip_feature = ifu.trainIpFeature(*ip_coefs,calculated_inputs,excitation_heights,T_t, T_at_the_start_of_input)

    # apply activation function
    overheating *= feature_activation_coefficients
    Delta_T *= feature_activation_coefficients
    T_der *= feature_activation_coefficients
    Delta_T_amb *= feature_activation_coefficients

    # new_ip_feature = trainIpFeature(ip_coefs,calculated_inputs,excitation_heights)
    # new_ip_feature = ipFeature( calculated_inputs, excitation_heights)
    # include the values that you will use to learn your feature
    # new_ip_feature = np.vstack( (calculated_inputs[1,:], excitation_heights))

    # feature_set = np.vstack( (T_t, Delta_T, Delta_T_amb, mean_temps, new_ip_feature) ) # 1,4,2,1,1 features
    # feature_set = np.vstack( (overheating, Delta_T, Delta_T_amb, new_ip_feature) ) # 1,4,2,1 features
    feature_set = np.vstack( (overheating, Delta_T, Delta_T_amb, T_der, np.zeros_like(overheating), new_ip_feature)  ) # 1,4,2,1,1,1 features

    return feature_set

def formFeaturesWithActivationForGlobalOptimization(ins, LAGS_AHEAD = 1):
    """"
    Form your feature vectors but zero out this time the contribution of all the features that do not contribute to the inputs when the inputs are activated.
    @param inputAtivationSequence : array like relevance or bells that replicates the thermal response of inputs and it's zero at all other times
    """
    # unpack
    (p,mean_temps,ip_coefs,boundary_, corner_) = ins

    # form activation coeffiecients
    input_activation_sequence = p.input_raw_features[0,:]
    max_activation_value = np.max(input_activation_sequence)
    # #print(f"DEBUG: Node {p.node} Max activation value : {max_activation_value}")
    feature_activation_coefficients = 1 - input_activation_sequence

    # calculate overheating with respect to the surounding area
    T_t = p.Delta_T[0,:]
    T_pow = (T_t**4) 
    Delta_T = np.sum(p.Delta_T[1:,:], axis = 0) 
    overheating = (T_t - mean_temps) 

    # calculate lag
    T_t_lag = np.zeros_like(T_t)
    T_t_lag[1:] = T_t[:-1]
    DTdt = (T_t - T_t_lag) 

    # keep ambient temperature only if you are in the periphery
    Delta_T_amb = np.zeros_like(p.Delta_T_ambient)
    T_der = np.zeros_like(T_t)
    if boundary_:
        # nohn = 4 - len(p.neighbor_nodes)
        # tmp = [T_t] * nohn
        # tmp += [np.zeros_like(T_t)] * (2 - nohn)
        # Delta_T_amb = np.vstack(tmp)
        
        if corner_:
            Delta_T_amb[0,:] = p.Delta_T_ambient[0,:]  
            Delta_T_amb[1,:] = p.Delta_T_ambient[0,:] 
        else:
            Delta_T_amb[0,:] = p.Delta_T_ambient[0,:] 
    else:
        dummy = 0
    # T_der = DTdt 

    # calculate inputs
    calculated_inputs = p.input_raw_features[1,:]
    excitation_heights = p.excitation_delay_torch_height_trajectory
    T_at_the_start_of_input = p.T_at_the_start_of_input

    # excitation_height_feature = excitationHeightFeature(excitation_heights) 

    # dump
    # overheating *= excitation_height_feature
    # Delta_T *= excitation_height_feature
    # Delta_T_amb *= excitation_ height_feature
    # T_der *= excitation_height_feature

    # new_ip_feature = ifu.trainIpFeature(*ip_coefs,calculated_inputs,excitation_heights,T_t,DTdt)

    # apply activation function
    overheating *= feature_activation_coefficients
    Delta_T *= feature_activation_coefficients
    T_der *= feature_activation_coefficients
    Delta_T_amb *= feature_activation_coefficients

    # new_ip_feature = trainIpFeature(ip_coefs,calculated_inputs,excitation_heights)
    # new_ip_feature = ipFeature( calculated_inputs, excitation_heights)
    # include the values that you will use to learn your feature
    # new_ip_feature = np.vstack( (calculated_inputs[1,:], excitation_heights))

    # feature_set = np.vstack( (T_t, Delta_T, Delta_T_amb, mean_temps, new_ip_feature) ) # 1,4,2,1,1 features
    # feature_set = np.vstack( (overheating, Delta_T, Delta_T_amb, new_ip_feature) ) # 1,4,2,1 features
    feature_set = np.vstack( (overheating, Delta_T, Delta_T_amb, T_der, np.zeros_like(overheating), calculated_inputs, excitation_heights)  ) # 1,4,2,1,1,1 features

    return feature_set

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

def _isOnCorner_(p,boundaries):
    
    [xMax,xMin,yMax,yMin] = boundaries
    coordinates = p.hallucinated_nodes_coordinates 

    count = 0
    for coordinate in coordinates:
        if (coordinate[0] < xMin or coordinate[0] > xMax) or (coordinate[1] < yMin or coordinate[1] > yMax):
            count += 1

    if count == 2:
        corner_ = True
    else:
        corner_ = False
    return corner_

def pointsOnBoundaries_(experiment):
    
    points_used_for_training = [p for p in experiment.Points if p._hasNeighbors_()]
    coordinate_list = [p.coordinates for p in points_used_for_training]
    coordinates = np.vstack(coordinate_list)
    boundaries = findDomainLimits(coordinates)
    on_boundary_ = [_isOnBoundary_(p,boundaries) for p in points_used_for_training]
    
    # find cornenrs as well
    # on_corner_ = [_isOnCorner_(p,boundaries) for p in points_used_for_training]
    on_corner_ = [False for p in points_used_for_training]

    return on_boundary_, on_corner_

def findEnclosedHallucinatedNodes(experiment, debug = False):
    """
    @returns hallucinated_nodes : node index
    @returns enclosed_unique_hallucinated_nodes_coordinates : coordinates of the hallucinated nodes
    @returns neighboring_nodes : index of the neighboring nodes
    """
    
    boundaries_,_ = pointsOnBoundaries_(experiment)
    enclosed_points = [p for (p,b_) in zip(experiment.Points,boundaries_) if b_ is False]
    points_used_in_the_experimet = [p for p in experiment.Points if p._hasNeighbors_()]

    # find unique hallucinated nodes - in coordinates
    enclosed_unique_hallucinated_nodes = set()
    for p in enclosed_points:
        for hallucinated_node in p.hallucinated_nodes_coordinates:
            enclosed_unique_hallucinated_nodes.add(hallucinated_node)
    enclosed_unique_hallucinated_nodes = list(enclosed_unique_hallucinated_nodes)
    # sort the hallucinated nodes
    sorted_enclosed_unique_hallucinated_nodes = np.vstack(sorted(enclosed_unique_hallucinated_nodes))

    # find the nodes which are neighboring with each hallucinated node
    acceptance_radius = np.sqrt(2*27**2) + 1
    neighbors = []
    for coordinates in sorted_enclosed_unique_hallucinated_nodes:
        
        tmp = []
        for p in points_used_in_the_experimet:
            distance_from_hn = np.linalg.norm(p.coordinates-coordinates)
            if distance_from_hn <= acceptance_radius:
                tmp.append(p.node)
            
        neighbors.append(tmp)
    
    hallucinated_nodes = np.arange(0,len(enclosed_unique_hallucinated_nodes))
    enclosed_unique_hallucinated_nodes_coordinates = np.vstack(enclosed_unique_hallucinated_nodes)
    neighboring_nodes = np.vstack(neighbors)

    if debug:
        print(f"hallucinated_nodes.shape : {hallucinated_nodes.shape} ")
        print(f"enclosed_unique_hallucinated_nodes_coordinates.shape : {enclosed_unique_hallucinated_nodes_coordinates.shape} ")
        print(f"neighboring_nodes.shape : {neighboring_nodes.shape} ")
    
    return hallucinated_nodes, enclosed_unique_hallucinated_nodes_coordinates, neighboring_nodes

def plotFeatures(p):
    fig = plt.figure(figsize = (16,18))

    ax1 = fig.add_subplot(411)
    ax1.plot(p.features[0,:], label = "overheating")
    ax1.plot(p.features[1,:], label = "deltaT")
    ax1.plot(p.features[2,:], label = "deltaT")
    ax1.plot(p.features[3,:], label = "deltaT")
    ax1.plot(p.features[4,:], label = "deltaT")
    ax1.set_title(f"Features of node {p.node}")
    ax1.legend()

    ax2 = fig.add_subplot(412)
    ax2.plot(p.features[5,:], label = "First ambient")
    ax2.plot(p.features[6,:], label = "Second ambient")
    ax2.legend()

    # fig2 = plt.figure(figsize = (16,18))
    # ax3 = fig2.add_subplot(211)
    ax3 = fig.add_subplot(413)
    ax3.plot(p.features[7,:], label = "dTdt")
    ax3.legend()

    # ax4 = fig2.add_subplot(212)
    ax4 = fig.add_subplot(414)
    ax4.plot(p.features[9,:], label = "input")
    ax4.legend()

def plotDelays(delay_model,x,y,results_folder = "results/general_dbging"):

    forecasts = delay_model.predict(x)
    heights =  np.asarray([i for i in np.linspace(0,0.4,6)]).reshape(-1,1)

    fig = plt.figure( figsize=(16,9) )
    ax = fig.add_subplot(111)
    ax.scatter(x,y,label = "forecasts")
    ax.scatter(x,forecasts,label = "predictions", color = "darkorange")
    ax.plot(heights,delay_model.predict(heights),label = "Model", color = "darkorange")
    ax.set_title( "Delay" )
    ax.set_xlim([0,0.4])
    ax.legend()

    plt.savefig(results_folder + "/delay_model.png")
    plt.close()

def findDeactivationStartingIdxs(point = None, feature_activation = None, debug = False):
    """
    Signals input sequence start.
    @ params point : _Point object
    @ params feature_activation : something like a part of bell sequence
    @ returns start_idxs : array with the first indexes of the input sequences  
    @ returns start_temperatures : array with the first temperatures of the input sequences. If the point is None then this is None as well   
    """
    # mask feature activation with a binary filter
    if np.any(feature_activation) == None:
        feature_activation = point.input_raw_features[0,:]

    masked_feature_activation = feature_activation.copy()
    masked_feature_activation[feature_activation != 0] = 1

    # take the derivative of the masked feature
    diff_masked_feature_activation = np.zeros_like(masked_feature_activation)
    diff_masked_feature_activation[:-1] = np.diff(masked_feature_activation)

    # Keep only the possitive derivative (activation)
    diff_masked_feature_activation[diff_masked_feature_activation<0] = 0

    # input start indexes 
    start_dixs = np.asarray(np.where(diff_masked_feature_activation>0)).squeeze()
    if np.min(start_dixs.shape) == 0:
        start_dixs = np.zeros((1,)).astype(np.int64)
    if point is not None:
        start_temperatures = point.T_t_use[start_dixs]
    else:
        start_temperatures = None


    if debug:
        plt.figure(figsize= (16,9) )
        plt.plot(point.T_t_use,label = "temperature")
        plt.plot(feature_activation,label = "activation seq")
        plt.plot(masked_feature_activation,label = "masked activation")
        plt.scatter(start_dixs,start_temperatures,label = "starting points",marker='x')
        plt.legend()
        plt.title(f"First input idxs node {point.node}")

    return start_dixs,start_temperatures

def inputStartingIdxsFromLengthscale( input_idxs, lengthscales):
    """
    @params input_idxs : array with input idxs
    @params lenghscales : array with lengthscales
    @returns idxs where input sequences start having a substantial value
    """
    offset = 3*lengthscales.astype(np.int64)
    out = input_idxs - offset
    out[out<0] = 0
    return out.astype(np.int64)

def inputStartingTemperaturesFromLengthscale( T_array, input_idxs, lengthscales):
    """
    @params T_array : array with Temperature history
    @params input_idxs : array with input idxs
    @params lenghscales : array with lengthscales
    @returns idxs where input sequences start having a substantial value
    """
    idxs = inputStartingIdxsFromLengthscale( input_idxs, lengthscales)
    T_before_input = T_array[idxs]
    return idxs, T_before_input 

def fillInputStartingTemperatureArray(starting_temperatures,starting_idxs,activation_function):
    """
    fill the T_start values for each point. It should produce a step function
    """
    starting_T_array = np.zeros_like(activation_function)

    starting_T_array[:starting_idxs[0]] = starting_temperatures[0]
    for i,(starting_idx,starting_temperature) in enumerate(zip(starting_idxs[1:],starting_temperatures[:-1])):
        prev_starting_idx = starting_idxs[ i ] # index of starting_idxs starts from 1 but in the enumerate, i starts from 0. So i will always lag one idx behind starting_idxs idx
        starting_T_array[prev_starting_idx:starting_idx] = starting_temperature
    starting_T_array[starting_idx:] = starting_temperatures[-1]

    return starting_T_array
    
def tryconvert(value, default, *types):
    """
    try to convert value to types and if you fail return default.
    """
    for t in types:
        try:
            return t(value)
        except (ValueError, TypeError):
            continue
    return default

def exportTrajectory(experiment,saving_location = "../results/general_dbging"):
    trajectory_x = experiment.trajectory[:,0]
    trajectory_y = experiment.trajectory[:,1]
    tool_deposition_height = experiment.tool_deposition_depth
    tool_statuses = experiment.statuses

    total_trajectory = np.stack((trajectory_x,trajectory_y,tool_deposition_height.T,tool_statuses),axis = -1)

    dict_out = {"x":trajectory_x,
                "y":trajectory_y,
                "z":tool_deposition_height,
                "status":tool_statuses
                }

    df_out = pd.DataFrame(dict_out)
    saving_name = saving_location + f"/reconstructed_trajectory_exp{experiment.experiment_id}.csv"
    df_out.to_csv(saving_name)
    print(f"Generated trajectory file {saving_name}")

# %%
if __name__ == "__main__":
    # test functions
    a = np.arange(0,10,dtype = np.float64)
    b = np.arange(10,20,dtype = np.float64)
    c = np.arange(10,20,dtype = np.float64)

    all = np.vstack((a,b,c))
    print(f"all.shape {all.shape}")
    neighbor_list = [ [1,2], [0,2], [0,1] ]
    res_delta,res = calculateMeanNeighborTemperature(all,neighbor_list)
    res2_delta,res2 = calculateMeanNeighborTemperature(all,neighbor_list,smoothing_window=5)
    print(f"res.shape {res.shape}")
    print("res:")
    print(res)
    print(f"res2.shape {res2.shape}")
    print("res2:")
    print(res2)

