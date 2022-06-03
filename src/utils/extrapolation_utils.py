
import numpy as np
import utils.generic_utils as gnu
import utils.online_optimization_utils as onopt

from copy import deepcopy as copy
from utils.generic_utils import setDevice
from scipy.interpolate import interp2d 

import torch
import gpytorch
import time 

from dtw import *

def prepareExtrapolation(validation_experiment, points_used_for_validation, delay_model, likelihoods, gp_models, GP_scaler, f, device = None, mean_model = True, samples = 100):

    device = setDevice(device)
    n = len(likelihoods.likelihoods)

    all_states = [p.excitation_delay_torch_height_trajectory for p in points_used_for_validation]
    unique_states =  np.unique(all_states)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        unique_states = torch.tensor(unique_states).float().to(device=device)
        GP_scaler = GP_scaler.to(device)
        likelihoods = likelihoods.to(device=device)
        gp_models = gp_models.to(device=device)
        infered_coefs = likelihoods( *gp_models(*[unique_states for i in range(n)] ) )
        means_np = np.vstack([((model.mean)*scale).squeeze().cpu().detach().numpy() for (model,scale) in zip(infered_coefs,GP_scaler)])
        if mean_model:
            infered_coefs_np = means_np
            samples = 1
        else:
            infered_coefs_samples = np.stack([(model.sample(sample_shape = torch.Size((samples,)))*scale).squeeze().cpu().detach().numpy() for (model,scale) in zip(infered_coefs,GP_scaler)], axis = -1)

            # make sure the signs are the same with the mean for mainting stability  
            signs = np.sign(means_np).T
            infered_coefs_np = []
            for infered_coefs_sample in infered_coefs_samples:
                infered_coefs_np.append(signs * np.abs(np.vstack(infered_coefs_sample)))
            infered_coefs_np = np.stack(infered_coefs_np,axis = 0)

    models_numbering = np.arange(n)
    if samples == 1:
        infered_coefs_np = infered_coefs_np.squeeze()
        weights_interpolator = interplWeights(unique_states,models_numbering,infered_coefs_np.T)
        coefficients, F_sequences, all_excitation_heights, T_ambient = calculatePathValues(points_used_for_validation, delay_model, weights_interpolator, models_numbering, validation_experiment, f)


    else:
        coefficients_rep = []
        input_coefficients_rep = []
        F_sequences_rep = []
        all_excitation_heights_rep = []
        T_ambient_rep = []
        for coefficient_samples in infered_coefs_np:
            weights_interpolator = interplWeights(unique_states,models_numbering,coefficient_samples)
            coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = calculatePathValues(points_used_for_validation, delay_model, weights_interpolator, models_numbering, validation_experiment, f)        

            coefficients_rep.append(coefficients_samp)
            F_sequences_rep.append(F_sequences_samp)
            all_excitation_heights_rep.append(all_excitation_heights_samp)
            T_ambient_rep.append(T_ambient_samp)
        
        coefficients = np.stack(coefficients_rep,axis=0)
        F_sequences = np.stack(F_sequences_rep,axis=0)
        all_excitation_heights = np.stack(all_excitation_heights_rep,axis=0)
        T_ambient = np.stack(T_ambient_rep,axis=0)

    return coefficients, F_sequences, all_excitation_heights, T_ambient

def interplWeights(unique_states,model_number,weight_values):
    return interp2d(unique_states,model_number,weight_values.T)

def calculatePathValues(points_used_for_validation, delay_model, weights_interpolator, models_arange, validation_experiment, f : onopt.fTerm):
    Fsequence_repository = []
    excited_nodes_ = []
    coefficient_repository = []
    excitation_heights_repository = []
    all_lengthscales = []
    all_peak_idxs = []
    array_length = 0
    for p in points_used_for_validation:

        array_length = np.max([np.max(p.T_t_use.shape),array_length])
        excitation_heights = p.excitation_delay_torch_height_trajectory
        excitation_heights_repository.append(excitation_heights)

        # ## delayed (and NOT perfectly synchronized) inputs
        if len(p.excitation_idxs)>0:

            inputs_to_delay_model = excitation_heights[p.excitation_idxs].reshape(-1,1)
            observed_peak_indexes = p.excitation_idxs.reshape(-1,1) 
            delays = delay_model.predict(inputs_to_delay_model).astype(np.int64)[:,None]

            min_elements = np.min([ (len(delays),len(observed_peak_indexes), len(p.excitation_delay_torch_height)) ])
            observed_peak_indexes = (observed_peak_indexes[:min_elements] + delays[:min_elements]).squeeze()

            # keep only those indexes that lie in your recorded horizon
            observed_peak_indexes = observed_peak_indexes[observed_peak_indexes<len(p.T_t_use)]

            # for lengthscales
            height_at_input = inputs_to_delay_model[:len(observed_peak_indexes),:]
        else:
            observed_peak_indexes = np.array([])
            min_elements = 0
            height_at_input = np.array([])

        infered_coefs_np = weights_interpolator(excitation_heights,models_arange)

        coefficient_repository.append(infered_coefs_np)
    
        if len(observed_peak_indexes)>0:
            lengthscales_for_point_input = infered_coefs_np[-1,observed_peak_indexes]
        else :
            lengthscales_for_point_input = np.array([])

        all_lengthscales.append(lengthscales_for_point_input)
        all_peak_idxs.append(observed_peak_indexes)
        ## sequence F_sequences calculating inputs
        # print(f"DEBUG node {p.node}")
        # Fsequence_repository.append(onopt.Fsequence(observed_peak_indexes,lengthscales_for_point_input,array_length,bell_resolution = 100))
        # f.update(lengthscales_for_point_input)
        # Fsequence_repository.append( f( observed_peak_indexes, array_length=len(infered_coefs_np) ))
        # if len(p.excitation_idxs)>0:
        #     debugFsec(observed_peak_indexes,lengthscales_for_point_input,array_length,p)

        ## sequence for activating the correct parts of the response
        if len(observed_peak_indexes)>0:
            excited_nodes_.append(True)
        else:
            excited_nodes_.append(False)

    f.update(all_lengthscales)
    Fsequence_repository = f(all_peak_idxs,array_length=array_length,multiple_params=True)

    coefficients = np.stack(coefficient_repository, axis=2) # 0 -> var, 1 -> sample, 2-> point
    F_sequences = np.vstack(Fsequence_repository) # 2-> point, 1 -> sample
    all_excitation_heights = np.vstack(excitation_heights_repository) # 0 -> sample, 1-> point
    T_ambient = validation_experiment.T_ambient[::validation_experiment.subsample]

    return coefficients, F_sequences, all_excitation_heights, T_ambient

def synchronizeWithDtwwrapper(ins):
    (validation_experiment, T_state_np,  starting_idx, steps, node) =ins
    # calculate dtw distance
    *_, normalized_distance, std_step_cost_increase, max_step_cost_increase, min_step_cost_increase  = synchronizeWithDtw( T_state_np[:,node], validation_experiment.Points[node].T_t_use[starting_idx : starting_idx+steps+1])

    return normalized_distance, std_step_cost_increase, min_step_cost_increase, max_step_cost_increase

def probabilisticPredictionsWrapper(validation_experiment, m, g, f, likelihoods, gp_models, GP_scaler , starting_idx = 300, steps = 4000, samples = 100, messages = False, mean_model = False, device = "cpu", scaler = None, delay_model = None, process_validation_experiment = True, dt = 0.5):

    # pre-process experiment: interpolate torch path, scale measurements, subsample, and find times that torch crosses each node
    if process_validation_experiment:
        _ = validation_experiment.trajectorySynchronization(J = 1,  lengthscale = 10, lengthscale_sharp = 10 )
        _ = validation_experiment.torchCrossingNodes()

        _ = validation_experiment.scaleMeasurements(copy(scaler))
        validation_experiment.subsampleTrajectories() 


    if steps + starting_idx > len(validation_experiment.Points[0].T_t_use):
        steps = len(validation_experiment.Points[0].T_t_use) - starting_idx - 1    

    # extract inputs
    n = len(likelihoods.likelihoods)
    points_used_for_validation = [ p for p in validation_experiment.Points if p._hasNeighbors_()]
    boundaries_ = gnu.pointsOnBoundaries_(validation_experiment)
    delta_0 = np.asarray(boundaries_)
    delta_1 = np.zeros_like(delta_0)

    # draw different weights and extrapolate with them
    extrapolations, all_elapsed = propabilisticExtrapolation( validation_experiment, points_used_for_validation, delay_model, m, g, f, likelihoods, gp_models, GP_scaler, mean_model, samples, device, messages, starting_idx, steps, dt = dt )
    
    return extrapolations,validation_experiment,all_elapsed

def propabilisticExtrapolation( validation_experiment, points_used_for_validation, delay_model, m, g, f, likelihoods, gp_models, GP_scaler, mean_model, samples, device, messages, starting_idx, steps, dt = 0.5):
    # draw different weights and extrapolate with them
    extrapolations = []
    ## prepare extrapolation
    coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = prepareExtrapolation(validation_experiment = validation_experiment, 
    points_used_for_validation = points_used_for_validation, 
    delay_model = delay_model,
    likelihoods = likelihoods, 
    gp_models = gp_models, 
    f = f,
    GP_scaler = GP_scaler,
    mean_model = mean_model, 
    samples=samples, 
    device = device)

    all_elaspsed = []
    if mean_model:
        samples = 1
        coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = [coefficients_samp], [F_sequences_samp], [all_excitation_heights_samp], [T_ambient_samp]
    
    for i,(coefficients, F_sequences, all_excitation_heights, T_ambient) in enumerate(zip(coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp)):
        if messages:
            print(f"Sampe {i+1}/{samples}")

        ## initialize roll out
        T_state = []
        T_dot_last = []
        T_state.append( np.hstack([ p.T_t_use[starting_idx] for p in validation_experiment.Points ]) )

        # tmp = []
        # for p in points_used_for_validation:
        #     tmp.append(p.T_t_use[starting_idx])
        #     if starting_idx>0:
        #         T_dot_last.append( p.T_t_use[starting_idx] - p.T_t_use[starting_idx-1] )
        #     else:
        #         T_dot_last.append( np.asarray([0]) )

        # T_state.append( np.hstack(tmp) )

        # total model propagation
        theta_M = coefficients[:3,:]
        theta_P = coefficients[3,:].T
        theta_G = coefficients[4:7,:]
        model = onopt.totalModel(F_sequences, all_excitation_heights, m = m, g = g, theta_g = theta_G, theta_m = theta_M, theta_p = theta_P)

        timesteps = np.arange(starting_idx,starting_idx+steps)*dt
        for t in timesteps:
            T_state.append( model(t,T_state[-1]))

        T_state_np = np.asarray(T_state)

        ## quick state propagation function
        # neighbor_list = [p.neighbor_nodes for p in validation_experiment.Points if p._hasNeighbors_()]
        # statePropagation = quickStatePropagationWithActivation(neighbor_list, delta_0, delta_1)

        # T_state_0 = T_state[-1]
        
        t = time.time()
        
        #TODO change to integrate the new stuff
        # T_state_np, _ = extrapolate(
        # statePropagation = statePropagation,
        # coefficients = coefficients,
        # all_excitation_heights = all_excitation_heights,
        # Fsequence = F_sequences,
        # T_ambient = T_ambient,
        # T_state_0 = T_state_0,
        # starting_idx = starting_idx,
        # num_of_steps = steps,
        # )
        
        elapsed = time.time() - t
        all_elaspsed.append(elapsed)
        extrapolations.append(T_state_np)

    return extrapolations,all_elaspsed

def extrapolate(statePropagation,coefficients,input_coefficients,all_excitation_heights,Fsequence,T_ambient,T_state_0,starting_idx,num_of_steps):
    # propagate state
    steps = num_of_steps
    T_state = [T_state_0]

    DEBUG_COEFS = []
    DEBUG_resps = []
    DEBUG_inputs = []

    for step in range(1,steps+1,1):

        ## pick pre-computed values
        F = Fsequence[:,starting_idx + step]
        excitation_heights = all_excitation_heights[:,starting_idx + step]
        feature_coef = coefficients[:,starting_idx + step,:]
        ip_feature_coef = input_coefficients[:,starting_idx + step,:]
        T_last = T_state[-1]
        T_amb = T_ambient[starting_idx + step]

        ## propagate state
        new_state, input_feature = statePropagation(
            T = T_last,
            T_boundary = T_amb,
            h = excitation_heights, 
            g_coef = ip_feature_coef,
            F = F,
            dyn_vectors = feature_coef)

        DEBUG_inputs.append(input_feature)
        T_state.append(new_state)

    DEBUG_COEFS = np.asarray(DEBUG_COEFS)

    DEBUG_resps = np.asarray(DEBUG_resps)

    DEBUG_inputs = np.asarray(DEBUG_inputs)

    return np.vstack(T_state),DEBUG_inputs

class quickStatePropagationWithActivation:
    def __init__(self,neighbor_nodes, delta_0, delta_1, feature_scaler = 1, d_grid = 27):
        """
        neighbor_nodes : list of iterables with neighboring nodes | len() : nop
        delta_0 : boolean array with selection values for first boundary | shape (nop,)
        delta_1 : boolean array with selection values for second boundary | shape (nop,)
        """
        nop = max(delta_0.shape)
        self.delta_0 = delta_0
        self.delta_1 = delta_1
        self.featureScaler = feature_scaler
        self.nop = nop
        self.associated_temperatures = np.zeros((nop,4))
        self.d_grid = d_grid
        self.features = np.zeros((nop,5))
        self.T_bound = np.zeros((nop,))
        self.T_start = np.zeros((nop,))

        # transform neighboring idxs in tuples (node, node_neighbor)
        idxs_for_T_array = []
        idxs_for_associated_T = []
        coefficients_for_substracting_T = []
        for node,neighbor_node_group in enumerate(neighbor_nodes):
            for i,neighbor_node in enumerate(neighbor_node_group):
                idxs_for_T_array.append(neighbor_node)
                idxs_for_associated_T.append((node,i))
            coefficients_for_substracting_T.append(i+1)

        self.neighbor_nodes = neighbor_nodes
        self.idxs_for_T_array = np.asarray(idxs_for_T_array)
        self.idxs_for_associated_T = np.asarray(idxs_for_associated_T)
        self.number_of_neighbors = np.asarray(coefficients_for_substracting_T)

    def __call__(self, T, T_boundary, h, g_coef, F, dyn_vectors):
        """
        T  | shape (nop,)
        T_mean  | shape (nop,)
        T_boundary  | shape (nop,)
        input_feature  | shape (nop,)
        activation_values  | shape (nop,)
        dyn_vector  | shape (nop,numberof_features)
        """

        activaation_values = 1 - F
        self.associated_temperatures[self.idxs_for_associated_T[:,0],self.idxs_for_associated_T[:,1]] = T[self.idxs_for_T_array]

        self.T_bound = (T - T_boundary)*activaation_values

        self.features[:,0] = T*activaation_values
        self.features[:,1] = (self.number_of_neighbors*T - np.sum(self.associated_temperatures, axis = -1))/(self.d_grid/100)*activaation_values#/ self.d_grid
        self.features[:,2] = self.T_bound*self.delta_0*activaation_values
        self.features[:,3] = self.T_bound*self.delta_1*activaation_values
        G_vals = onopt.G( *g_coef, F = F, h = h,T = T)
        self.features[:,4] =  G_vals

        DT = np.sum(self.features*dyn_vectors.T/self.featureScaler,axis = -1)
        return T+DT , G_vals

    def debugLaplacians(self,T):
        # build differences
        differences = []
        for (T_i,neighbors) in zip(T,self.neighbor_nodes):
            tmp = []
            for neighb in neighbors:
                tmp.append(T_i-T[neighb])

            differences.append(tmp)

        # compare
        lap = (self.number_of_neighbors*T - np.sum(self.associated_temperatures, axis = -1))
        lap_test = np.asarray( [np.sum(diffs) for diffs in differences] )

        same_ = np.all(lap,lap_test)
        return same_

def calculateStdsOnCentralPlate(neighbors,temperature_array):
    """
    Calculate standard deviations in T and delta T field without taking the nodes on the boundaries.
    @params neighbors: list of lists with neighbors. Each sublist contains the neighbors of each node
    @params temperature array : n x num_of_points_on_grid
    """
    # select the data in the central part of the plate
    T_idxs_NOT_to_keep = [False if len(neigbs)==4 else True for neigbs in neighbors]
    T_idxs_NOT_to_keep = np.asarray(T_idxs_NOT_to_keep)

    # build a matrix where if the (i,j) tile is 1 then i is neighboring j. The Matrix should be symmetric
    neighborhood = np.zeros((len(neighbors),len(neighbors)))
    for i,neighbor_idxs in enumerate(neighbors) :
        for neighb in neighbor_idxs:
            neighborhood[i,neighb] = 1
    
    # screen out results
    neighborhood_to_keep = neighborhood.copy() 
    neighborhood_to_keep[T_idxs_NOT_to_keep,:] = 0
    neighborhood_to_keep[:,T_idxs_NOT_to_keep] = 0

    # now keep only the upper triangular part for keeping only one copy of each neighbor pairs
    unique_neighborhood = np.triu(neighborhood_to_keep)
    unique_neighbors = np.where(unique_neighborhood > 0.5)
    unique_neighbors_np = np.vstack(unique_neighbors).T

    # build differences
    Delta_T = []
    for neighbor_idxs in unique_neighbors_np:
        Delta_T.append( np.abs( temperature_array[:, neighbor_idxs[0]] - temperature_array[:, neighbor_idxs[1]]) )

    Delta_T = np.vstack(Delta_T).T

    # calculate stds
    std_delta_T = np.std(Delta_T,axis = -1)
    mean_delta_T = np.mean(Delta_T,axis = -1)
    std_T = np.std(temperature_array,axis = -1)
    mean_T = np.mean(temperature_array,axis = -1)

    return mean_T, std_T, mean_delta_T, std_delta_T

def synchronizeWithDtw(signal1,signal2):
    # return sync_signal1, sync_signal2, distance
    alignment = dtw( signal1, signal2, keep_internals=True)
    # normalized_distance = alignment.normalizedDistance
    
    index1 = alignment.index1
    sync_signal1 = signal1[index1]
    
    index2 = alignment.index2
    sync_signal2 = signal2[index2]

    # find the indexes where the second signal is significant (important indexes)
    substantial_idxs = np.where(np.abs(sync_signal2)>0.05)
    normalized_distance = np.mean( np.abs( (sync_signal1[substantial_idxs] - sync_signal2[substantial_idxs] )/sync_signal2[substantial_idxs] ) )

    costMatrix = alignment.costMatrix
    costsPerStep = costMatrix[index1,index2]
    diffCostsPerStep = np.diff(costsPerStep)
    std_step_cost_increase = np.std(diffCostsPerStep)
    max_step_cost_increase = diffCostsPerStep.max()
    min_step_cost_increase = diffCostsPerStep.min()

    return sync_signal1, sync_signal2, normalized_distance,std_step_cost_increase, max_step_cost_increase, min_step_cost_increase

def unscale(responses,scaler):
    responses_unscale = responses.T
    unscaled_responses_data = np.zeros_like(responses_unscale)
    for i,response_unscale in enumerate(responses_unscale):
        to_scaler = np.vstack([response_unscale,response_unscale*0]).T
        unscaled_responses_data[i,:] = scaler.inverse_transform(to_scaler).T[0,:]

    return unscaled_responses_data.T
