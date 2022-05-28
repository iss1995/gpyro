
import numpy as np
import utils.helper_func as hlp
import utils.online_optimization_utils as onopt

from copy import deepcopy as copy
from utils.generic_utils import setDevice
from scipy.interpolate import interp2d 

import torch
import gpytorch
import time 

from dtw import *

def prepareExtrapolation(validation_experiment, points_used_for_validation, delay_model, likelihoods, gp_models, GP_scaler, device = None, mean_model = True, samples = 100):

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
        coefficients, input_coefficients, F_sequences, all_excitation_heights, T_ambient = calculatePathValues(points_used_for_validation, delay_model, weights_interpolator, models_numbering, validation_experiment)


    else:
        coefficients_rep = []
        input_coefficients_rep = []
        F_sequences_rep = []
        all_excitation_heights_rep = []
        T_ambient_rep = []
        for coefficient_samples in infered_coefs_np:
            weights_interpolator = interplWeights(unique_states,models_numbering,coefficient_samples)
            coefficients_samp, input_coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = calculatePathValues(points_used_for_validation, delay_model, weights_interpolator, models_numbering, validation_experiment)        

            coefficients_rep.append(coefficients_samp)
            input_coefficients_rep.append(input_coefficients_samp)
            F_sequences_rep.append(F_sequences_samp)
            all_excitation_heights_rep.append(all_excitation_heights_samp)
            T_ambient_rep.append(T_ambient_samp)
        
        coefficients = np.stack(coefficients_rep,axis=0)
        input_coefficients = np.stack(input_coefficients_rep,axis=0)
        F_sequences = np.stack(F_sequences_rep,axis=0)
        all_excitation_heights = np.stack(all_excitation_heights_rep,axis=0)
        T_ambient = np.stack(T_ambient_rep,axis=0)

    return coefficients, input_coefficients, F_sequences, all_excitation_heights, T_ambient

def interplWeights(unique_states,model_number,weight_values):
    return interp2d(unique_states,model_number,weight_values.T)

def calculatePathValues(points_used_for_validation, delay_model,weights_interpolator, models_arange, validation_experiment):
    Fsequence_repository = []
    input_starting_idxs_repository = []
    excited_nodes_ = []
    coefficient_repository = []
    input_model_coefficient_repository = []
    excitation_heights_repository = []
    for p in points_used_for_validation:

        array_length = np.max(p.T_t_use.shape)
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

        ## sequence F_sequences calculating inputs
        # print(f"DEBUG node {p.node}")
        Fsequence_repository.append(onopt.Fsequence(observed_peak_indexes,lengthscales_for_point_input,array_length,bell_resolution = 100))
        # if len(p.excitation_idxs)>0:
        #     debugFsec(observed_peak_indexes,lengthscales_for_point_input,array_length,p)

        ## sequence for activating the correct parts of the response
        if len(observed_peak_indexes)>0:
            excited_nodes_.append(True)
        else:
            excited_nodes_.append(False)

    coefficients = np.stack(coefficient_repository, axis=2)[:5,:,:] # 0 -> var, 1 -> sample, 2-> point
    input_coefficients = np.stack(coefficient_repository, axis=2)[5:10,:,:] # 0 -> var, 1 -> sample, 2-> point
    F_sequences = np.vstack(Fsequence_repository) # 2-> point, 1 -> sample
    all_excitation_heights = np.vstack(excitation_heights_repository) # 0 -> sample, 1-> point
    T_ambient = validation_experiment.T_ambient[::validation_experiment.subsample]

    return coefficients, input_coefficients, F_sequences, all_excitation_heights, T_ambient

def debugFsec(observed_peak_indexes,lengthscales_for_point_input,array_length,p):
    import matplotlib.pyplot as plt
    Fsec = onopt.Fsequence(observed_peak_indexes,lengthscales_for_point_input,array_length,bell_resolution = 100)
    fig = plt.figure(figsize=(16,9))
    plt.plot(Fsec)
    plt.title(p.node)
    plt.show()

def synchronizeWithDtwwrapper(ins):
    (validation_experiment, T_state_np,  starting_idx, steps, node) =ins
    # calculate dtw distance
    *_, normalized_distance, std_step_cost_increase, max_step_cost_increase, min_step_cost_increase  = synchronizeWithDtw( T_state_np[:,node], validation_experiment.Points[node].T_t_use[starting_idx : starting_idx+steps+1])

    return normalized_distance, std_step_cost_increase, min_step_cost_increase, max_step_cost_increase

def probabilisticPredictionsWrapper(validation_experiment, likelihoods, gp_models, GP_scaler , starting_idx = 300, steps = 4000, samples = 100, messages = False, mean_model = False, device = "cpu", scaler = None, delay_model = None, process_validation_experiment = True):

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
    boundaries_, corners_ = hlp.pointsOnBoundaries_(validation_experiment)
    delta_0 = np.asarray(boundaries_)
    delta_1 = np.asarray(corners_)

    # draw different weights and extrapolate with them
    extrapolations, all_elapsed = propabilisticExtrapolation( validation_experiment, points_used_for_validation, delay_model, likelihoods, gp_models, GP_scaler, mean_model, samples, device, messages, starting_idx, steps, delta_0, delta_1)
    
    return extrapolations,validation_experiment,all_elapsed

def propabilisticExtrapolation( validation_experiment, points_used_for_validation, delay_model, likelihoods, gp_models, GP_scaler, mean_model, samples, device, messages, starting_idx, steps, delta_0, delta_1):
    # draw different weights and extrapolate with them
    extrapolations = []
    ## prepare extrapolation
    coefficients_samp, input_coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = prepareExtrapolation(validation_experiment = validation_experiment, 
    points_used_for_validation = points_used_for_validation, 
    delay_model = delay_model,
    likelihoods = likelihoods, 
    gp_models = gp_models, 
    GP_scaler = GP_scaler,
    mean_model = mean_model, 
    samples=samples, 
    device = device)

    all_elaspsed = []
    if mean_model:
        samples = 1
        coefficients_samp, input_coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = [coefficients_samp], [input_coefficients_samp], [F_sequences_samp], [all_excitation_heights_samp], [T_ambient_samp]
    
    for i,(coefficients, input_coefficients, F_sequences, all_excitation_heights, T_ambient) in enumerate(zip(coefficients_samp, input_coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp)):
        if messages:
            print(f"Sampe {i+1}/{samples}")

        ## initialize roll out
        T_state = []
        T_dot_last = []
        T_state.append( np.hstack([ p.T_t_use[starting_idx] for p in validation_experiment.Points ]) )

        tmp = []
        for p in points_used_for_validation:
            tmp.append(p.T_t_use[starting_idx])
            if starting_idx>0:
                T_dot_last.append( p.T_t_use[starting_idx] - p.T_t_use[starting_idx-1] )
            else:
                T_dot_last.append( np.asarray([0]) )

        T_state.append( np.hstack(tmp) )

        ## quick state propagation function
        neighbor_list = [p.neighbor_nodes for p in validation_experiment.Points if p._hasNeighbors_()]
        statePropagation = quickStatePropagationWithActivation(neighbor_list, delta_0, delta_1)

        T_state_0 = T_state[-1]
        
        t = time.time()
        
        T_state_np, _ = extrapolate(
        statePropagation = statePropagation,
        coefficients = coefficients,
        input_coefficients = input_coefficients,
        all_excitation_heights = all_excitation_heights,
        Fsequence = F_sequences,
        T_ambient = T_ambient,
        T_state_0 = T_state_0,
        starting_idx = starting_idx,
        num_of_steps = steps,
        )
        
        elapsed = time.time() - t
        all_elaspsed.append(elapsed)
        extrapolations.append(T_state_np)

    return extrapolations,all_elaspsed

def assesTrajectory(layer,validation_experiment,likelihoods,models,window_size, GP_scaler, delay_model, RESULTS_FOLDER = "results/general_dbging", device = None, debug = False, lengthscaleModel= None, start_from = 1400, num_of_steps = 1000):
    """
    validate with response in bewtween layers
    """

    ## extract inputs
    points_used_for_validation = [ p for p in validation_experiment.Points if p._hasNeighbors_()]
    boundaries_, corners_ = hlp.pointsOnBoundaries_(validation_experiment)
    delta_0 = np.asarray(boundaries_)
    delta_1 = np.asarray(corners_)

    ## for prediction with neighbors' mean temps
    neighbor_list = [p.neighbor_nodes for p in validation_experiment.Points if p._hasNeighbors_()]

    ## prepare extrapolation
    coefficients, input_coefficients, F_sequences, all_excitation_heights, T_ambient = prepareExtrapolation(validation_experiment, points_used_for_validation, delay_model, likelihoods, models, GP_scaler, device = device)

    ## initialize roll out
    starting_idx = start_from

    T_state = []
    T_dot_last = []
    T_state.append( np.hstack([ p.T_t_use[starting_idx] for p in validation_experiment.Points ]) )

    tmp = []
    for p in points_used_for_validation:
        tmp.append(p.T_t_use[starting_idx])
        if starting_idx>0:
            T_dot_last.append( p.T_t_use[starting_idx] - p.T_t_use[starting_idx-1] )
        else:
            T_dot_last.append( np.asarray([0]) )

    T_state.append( np.hstack(tmp) )

    ## test propagation function
    neighbor_list = [p.neighbor_nodes for p in validation_experiment.Points if p._hasNeighbors_()]
    statePropagation = quickStatePropagationWithActivation(neighbor_list, delta_0, delta_1)
    T_last = T_state[-1]

    ## extrapolate 
    T_state_0 = T_last
    t = time.time()
    T_state_np, DEBUG_inputs = extrapolate(
        statePropagation = statePropagation,
        coefficients = coefficients,
        input_coefficients = input_coefficients,
        all_excitation_heights = all_excitation_heights,
        Fsequence = F_sequences,
        T_ambient = T_ambient,
        T_state_0 = T_state_0,
        starting_idx = starting_idx,
        num_of_steps = num_of_steps,
    )
    elapsed = time.time() - t

    ## assess
    mean_T, std_T, mean_delta_T, std_delta_T = calculateStdsOnCentralPlate(neighbor_list,T_state_np)

    laplacians = calculateLaplacians(neighbor_list, T_state_np)
    mean_laplacians, std_laplacians, mean_delta_laplacians, std_delta_laplacians = calculateStdsOnCentralPlate(neighbor_list, laplacians)

    return mean_T, std_T, mean_delta_T, std_delta_T, mean_laplacians, std_laplacians, mean_delta_laplacians, std_delta_laplacians, T_state_np,elapsed, DEBUG_inputs 

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

def calculateLaplacians(neighbors,temperature_array):
    """
    Calculate Laplacians, nodes on boundaries consider a delta T = T.
    @params neighbors: list of lists with neighbors. Each sublist contains the neighbors of each node
    @params temperature array : n x num_of_points_on_grid
    """

    temperature_array_shape = temperature_array.shape
    # build matrix with all temperatures associated with the laplacian at a particular point.
    associated_temperatures = np.zeros((temperature_array_shape[0],4))
    laplacians = np.zeros_like(temperature_array)

    for i,neighbor_idxs in enumerate(neighbors) :
        associated_temperatures[:,:len(neighbor_idxs)] = temperature_array[:,neighbor_idxs]
        laplacians[:,i] = np.sum(associated_temperatures, axis = -1) - len(neighbor_idxs)*temperature_array[:,i]
        associated_temperatures *= 0

    return laplacians

def calculateErrorPerLayer(array,nominal_array,points_on_grid, nominal_array_min_value_for_rel_error = 0.05):
    """
    Check the error between an array and its nominal counterpart at the end of each layer.
    @params array : field value with dimensions n x num_of_points_on_grid
    @params nominal_array : field value with dimensions n x num_of_points_on_grid
    @params points_on_grid : list of _Point objects on grid

    @returns error : euclidean norm between array and nominal with dimensions n x num_of_points_on_grid
    @returns idxs_to_evaluate : idxs with steps that correspond to the in-between layer torch movement
    """

    # first find the first and the last deposition idx on each layer
    first_deposition_idx_per_layer, last_deposition_idx_per_layer = firstAndLastExcitationIdxsPerLayer(points_on_grid)

    # now, evaluate the given arrays at the midpoint of each interval between the layers
    idxs_to_evaluate = []
    for (first_excitation_idx_next_layer, last_excitation_idx_previous_layer) in zip(first_deposition_idx_per_layer[1:], last_deposition_idx_per_layer[:-1]):
        idxs_to_evaluate.append( int((last_excitation_idx_previous_layer + first_excitation_idx_next_layer)/2 ) )

    idxs_to_evaluate.append(last_deposition_idx_per_layer[-1])

    # idxs_for_relative_errors = np.abs(nominal_array) > nominal_array_min_value_for_rel_error
    # error = np.zeros_like(array)
    # error[idxs_for_relative_errors] = np.abs( (array[idxs_for_relative_errors] - nominal_array[idxs_for_relative_errors])) /nominal_array[idxs_for_relative_errors]
    error = array - nominal_array
    return error, np.asarray(idxs_to_evaluate)

def synchronizeWithDtw(signal1,signal2):
    # distance, path = fastdtw( signal1, signal2, dist = euclidean)

    # idxs_for_signal2 = [p[1] for p in path]
    # sync_signal2 = signal2[idxs_for_signal2]
    # idxs_for_signal1 = [p[0] for p in path]
    # sync_signal1 = signal1[idxs_for_signal1]

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

def firstAndLastExcitationIdxsPerLayer(points_on_grid):
    """
    Find the first and the last deposition idxs where material is deposited on a layer.
    @params points_on_grid : list of _Point objects on grid
    @return first_deposition_idx_per_layer : list with first idx that material is deposited on layer
    @return last_deposition_idx_per_layer : list with last idx that material is deposited on layer
    """

    # first define all excitation levels (same like layers) and excited points
    excitation_levels = [p.excitation_delay_torch_height for p in points_on_grid]
    unique_torch_heights = np.unique(np.hstack(excitation_levels))
    excited_points = [p for p in points_on_grid if len(p.excitation_idxs)>0]

    # now find the the first and the last node excited by the torch on each excitation level
    first_deposition_idx_per_layer = []
    last_deposition_idx_per_layer = []
    for torch_height in unique_torch_heights:
        min_excitation_idx = 1e10
        max_excitation_idx = -1
        for p in excited_points:
            excitations_on_layer = np.where(p.excitation_delay_torch_height == torch_height)
            excitations_on_layer = [excitation for excitation in excitations_on_layer[0] if excitation<len(p.excitation_idxs)]
            idxs_of_torch_crossings = p.excitation_idxs[excitations_on_layer]

            if len(idxs_of_torch_crossings)>0:
                first_excitation_of_node = np.min(idxs_of_torch_crossings)
                last_excitation_of_node = np.max(idxs_of_torch_crossings)

                if min_excitation_idx > first_excitation_of_node:
                    min_excitation_idx = first_excitation_of_node

                if max_excitation_idx < last_excitation_of_node:
                    max_excitation_idx = last_excitation_of_node

        first_deposition_idx_per_layer.append(min_excitation_idx)
        last_deposition_idx_per_layer.append(max_excitation_idx)

    return first_deposition_idx_per_layer, last_deposition_idx_per_layer

def unscale(responses,scaler):
    responses_unscale = responses.T
    unscaled_responses_data = np.zeros_like(responses_unscale)
    for i,response_unscale in enumerate(responses_unscale):
        to_scaler = np.vstack([response_unscale,response_unscale*0]).T
        unscaled_responses_data[i,:] = scaler.inverse_transform(to_scaler).T[0,:]

    return unscaled_responses_data.T

    
def probabilisticPredictionsWrapper_components(validation_experiment, likelihoods, gp_models, GP_scaler , starting_idx = 300, steps = 4000, samples = 100, messages = False, mean_model = False, device = "cpu", scaler = None, delay_model = None, process_validation_experiment = True):

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
    boundaries_, corners_ = hlp.pointsOnBoundaries_(validation_experiment)
    delta_0 = np.asarray(boundaries_)
    delta_1 = np.asarray(corners_)

    # draw different weights and extrapolate with them
    extrapolations, all_elapsed,M_val, G_val, F_val = propabilisticExtrapolation_components( validation_experiment, points_used_for_validation, delay_model, likelihoods, gp_models, GP_scaler, mean_model, samples, device, messages, starting_idx, steps, delta_0, delta_1)
    
    return extrapolations,validation_experiment,all_elapsed,M_val, G_val, F_val


def propabilisticExtrapolation_components( validation_experiment, points_used_for_validation, delay_model, likelihoods, gp_models, GP_scaler, mean_model, samples, device, messages, starting_idx, steps, delta_0, delta_1):
    # draw different weights and extrapolate with them
    extrapolations = []
    ## prepare extrapolation
    coefficients_samp, input_coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = prepareExtrapolation(validation_experiment = validation_experiment, 
    points_used_for_validation = points_used_for_validation, 
    delay_model = delay_model,
    likelihoods = likelihoods, 
    gp_models = gp_models, 
    GP_scaler = GP_scaler,
    mean_model = mean_model, 
    samples=samples, 
    device = device)

    all_elaspsed = []
    if mean_model:
        samples = 1
        coefficients_samp, input_coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = [coefficients_samp], [input_coefficients_samp], [F_sequences_samp], [all_excitation_heights_samp], [T_ambient_samp]
    
    for i,(coefficients, input_coefficients, F_sequences, all_excitation_heights, T_ambient) in enumerate(zip(coefficients_samp, input_coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp)):
        if messages:
            print(f"Sampe {i+1}/{samples}")

        ## initialize roll out
        T_state = []
        T_dot_last = []
        T_state.append( np.hstack([ p.T_t_use[starting_idx] for p in validation_experiment.Points ]) )

        tmp = []
        for p in points_used_for_validation:
            tmp.append(p.T_t_use[starting_idx])
            if starting_idx>0:
                T_dot_last.append( p.T_t_use[starting_idx] - p.T_t_use[starting_idx-1] )
            else:
                T_dot_last.append( np.asarray([0]) )

        T_state.append( np.hstack(tmp) )

        ## quick state propagation function
        neighbor_list = [p.neighbor_nodes for p in validation_experiment.Points if p._hasNeighbors_()]
        statePropagation = quickStatePropagationWithActivation_components(neighbor_list, delta_0, delta_1)

        T_state_0 = T_state[-1]
        
        t = time.time()
        
        T_state_np, Inputs, M_val, G_val, F_val = extrapolate_components(
        statePropagation = statePropagation,
        coefficients = coefficients,
        input_coefficients = input_coefficients,
        all_excitation_heights = all_excitation_heights,
        Fsequence = F_sequences,
        T_ambient = T_ambient,
        T_state_0 = T_state_0,
        starting_idx = starting_idx,
        num_of_steps = steps,
        )
        
        elapsed = time.time() - t
        all_elaspsed.append(elapsed)
        extrapolations.append(T_state_np)

    return extrapolations,all_elaspsed,M_val, G_val, F_val


def extrapolate_components(statePropagation,coefficients,input_coefficients,all_excitation_heights,Fsequence,T_ambient,T_state_0,starting_idx,num_of_steps):
    # propagate state
    steps = num_of_steps
    T_state = [T_state_0]

    DEBUG_COEFS = []
    DEBUG_resps = []
    DEBUG_inputs = []
    M,G,Fs = [], [], []

    for step in range(1,steps+1,1):

        ## pick pre-computed values
        F = Fsequence[:,starting_idx + step]
        excitation_heights = all_excitation_heights[:,starting_idx + step]
        feature_coef = coefficients[:,starting_idx + step,:]
        ip_feature_coef = input_coefficients[:,starting_idx + step,:]
        T_last = T_state[-1]
        T_amb = T_ambient[starting_idx + step]
        input_feature = 0

        ## propagate state
        new_state,  M_val, G_val, F_val = statePropagation(
            T = T_last,
            T_boundary = T_amb,
            h = excitation_heights, 
            g_coef = ip_feature_coef,
            F = F,
            dyn_vectors = feature_coef)

        DEBUG_inputs.append(input_feature)
        T_state.append(new_state)
        M.append(M_val)
        G.append(G_val)
        Fs.append(F_val)


    DEBUG_COEFS = np.asarray(DEBUG_COEFS)

    DEBUG_resps = np.asarray(DEBUG_resps)

    DEBUG_inputs = np.asarray(DEBUG_inputs)
    
    M = np.asarray(M)
    G = np.asarray(G)
    Fs = np.asarray(Fs)


    return np.vstack(T_state),DEBUG_inputs,M,G,Fs


class quickStatePropagationWithActivation_components:
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
        M_vals = np.sum(self.features[:,:3]*dyn_vectors[:3,:].T/self.featureScaler,axis = -1)
        self.features[:,4] =  G_vals

        DT = np.sum(self.features*dyn_vectors.T/self.featureScaler,axis = -1)
        return T+DT ,M_vals, G_vals, F

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
