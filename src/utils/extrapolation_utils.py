
from statistics import LinearRegression
import numpy as np
from sklearn.preprocessing import MinMaxScaler as Scaler
from data_processing.preprocessor import preProcessor
from utils.online_optimization_utils import fTerm, gTerm, mTerm
import utils.online_optimization_utils as onopt
import utils.visualizations as vis
import multiprocessing_on_dill as mp
from data_processing._Experiment import _Experiment
from copy import deepcopy as copy
from utils.generic_utils import setDevice
from scipy.interpolate import interp1d,interp2d 

import torch
import gpytorch
import time 

from dtw import *


def prepareExtrapolation(validation_experiment : _Experiment, points_used_for_validation : list, delay_model : LinearRegression, likelihoods : gpytorch.likelihoods.LikelihoodList, 
    gp_models : gpytorch.models.IndependentModelList, GP_scaler : torch.Tensor, f : fTerm, device = None, mean_model_ = True, samples = 100, pool: mp.Pool or None = None, processes : int = None):
    """
    Prepare the extrapolation
    @param validation_experiment: the experiment to extrapolate
    @param points_used_for_validation: the points to use for extrapolation
    @param delay_model: the delay model to use
    @param likelihoods: the likelihoods
    @param gp_models: the gp models
    @param GP_scaler: the scaler for the GP models
    @param f: the f term
    @param device: the device to use (cuda or cpu)
    @param mean_model_: whether to only calculate the mean model
    @param samples: the number of samples to needed approximate the distribution
    @param pool: the pool to use
    @param processes: the number of processes to use
    @return: the extrapolation
    @return: the extrapolation times
    """
    device = setDevice(device)
    n = len(likelihoods.likelihoods)

    all_states = [p.excitation_delay_torch_height_trajectory for p in points_used_for_validation]
    unique_states =  np.unique(all_states)

    # precalculate the weights for the different states 
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        unique_states = torch.tensor(unique_states).float().to(device=device)
        GP_scaler = GP_scaler.to(device)
        likelihoods = likelihoods.to(device=device)
        gp_models = gp_models.to(device=device)
        infered_coefs = likelihoods( *gp_models(*[unique_states for i in range(n)] ) )
        means_np = np.vstack([((model.mean)*scale).squeeze().cpu().detach().numpy() for (model,scale) in zip(infered_coefs,GP_scaler)])
        if mean_model_:
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
        coefficients, F_sequences, all_excitation_heights, T_ambient = precalculateExtrapolationValues(unique_states,models_numbering,infered_coefs_np.T,delay_model,validation_experiment,f)

    else:
        #TODO: parallelize this block

        unique_states_list = [unique_states for i in range(samples)]
        models_numbering_list = [models_numbering for i in range(samples)]
        delay_model_list = [delay_model for i in range(samples)]
        validation_experiment_list = [validation_experiment for i in range(samples)]
        f_list = [f for i in range(samples)]
        if pool is None:
            if processes is None or processes <1:
                processes = mp.cpu_count() -1

            with mp.Pool(processes=processes) as pool:
                out = pool.starmap(precalculateExtrapolationValues, zip(unique_states_list,models_numbering_list,infered_coefs_np,
                                        delay_model_list,validation_experiment_list,f_list))
                
        else:
            out = pool.starmap(precalculateExtrapolationValues, zip(unique_states_list,models_numbering_list,infered_coefs_np,
                                                                delay_model_list,validation_experiment_list,f_list))

        # unpacking the output
        coefficients_rep = []
        F_sequences_rep = []
        all_excitation_heights_rep = []
        T_ambient_rep = []
        for val in out:
     
            coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = val
            coefficients_rep.append(coefficients_samp)
            F_sequences_rep.append(F_sequences_samp)
            all_excitation_heights_rep.append(all_excitation_heights_samp)
            T_ambient_rep.append(T_ambient_samp)
        
        coefficients = np.stack(coefficients_rep,axis=0)
        F_sequences = np.stack(F_sequences_rep,axis=0)
        all_excitation_heights = np.stack(all_excitation_heights_rep,axis=0)
        T_ambient = np.stack(T_ambient_rep,axis=0)

    return coefficients, F_sequences, all_excitation_heights, T_ambient

def precalculateExtrapolationValues(unique_states,models_numbering,coefficient_samples,delay_model,validation_experiment,f):
    weights_interpolator = interplWeights(unique_states,models_numbering,coefficient_samples)
    coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = calculatePathValues(delay_model, weights_interpolator, models_numbering, validation_experiment, f)        

    return coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp

def interplWeights(unique_states,model_number,weight_values):
    return interp2d(unique_states,model_number,weight_values.T)

def calculatePathValues( delay_model : interp1d, weights_interpolator : interp2d, models_arange , validation_experiment : _Experiment, f : onopt.fTerm):
    """
    Precalculate the values of the extrapolation path for the given experiment and the given delay model
    @param delay_model: the delay model to use
    @param weights_interpolator: the interpolator for the weights
    @param models_arange: the models numbering
    @param validation_experiment: the experiment to use
    @param f: the f term to use
    @return: the coefficients for the extrapolation path, the F sequences, the excitation heights and the ambient temperatures
    """
    Fsequence_repository = []
    excited_nodes_ = []
    coefficient_repository = []
    excitation_heights_repository = []
    all_lengthscales = []
    all_peak_idxs = []
    array_length = 0
    points_used_for_validation = validation_experiment.Points
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

def probabilisticPredictionsWrapper(validation_experiment : _Experiment, m : mTerm, g : gTerm, f : fTerm, likelihoods : gpytorch.likelihoods.LikelihoodList, gp_models : gpytorch.models.IndependentModelList, GP_scaler : torch.Tensor, 
starting_idx :int = 300, steps :int = 4000, samples :int = 100, messages_ : bool = False, mean_model_ : bool = False, device :str = "cpu", scaler : Scaler or None = None, delay_model : interp1d or None = None, process_validation_experiment_  : bool = True, dt : float = 0.5, number_of_concurrent_processes : int or None= None):
    """
    Replicate the thermal fields of a validation experiment using the GP models and the likelihoods
    @param validation_experiment: the experiment to replicate
    @param m: the m term
    @param g: the g term
    @param f: the f term
    @param likelihoods: the likelihoods
    @param gp_models: the gp models
    @param GP_scaler: the scaler for the GP models
    @param starting_idx: the starting index of the validation experiment
    @param steps: the number of steps to replicate
    @param samples: the number of samples to needed approximate the distribution
    @param messages_: whether to print messages
    @param mean_model_: whether to only calculate the mean model
    @param device: the device to use (cuda or cpu)
    @param scaler: the scaler to use
    @param delay_model: the delay model to use
    @param process_validation_experiment_: whether to process the validation experiment
    @param dt: the timestep to use
    @param number_of_concurrent_processes: the number of concurrent processes to use
    @return: the replicated thermal fields
    @return: the processed validation experiment
    @return: extrapolation times
    """

    # pre-process experiment: interpolate torch path, scale measurements, subsample, and find times that torch crosses each node
    if process_validation_experiment_:
        _ = validation_experiment.trajectorySynchronization(J = 1,  lengthscale = 10, lengthscale_sharp = 10 )
        _ = validation_experiment.torchCrossingNodes()

        _ = validation_experiment.scaleMeasurements(copy(scaler))
        validation_experiment.subsampleTrajectories() 


    if steps + starting_idx > len(validation_experiment.Points[0].T_t_use):
        steps = len(validation_experiment.Points[0].T_t_use) - starting_idx - 1    

    # extract inputs
    points_used_for_validation = [ p for p in validation_experiment.Points if p._hasNeighbors_()]

    # draw different weights and extrapolate with them
    extrapolations, all_elapsed = propabilisticExtrapolation( validation_experiment, points_used_for_validation, delay_model, m, g, f, likelihoods, gp_models, 
                        GP_scaler, mean_model_, samples, device, messages_, starting_idx, steps, dt = dt, number_of_concurrent_processes = number_of_concurrent_processes )
    
    return extrapolations,validation_experiment,all_elapsed

def propabilisticExtrapolation( validation_experiment : _Experiment, points_used_for_validation : list, delay_model : LinearRegression, m : mTerm, g : gTerm, f : fTerm, 
    likelihoods : gpytorch.likelihoods.LikelihoodList, gp_models : gpytorch.models.IndependentModelList, GP_scaler : torch.Tensor, mean_model_ : bool, samples : int, 
    device = None , messages_ = False, starting_idx = 300, steps = 4000, dt = 0.5, number_of_concurrent_processes : int or None = None, pool : mp.Pool or None = None):
    """
    Propabilistic extrapolation of the validation experiment
    @param validation_experiment: the experiment to extrapolate
    @param points_used_for_validation: the points used for validation
    @param delay_model: the delay model to use
    @param m: the m term
    @param g: the g term
    @param f: the f term
    @param likelihoods: the likelihoods
    @param gp_models: the gp models
    @param GP_scaler: the scaler for the GP models
    @param mean_model_: whether to only calculate the mean model
    @param samples: the number of samples to needed approximate the distribution
    @param device: the device to use (cuda or cpu)
    @param messages_: whether to print messages
    @param starting_idx: the starting index of the validation experiment
    @param steps: the number of steps to replicate
    @param dt: the timestep to use
    @param number_of_concurrent_processes: the number of concurrent processes to use
    @param pool: the pool to use
    @return: the replicated thermal fields
    @return: the elapsed time
    """
    
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
    mean_model_ = mean_model_, 
    samples=samples, 
    device = device,
    processes=number_of_concurrent_processes,
    pool = pool)

    all_elaspsed = []
    if mean_model_:
        samples = 1
        coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp = [coefficients_samp], [F_sequences_samp], [all_excitation_heights_samp], [T_ambient_samp]
    
    t = time.time()

    ins = []
    for i,(coefficients, F_sequences, all_excitation_heights, T_ambient) in enumerate(zip(coefficients_samp, F_sequences_samp, all_excitation_heights_samp, T_ambient_samp)):
        ins.append((i, samples, validation_experiment, coefficients, F_sequences, all_excitation_heights, m, g, starting_idx, steps, dt, messages_))

    if mean_model_ or number_of_concurrent_processes == 1:
        extrapolations = [unrollPathsWrapper(ins[0])]
    else:
        if pool is None:
            if number_of_concurrent_processes is None:
                number_of_concurrent_processes = np.min([len(ins), mp.cpu_count()-1])
            with mp.Pool(number_of_concurrent_processes) as pool:
                extrapolations = pool.map(unrollPathsWrapper,ins)    
        else:
            extrapolations = pool.map(unrollPathsWrapper,ins)

    elapsed = time.time() - t
    all_elaspsed.append(elapsed/len(coefficients_samp))

    return extrapolations,all_elaspsed

def unrollPathsWrapper(args):
    try:
        out = unrollPaths(*args)
    except Exception as e:
        print(e)
        out = np.NaN
    return out

def unrollPaths( i, samples, validation_experiment, coefficients, F_sequences, all_excitation_heights, m, g, starting_idx = 300, steps = 4000, dt = 0.5, messages_ = False):
        if messages_:
            print(f"Sampe {i+1}/{samples}")

        ## initialize roll out
        T_state = []
        T_dot_last = []
        T_state.append( np.hstack([ p.T_t_use[starting_idx] for p in validation_experiment.Points ]) )

        # total model propagation
        theta_M = coefficients[:3,:]
        theta_P = coefficients[3,:].T
        theta_G = coefficients[4:7,:]
        model = onopt.totalModel(F_sequences, all_excitation_heights, m = m, g = g, theta_g = theta_G, theta_m = theta_M, theta_p = theta_P)

        timesteps = np.arange(starting_idx,starting_idx+steps)*dt
        for t in timesteps:
            T_state.append( model(t,T_state[-1]))

        T_state_np = np.asarray(T_state)
        
        return T_state_np

def synchronizeWithDtw(signal1 : np.ndarray, signal2 : np.ndarray):
    """
    Synchronizes two signals using dynamic time warping
    @param signal1: the first signal
    @param signal2: the second signal
    @return sync_signal1: signal 1 synchronized (and distorted to fit signal 2)
    @return sync_signal2: signal 2 synchronized (and distorted to fit signal 1)
    @return distances between the synchronized signals
    """

    alignment = dtw( signal1, signal2, keep_internals=True)
    
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

def unscale(responses : np.ndarray, scaler : Scaler) -> np.ndarray:
    """
    Unscale the responses
    @param responses: the responses to unscale
    @param scaler: the scaler to use
    @return: the unscaled responses
    """
    responses_unscale = responses.T
    unscaled_responses_data = np.zeros_like(responses_unscale)
    for i,response_unscale in enumerate(responses_unscale):
        to_scaler = np.vstack([response_unscale,response_unscale*0]).T
        unscaled_responses_data[i,:] = scaler.inverse_transform(to_scaler).T[0,:]

    return unscaled_responses_data.T

def eval( m : mTerm, g : gTerm, f : fTerm, file_to_evaluate : str, validation_experiment : _Experiment, 
    likelihoods : gpytorch.likelihoods.LikelihoodList, models : gpytorch.models.IndependentModelList, GP_weights_normalizers : torch.Tensor, 
    prc : preProcessor, delay_model : LinearRegression , save_plots_ = False, RESULTS_FOLDER = "./"  , starting_point = 300, steps = 4000, 
    number_of_concurrent_processes : int or None = None, pool : mp.Pool or None = None) -> float:
    """
    Evaluates the model on a given experiment.
    @param m : m term of the model
    @param g : g term of the model
    @param f : f term of the model
    @param file_to_evaluate : experiment id to evaluate
    @param validation_experiment : experiment to evaluate (corresponding to file_to_evaluate)
    @param likelihoods : likelihoods of the model
    @param models : GP models
    @param GP_weights_normalizers : normalizers of the GP weights
    @param prc : preprocessor of the model
    @param delay_model : delay model
    @param save_plots_ : whether to save the plots or not
    @param RESULTS_FOLDER : folder to save the plots
    @param starting_point : starting point of the roll out
    @param steps : number of steps of the roll out
    @param number_of_concurrent_processes : number of concurrent processes to use
    @param pool : pool of concurrent processes to use
    @return :  mean DTW error
    """
    file = file_to_evaluate
    all_extrapolation_times = []

    # sample mean parameters (most probable system) and unroll the model    
    mean_extrapolation,validation_experiment,all_elapsed_times = probabilisticPredictionsWrapper(copy(validation_experiment), m, g, f, likelihoods, models, GP_weights_normalizers , starting_idx = 300, steps = 4000, samples = 1, messages_ = False, mean_model_=True, device = "cpu", scaler = copy(prc.scaler), delay_model = delay_model)
    mean_extrapolation = mean_extrapolation[0]

    if save_plots_:
        # if needed sample more system extrapolations and unroll them to calculate their statistics
        sampled_extrapolations,*_ = probabilisticPredictionsWrapper(copy(validation_experiment), m, g, f, likelihoods, models, GP_weights_normalizers , starting_idx = 300, steps = 4000, samples = 100, messages_ = False, scaler = copy(prc.scaler), delay_model = delay_model, process_validation_experiment_ = False , number_of_concurrent_processes=number_of_concurrent_processes)

        stacked_sampled_extrapolations = np.stack(sampled_extrapolations,axis=0)
        std_sampled_extrapolations = np.std(stacked_sampled_extrapolations,axis = 0)

        ub = mean_extrapolation + 3*std_sampled_extrapolations
        lb = mean_extrapolation - 3*std_sampled_extrapolations

    T_state_nominal = np.asarray( [p.T_t_use[starting_point:starting_point+steps+1] for p in validation_experiment.Points if p._hasNeighbors_()] ).T

    # undo scaling
    unscaled_responses = unscale(mean_extrapolation,prc.scaler)
    unscaled_targets = unscale(T_state_nominal,prc.scaler)
    if save_plots_:
        ub_unscaled = unscale(ub,prc.scaler)
        lb_unscaled = unscale(lb,prc.scaler)
    timestamps = np.linspace(starting_point, starting_point + len(unscaled_targets),len(unscaled_targets))*0.5

    if len(unscaled_targets)<len(unscaled_responses):
        unscaled_responses = unscaled_responses[:len(unscaled_targets),:]
        ub_unscaled = ub_unscaled[:len(unscaled_targets),:]
        lb_unscaled = lb_unscaled[:len(unscaled_targets),:]
        timestamps = timestamps[:len(unscaled_targets)]

    # select points to calculate the metrics (exclude points in boundaries because of corrupted measurements in many datasets)
    central_plate_points = [p for p in validation_experiment.Points if len(p.neighbor_nodes)==4]
    T_idxs_to_keep = [True if len(p.neighbor_nodes)==4 else False for p in validation_experiment.Points]
    T_idxs_to_keep = np.asarray(T_idxs_to_keep)

    ins = []
    all_mean_T_distances = np.zeros((len(central_plate_points),))
    nodes_to_plot = [p.node for p in central_plate_points]

    for node in nodes_to_plot:
        # if you don't plot them one by one prepare the inputs for doing that in parallel
        ins.append( (copy(validation_experiment), mean_extrapolation, starting_point, steps, node) )

    results = []
    if number_of_concurrent_processes is None:
        number_of_concurrent_processes = np.min( [len(nodes_to_plot), int(mp.cpu_count()*0.8)] )

    # calculate the metrics
    if pool is None:
        with mp.Pool( processes = number_of_concurrent_processes ) as pool:
            results = pool.map( synchronizeWithDtwwrapper, ins )
    else:
        results = pool.map( synchronizeWithDtwwrapper, ins )

    # These distances are the DTW distances of each roll out. So they are already time-averaged
    for i,(in_i,result) in enumerate(zip(ins,results)):
        (*_, node)  = in_i
        distance_i, distance_std, distance_min, distance_max = result
        all_mean_T_distances[i] = distance_i
    ins = []

    mean_dtw_rel_error = np.mean(all_mean_T_distances) # spatial average of mean Distance in timeseries of nodes

    # write performance
    time_elapsed = np.mean(all_elapsed_times)
    all_extrapolation_times.append(time_elapsed)

    if save_plots_:
        if file == "T25" or file == "T17":
        # if 0:
            vis.plotNodeUncertainEvolution( unscaled_responses[:,nodes_to_plot], lb_unscaled[:,nodes_to_plot], ub_unscaled[:,nodes_to_plot], timestamps, unscaled_targets[:,nodes_to_plot], RESULTS_FOLDER ,file )
            # vis.plotNodeEvolution( unscaled_responses, timestamps, unscaled_targets, RESULTS_FOLDER ,file )

            step_to_plot = [1825]
            point_locations = np.vstack([p.coordinates for p in central_plate_points]).T
            max_T = max( np.max( unscaled_responses[step_to_plot,:] ), np.max( unscaled_targets[step_to_plot,:] ) )
            min_T = min( np.min( unscaled_responses[step_to_plot,:] ), np.min( unscaled_targets[step_to_plot,:] ) )
            colorbar_scaling = np.array([max_T, min_T])

            vis.plotContour2(unscaled_responses[:,T_idxs_to_keep],point_locations,np.array(step_to_plot),d_grid = 27, result_folder = RESULTS_FOLDER, field_value_name_id = "CoarseTemperature_"+ file, colorbar_scaling = colorbar_scaling)
            vis.plotContour2(unscaled_targets[:,T_idxs_to_keep],point_locations,np.array(step_to_plot),d_grid = 27, result_folder = RESULTS_FOLDER, field_value_name_id = "CoarseNominalTemperature_"+ file, colorbar_scaling = colorbar_scaling)

    return mean_dtw_rel_error

def evalNoDTW( m : mTerm, g : gTerm, f : fTerm, file_to_evaluate : str, validation_experiment : _Experiment, 
    likelihoods : gpytorch.likelihoods.LikelihoodList, models : gpytorch.models.IndependentModelList, GP_weights_normalizers : torch.Tensor, 
    prc : preProcessor, delay_model : LinearRegression , save_plots_ = False, RESULTS_FOLDER = "./"  , starting_point = 300, steps = 4000, 
    number_of_concurrent_processes : int or None = None, pool : mp.Pool or None = None):
    """
    Evaluates the model on a given experiment.
    @param m : m term of the model
    @param g : g term of the model
    @param f : f term of the model
    @param file_to_evaluate : experiment id to evaluate
    @param validation_experiment : experiment to evaluate (corresponding to file_to_evaluate)
    @param likelihoods : likelihoods of the model
    @param models : GP models
    @param GP_weights_normalizers : normalizers of the GP weights
    @param prc : preprocessor of the model
    @param delay_model : delay model
    @param save_plots_ : whether to save the plots or not
    @param RESULTS_FOLDER : folder to save the plots
    @param starting_point : starting point of the roll out
    @param steps : number of steps of the roll out
    @param number_of_concurrent_processes : number of concurrent processes to use
    @param pool : pool of concurrent processes to use
    @return : mean asynchronous MAPE (including contribution from the delay model)
    """
    
    file = file_to_evaluate
    all_extrapolation_times = []

    # sample mean parameters (most probable system) and unroll the model    
    mean_extrapolation,validation_experiment,all_elapsed_times = probabilisticPredictionsWrapper(copy(validation_experiment), m, g, f, likelihoods, models, GP_weights_normalizers , starting_idx = 300, steps = 4000, samples = 1, messages_ = False, mean_model_=True, device = "cpu", scaler = copy(prc.scaler), delay_model = delay_model)
    mean_extrapolation = mean_extrapolation[0]

    if save_plots_:
        # if needed sample more system extrapolations and unroll them to calculate their statistics
        sampled_extrapolations,*_ = probabilisticPredictionsWrapper(copy(validation_experiment), m, g, f, likelihoods, models, GP_weights_normalizers , starting_idx = 300, steps = 4000, samples = 100, messages_ = False, scaler = copy(prc.scaler), delay_model = delay_model, process_validation_experiment_ = False , number_of_concurrent_processes=number_of_concurrent_processes)

        stacked_sampled_extrapolations = np.stack(sampled_extrapolations,axis=0)
        std_sampled_extrapolations = np.std(stacked_sampled_extrapolations,axis = 0)

        ub = mean_extrapolation + 3*std_sampled_extrapolations
        lb = mean_extrapolation - 3*std_sampled_extrapolations

    T_state_nominal = np.asarray( [p.T_t_use[starting_point:starting_point+steps+1] for p in validation_experiment.Points if p._hasNeighbors_()] ).T

    # undo the scaling
    unscaled_responses = unscale(mean_extrapolation,prc.scaler)
    unscaled_targets = unscale(T_state_nominal,prc.scaler)
    if save_plots_:
        ub_unscaled = unscale(ub,prc.scaler)
        lb_unscaled = unscale(lb,prc.scaler)
    timestamps = np.linspace(starting_point, starting_point + len(unscaled_targets),len(unscaled_targets))*0.5

    if len(unscaled_targets)<len(unscaled_responses):
        unscaled_responses = unscaled_responses[:len(unscaled_targets),:]
        ub_unscaled = ub_unscaled[:len(unscaled_targets),:]
        lb_unscaled = lb_unscaled[:len(unscaled_targets),:]
        timestamps = timestamps[:len(unscaled_targets)]

    # select points to calculate the metrics (exclude points in boundaries because of corrupted measurements in many datasets)
    central_plate_points = [p for p in validation_experiment.Points if len(p.neighbor_nodes)==4]
    T_idxs_to_keep = [True if len(p.neighbor_nodes)==4 else False for p in validation_experiment.Points]
    T_idxs_to_keep = np.asarray(T_idxs_to_keep)

    # calculate the metrics
    nodes_to_plot = [p.node for p in central_plate_points]
    responses_eval = unscaled_responses[:,T_idxs_to_keep]
    targets_eval = unscaled_targets[:,T_idxs_to_keep]
    mare = np.mean( np.abs(responses_eval-targets_eval)/(np.abs(targets_eval))+1e-4 ) # spatial average of mean Distance in timeseries of nodes

    # write performance
    time_elapsed = np.mean(all_elapsed_times)
    all_extrapolation_times.append(time_elapsed)

    if save_plots_:
        if file == "T25" or file == "T17":
        # if 0:
            vis.plotNodeUncertainEvolution( unscaled_responses[:,nodes_to_plot], lb_unscaled[:,nodes_to_plot], ub_unscaled[:,nodes_to_plot], timestamps, unscaled_targets[:,nodes_to_plot], RESULTS_FOLDER ,file )
            # vis.plotNodeEvolution( unscaled_responses, timestamps, unscaled_targets, RESULTS_FOLDER ,file )

            step_to_plot = [1825]
            point_locations = np.vstack([p.coordinates for p in central_plate_points]).T
            max_T = max( np.max( unscaled_responses[step_to_plot,:] ), np.max( unscaled_targets[step_to_plot,:] ) )
            min_T = min( np.min( unscaled_responses[step_to_plot,:] ), np.min( unscaled_targets[step_to_plot,:] ) )
            colorbar_scaling = np.array([max_T, min_T])

            vis.plotContour2(unscaled_responses[:,T_idxs_to_keep],point_locations,np.array(step_to_plot),d_grid = 27, result_folder = RESULTS_FOLDER, field_value_name_id = "CoarseTemperature_"+ file, colorbar_scaling = colorbar_scaling)
            vis.plotContour2(unscaled_targets[:,T_idxs_to_keep],point_locations,np.array(step_to_plot),d_grid = 27, result_folder = RESULTS_FOLDER, field_value_name_id = "CoarseNominalTemperature_"+ file, colorbar_scaling = colorbar_scaling)

    return mare

if __name__ == "__main__":
    
    dummy = 1