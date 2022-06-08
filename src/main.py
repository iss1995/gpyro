# %%
import numpy as np
import pandas as pd
import multiprocessing_on_dill as mp

import data_processing._config as config

import torch
import csv
import pickle
import time

import utils.online_optimization_utils as onopt
import utils.extrapolation_utils as exu
import utils.visualizations as vis

from functools import partial
from sklearn.linear_model import LinearRegression
from data_processing.preprocessor import preProcessor
from scipy.signal import butter, filtfilt
from copy import deepcopy as copy
from math import ceil

def executeMain(hyperparameters,save_plots_ = False):
    try:
        result = main(hyperparameters,save_plots_)
    except Exception as e:
        result = float('nan')
        print(f"In {hyperparameters}:")
        print(e)

    return result

def multiprocess_eval( func, processes = 2):
    def wrapper(*args):
        with mp.Pool(processes) as pool:
            results = pool.map(func,args)
        return results
    return wrapper
# %%
def main(save_plots_ = False):
    # preprocess training data and create scaler
    ###############################################################################################
    # save_plots_ = True
    print("Pre-processing...")
    FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER,RESULTS_FOLDER_GP,RESULTS_FOLDER_MODEL,d_grid,bounds_f,F_reg,bounds_g,G_reg,bounds_m,M_reg,output_scale,length_mean,length_var,param_f0,param_g0,param_m0 = config.config()
    prc = preProcessor(FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER,subsample = 5)
    _ = prc.loadData(d_grid = d_grid, deposition_height_of_boundary =  0 )
    id_tag = prc.plotter.time

    experiment_ids = [exp.experiment_id for exp in prc.experiments]
    experiment_to_use = [i for i,id in enumerate(experiment_ids) if id == "T1"][0]
    experiment = prc.experiments[experiment_to_use]
    _ = experiment.trajectorySynchronization(J = 1, lengthscale = 10, lengthscale_sharp = 10 )
    _ = experiment.torchCrossingNodes()

    # %%
    # prepare scaler
    ###############################################################################################
    measurements_for_scaler = []
    points_used_for_training = [p for p in experiment.Points]
    T_t, relevance_coef = [],[]
    b,a = butter(8,0.5)
    for p in points_used_for_training:
        # filter data for artifacts
        T_t.append(filtfilt(b,a,p.T_t_use))
        relevance_coef.append(p.relevance_coef)


    T_t_np = np.hstack( T_t )
    relevance_coef_np = np.hstack( relevance_coef )
    feature_data_rep = [T_t_np, relevance_coef_np]
    feature_data = np.vstack(feature_data_rep).T

    prc.scaler.fit(feature_data)

    _ = experiment.scaleMeasurements(prc.scaler)

    # %%
    # Synchronize training data
    ###############################################################################################
    _ = experiment.formRawInputsWithSynchronousExcitationMultithreadWrapper(  )

    # %%
    # prepare opt
    ###############################################################################################
    points_used_for_training = [p for p in experiment.Points if p._hasNeighbors_()]

    # on_corner_ all false 
    # TODO: remove on_corner_ ?
    on_boundary_ = onopt.pointsOnBoundaries_(experiment)

    layer_idxs, unique_height_values = onopt.splitToLayers(points_used_for_training=points_used_for_training,tool_height_trajectory = experiment.tool_deposition_depth[::experiment.subsample], debug = False)

    # TODO: define the trajectory height of boundary nodes as the opposite trajectory height of their closest excited node  
    points_used_for_training = onopt.statesForBoundaries( points_used_for_training ,on_boundary_ )

    states = np.unique([p.excitation_delay_torch_height_trajectory for p in points_used_for_training])
    HORIZON = 1

    neighbor_list = [p.neighbor_nodes for p in experiment.Points if p._hasNeighbors_()]

    # %%
    # optimization
    ####################################################################################################
    # import importlib
    # importlib.reload(onopt)
    g = onopt.gTerm(params = param_g0)
    f = onopt.fTerm(params = param_f0)
    m = onopt.mTerm(neighbors = neighbor_list ,params = param_m0)

    epochs = 1
    underestimate_lengthscales = 0
    all_training_times_per_state = []

    param_f0 = np.asarray([1.])
    # bounds_f = scipy.optimize.Bounds([0.05],[np.inf])
    bounds_f = ([0.05],[np.inf])

    param_m0 = np.asarray([-.1,-.1,-.1, 1])
    # bounds_m = scipy.optimize.Bounds([-np.inf, -np.inf, -np.inf],[0, 0, 0])
    bounds_m = ([-np.inf, -np.inf, -np.inf, 0],[0, 0, 0, np.inf])

    param_g0 = np.asarray([0.1,0.1,0.1])
    # bounds_g = scipy.optimize.Bounds([-np.inf, -np.inf, -np.inf],[np.inf, np.inf, np.inf])
    bounds_g = ([-np.inf, -np.inf, -np.inf],[np.inf, np.inf, np.inf])
    kwargs = {"bounds_f":bounds_f, 
            "bounds_g":bounds_g, 
            "bounds_m":bounds_m, 
            "F_reg":F_reg,
            "G_reg": G_reg,
            "M_reg": M_reg,
            "param_f0": param_f0,
            "param_g0": param_g0,
            "param_m0": param_m0,
            "epochs" : epochs}
    # f_parameters_per_building_layer_height, g_parameters_per_building_layer_height, m_parameters_per_building_layer_height, all_training_times_per_state = onopt.onlineOptimization(layer_idxs, states, points_used_for_training, **kwargs )
    f_parameters_per_building_layer_height, g_parameters_per_building_layer_height, m_parameters_per_building_layer_height, all_training_times_per_state = onopt.batchOptimization(states, points_used_for_training, m , f, g, **kwargs )

    print(f"Mean Training time per state {np.mean(all_training_times_per_state)}")
    # %%
    # learn a simple delay model
    ###############################################################################################
    ## find the temperatures when the inputs are activated
    excited_points = [ p for p in points_used_for_training if len(p.input_idxs)>0]
    delay_model = onopt.learnDelay(excited_points)

    # %%
    # save results and prepare for the GPs
    ####################################################################################################
    # save m, g, f
    m_repository = np.asarray(m_parameters_per_building_layer_height[-1])
    g_repository = np.asarray(g_parameters_per_building_layer_height[-1])
    f_repository = np.asarray(f_parameters_per_building_layer_height[-1])[:,None]

    # overwirte elements of g and f where no training data were available. Assign values that will improve GP fit.
    # are you sure it helps? maybe 0 better?
    states_with_no_excitation = np.where(states <= 0)[0]
    m_repository[states_with_no_excitation[:-1],-1] = m_repository[states_with_no_excitation[-1],-1]
    g_repository[states_with_no_excitation[:-1],:] = g_repository[states_with_no_excitation[-1],:]
    f_repository[states_with_no_excitation[:-1],:] = f_repository[states_with_no_excitation[-1],:]

    # overwrite input coefs for not activated elemets
    # m_repository[0,-1] = m_repository[-1,-1]

    parameters = np.concatenate((m_repository,g_repository,f_repository),axis = 1).T
    export_name = RESULTS_FOLDER_MODEL + "final_models"
    df = pd.DataFrame(parameters)
    df.to_csv(export_name + '.csv', header = False, index = False)

    #print(f"Models' table dimensions {parameters.shape} ")

    # save delay
    with open( RESULTS_FOLDER_MODEL + "delay_model.sav","wb") as file:
        pickle.dump(delay_model, file)

    # %%
    # TRAIN GP
    ###############################################################################################
    # TODO: check if cuda is available
    # set prefered device for optimization
    models,likelihoods, GP_weights_normalizers, band_nominal_height_tensor, device = onopt.initializeGPModels(parameters,states,device_tp_optimize = 'cpu', output_scale = output_scale, length_mean = length_mean, length_var = length_var)
    #print("GP model optimization starting:")

    # define hyperparams
    gp_model_to_load = '' # if empty, then optimize
    max_training_iter = 4000
    thress = 5e-4
    gp_opt_kwargs = {"learning_rate" : 0.1,
                    "training_iter": max_training_iter,
                    "loss_thress":thress,
                    "no_improvement_thress": 50,
                    "gp_model_to_load":gp_model_to_load}

    # run optimization or load specified model
    models, likelihoods, gp_data = onopt.GPOpt(copy(models),copy(likelihoods),**gp_opt_kwargs)

    # check if you loaded data
    if gp_data is not None:
        T_vector = gp_data["temperatures"]
        parameters = gp_data["weights"]

    # save model if you didn't load it
    if not gp_model_to_load:
        export_name = RESULTS_FOLDER_GP + f"noLags_{id_tag}"
        torch.save(models.state_dict(), export_name + ".pth")
        torch.save(likelihoods.state_dict(),export_name + "_like.pth")

        with open(export_name + '_data.csv',"w+") as data_csv:
            csvWriter = csv.writer(data_csv,delimiter=',')
            csvWriter.writerows([ np.vstack( (band_nominal_height_tensor.T, parameters) ) ])

    # plot GPs
    if save_plots_: 
        vis.plotGPWeights( likelihoods, models, RESULTS_FOLDER, states=states, device = None, xticks = None, yticks = [0,0.5,1.0,1.5], id = "",title = None)
        vis.plotWeightsSubplot(likelihoods, models, RESULTS_FOLDER, states=states, weights_in_subplot = [0,1,2], device = None, xticks = None, yticks = [0,0.5,1.0,1.5], id = "",title = None)

    # %%
    # Validate with online state propagation
    ###############################################################################################
    # import importlib
    # # importlib.reload(onopt)
    # importlib.reload(exu)

    g = onopt.gTerm(params = g_repository)
    f = onopt.fTerm(params = f_repository)
    m = onopt.mTerm(neighbors = neighbor_list ,params = m_repository)
    starting_point, steps  = 300, 4000

    experiment_range = np.arange(2,len(experiment_ids))
    files_to_evaluate = [f"T{i}" for i in experiment_range]
    validation_experiments = [prc.experiments[experiment_to_use] for experiment_to_use in experiment_range]
    
    number_of_concurrent_processes = 5
    all_mean_dtw_distances = []
    for (validation_experiment, file_id) in zip(validation_experiments,files_to_evaluate):    
        all_mean_dtw_distances.append( exu.safe_eval(m, g, f, file_id, validation_experiment, likelihoods, models, GP_weights_normalizers, prc, delay_model , save_plots_ , RESULTS_FOLDER  , starting_point , steps, number_of_concurrent_processes ))

    all_mean_dtw_distances = np.asarray(all_mean_dtw_distances)
    failed_idxs = np.isnan(all_mean_dtw_distances)
    T_Mean_error = np.mean(all_mean_dtw_distances[~failed_idxs]) * 100
    print(f"\nDTW mean relative error: {T_Mean_error}%\tSuccesful validations {len(all_mean_dtw_distances)-np.sum(failed_idxs)}")
    return T_Mean_error
# %%
if __name__ == "__main__":
    _ = main(True)
# %%
