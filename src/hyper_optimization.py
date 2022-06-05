# %%
import numpy as np
import pandas as pd
import multiprocessing_on_dill as mp

import scipy
import data_processing._config as config
import data_processing.preprocessor as preprocessor
from pathos.pools import ParallelPool as Pool

import torch
import csv
import pickle
import time
import warnings
import optuna
import matplotlib.pyplot as plt

import utils.online_optimization_utils as onopt
import utils.extrapolation_utils as exu
import utils.visualizations as vis

from scipy.signal import butter, filtfilt
from copy import deepcopy as copy
from math import ceil

def executeMain(hyperparameters,kwargs,save_plots_ = False):
    try:
        result = evaluate_hyperparamters(hyperparameters,**kwargs)
    except Exception as e:
        result = float('nan')
        print(f"In {hyperparameters}:")
        print(e)

    return result

# %%
def main(save_plots_ = False):

    warnings.filterwarnings("ignore")
    # preprocess training data and create scaler
    ###############################################################################################
    # save_plots_ = True
    print("Post-processing...")
    FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER,RESULTS_FOLDER_GP,RESULTS_FOLDER_MODEL,d_grid,bounds_f,F_reg,bounds_g,G_reg,bounds_m,M_reg,output_scale,length_mean,length_var,param_f0,param_g0,param_m0 = config.config()
    FILES_FOLDER = "./Gpyro-TD/"
    prc = preprocessor.preProcessor(FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER,subsample = 5)
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
    
    kwargs = {
        "bounds_f": bounds_f, "bounds_g" : bounds_g, "bounds_m" : bounds_m, "param_f0" : param_f0, "param_g0" : param_g0, 
        "param_m0" : param_m0,"points_used_for_training" : points_used_for_training, "states" : states,"RESULTS_FOLDER" : RESULTS_FOLDER, 
        "neighbor_list" : neighbor_list, "experiment_ids" : experiment_ids,"prc" : prc 
    }
    obj = objective(kwargs)
    N = 200
    # Let us minimize the objective function above.

    print(f"Running {N} trials...")
    study = optuna.create_study()
    print(study.study_name)
    study.optimize(obj, n_trials=N)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    fig = plt.figure(figsize=(42,42))
    optuna.visualization.plot_contour(study)
    plt.savefig(f"{study.study_name}.pdf")

    return None

def evaluate_hyperparamters( hyperparameters, bounds_f, bounds_g, bounds_m, param_f0, param_g0, param_m0, points_used_for_training, states, RESULTS_FOLDER, neighbor_list, experiment_ids, prc ):
    save_plots_ = False
    (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = hyperparameters

    g = onopt.gTerm(params = param_g0)
    f = onopt.fTerm(params = param_f0)
    m = onopt.mTerm(neighbors = neighbor_list ,params = param_m0)

    kwargs = {"bounds_f":bounds_f, 
            "bounds_g":bounds_g, 
            "bounds_m":bounds_m, 
            "F_reg":F_reg,
            "G_reg": G_reg,
            "M_reg": M_reg,
            "param_f0": param_f0,
            "param_g0": param_g0,
            "param_m0": param_m0}
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
    validation_experiments = [copy(prc.experiments[experiment_to_use]) for experiment_to_use in experiment_range]
    extrap_in = [(copy(m), copy(g), copy(f), file_id, validation_experiment, copy(likelihoods), copy(models), copy(GP_weights_normalizers), copy(prc), copy(delay_model), save_plots_, RESULTS_FOLDER, starting_point, steps) for (file_id,validation_experiment) in zip(files_to_evaluate,validation_experiments) ]

    processes = np.min( [len(extrap_in), int(mp.cpu_count()*0.8)] )
    # decorator = partial(multiprocess_eval, processes = processes)

    # @decorator
    with mp.Pool( processes ) as pool:
        all_mean_dtw_distances = pool.map_async(exu.safe_eval,extrap_in).get()
        # all_mean_dtw_distances.wait()
        # all_mean_dtw_distances.wait()
        # time.sleep(300)
        pool.join() 

    failed_idxs = np.isnan(all_mean_dtw_distances)
    # all_mean_dtw_distances = eval(extrap_in)
    T_Mean_error = np.mean(all_mean_dtw_distances[~failed_idxs]) * 100
    print(f"\nDTW mean relative error: {T_Mean_error}%")
    return T_Mean_error 

class objective:
    def __init__(self,kwargs):
        self.kwargs = kwargs

    def __call__(self,trial):
        g_reg = trial.suggest_float("g_reg", -6, 0)
        # g_reg = -5
        f_reg = trial.suggest_float("f_reg", -6, 0)
        # f_reg = -3.4917008457058287
        m_reg = trial.suggest_float("m_reg", -6,0)
        # m_reg = -2.298677092036098
        output_scale = trial.suggest_float("output_scale", -1, 1)
        length_mean = trial.suggest_float("length_mean", -3, 0)
        length_var = trial.suggest_float("length_var", -3, 0)

        hyperparams = (10**g_reg, 10**f_reg, 10**m_reg, 10**output_scale, 10**length_mean, 10**length_var)
        return executeMain(hyperparams, self.kwargs)

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    _ = main(True)
# %%
