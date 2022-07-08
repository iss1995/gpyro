# %%
import numpy as np
import multiprocessing_on_dill as mp

import data_processing._config as config
import data_processing.preprocessor as preprocessor
# from pathos.pools import ParallelPool as Pool

import torch, random
import warnings
import optuna

import utils.online_optimization_utils as onopt
import utils.extrapolation_utils as exu

from scipy.signal import butter, filtfilt
from copy import deepcopy as copy

def executeMain(hyperparameters,kwargs,save_plots_ = False):
    try:
        result = evaluate_hyperparamters(hyperparameters,**kwargs)
    except Exception as e:
        result = float('nan')
        print(f"In {hyperparameters}:")
        print(e)

    return result

# %%
def main(N = 100,save_plots_ = False):

    warnings.filterwarnings("ignore")
    # preprocess training data and create scaler
    ###############################################################################################
    # save_plots_ = True
    print("Post-processing...")

    FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER,RESULTS_FOLDER_GP,RESULTS_FOLDER_MODEL,d_grid,output_scale,length_mean,length_var,opt_kwargs = config.config()
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

    
    kwargs = {
        "bounds_f": opt_kwargs["bounds_f"], "bounds_g" : opt_kwargs["bounds_g"], "bounds_m" : opt_kwargs["bounds_m"], "param_f0" : opt_kwargs["param_f0"], "param_g0" : opt_kwargs["param_g0"], 
        "param_m0" : opt_kwargs["param_m0"],"points_used_for_training" : points_used_for_training, "states" : states,"RESULTS_FOLDER" : RESULTS_FOLDER, 
        "neighbor_list" : neighbor_list, "experiment_ids" : experiment_ids,"prc" : prc ,"epochs" : epochs, "HORIZON" : HORIZON, "underestimate_lengthscales" : underestimate_lengthscales,
    }
    obj = objective(kwargs)
    # Let us minimize the objective function above.

    print(f"Running {N} trials...")
    study = optuna.create_study(study_name = "gpyro_hyperparameter_opt_"+id_tag ,direction="minimize")
    print(study.study_name)
    study.optimize(obj, n_trials=N)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f"{study.study_name}.pdf")

    return None

def evaluate_hyperparamters( hyperparameters, bounds_f, bounds_g, bounds_m, param_f0, param_g0, param_m0, points_used_for_training, states, RESULTS_FOLDER, neighbor_list, experiment_ids, prc, seed = 0, epochs = 20, HORIZON = 1, underestimate_lengthscales = 0 ):
    save_plots_ = False
    (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = hyperparameters

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
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
            "param_m0": param_m0,
            "epochs" : epochs,
            "perturbation_magnitude" : 0.0}
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
    states_with_no_excitation = np.where(states < 0)[0][::-1]
    state_with_first_excitation = np.where(states == 0)[0][0]
    # m_repository[states_with_no_excitation[:-1],-1] = m_repository[states_with_no_excitation[-1],-1]
    # g_repository[states_with_no_excitation[:-1],:] = g_repository[states_with_no_excitation[-1],:]
    # f_repository[states_with_no_excitation[:-1],:] = f_repository[states_with_no_excitation[-1],:]
    # m_repository[states_with_no_excitation[:-1],-1] = -m_repository[:states_with_no_excitation[-1],-1]
    # g_repository[states_with_no_excitation[:-1],:] = -g_repository[:states_with_no_excitation[-1],:]
    # f_repository[states_with_no_excitation[:-1],:] = -f_repository[:states_with_no_excitation[-1],:]
    m_repository[states_with_no_excitation,-1] = m_repository[state_with_first_excitation,-1] - \
            (m_repository[state_with_first_excitation+1:,-1] - m_repository[state_with_first_excitation,-1])
    g_repository[states_with_no_excitation,:] = g_repository[state_with_first_excitation,:] -\
            ( g_repository[state_with_first_excitation+1:,:] - g_repository[state_with_first_excitation,:] )
    f_repository[states_with_no_excitation,:] = f_repository[state_with_first_excitation,:] -\
            ( f_repository[state_with_first_excitation+1:,:] - f_repository[state_with_first_excitation,:] )

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
                    "gp_model_to_load":gp_model_to_load,
                    "_device" : "cpu"}

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
    # files_to_evaluate = [f"T26"]
    validation_experiments = [exp for exp in prc.experiments if exp.experiment_id in files_to_evaluate]

    # put files in order
    files_to_evaluate = [exp.experiment_id for exp in validation_experiments]

    number_of_concurrent_processes = mp.cpu_count()-1
    all_mean_dtw_distances = []
    # with mp.Pool(number_of_concurrent_processes) as pool:
    pool = None
    try:
        for (validation_experiment, file_id) in zip(validation_experiments,files_to_evaluate):    
            all_mean_dtw_distances.append( exu.safe_eval(m, g, f, file_id, validation_experiment, likelihoods, models, GP_weights_normalizers, prc, delay_model , save_plots_ , RESULTS_FOLDER  , starting_point , steps, number_of_concurrent_processes, pool ))
    except Exception as e:
        print(e)

    all_mean_dtw_distances = np.asarray(all_mean_dtw_distances)
    failed_idxs = np.isnan(all_mean_dtw_distances)

    T_Mean_error = np.mean(all_mean_dtw_distances[~failed_idxs]) * 100
    print(f"\nDTW mean relative error: {T_Mean_error}%\n(G_reg, F_reg, M_reg, output_scale, length_mean, length_var) {hyperparameters}"+
    "\n______________________________________________________________________________")
    return T_Mean_error 

class objective:
    def __init__(self,kwargs):
        self.kwargs = kwargs

    def __call__(self,trial):
        g_reg = trial.suggest_float("g_reg", -4, -1)
        # g_reg = -5
        f_reg = trial.suggest_float("f_reg", -7, -4)
        # f_reg = -3.4917008457058287
        m_reg = trial.suggest_float("m_reg", -4, -1)
        # m_reg = -2.298677092036098
        output_scale = np.log(0.5)
        length_mean = np.log(0.1)
        length_var = np.log(0.05)
        # output_scale = trial.suggest_float("output_scale", -1, 0)
        # length_mean = trial.suggest_float("length_mean", -2, 0)
        # length_var = trial.suggest_float("length_var", -3, -1)


        hyperparams = (10**g_reg, 10**f_reg, 10**m_reg, 10**output_scale, 10**length_mean, 10**length_var)
        return executeMain(hyperparams, self.kwargs)

if __name__ == "__main__":

    SEED = 0
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    warnings.filterwarnings("ignore")
    N = 100
    _ = main(N=N,save_plots_=False)
# %%
