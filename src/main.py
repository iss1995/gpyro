# %%
import numpy as np
import pandas as pd
import multiprocessing as mp
import data_processing._config as config
import data_processing.preprocessor as preprocessor

import torch
import csv
import pickle

import utils.online_optimization_utils as onopt
import utils.extrapolation_utils as exu
import utils.visualizations as vis

from scipy.signal import butter, filtfilt
from copy import deepcopy as copy

def executeMain(hyperparameters,save_plots_ = False):
    try:
        result = main(hyperparameters,save_plots_)
    except Exception as e:
        result = float('nan')
        print(f"In {hyperparameters}:")
        print(e)

    return result

# %%
def main(save_plots_ = False):
    # preprocess training data and create scaler
    ###############################################################################################
    print("Post-processing...")
    FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER,RESULTS_FOLDER_GP,RESULTS_FOLDER_MODEL,d_grid,bounds_f,F_reg,bounds_g,G_reg,bounds_m,M_reg,output_scale,length_mean,length_var,param_f0,param_g0,param_m0 = config.config()
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
    # %%
    # optimization
    ####################################################################################################
    kwargs = {"bounds_f":bounds_f, 
            "bounds_g":bounds_g, 
            "bounds_m":bounds_m, 
            "F_reg":F_reg,
            "G_reg": G_reg,
            "M_reg": M_reg,
            "param_f0": param_f0,
            "param_g0": param_g0,
            "param_m0": param_m0,
            "on_boundary_": on_boundary_,
            "d_grid": d_grid}
    # f_parameters_per_building_layer_height, g_parameters_per_building_layer_height, m_parameters_per_building_layer_height, all_training_times_per_state = onopt.onlineOptimization(layer_idxs, states, points_used_for_training, **kwargs )
    f_parameters_per_building_layer_height, g_parameters_per_building_layer_height, m_parameters_per_building_layer_height, all_training_times_per_state = onopt.batchOptimization(layer_idxs, states, points_used_for_training, **kwargs )

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

    max_input_coef_idx = np.argmax(m_repository[:,-1])
    max_input_coef = m_repository[max_input_coef_idx,-1]

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
    starting_point = 300
    steps = 4000
    files_to_evaluate = [f"T{i}" for i in range(2,28)]
    
    results = []
    all_dtw_distances = []
    all_mean_dtw_distances = []
    all_extrapolation_times = []

    print("Validating...")
    for file in files_to_evaluate:
        print(f"\tCurrent file : {file}")
        target_validation_experiment = file
        experiment_to_use = [i for i,id in enumerate(experiment_ids) if id == target_validation_experiment ][0]
        experiment_to_validate = experiment_to_use
        validation_experiment = copy(prc.experiments[ experiment_to_validate ])
        
        ## for prediction with neighbors' mean temps
        neighbor_list = [p.neighbor_nodes for p in validation_experiment.Points if p._hasNeighbors_()]
        
        mean_extrapolation,validation_experiment,all_elapsed_times = exu.probabilisticPredictionsWrapper(copy(validation_experiment), likelihoods, models, GP_weights_normalizers , starting_idx = 300, steps = 4000, samples = 1, messages = False, mean_model=True, device = "cpu", scaler = copy(prc.scaler), delay_model = delay_model)
        mean_extrapolation = mean_extrapolation[0]

        if save_plots_:
            sampled_extrapolations,*_ = exu.probabilisticPredictionsWrapper(copy(validation_experiment), likelihoods, models, GP_weights_normalizers , starting_idx = 300, steps = 4000, samples = 100, messages = False, scaler = copy(prc.scaler), delay_model = delay_model, process_validation_experiment = False)

            stacked_sampled_extrapolations = np.stack(sampled_extrapolations,axis=0)
            std_sampled_extrapolations = np.std(stacked_sampled_extrapolations,axis = 0)

            ub = mean_extrapolation + 3*std_sampled_extrapolations
            lb = mean_extrapolation - 3*std_sampled_extrapolations

        T_state_nominal = np.asarray( [p.T_t_use[starting_point:starting_point+steps+1] for p in validation_experiment.Points if p._hasNeighbors_()] ).T

        unscaled_responses = exu.unscale(mean_extrapolation,prc.scaler)
        unscaled_targets = exu.unscale(T_state_nominal,prc.scaler)
        if save_plots_:
            ub_unscaled = exu.unscale(ub,prc.scaler)
            lb_unscaled = exu.unscale(lb,prc.scaler)
        timestamps = np.linspace(starting_point, starting_point + len(unscaled_targets),len(unscaled_targets))*0.5

        if len(unscaled_targets)<len(unscaled_responses):
            unscaled_responses = unscaled_responses[:len(unscaled_targets),:]
            ub_unscaled = ub_unscaled[:len(unscaled_targets),:]
            lb_unscaled = lb_unscaled[:len(unscaled_targets),:]
            timestamps = timestamps[:len(unscaled_targets)]


        central_plate_points = [p for p in validation_experiment.Points if len(p.neighbor_nodes)==4]
        T_idxs_to_keep = [True if len(p.neighbor_nodes)==4 else False for p in validation_experiment.Points]
        T_idxs_to_keep = np.asarray(T_idxs_to_keep)

        ins = []
        all_mean_T_distances = np.zeros((len(central_plate_points),))
        all_std_T_distances = np.zeros((len(central_plate_points),))
        all_min_T_distances = np.zeros((len(central_plate_points),))
        all_max_T_distances = np.zeros((len(central_plate_points),))
        nodes_to_plot = [p.node for p in central_plate_points]

        for node in nodes_to_plot:
            # if you don't plot them one by one prepare the inputs for doing that in parallel
            ins.append( (copy(validation_experiment), mean_extrapolation, starting_point, steps, node) )

        # run in parallel the plotting fun
        results = []
        with mp.Pool( processes = np.min( [2 , len(nodes_to_plot) ]) ) as pool:
            # mean_T_distances = pool.map( tempExtrapolationPlot, ins )
            results = pool.map( exu.synchronizeWithDtwwrapper, ins )

        # These distances are the DTW distances of each roll out. So they are already time-averaged
        for i,(in_i,result) in enumerate(zip(ins,results)):
            (*_, node)  = in_i
            distance_i, distance_std, distance_min, distance_max = result
            all_mean_T_distances[i] = distance_i
            all_std_T_distances[i] = distance_std
            all_min_T_distances[i] = distance_min
            all_max_T_distances[i] = distance_max
        
        all_dtw_distances.append(all_mean_T_distances)

        mean_dtw_rel_error = np.mean(all_mean_T_distances) # spatial average of mean Distance in timeseries of nodes
        all_mean_dtw_distances.append(mean_dtw_rel_error)

        # write performance
        time_elapsed = np.mean(all_elapsed_times)
        all_extrapolation_times.append(time_elapsed)

        if save_plots_:
            if file == "T26":
                vis.plotNodeUncertainEvolution( unscaled_responses[:,nodes_to_plot], lb_unscaled[:,nodes_to_plot], ub_unscaled[:,nodes_to_plot], timestamps, unscaled_targets[:,nodes_to_plot], RESULTS_FOLDER ,file )
                # vis.plotNodeEvolution( unscaled_responses, timestamps, unscaled_targets, RESULTS_FOLDER ,file )

                step_to_plot = [1825]
                point_locations = np.vstack([p.coordinates for p in central_plate_points]).T
                max_T = max( np.max( unscaled_responses[step_to_plot,:] ), np.max( unscaled_targets[step_to_plot,:] ) )
                min_T = min( np.min( unscaled_responses[step_to_plot,:] ), np.min( unscaled_targets[step_to_plot,:] ) )
                colorbar_scaling = np.array([max_T, min_T])

                vis.plotContour2(unscaled_responses[:,T_idxs_to_keep],point_locations,np.array(step_to_plot),d_grid = 27, result_folder = RESULTS_FOLDER, field_value_name_id = "CoarseTemperature_"+ file, colorbar_scaling = colorbar_scaling)
                vis.plotContour2(unscaled_targets[:,T_idxs_to_keep],point_locations,np.array(step_to_plot),d_grid = 27, result_folder = RESULTS_FOLDER, field_value_name_id = "CoarseNominalTemperature_"+ file, colorbar_scaling = colorbar_scaling)

    # plot GPs
    if save_plots_: 
        vis.plotGPWeights( likelihoods, models, RESULTS_FOLDER, states=states, device = None, xticks = None, yticks = [0,0.5,1.0,1.5], id = "",title = None)
        vis.plotWeightsSubplot(likelihoods, models, RESULTS_FOLDER, states=states, weights_in_subplot = [0,1,2], device = None, xticks = None, yticks = [0,0.5,1.0,1.5], id = "",title = None)

    T_Mean_error = np.mean(all_mean_dtw_distances) * 100
    print(f"\nDTW mean relative error: {T_Mean_error}%")
    return T_Mean_error


# %%
if __name__ == "__main__":
    _ = main(True)
# %%
