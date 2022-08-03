import os,sys
import warnings
import numpy as np
from scipy.optimize import Bounds
from datetime import datetime

def config():
        warnings.filterwarnings("ignore")

        # check if interactive shell
        if os.isatty(sys.stdout.fileno()):
                PATH_TO_PARENT = "./"
        else:
                PATH_TO_PARENT = "../"

        FILES_FOLDER = PATH_TO_PARENT+"Gpyro-TD/" #
        POINT_TEMPERATURE_FILES = "temperatures/T*.csv" # no time information on thermal imaging camera
        TRAJECTORY_FILES = "Coordinate_Time/Coordinates_T*.csv"
        POINT_COORDINATES = "Coordinate_Time/point_coordinates.csv"

        run_id = "{}/".format(datetime.now().strftime("%m%d%y/%H%M%S"))
        RESULTS_FOLDER = PATH_TO_PARENT + "results/plots/" + run_id
        RESULTS_FOLDER_ERROR_EVOLUTION = RESULTS_FOLDER + "/Error_evolution/"
        RESULTS_FOLDER_STATISTICS = RESULTS_FOLDER + "/Statistics/"
        RESULTS_FOLDER_MODEL = PATH_TO_PARENT + "results/models/linear/" + run_id
        RESULTS_FOLDER_GP = PATH_TO_PARENT + "results/models/gp/" + run_id
        BACKUP_FOLDER = PATH_TO_PARENT + "back_up/" + run_id

        os.makedirs(BACKUP_FOLDER, exist_ok = True)
        os.makedirs(RESULTS_FOLDER, exist_ok = True)
        os.makedirs(RESULTS_FOLDER + "/nodes_time/", exist_ok = True)
        os.makedirs(RESULTS_FOLDER + "/DEBUG/" , exist_ok = True)
        os.makedirs(RESULTS_FOLDER_STATISTICS, exist_ok = True)
        os.makedirs(RESULTS_FOLDER_ERROR_EVOLUTION, exist_ok = True)
        os.makedirs(RESULTS_FOLDER_MODEL, exist_ok = True)
        os.makedirs(RESULTS_FOLDER_GP, exist_ok= True)

        d_grid = 27

        # G_reg, F_reg, M_reg, output_scale, length_mean, length_var = (1.3494328312494594e-06, 3.8252949492342524e-05, 0.05727953475125568, 1.1575308075243074, 0.5469046384472692, 0.06122193346017071)
        # G_reg, F_reg, M_reg, output_scale, length_mean, length_var = (2.965568341672765e-05, 0.0058595727258403405, 0.004350615438521492, 0.6256305896991189, 0.2802618031752034, 0.01993745968749789)
        # (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = (0.00260910580042891, 3.5085344790466794e-05, 0.008394886320272292, 0.6954645807725358, 0.4168718470022085, 0.004747836713657)
        # (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = (0.0066683951679123174, 0.00036449461447877454, 0.0013879306498389356, 1.9310377024123906, 0.13086750735377498, 0.9068099667753063)
        # (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = (0.08193669809780384, 9.50100715892454e-06, 0.014482733931970656, 0.47306714997437277, 0.19087101032988138, 0.0041883678096251205) # gradient descent
        # (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = (0.2025730741381932, 3.512553288229271e-06, 0.07020475588025676, 0.5453833163127719, 0.01113564687660983, 0.033583130119869575) # l-bfgs-b 4.75%
        # (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = (0.01015004806480582, 1.4040897334213959e-06, 0.004896972970802323, 0.2998779995941545, 0.028325828878155296, 0.005791083487117619) # l-bfgs-b gd 2.42%
        (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = (5.e-3, 1.e-06, 1e-1, 0.5, 0.25, 0.05) 
        epochs = 10
        underestimate_lengthscales = 0

        param_f0 = np.asarray([1.])
        bounds_f = Bounds([0.05],[np.inf])
        # bounds_f = ([0.05],[np.inf])

        param_g0 = np.asarray([0.1,0.1,0.1])
        # bounds_g = Bounds([-np.inf, -np.inf, -np.inf],[np.inf, np.inf, np.inf])
        # bounds_g = ([-np.inf, -np.inf, -np.inf],[np.inf, np.inf, np.inf])
        bounds_g = ([0, 0, 0],[np.inf, np.inf, np.inf])

        param_m0 = np.asarray([-.1,-.1,-.1, 1])
        # bounds_m = Bounds([-np.inf, -np.inf, -np.inf, 0],[0, 0, 0, np.inf])
        bounds_m = ([-np.inf, -np.inf, -np.inf, 0],[0, 0, 0, np.inf])

        opt_kwargs = {"bounds_f":bounds_f, 
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

        return FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER,RESULTS_FOLDER_GP,RESULTS_FOLDER_MODEL,d_grid,output_scale,length_mean,length_var,opt_kwargs


def configHyperparameters():
        G_reg, F_reg, M_reg, output_scale, length_mean, length_var = (1.3494328312494594e-06, 3.8252949492342524e-05, 0.05727953475125568, 1.1575308075243074, 0.5469046384472692, 0.06122193346017071)
        return G_reg, F_reg, M_reg, output_scale, length_mean, length_var