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

        (G_reg, F_reg, M_reg, output_scale, length_mean, length_var) = (10**-1.21, 1.e-05, 10**-2.23, 0.5, 0.1, 0.05) 
        epochs = 10
        underestimate_lengthscales = 0

        param_f0 = np.asarray([1.])
        bounds_f = Bounds([0.05],[np.inf])

        param_g0 = np.asarray([0.1,0.1,0.1])
        bounds_g = ([0, 0, 0],[np.inf, np.inf, np.inf])

        param_m0 = np.asarray([-.1,-.1,-.1, 1])
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
