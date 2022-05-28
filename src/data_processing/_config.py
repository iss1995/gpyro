import os
import warnings
import numpy as np
from scipy.optimize import Bounds
from datetime import datetime

def config():
        warnings.filterwarnings("ignore")

        PATH_TO_PARENT = "./"
        FILES_FOLDER = PATH_TO_PARENT+"./Gpyro-TD/" #
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

        G_reg, F_reg, M_reg, output_scale, length_mean, length_var = (1.3494328312494594e-06, 3.8252949492342524e-05, 0.05727953475125568, 1.1575308075243074, 0.5469046384472692, 0.06122193346017071)
        bounds_m = (np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0]),
                np.array([0, 0, 0, 0, np.inf ]))
        params_m0 = -np.ones( (5,) )*0.1 # one for T,Delta_T,amb,in1,in2 # one for T,Delta_T,amb,in1,in2
        params_m0[-1]= 0.1 # one for T,Delta_T,amb,in1,in2

        bounds_g = Bounds([-np.inf, -np.inf, -np.inf, -np.inf, 0.2, 0, -np.inf], 
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf ]) 
        param_g0 = np.ones((7))*0.1

        bounds_f = Bounds([-np.inf, -np.inf, -np.inf, -np.inf, 0.2, 0, -np.inf], 
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf ]) 
        param_f0 = np.zeros((7))
        param_f0[-3] = 1 # lengthscale

        return FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER,RESULTS_FOLDER_GP,RESULTS_FOLDER_MODEL,d_grid,bounds_f,F_reg,bounds_g,G_reg,bounds_m,M_reg,output_scale,length_mean,length_var,param_f0,param_g0,params_m0