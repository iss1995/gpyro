
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import utils.online_optimization_utils as onopt
import copy
from numpy.core.numeric import zeros_like

from scipy.interpolate import interp1d
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import minimize
from numpy.linalg import norm
from utils.generic_utils import rolling_window,moving_average,checkConcave_
# from layers.predefined_filters import hankel

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 24}
mpl.rc('font', **font)
mpl.rcParams['lines.markersize'] = 20
# mpl.use('GTK3Cairo')

square_fig = (12,12)
long_fig = (16,9)

# %% 
class _Experiment:
    #TODO: Parent is inheriting the state of the parent at the moment when the self is passed. Afterwards, it is not updated.
    # It's like having a copy of the current state of your parent, but you don't get the new information added. Fix that. 
    def __init__(self,Points,Parent_plotter, Timestamps,T_ambient = 250, subsample = 1,id = None, deposition_height_of_non_excited = -0.4, deposition_height_of_boundary = -0.6 ,results_folder = './results/'):
        """
        Points -> list of _Point objects
        """
        self.results_folder = results_folder
        self.Points = Points
        self.Parent_plotter = Parent_plotter
        self.experiment_id = id 
        self.T_ambient = T_ambient
        self.max_hallucinated_nodes = 2
        self.temperature_timestamps = Timestamps
        self.nots_to_use = None
        self.Time = None # casted to the nots_to_use
        self.Time_raw = None # casted to number of images your dataset has
        self.traj_x = None
        self.traj_y = None
        self.traj_z = None
        self.traj_t = None
        self.traj_status = None
        self.tool_deposition_depth = None
        self.trajectory = None
        self.statuses = None
        self.hocus_pocus_plots = None
        self.hocus_pocus_ax1 = None
        self.hocus_pocus_ax2 = None
        self.T = None
        self.k = None
        self.J = None
        self.conv_filters = None
        self.hocusPocusKwargs = {}
        self.subsample = subsample # subsampling step : Subsampling will have an effect on convolved features
        self.evaluation_mode_ = False # True if you compute features online and propagate state based on model
        self.timestep = None
        self.use_delay_model_ = False
        self.delays = None
        self.deposition_height_of_non_excited = deposition_height_of_non_excited 
        self.deposition_height_of_boundary = deposition_height_of_boundary

    def _calculateAmbientTempeature(self,nodes = None, w = 3000):
        '''
        @params nodes : list of nodes to use for calculating the ambient temperature mean 
        '''

        if nodes is None:
            self.T_ambient = np.ones_like(self.Points[0].T_t)*self.T_ambient
        else:
            points = [self.Points[n] for n in nodes]

            T_amb = points[0].T_t
            for p in points:
                T_amb += p.T_t
            
            self.T_ambient = moving_average(T_amb/len(nodes), w)

            def dbgTamb():
                f = plt.figure(figsize=(16,9))
                plt.plot(self.T_ambient)
                plt.title(f"T ambient for window {w}")
                plt.savefig("results/general_dbging/T_amb.png")
                plt.close(f)
            
    def _fillGapsInMeas(self,kwargs_interp1d = {}):
        """
        Finding missing measurements based on their timestamp and interpolate for keeping the timestep steady.
        kwargs refer to interp1d
        """
        dts = np.diff(self.temperature_timestamps)
        flag_idxs = np.where(dts>1.0)[0]
        
        # if there are gaps in measurements
        if len(flag_idxs)>0:
            # iterate over points to change their temperature profiles
            for p in self.Points:
                interpolator = interp1d(self.temperature_timestamps,p.T_t,**kwargs_interp1d)

                # fill every gap
                added_elements = 0
                new_T_t = list(p.T_t)
                for idx in flag_idxs:
                    k = 1
                    t_init = self.temperature_timestamps[idx]
                    t_stop = self.temperature_timestamps[idx+1]
                    while (t_init + k)<t_stop:
                        new_T_t.insert( int(idx + k + added_elements), interpolator( t_init + k ))
                        added_elements += 1
                        k += 1

                # assign the newly interpolated values to the point  
                p.T_t = np.hstack(new_T_t)

    def _setTraj(self,traj):
        """
        indexing : 0-> experiment, 1-> timestep
        """
        self.traj_points = np.atleast_2d(traj) 
        # self.traj_x = self.traj_points[:,0] # mm
        self.traj_x = -self.traj_points[:,1] # mm
        # self.traj_y = self.traj_points[:,1] # mm
        self.traj_y = -self.traj_points[:,0] # mm
        self.traj_z = self.traj_points[:,2] # mm
        self.traj_t = self.traj_points[:,3] # sec
        self.traj_status = self.traj_points[:,4] # bool : 0 -> torch off, 1 -> torch on
    
    def trajectorySynchronization(self,J=20 , lengthscale = 10, lengthscale_sharp = 10):
        """
        Populate trajectories. After this function is called, each trajectory should have as many elements as the corresponding temperatures
        """
        # Predifine number of trajectory points and then assign them to tool trajectory edges.
        # compute timesteps and time for each experiment
        
        # first fill the gaps in measurements
        ##print("Synchronizing trajectories...")
        self._fillGapsInMeas()

        # now, find at which node you depoisited last
        last_point = np.atleast_1d( (self.traj_x[-1],self.traj_y[-1]))
        all_coordinates = np.asarray( [p.coordinates for p in self.Points], dtype = "object")
        amin = np.argmin([norm(last_point - coordinates) for coordinates in all_coordinates])
        
        # find the sample where material was deposited for the last time and the last sample in general
        nots = self.Points[amin].T_t.shape[0] # number of samples

        self.Ts = 0.1 
        nots_on_trajectory = int(self.traj_t[-1]/self.Ts)

        # Robust solution to data files uncertainty
        nots_to_use = min((nots,nots_on_trajectory))
        self.nots_to_use = nots_to_use

        # self.Time = np.linspace(0,self.Ts*(nots-1),nots)
        self.Time_raw = np.linspace(0,self.Ts*(nots-1),nots)
        self.Time = np.linspace(0,self.Ts*(nots_to_use-1),nots_to_use)
        
        self.Time_on_traj = np.linspace(0,self.Ts*(nots_to_use-1),nots_to_use)

        traj_elem = self.traj_t.shape[0]
        self.trajectory = np.zeros((nots_to_use,2))
        self.statuses = np.zeros((nots_to_use,))
        self.tool_deposition_depth = np.zeros((nots_to_use,))

        # iterate over all the edges of the tool movement and break them in small pieces
        i = 0
        for j,t_step in enumerate(self.Time_on_traj):
            if t_step > self.traj_t[i+1]: # check if you access element 0 -> yes
                if i != traj_elem-1:
                    i += 1

            DeltaX_i = self.traj_x[i+1] - self.traj_x[i]
            DeltaY_i = self.traj_y[i+1] - self.traj_y[i]

            DeltaZ_i = self.traj_z[i+1] - self.traj_z[i]
            DeltaT_i = self.traj_t[i+1] - self.traj_t[i]
            status_i = self.traj_status[i]

            tk_ti1 = t_step - self.traj_t[i]
            if DeltaT_i>0:
                mean_vel_x = DeltaX_i/DeltaT_i
                mean_vel_y = DeltaY_i/DeltaT_i
                mean_vel_z = DeltaZ_i/DeltaT_i
            else:
                mean_vel_x = 0
                mean_vel_y = 0
                mean_vel_z = 0

            self.trajectory[j,:] = (tk_ti1*mean_vel_x + self.traj_x[i], tk_ti1*mean_vel_y + self.traj_y[i])
            self.tool_deposition_depth[j] =  tk_ti1*mean_vel_z + self.traj_z[i]
            self.statuses[j] = status_i

        # compute features relevant to boundary conditions
        # create windows for trajectory components
        traj_windows = [] # one for every coordinate in trajectory
        for traj_component in self.trajectory.T:
            initializer = np.full((J,),1e10) # initialize with something far so your relevance gives you zero
            expanded_traj_component = np.hstack((initializer,traj_component))
            traj_windows.append( rolling_window( expanded_traj_component, 2*J )[J:-1] ) # J in the past and J in the feature, discard initializer and last pad
        
        # compute relevance
        i = 0
        while lengthscale <= 0:
            try:
                lengthscale = norm(self.Points[i].coordinates-self.Points[i].neighbor_nodes_coordinates[0])/2
            except:
                i += 1

        sqrt2 = np.sqrt(2)
        for p in self.Points:
            p_0 = np.full( (traj_windows[0].shape[0],) , p.coordinates[0])
            p_1 = np.full( (traj_windows[0].shape[0],) , p.coordinates[1])

            d_i_xy = np.sum( (np.exp(-((traj_windows[0].T - p_0)/lengthscale)**2/sqrt2) *
                            np.exp(-((traj_windows[1].T - p_1)/lengthscale)**2/sqrt2) * self.statuses).T , axis = 1 )/J

            #for synchronization
            
            d_i_xy_sharp = np.sum( (np.exp(-((traj_windows[0].T - p_0)/lengthscale_sharp)**2/sqrt2) *
                            np.exp(-((traj_windows[1].T - p_1)/lengthscale_sharp)**2/sqrt2) * self.statuses).T , axis = 1 )/J
            
            p._setRelevanceCoef(d_i_xy)
            p._setRelevanceCoef_sharp(d_i_xy_sharp)

            p._set_T_t_use_node(p.T_t[:self.nots_to_use]) # also pass the truncated temperature vector

        self.T_ambient = self.T_ambient[:self.nots_to_use] 

        #print("Done!")

        return self.trajectory,self.statuses

    def torchCrossingNodes(self):
        
        idxs_torch_on = np.where(self.statuses == 1)[0]
        trajectory_torch_on = self.trajectory[idxs_torch_on,:]
        time_torch_on = self.Time[idxs_torch_on]
        lot = idxs_torch_on.shape[0]

        # points_locations = set([])
        for p in self.Points:
            # points_locations.add(p.coordinates)
            
            # find the distances between every trajectory point and itself
            coord_vals = np.atleast_2d(np.full((lot,2),p.coordinates))
            distances = norm(trajectory_torch_on - coord_vals,axis = 1)

            # find adjacent elements in vectors 
            crossing_idxs = np.where(distances < 10)[0] #  intervalas where torch is close to point
            diff_between_idxs = np.diff(crossing_idxs)
            idxs_of_boundaries_of_adjucent_areas = np.where( diff_between_idxs > 7 )[0] # areas will be separated by some thousand pictures, 7 could also be 1 but just for robustness leave it like that
            
            # adjacent element groups
            adjacent_element_pools = []
            low_bound = 0
            for idx in idxs_of_boundaries_of_adjucent_areas:
                adjacent_element_pools.append(crossing_idxs[low_bound:idx+1])
                low_bound = idx+1
            
            # define the point in trajectory closest to the point on the grid

            time_of_crossings = []
            idx_of_crossing = []
            min_dist_idxs = [] # for debuging
            for element_pool in adjacent_element_pools:
                argmin_dist = np.argmin(distances[element_pool])
                min_dist_point_idx = element_pool[argmin_dist]

                # #print(f"DEBUG trajectory_on best loc : {trajectory_torch_on[min_dist_point_idx]} ")
                
                time_of_crossings.append(time_torch_on[min_dist_point_idx])
                global_idx_frame_idx =  idxs_torch_on[min_dist_point_idx]

                idx_of_crossing.append(global_idx_frame_idx)
                min_dist_idxs.append(min_dist_point_idx)

            p._setExcitation_times(time_of_crossings)
            p._setExcitation_idxs(idx_of_crossing)

            excitation_temperatures = p.T_t_use[idx_of_crossing]
            p._setExcitation_temperatures(excitation_temperatures)
        
        all_times = []
        for p in self.Points:
            all_times.append(p.excitation_times)
        all_times_excited = sorted(np.hstack(all_times))

        return None

    def scaleMeasurements(self,scaler_func):
        '''
        have to call after trajectorySynchronization and torchCrossingNodes
        '''
        self.tool_deposition_depth /= np.max(np.abs(self.tool_deposition_depth)) 

        # scale ambient temperature as well
        (self.T_ambient,_) =  scaler_func.transform(np.vstack((self.T_ambient,np.zeros_like(self.T_ambient))).T).T
        
        for p in self.Points:
            feature_data_rep = [p.T_t_use, p.relevance_coef]
            
            feature_data = np.vstack(feature_data_rep).T
            scaled_feature_data = scaler_func.transform(feature_data).T

            p.T_t_use = scaled_feature_data[ :1, : ].squeeze()
            p.relevance_coef = scaled_feature_data[ 1:, : ].squeeze()

    def exportAllTemperatures(self):
        """
        Export the temperatures of all nodes in this experiment.
        @returns all_temps_in_experiment : array including temperatures with indexes # 0->timestep, 1->node
        """
        all_temps_in_experiment = np.zeros((self.Time.shape[0],len(self.Points)))
        for i,p in enumerate(self.Points):
            all_temps_in_experiment[:,i] = p.T_t[:self.nots_to_use]
        
        return all_temps_in_experiment

    @ staticmethod
    def calculateExcitationIdxs(peak,T_t_windows,delta_peaks_kwargs,peak_window_length,T_t_use,relevance_peak_idxs,optimization_tolerance = 1e-3, sort_out_delta_res_coef = 0.1):
        """
        when do the temperature signals show up on your plate?
        """
        N = len(relevance_peak_idxs)
        # find where the peaks are by looking which parts of the dataset look like your model peak
        fun = lambda a: a*peak
        residuals = []
        for window in T_t_windows:
            obj = lambda a: np.linalg.norm(window - fun(a)) 
            res = minimize( obj, 1, tol = optimization_tolerance)
            # #print(res.x)
            residuals.append(res.fun)

        # Find how residuals of peak-similarity change
        residuals = np.asarray(residuals)
        delta_res = np.diff(residuals) 

        b,a = butter(8,0.1)
        delta_res_filt = filtfilt(b,a,delta_res)

        delta_res_peak_idxs, delta_res_peak_heights = find_peaks(delta_res_filt, **delta_peaks_kwargs)

        # At peaks, from a good fit you go to bad fit fast -> High delta residuals -> find the highest 
        # delta_peaks_kwargs["height"] = sort_out_delta_res_coef*np.max(delta_res_filt)
        # delta_res_peak_idxs, delta_res_peak_heights = find_peaks(delta_res_filt, **delta_peaks_kwargs)

        # try to only keep the thermal peaks corresponding to an excitation and clear relevance indexes from relevance peaks happening after the last peak in the thermal data 
        relevance_peak_idxs_to_keep = relevance_peak_idxs.copy()
        N_old = N+1
        while N_old!=N:
            idxs_of_main_excitation_peaks = np.argsort(delta_res_peak_heights['peak_heights'])[::-1][:N]
            input_idxs = delta_res_peak_idxs [ np.sort(idxs_of_main_excitation_peaks) ] +int(peak_window_length) #+int(peak_window_length*0.5) 

            # clear relevance indexes from relevance peaks happening after the last peak in the thermal data
            last_peak_idx = input_idxs[-1]
            relevance_peak_idxs_to_keep = relevance_peak_idxs[relevance_peak_idxs<last_peak_idx]
            N_old = N
            N = len(relevance_peak_idxs_to_keep)

        # put a thresshold on input_idxs: not larger than the length of the array
        input_idxs[input_idxs >= len(T_t_use)] = len(T_t_use) - 1

        # keep only the peaks that result in positive delays
        input_idxs = _Experiment.clearNegativeDelays(input_idxs, relevance_peak_idxs_to_keep, T_t_use)
            
        # map elements outside the vector
        input_idxs = np.asarray([e if e<T_t_use.shape[0] else T_t_use.shape[0]-1 for e in input_idxs])
        return input_idxs, delta_res, delta_res_filt

    @staticmethod
    def clearNegativeDelays(input_idxs, relevance_peak_idxs, T_t_use):
        delays = input_idxs - relevance_peak_idxs
        negative_delays = np.sum(delays<0)
        new_excitation_idxs = copy.copy(input_idxs)

        while negative_delays>0:
            # find first negative delay
            negative_delay_idxs = np.where(delays<0)[0]
            first_negative_delay_idx = negative_delay_idxs[0]

            # check which peak has a hiher temperature, the one admitting negative delay or its closest one?
            ## just take the previous one, your goal is anyhow to have positive delays, even if the next peak is closer than the previous, deleting the next wouldn't change the negative delay
            if first_negative_delay_idx == 0:
                delay_idx_to_delete = first_negative_delay_idx
            else:
                discriminant = np.argmin( (T_t_use[input_idxs[first_negative_delay_idx - 1]], T_t_use[input_idxs[first_negative_delay_idx]]) )
                if discriminant == 0:
                    delay_idx_to_delete = first_negative_delay_idx - 1
                else:
                    delay_idx_to_delete = first_negative_delay_idx

            # delete first negative delay
            mask = np.ones_like(new_excitation_idxs, dtype = bool)
            mask[delay_idx_to_delete] = False
            new_excitation_idxs = new_excitation_idxs[mask]

            # check if delays are fine
            delays = new_excitation_idxs - relevance_peak_idxs[:len(new_excitation_idxs)]
            negative_delays = np.sum(delays<0)     
        
        return new_excitation_idxs

    @ staticmethod
    def lookForHigherTsAndCorrectPeaks(input_idxs,half_window_size,T_t_use,W):
        """
        Usually you define as a peak a value that's not exactly on the T peak. Here you look at an area around the point and try to understand if you can actually pick a more fitting neighbor.
        """
        new_excitation_idxs = []
        for idx in input_idxs:
            window_idxs = [i for i in range( max( idx-half_window_size , 0 ) ,min( idx+half_window_size+1 , W ) ) ]
            T_t_use_window = T_t_use[window_idxs]
            max_idx = np.argmax(T_t_use_window)

            # check if you found your peak or just the boundaries are higher than the peaks
            if max_idx == 0 or max_idx == len(window_idxs):
                
                temp_T_t_use_window = T_t_use_window
                temp_T_t_use_window[max_idx] = np.nan
                new_max_idx = np.argmax(temp_T_t_use_window)

                # find a max that it's concave
                while not checkConcave_(new_max_idx,T_t_use):
                    temp_T_t_use_window[new_max_idx] = np.nan
                    new_max_idx = np.argmax(temp_T_t_use_window)
                    if np.isnan(temp_T_t_use_window[new_max_idx]):
                        new_max_idx = max_idx
                        break
                        
                max_idx = new_max_idx
    

            new_excitation_idx = max( idx-half_window_size , 0 )+ max_idx
            new_excitation_idxs.append(new_excitation_idx)

        return np.asarray(new_excitation_idxs)

    @staticmethod
    def offsetRelevancePeaks(relevance, peak_window_length, input_idxs, relevance_peak_idxs):
        W = len(relevance)
        relevance_new = relevance * 0
        peak_window_length_new = 3*peak_window_length
        # #print(f"node: {p.node}")
        for (delayed,normal) in zip(input_idxs,relevance_peak_idxs):
            # #print(f"\tnormal {normal} vs delayed {delayed}")

            idx_set = normal+peak_window_length_new - max(normal-peak_window_length_new,0) 
            idx_edge_d =   min( delayed+peak_window_length_new, W) # for not overflowing
            idx_edge_n =  min( normal+peak_window_length_new, W) # for not overflowing

            # #print(f"\t\tfirst {idx_edge_n - idx_set } second {  idx_edge_d - idx_set  }")
            
            relevance_new[ idx_edge_d-idx_set : idx_edge_d ] = relevance[ idx_edge_n-idx_set : idx_edge_n ]
            # idx_set = normal+10 - max(normal-10,0) 
            # relevance_new[ delayed+10-idx_set : delayed+10 ] = relevance[ normal+10-idx_set : normal+10 ]   
        return relevance_new 

    @staticmethod
    def synchronizeExcitation(p,subsample = 5, peak_window_length = 8, optimization_tolerance = 1e-6, half_window_size = 10,relevance_peaks = {}, delta_peaks_kwargs = {}, debug = False, RESULTS_FOLDER = "../results/general_dbging/" ):

        T_t_use = p.T_t_use[::subsample]
        relevance = p.relevance_coef[::subsample]
        relevance_sharp = p.relevance_coef_sharp[::subsample] # take the sharp ones, helps in finding peaks

        if not relevance_peaks:
            # first check the relevance is high (you are on the truss structure or not)
            max_rel = np.max(relevance)

            if max_rel>0.5:
                # relevance_peaks = {"height" : np.max(relevance_sharp)*0.5, "distance" : 20}
                # for discriminating between nodes that get excited and those who don't I use an absolute height value 
                relevance_peaks = {"height" : 0.5, "distance" : 20}   
            else:
                # if you are not on the truss structure then just push the sysem towards not finding any peaks
                relevance_peaks = {"height" : 50, "distance" : 20}   

        if not delta_peaks_kwargs:
            delta_peaks_kwargs = { "height" : 0.0, "distance" : 20}
        
        relevance_peak_idxs,_ = find_peaks(relevance_sharp, **relevance_peaks)
        
        # if node gets excited 
        if len(relevance_peak_idxs)>0:
            peak = relevance_sharp[ relevance_peak_idxs[1]-peak_window_length : relevance_peak_idxs[1]+int(peak_window_length*0.5)]
        # if node is not excited
        else:
            peak = []

        w = len(peak)

        # if you found excitations
        if w>0:
            # find the idxs of the temperature peaks
            T_t_windows = rolling_window(T_t_use,w)
            input_idxs, delta_res, delta_res_filt = _Experiment.calculateExcitationIdxs(peak, T_t_windows, delta_peaks_kwargs, peak_window_length, T_t_use, relevance_peak_idxs,optimization_tolerance)

            peak_temperatures = T_t_use[input_idxs]
            delta_peak_temperatures = np.abs( np.diff(peak_temperatures) )

            ################################################################################################################
            ## ways to filter your peaks if needed
            # input_idxs = _Experiment.filterThermalPeaks(peak_temperatures, input_idxs, limit_coef = 0.4)
            ################################################################################################################

            # get rid of negative idxs and put them in order            
            input_idxs[ input_idxs<0 ] = 0
            input_idxs.sort()
            
            # now that you found the excitation idxs look in an area around them for higher T values and correct them if necessary.
            W = len(relevance)
            input_idxs = _Experiment.lookForHigherTsAndCorrectPeaks(input_idxs,half_window_size,T_t_use,W)

            # move peaks to delayed positions
            relevance_new = _Experiment.offsetRelevancePeaks(relevance, peak_window_length, input_idxs, relevance_peak_idxs)

        # if you didn't find any exitation
        else:
            relevance_new = zeros_like(relevance_sharp)
            input_idxs = np.array([])
            relevance_peak_idxs = np.array([]) 
            delta_res = np.array([])
            delta_res_filt = np.array([])

        if debug:

            from datetime import datetime
            time = datetime.now().strftime("%m%d_%H%M%S")
            if len(delta_res)>0:
                
                f = plt.figure(figsize=(16,18))

                ax = f.add_subplot(211)
                ax.plot(T_t_use, label = "Temperature" )
                ax.plot(delta_res, label = "Delta Residuals" )
                ax.set_title("Optimization results")
                ax.legend()

                ax1 = f.add_subplot(212)
                ax1.plot(T_t_use)
                ax1.plot(delta_res_filt )
                ax1.scatter(input_idxs,T_t_use[input_idxs], label = "peaks", marker = 'x', linewidths = 2)
                # ax1.legend()  

                plt.savefig(RESULTS_FOLDER + f"point_{p.node}_sync_excite_{time}.png")   

                plt.close(f)

        return relevance_new, input_idxs, relevance_peak_idxs
    
    def formRawInputsWithSynchronousExcitationMultithreadWrapper(self, synchronizeExcitation_kwargs = {}):
        '''
        Compute Raw inputs for all nodes. This means input values before convolution.

        @param experiment_idxs : subscriptable with indexes of experiments to use
        @returns Delta_T : List of lists with the Delta_Ts of neighbors. Array indexing:[a][b]: a -> point, b -> T_point_a-T_point_b
        '''
        points_with_neighbors = [p for p in self.Points if p._hasNeighbors_()]

        # #print("Forming Raw Inputs...")
        with mp.Pool( processes = mp.cpu_count() - 1 ) as pool:
            results = pool.map(self.formRawInputsWithSynchronousExcitation,points_with_neighbors)
        
        for (res,point) in zip(results,points_with_neighbors):
            temps, deltaTemps, inputs, T_t_use, tool_height_trajectory, tool_heights, input_idxs, excitation_idxs, excitation_temperatures, point_delays, relevance, relevance_sharp, delayed_relevance = res

            point._setDeltaT(temps)
            point._setDeltaTambient(deltaTemps)
            point._setInputRawFeatures(inputs)
            point._set_T_t_use_node(T_t_use)
            point._setInputIdxs(input_idxs)
            point._setExcitation_idxs(excitation_idxs)
            point._setExcitation_temperatures(excitation_temperatures)
            point._setExcitation_delay(point_delays)
            point._setRelevanceCoef(relevance)
            point._setDelayedRelevanceCoef(delayed_relevance)
            point._setRelevanceCoef_sharp(relevance_sharp)
            point._setExcitation_excitationDelayTorchHeight(tool_heights)
            point._setExcitation_excitationDelayTorchHeightTrajectory(tool_height_trajectory)
            point._setExcitation_TatTheStartOfInput(np.zeros_like(tool_height_trajectory))

    def formRawInputsWithSynchronousExcitationWrapper(self, synchronizeExcitation_kwargs = {}):
        '''
        Compute Raw inputs for all nodes. This means input values before convolution.

        @param experiment_idxs : subscriptable with indexes of experiments to use
        @returns Delta_T : List of lists with the Delta_Ts of neighbors. Array indexing:[a][b]: a -> point, b -> T_point_a-T_point_b
        '''
        points_with_neighbors = [p for p in self.Points if p._hasNeighbors_()]
        # points_with_neighbors = [p for p in self.Points[54:56]]
        
        # #print("Forming Raw Inputs...")
        T_t_useS = []
        for point in points_with_neighbors:
            # #print(f"point : {point.node}")
            _, _, _, T_t_use, *_ = self.formRawInputsWithSynchronousExcitation(point,**synchronizeExcitation_kwargs)
            T_t_useS.append(T_t_use)
        
        for (point, T_t_use) in zip (points_with_neighbors,T_t_useS):
            point._set_T_t_use_node(T_t_use)

    def calculateToolDepositionDepth(self,tool_height_trajectory,relevance_peak_idxs,robustness_offset = 5):
        # excitation_tool_heights = self.tool_deposition_depth[::self.subsample][relevance_peak_idxs-robustness_offset]
        excitation_tool_heights_trajectory = self.tool_deposition_depth[::self.subsample]
        # check status to see if you are on air or not.
        excitation_tool_statuses_trajectory = self.statuses[::self.subsample][relevance_peak_idxs-robustness_offset]

        W = len(relevance_peak_idxs)
        for i,idx in enumerate(relevance_peak_idxs):
            if i== 0:
                final_idx = int((relevance_peak_idxs[i+1] + idx)/2)
                
                # select as height the most prevalent value
                heights, counts = np.unique( excitation_tool_heights_trajectory[:final_idx] , return_counts=True)
                height = heights[np.argmax(counts)]
                
                tool_height_trajectory[:final_idx] = height 
            elif i<W-1:
                final_idx = int((relevance_peak_idxs[i+1] + idx)/2)
                first_idx = int((relevance_peak_idxs[i-1] + idx)/2)
                
                # select as height the most prevalent value
                heights, counts = np.unique( excitation_tool_heights_trajectory[first_idx:final_idx] , return_counts=True)
                height = heights[np.argmax(counts)]
                
                tool_height_trajectory[first_idx:final_idx] = height 
            else:
                first_idx = int((relevance_peak_idxs[i-1] + idx)/2)

                # select as height the most prevalent value
                heights, counts = np.unique( excitation_tool_heights_trajectory[first_idx:] , return_counts=True)
                height = heights[np.argmax(counts)]
                
                tool_height_trajectory[first_idx:] = height 
        
        # for i,(height,idx) in enumerate(zip(excitation_tool_heights,relevance_peak_idxs)):
        #     if i== 0:
        #         final_idx = int((relevance_peak_idxs[i+1] + idx)/2)
        #         tool_height_trajectory[:final_idx] = height 
        #     elif i<W-1:
        #         final_idx = int((relevance_peak_idxs[i+1] + idx)/2)
        #         first_idx = int((relevance_peak_idxs[i-1] + idx)/2)
        #         tool_height_trajectory[first_idx:final_idx] = height 
        #     else:
        #         first_idx = int((relevance_peak_idxs[i-1] + idx)/2)
        #         tool_height_trajectory[first_idx:] = height 

        
        return tool_height_trajectory

    def formRawInputsWithSynchronousExcitation(self, point, synchronizeExcitation_kwargs = {}, debug = False, RESULTS_FOLDER = "../results/general_dbging/"):
        '''
        Compute Raw inputs for all nodes. This means input values before convolution.

        @param experiment_idxs : subscriptable with indexes of experiments to use
        @returns Delta_T : List of lists with the Delta_Ts of neighbors. Array indexing:[a][b]: a -> point, b -> T_point_a-T_point_b
        '''
        
        # calculate separate delay for each node - no aliasing 
        # #print(f"\tNode:{point.node}")

        # define timeseries
        point_T_t = point.T_t_use[::self.subsample] 
        ambient = self.T_ambient[::self.subsample]
        neighboring_nodes_T_ts = [self.Points[neighb_idx].T_t_use[::self.subsample] for neighb_idx in point.neighbor_nodes]
        
        # synchronize excitation and create respective tool height series
        robustness_offset = 0
        relevance_d, input_idxs, relevance_peak_idxs = self.synchronizeExcitation( point, subsample = self.subsample, debug = debug, RESULTS_FOLDER = RESULTS_FOLDER, **synchronizeExcitation_kwargs)
        
        tool_heights = np.array([])
        tool_height_trajectory = np.ones_like(relevance_d)*(self.deposition_height_of_non_excited) # -1 for non excited points
        # tool_height_trajectory = np.zeros_like(relevance_d)
        # if node gets excited
        if len(relevance_peak_idxs)>0:
            tool_height_trajectory = self.calculateToolDepositionDepth(tool_height_trajectory,relevance_peak_idxs,robustness_offset)
            tool_heights = tool_height_trajectory[input_idxs]

        # assign special height to the plate nodes on boundary
        else:
            if len(point.neighbor_nodes)<4:
                tool_height_trajectory = np.ones_like(relevance_d)*(self.deposition_height_of_boundary) # -1 for non excited points

        if debug:
            f = plt.figure(figsize = (16,18))

            # ax = f.add_subplot(211)
            # ax.plot(tool_height_trajectory,label = "tool height")
            # ax.plot(point.relevance_coef[::self.subsample],label = "relevance")
            # ax.plot(relevance_d,label = "delayed relevance")
            # ax.set_title("Constructed tool height trajectory")
            # ax.set_ylabel("Height")
            # ax.set_xlabel("Samples")
            # ax.legend()
            
            ax1 = f.add_subplot(111)
            ax1.plot(point_T_t,label = "Temperature")
            ax1.plot(point.relevance_coef[::self.subsample],label = "relevance")
            ax1.plot(relevance_d,label = "delayed relevance")
            ax1.set_title("Delayed relevance")
            ax1.set_ylabel("Relevance")
            ax1.set_xlabel("Samples")
            ax1.legend()

            plt.savefig(RESULTS_FOLDER + f"point_{point.node}_tool_traj.png")
            plt.close(f)

        # If point is in the grid
        if neighboring_nodes_T_ts:
            
            # fill missing nodes with 0 for not getting extra contribiution
            # neighboring_nodes_T_ts += [np.zeros_like(point_T_t)]*len(point.hallucinated_nodes)
            
            # introduce a lag in data so that you can train on node i and predict - why??
            nodes_with_lags = []
            for T_t in neighboring_nodes_T_ts:
                nodes_with_lags.append(point_T_t - T_t)

            # zero out the contribution of non-existing neighbors
            nodes_with_lags += [np.zeros_like(point_T_t)]*len(point.hallucinated_nodes)

            # create raw feature vector from temperatures. Set 0 for hallucinated nodes
            temps = np.vstack(( point_T_t, np.vstack(nodes_with_lags)))
            
            # create the delta T for the boundary conditions
            point_delta_T_ambient = []
            
            point_delta_T_ambient = [point_T_t - ambient]*len(point.hallucinated_nodes)
            # point_delta_T_ambient = [point_T_t - self.T_ambient[::self.subsample]]*len(point.hallucinated_nodes)
            point_delta_T_ambient += [np.zeros_like(point_T_t)]*(self.max_hallucinated_nodes - len(point.hallucinated_nodes))
            deltaTemps = np.vstack(point_delta_T_ambient)
            
            # create Input Features
            IF = []
            IF.append(relevance_d * tool_height_trajectory ) # introduce height component
            IF.append(relevance_d) # introduce energy input component
            inputs = np.vstack(IF)

        else:
            temps = None
            deltaTemps = None
            inputs = None

        # resample other stuff as well
        excitation_idxs = np.asarray(point.excitation_idxs/self.subsample,dtype = np.int64)
        excitation_temperatures = point_T_t[excitation_idxs]

        relevance = point.relevance_coef[::self.subsample]
        relevance_sharp = point.relevance_coef_sharp[::self.subsample]
        delayed_relevance = relevance_d

        # define delay times
        delays_to_count = np.min([len(input_idxs),len(excitation_idxs)])
        point_delays = input_idxs[:delays_to_count] - excitation_idxs[:delays_to_count]

        point._setDeltaT(temps)
        point._setDeltaTambient(deltaTemps)
        point._setInputRawFeatures(inputs)
        point._setInputIdxs(input_idxs)
        point._setExcitation_idxs(excitation_idxs)
        point._setExcitation_temperatures(excitation_temperatures)
        point._setExcitation_delay(point_delays)
        point._setRelevanceCoef(relevance)
        point._setDelayedRelevanceCoef(delayed_relevance)
        point._setRelevanceCoef_sharp(relevance_sharp)
        point._setExcitation_excitationDelayTorchHeight(tool_heights)
        point._setExcitation_excitationDelayTorchHeightTrajectory(tool_height_trajectory)
        point._setExcitation_TatTheStartOfInput(np.zeros_like(tool_height_trajectory))

        # now subsample all points to use. If you do beforehand then you double subsample stuff
        T_t_use = point_T_t
        # if not self.evaluation_mode_: 
        #     point._set_T_t_use_node(T_t_use)

        return temps, deltaTemps, inputs, T_t_use, tool_height_trajectory, tool_heights, input_idxs, excitation_idxs, excitation_temperatures, point_delays, relevance, relevance_sharp, delayed_relevance

    def subsampleTrajectories(self):
        """
        Subsample the trajectories you need for extrapolation. Make assessment faster.
        """
        for point in self.Points:
            point._set_T_t_use_node( point.T_t_use[::self.subsample] )
            point._setRelevanceCoef( point.relevance_coef[::self.subsample] )
            point._setRelevanceCoef_sharp( point.relevance_coef_sharp[::self.subsample] )
            point._setExcitation_idxs( (point.excitation_idxs/self.subsample).astype(np.int64) )
            point._setExcitation_temperatures( point.T_t_use[ point.excitation_idxs ] )

            # tool_height_trajectory = self.calculateToolDepositionDepth(np.zeros_like(point.T_t_use), point.excitation_idxs, robust_idx )
            tool_height_trajectory = self.calculateToolDepositionDepth(np.ones_like(point.T_t_use)*(self.deposition_height_of_non_excited), point.excitation_idxs )

            # special case for nodes on boundary
            if len(point.neighbor_nodes)<4:
                tool_height_trajectory = np.ones_like(point.T_t_use)*(self.deposition_height_of_boundary) # -1 for non excited points

            tool_height = tool_height_trajectory[point.excitation_idxs]
            point._setExcitation_excitationDelayTorchHeight(tool_height)
            
            point._setExcitation_excitationDelayTorchHeightTrajectory(tool_height_trajectory)
            point._setExcitation_TatTheStartOfInput(np.zeros_like(tool_height_trajectory))

            # point._setDeltaTambient( deltaTemps )
            # point._setInputRawFeatures(inputs)
            # point._set_T_t_use_node(T_t_use)
            # point._setInputIdxs(input_idxs)
            # point._setExcitation_delay(point_delay)
        
        on_boundary_ = onopt.pointsOnBoundaries_(self)
        self.Points = onopt.statesForBoundaries( self.Points ,on_boundary_ )

    def _copy(self):
        return copy.deepcopy(self)   
