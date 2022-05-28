
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import utils.online_optimization_utils as onopt
import os
import copy
from numpy.core.numeric import zeros_like

from scipy.interpolate import interp1d
from scipy.signal import find_peaks, butter, filtfilt
from scipy.optimize import minimize
from numpy.linalg import norm
from utils.generic_utils import fromDynVecToDynamics,rolling_window, piecewiseLinear,moving_average,checkConcave_
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
            
            # dbgTamb()

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
    
        # ("\ttraj_x shape {}, traj shape {}, (first,last) status : {}".format(self.traj_x.shape,self.traj.shape,(self.traj_status[0],self.traj_status[-1])))#debug

    def scaleMeasurements(self,scaler_func):
        '''
        have to call after trajectorySynchronization and torchCrossingNodes
        '''
        self.tool_deposition_depth /= np.max(np.abs(self.tool_deposition_depth)) 

        # scale ambient temperature as well
        (self.T_ambient,_) =  scaler_func.transform(np.vstack((self.T_ambient,np.zeros_like(self.T_ambient))).T).T
        
        for p in self.Points:
            feature_data_rep = [p.T_t_use, p.relevance_coef]
            # feature_data_len = [np.min(feat.shape) for feat in feature_data_rep]
            # index_bounds = np.cumsum(feature_data_len)
            
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
        # last_node_temp_profile = self.Points[amin].T_t
        # peaks = find_peaks(last_node_temp_profile, prominence = 10)
        # nots_on_trajectory = peaks[0][-1]

        # define timestep and time intervals
        # self.Ts = self.traj_t[-1]/nots
        # self.Ts = self.traj_t[-1]/nots_on_trajectory
        self.Ts = 0.1 # LOL this worked better than anything else
        nots_on_trajectory = int(self.traj_t[-1]/self.Ts)

        # Robust solution to data files uncertainty
        nots_to_use = min((nots,nots_on_trajectory))
        self.nots_to_use = nots_to_use

        # self.Time = np.linspace(0,self.Ts*(nots-1),nots)
        self.Time_raw = np.linspace(0,self.Ts*(nots-1),nots)
        self.Time = np.linspace(0,self.Ts*(nots_to_use-1),nots_to_use)
        
        # self.Time_on_traj = np.linspace(0,self.Ts*(nots_on_trajectory-1),nots_on_trajectory)
        # self.Time_on_traj = np.linspace(0,self.Ts*(nots_on_trajectory-1),nots_on_trajectory)
        self.Time_on_traj = np.linspace(0,self.Ts*(nots_to_use-1),nots_to_use)

        traj_elem = self.traj_t.shape[0]
        # self.trajectory = np.zeros((nots,2))
        # self.statuses = np.zeros((nots,))
        # self.tool_deposition_depth = np.zeros((nots,))
        self.trajectory = np.zeros((nots_to_use,2))
        self.statuses = np.zeros((nots_to_use,))
        self.tool_deposition_depth = np.zeros((nots_to_use,))
        # self.trajectory = np.zeros((nots_on_trajectory,2))
        # self.statuses = np.zeros((nots_on_trajectory,))

        # iterate over all the edges of the tool movement and break them in small pieces
        i = 0
        for j,t_step in enumerate(self.Time_on_traj):
        # for j,t_step in enumerate(self.Time):
            # #print("t_step",t_step)
            if t_step > self.traj_t[i+1]: # check if you access element 0 -> yes
                if i != traj_elem-1:
                    i += 1

            # DeltaX_i = point2[0] - point1[0]
            DeltaX_i = self.traj_x[i+1] - self.traj_x[i]
            # DeltaY_i = point2[1] - point1[1]
            DeltaY_i = self.traj_y[i+1] - self.traj_y[i]

            DeltaZ_i = self.traj_z[i+1] - self.traj_z[i]
            # DeltaT_i = point2[3] - point1[3]
            DeltaT_i = self.traj_t[i+1] - self.traj_t[i]
            # status_i = point1[4]
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
            # #print("\tmean_vel_x and y: {}\t{}".format(mean_vel_x,mean_vel_y))

            # experiment_trajectory.append(( tk_ti1*mean_vel_x + point1[0], tk_ti1*mean_vel_y + point1[1]) )
            self.trajectory[j,:] = (tk_ti1*mean_vel_x + self.traj_x[i], tk_ti1*mean_vel_y + self.traj_y[i])
            self.tool_deposition_depth[j] =  tk_ti1*mean_vel_z + self.traj_z[i]
            # #print("\tcurrent position",experiment_trajectory[-1])
            # experiment_statuses.append(status_i)
            self.statuses[j] = status_i

        # #print("T_t.shape = {}, trajectory.shape = {}, statuses.shape = {}".format(self.Points[0].T_t.shape,self.trajectory.shape,self.statuses.shape))
        # 
        # plt.figure("Xs")
        # plt.plot(self.Time,self.trajectory[:,0])
        # plt.title('X')
        # plt.xlabel('Distance (mm)')
        # plt.ylabel("Time (s)")
        #
        # plt.figure("Ys")
        # plt.plot(self.Time,self.trajectory[:,0])
        # plt.title('Y')
        # plt.xlabel('Distance (mm)')
        # plt.ylabel("Time (s)")
        #

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
            
            # f = plt.figure("debug crossings")
            # plt.plot(distances)
            # for element_pool in adjacent_element_pools:
            #     plt.plot(element_pool,distances[element_pool],color = "orange")
            # plt.show()
            # plt.close(f)

            time_of_crossings = []
            idx_of_crossing = []
            min_dist_idxs = [] # for debuging
            for element_pool in adjacent_element_pools:
                argmin_dist = np.argmin(distances[element_pool])
                min_dist_point_idx = element_pool[argmin_dist]

                # #print(f"DEBUG trajectory_on best loc : {trajectory_torch_on[min_dist_point_idx]} ")
                
                time_of_crossings.append(time_torch_on[min_dist_point_idx])
                global_idx_frame_idx =  idxs_torch_on[min_dist_point_idx]

                # #print(f"DEBUG trajectory best loc : {self.trajectory[global_idx_frame_idx]}, torch status : {self.statuses[global_idx_frame_idx]} ")
                # take only the excitations your datasets allow to use
                # if global_idx_frame_idx<self.nots_to_use:
                idx_of_crossing.append(global_idx_frame_idx)
                min_dist_idxs.append(min_dist_point_idx)
                # else:
                #     time_of_crossings.pop()
                # plt.scatter( min_dist_point_idx, distances[min_dist_point_idx], marker = 'x', color = 'black')

            p._setExcitation_times(time_of_crossings)

            # idx_of_crossing = []
            # for time in time_of_crossings:
            #     idx_of_crossing.append(np.argmin(np.abs(time - self.Time)))

            p._setExcitation_idxs(idx_of_crossing)

            excitation_temperatures = p.T_t_use[idx_of_crossing]
            p._setExcitation_temperatures(excitation_temperatures)
            
            # plt.figure("times")
            # plt.plot(self.Time, p.T_t)
            # plt.plot(self.Time[idx_of_crossing], p.T_t[idx_of_crossing], marker = 'x')
            # plt.show()
        
        all_times = []
        for p in self.Points:
            all_times.append(p.excitation_times)
        all_times_excited = sorted(np.hstack(all_times))

        return None

    def formDeltaT(self):
        '''
        Compute Delta Temperature between nodes. For node i every Delta T is Ti-Tneighbour.

        @param experiment_idxs : subscriptable with indexes of experiments to use
        @returns Delta_T : List of lists with the Delta_Ts of neighbors. Array indexing:[a][b]: a -> point, b -> T_point_a-T_point_b
        '''
        tool_height = self.tool_deposition_depth[::self.subsample]
        for point in self.Points:
            # temperature timeseries for the neighbors of the current node
            neighboring_nodes_T_ts = [self.Points[neighb_idx].T_t_use[::self.subsample] for neighb_idx in point.neighbor_nodes]
            point_T_t = point.T_t_use[::self.subsample] 
            relevance = point.relevance_coef[::self.subsample]
            # If point is in the grid
            if neighboring_nodes_T_ts:
                
                # fill missing nodes with T_t of your current node to get a 0 Delta_T
                neighboring_nodes_T_ts += [point_T_t]*len(point.hallucinated_nodes)

                # create the Delta Ts between point and neighbors. Consider T_p -T_neigh or 0 for hallucinated nodes
                point_delta_T = []
                for T_t_i in neighboring_nodes_T_ts:
                    point_delta_T.append(point_T_t - T_t_i)
                
                point._setDeltaT(np.vstack(point_delta_T))

            
                # create the delta T for the boundary conditions
                point_delta_T_ambient = []
                
                point_delta_T_ambient = [point_T_t - self.T_ambient]*len(point.hallucinated_nodes)
                point_delta_T_ambient += [np.zeros_like(point_T_t)]*(self.max_hallucinated_nodes - len(point.hallucinated_nodes))
                
                # point_delta_T_ambient = [point_T_t- relevance]*len(point.hallucinated_nodes) # non constant T_ambient
                # point_delta_T_ambient += [np.zeros_like(point_T_t)]*(self.max_hallucinated_nodes - len(point.hallucinated_nodes)) # non constant T_ambient

                # point_delta_T_ambient += [point_T_t- tool_height]*len(point.hallucinated_nodes) # non constant T_ambient
                # point_delta_T_ambient += [np.zeros_like(point_T_t)]*(self.max_hallucinated_nodes - len(point.hallucinated_nodes)) # non constant T_ambient

                point._setDeltaTambient(point_delta_T_ambient)

            else:
                point._setDeltaT(None)
                point._setDeltaTambient(None)

        # now subsample all points to use. If you do beforehand then you double subsample stuff 
        for point in self.Points:
            point._set_T_t_use_node(point.T_t_use[::self.subsample])

    @staticmethod
    def calculateDelay( delay_model, torch_height, statuses):
        '''
        calculate delay linear model compared to height. if torch is off return 0 delays 
        '''
        # return (statuses*(delay_model[0]*torch_height + delay_model[1])).astype(np.int64)
        # return (statuses*np.polyval(delay_model,torch_height)).astype(np.int64)
        # out = (piecewiseLinear(torch_height,*delay_model)*statuses).astype(np.int64) 
        out = (piecewiseLinear(torch_height,*delay_model)).astype(np.int64) 
        negative_ = out<0
        out[ negative_ ] = 0
        return out

    @staticmethod       
    def formDelayedIdxs( idxs_of_absolute_frame, delays):
        '''
        calculate the delayed idxs for the relevance. The idxs can never decrease, so you get rid of alliased relevance peaks. 
        '''
        # delayed_relevance_idxs = idxs_of_absolute_frame - delays
        delayed_relevance_idxs = (idxs_of_absolute_frame - delays).tolist()
        transformed = [delayed_relevance_idxs[0] if delayed_relevance_idxs[0]>=0 else idxs_of_absolute_frame[0]]

        for (del_idx,normal_idx) in zip(delayed_relevance_idxs[1:],idxs_of_absolute_frame[1:]):
    
            # treat negative idxs
            # if del_idx<=0:
            #     transformed.append(0)
            if del_idx<transformed[-1]:
                transformed.append(transformed[ -1])
                # transformed.append(normal_idx)
            else:
                transformed.append(del_idx)
        # negative_ = np.where(delayed_relevance_idxs<0)
        # delayed_relevance_idxs[ negative_ ] = negative_
        # transformed = delayed_relevance_idxs

        # return delayed_relevance_idxs
        return np.asarray(transformed, dtype = np.int64)

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
            
        # input_idxs get updated after their sanity check 
        # input_idxs = delta_res_peak_idxs [ np.argsort(delta_res_peak_heights['peak_heights'])[::-1][:N] ] +int(peak_window_length*0.5) 

        # delays = np.sort(input_idxs) - relevance_peak_idxs
        # number_of_peaks_that_have_appeared_in_T_t_already = np.sum(delays>0)
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

            # # check which is the closest peak to the one admitting a negative delay
            # if first_negative_delay_idx == 0:
            #     closest_excitation_idx_to_negative_delay = new_excitation_idxs[1]
            # elif first_negative_delay_idx == len(new_excitation_idxs)-1:
            #     closest_excitation_idx_to_negative_delay = new_excitation_idxs[ len(new_excitation_idxs)-2 ]
            # else:
            #     closest_excitation_idx_to_negative_delay = np.min( new_excitation_idxs[first_negative_delay_idx - 1], new_excitation_idxs[first_negative_delay_idx + 1] )

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
    def filterThermalPeaks(peak_temperatures, input_idxs, limit_coef = 0.4):
        # ordered_idxs_delta_peak_temperatures = np.argsort(delta_peak_temperatures)[::-1]
        # extra_peaks = 0
        # for idx in ordered_idxs_delta_peak_temperatures:
        #     if peak_temperatures[idx]<0.2:
        #         extra_peaks =+1
        #     # if idx>0 :
        #     #     check =  delta_peak_temperatures[idx]/delta_peak_temperatures[idx-1]
        #     # else:
        #     #     check =  delta_peak_temperatures[idx]/delta_peak_temperatures[idx+1]

        #     # if check>5:
        #     #     extra_peaks =+1

        excitation_idxs_new = []
        for (peak_temp,excitation_idx) in zip(peak_temperatures,input_idxs):
            if excitation_idx>200:
                if peak_temp>limit_coef*max(peak_temperatures):
                    excitation_idxs_new.append(excitation_idx)
            else:
                excitation_idxs_new.append(excitation_idx)

        excitation_idxs_new = np.asarray(excitation_idxs_new)

        # excitation_idxs_new = []
        # for idx in input_idxs:
        #     select_idxs = input_idxs[ input_idxs != idx ]
        #     rest_peaks = T_t_use[ select_idxs ]
        #     rest_mean = np.mean(rest_peaks)
        #     if T_t_use[idx]>rest_mean*0.5:
        #         excitation_idxs_new.append(idx)
        
        # excitation_idxs_new = np.asarray(excitation_idxs_new, dtype= np.int64)

        # update excitation idxs
        # input_idxs = delta_res_peak_idxs [ np.argsort(delta_res_peak_heights['peak_heights'])[::-1][:N-extra_peaks] ] 
        return excitation_idxs_new

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


        # #print("Done!")
    
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
        # #print("Done!")

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

    def formRawInputs(self, T = None, timestep = None, delay_model = None):
        '''
        Compute Raw inputs for all nodes. This means input values before convolution.

        @param experiment_idxs : subscriptable with indexes of experiments to use
        @returns Delta_T : List of lists with the Delta_Ts of neighbors. Array indexing:[a][b]: a -> point, b -> T_point_a-T_point_b
        '''
        points_with_neighbors = [p for p in self.Points if p.neighbor_nodes]
        
        # insert delay?
        if delay_model is None:
            assert not self.use_delay_model_, "You must provide a delay model if use_delay_model_ is true"
            delay_model = np.array([0,0,0],dtype = np.int64)

        # # with this approach you keep the same delay for all the points in the system (not correct for last excited point - aliasing)
        # self.delays = self.calculateDelay(delay_model,self.tool_deposition_depth,self.statuses) 

        for point in points_with_neighbors:
            
            tool_height = np.copy(self.tool_deposition_depth[::self.subsample])
            relevance = point.relevance_coef[::self.subsample]
            ambient = self.T_ambient[::self.subsample]
            # delays = self.delays[::self.subsample]

            # calculate separate delay for each node - no aliasing 
            robustness_offset = 5
            # excitation_idxs = (point.excitation_idxs/self.subsample).astype(np.int64) # -robustness_offset for always taking the correct height and not elevated ones
            excitation_idxs = point.excitation_idxs
            excitation_idxs[ excitation_idxs<0 ] = excitation_idxs[ excitation_idxs<0 ]  
            
            # tmp = tool_height
            tool_height_in_last_excitation = np.copy( tool_height )

            # tool_height_in_last_excitation[ :excitation_idxs[0] ] = tool_height[excitation_idxs[0]]
            tool_height_in_last_excitation[ :excitation_idxs[0] ] = 0
            for i in range(0,excitation_idxs.shape[0]-1,1):

                # #print(f"\tDBG: node {point.node} | excitation {i}, idx {excitation_idxs[i]} -- excitation {i+1}, idx {excitation_idxs[i+1]} ")
                # #print(f"\tDBG: node {point.node} | tool_height[idx_i] {tool_height[ excitation_idxs[i] ]} ")
                idxs_to_look_at = [j for j in  range(excitation_idxs[i] - robustness_offset,excitation_idxs[i] + robustness_offset + 1) if j>=0]
                tool_height_in_last_excitation[ excitation_idxs[i]:excitation_idxs[i+1] ] = np.min( tool_height[idxs_to_look_at])

            tool_height_in_last_excitation[ excitation_idxs[-1]: ] = tool_height[ excitation_idxs[-1] ]
            delays = self.calculateDelay( delay_model, tool_height_in_last_excitation, self.statuses[::self.subsample]) 
            # #print(f"\tDBG: node {point.node} | last excitation {excitation_idxs[-1]} ")
            # temperature timeseries for the neighbors of the current node
            if T is None:
                neighboring_nodes_T_ts = [self.Points[neighb_idx].T_t_use[::self.subsample] for neighb_idx in point.neighbor_nodes]
                point_T_t = point.T_t_use[::self.subsample] 
                first_idx = - len(point_T_t)
                idxs_of_absolute_frame = np.asarray( [i for i in range( len(point.T_t_use[::self.subsample]) + first_idx, len(point.T_t_use[::self.subsample]), 1)], dtype= np.int64 )
                ambient = self.T_ambient[:len(point_T_t)]

                # relevance = point.relevance_coef[::self.subsample]
            else:
                if T > len(point.T_t_use):
                    first_idx = - len(point.T_t_use)
                    relative_position = 0 # gives relative current position of the current frame wrt to the subsampled index frame  
                else:
                    first_idx = -T
                    relative_position = timestep 
                
                neighboring_nodes_T_ts = [np.vstack(self.Points[neighb_idx].T_t_use[first_idx:]).squeeze() for neighb_idx in point.neighbor_nodes]
                point_T_t = np.atleast_1d( np.vstack( point.T_t_use[first_idx:] ).squeeze() )
                idxs_of_absolute_frame = np.asarray( [i for i in range( len(point.T_t_use) + first_idx, len(point.T_t_use), 1)], dtype= np.int64 )
                # ambient = self.T_ambient[::self.subsample] [first_idx : first_idx + len(point_T_t)]

            # calculate delay and its constraints
            tool_height  = tool_height[ idxs_of_absolute_frame ] # first_idx is negative
            delays = delays[ idxs_of_absolute_frame ]
            tool_height_excited = tool_height_in_last_excitation[ idxs_of_absolute_frame ]
            ambient = ambient[idxs_of_absolute_frame]
            delayed_relevance_idxs = self.formDelayedIdxs(idxs_of_absolute_frame, delays)
            relevance_d = relevance[ delayed_relevance_idxs ] # first_idx is negative
            
            # def correctDelayedRelevance(relevance_d):
            #     '''
            #     check for high plateaus in relevance and correct them.
            #     '''
            #     max_rel = np.max(relevance_d)
            #     peaks, plateaus = find_peaks(relevance_d, height = 0.1*max_rel , plateau_size = 2)

            #     for i in range(len(plateaus["plateau_sizes"])):
            #         #print(plateaus["plateau_sizes"][i])
            #         relevance_d[ plateaus["left_edges"][i] : plateaus["right_edges"][i]+1 ] = 0
            #         relevance_d[ plateaus["right_edges"][i] ] = 0

            #     return relevance_d
            
            # relevance_d = correctDelayedRelevance(relevance_d)

            # def debugIndexing():
            #     f = plt.figure(figsize = (16,9))
            #     plt.plot(point_T_t[:], label = 'Thermal history')
            #     plt.plot(relevance[:], linestyle = '-.', alpha = 0.75, label = 'relevance')
            #     plt.plot(relevance_d[:], linestyle = '-.', alpha = 0.75, label = 'Delayed relevance')
            #     # plt.plot(self.Time[::self.subsample][idxs_of_absolute_frame],point_T_t, label = 'Thermal history')
            #     # plt.plot(self.Time[::self.subsample][idxs_of_absolute_frame],relevance, linestyle = '-.', alpha = 0.75, label = 'Delayed relevance')
            #     plt.title(f"DBG indexing node {point.node}")
            #     plt.xlabel("samples")
            #     plt.legend()
            #     plt.savefig("./results/general_dbging/" + f"indexing_node_{point.node}")
            #     plt.close(f)

            #     f = plt.figure(figsize = (16,9))
            #     plt.plot(delays[:], label = 'Delays')
            #     plt.plot(tool_height[:]*max(delays), linestyle = '-.', alpha = 0.75, label = 'tool height')
            #     plt.plot(tool_height_in_last_excitation[:]*max(delays), linestyle = '--', alpha = 0.75, label = 'tool height excitaiton')
            #     plt.title(f"DBG delays node {point.node}")
            #     plt.xlabel("samples")
            #     plt.legend()
            #     plt.savefig("./results/general_dbging/" + f"Delays_node_{point.node}")
            #     plt.close(f)

            #     f = plt.figure(figsize = (16,9))
            #     plt.plot(1000*relevance[:] + 0, linestyle = '-.', alpha = 0.75, label = 'relevance')
            #     plt.plot(1000*relevance_d[:] + 0, linestyle = '-.', alpha = 0.75, label = 'Delayed relevance')
            #     plt.plot(delayed_relevance_idxs[:], label = 'delayed_relevance_idxs')
            #     plt.plot(idxs_of_absolute_frame[:], linestyle = '-.', alpha = 0.75, label = 'idxs_of_absolute_frame')
            #     plt.title(f"DBG delays node {point.node}")
            #     plt.xlabel("samples")
            #     plt.legend()
            #     plt.savefig("./results/general_dbging/" + f"idxs_{point.node}")
            #     plt.close(f)

            # def debugIndexingOnline():
            #     f = plt.figure(figsize = (16,9))
            #     plt.plot(point_T_t[:], label = 'Thermal history')
            #     plt.plot(relevance[:], linestyle = '-.', alpha = 0.75, label = 'relevance')
            #     plt.plot(relevance_d[:], linestyle = '-.', alpha = 0.75, label = 'Delayed relevance')
            #     # plt.plot(self.Time[::self.subsample][idxs_of_absolute_frame],point_T_t, label = 'Thermal history')
            #     # plt.plot(self.Time[::self.subsample][idxs_of_absolute_frame],relevance, linestyle = '-.', alpha = 0.75, label = 'Delayed relevance')
            #     plt.title(f"DBG indexing node {point.node}")
            #     plt.xlabel("samples")
            #     plt.legend()
            #     plt.savefig("./results/general_dbging/" + f"indexing_node_{point.node}")
            #     plt.close(f)

            #     f = plt.figure(figsize = (16,9))
            #     plt.plot(delays[:], label = 'Delays')
            #     plt.plot(tool_height[:]*max(delays), linestyle = '-.', alpha = 0.75, label = 'tool height')
            #     plt.plot(tool_height_in_last_excitation[:]*max(delays), linestyle = '--', alpha = 0.75, label = 'tool height excitaiton')
            #     plt.title(f"DBG delays node {point.node}")
            #     plt.xlabel("samples")
            #     plt.legend()
            #     plt.savefig("./results/general_dbging/" + f"Delays_node_{point.node}")
            #     plt.close(f)

            #     f = plt.figure(figsize = (16,9))
            #     plt.plot(1000*relevance[:] + 0, linestyle = '-.', alpha = 0.75, label = 'relevance')
            #     plt.plot(1000*relevance_d[:] + 0, linestyle = '-.', alpha = 0.75, label = 'Delayed relevance')
            #     plt.plot(delayed_relevance_idxs[:], label = 'delayed_relevance_idxs')
            #     plt.plot(idxs_of_absolute_frame[:], linestyle = '-.', alpha = 0.75, label = 'idxs_of_absolute_frame')
            #     plt.title(f"DBG delays node {point.node}")
            #     plt.xlabel("samples")
            #     plt.legend()
            #     plt.savefig("./results/general_dbging/" + f"idxs_{point.node}")
            #     plt.close(f)

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
                point._setDeltaT(np.vstack(( point_T_t, np.vstack(nodes_with_lags))))
                tmp = np.vstack(( point_T_t, np.vstack(nodes_with_lags)))
                # create the delta T for the boundary conditions
                point_delta_T_ambient = []
                
                point_delta_T_ambient = [point_T_t - ambient]*len(point.hallucinated_nodes)
                # point_delta_T_ambient = [point_T_t - self.T_ambient[::self.subsample]]*len(point.hallucinated_nodes)
                point_delta_T_ambient += [np.zeros_like(point_T_t)]*(self.max_hallucinated_nodes - len(point.hallucinated_nodes))
                
                point._setDeltaTambient(np.vstack(point_delta_T_ambient))
                tmp = np.vstack(point_delta_T_ambient)
                # create Input Features
                IF = []
                # IF.append(relevance/(tool_height + 1)) # introduce height component
                # IF.append(relevance_d * tool_height ) # introduce height component
                IF.append(relevance_d * tool_height_excited ) # introduce height component
                IF.append(relevance_d) # introduce energy input component
                # IF.append(tool_height) # introduce energy input component

                point._setInputRawFeatures(np.vstack(IF))
   
            else:
                point._setDeltaT(None)
                point._setDeltaTambient(None)
                point._setInputRawFeatures(None)

        # now subsample all points to use. If you do beforehand then you double subsample stuff
        if not self.evaluation_mode_: 
            for point in self.Points:
                point._set_T_t_use_node(point.T_t_use[::self.subsample])
    
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
        
        on_boundary_,on_corner_ = onopt.pointsOnBoundaries_(self)
        self.Points = onopt.statesForBoundaries( self.Points ,on_boundary_ )

    @staticmethod
    def excitationsPerLayer(zs):
        '''
        Calculate how many times you aim to cross the node on each layer
        @params zs : array of torch height on each crossing
        @returns array of the same length as zs with number of excitations on the corresponding layer
        '''
        all_values, inverse_idxs, counts = np.unique(zs, return_inverse = True, return_counts = True)
        excitation_array = counts[inverse_idxs]
        return excitation_array

    @staticmethod
    def excitationsPerLayerSoFar(zs):
        '''
        Calculate how many times you have crossed already the node on each layer
        @params zs : array of torch height on each crossing
        @returns array of the same length as zs with number of excitations on the corresponding layer so far
        '''
        times_crossed = [0]
        for i in range(1,len(zs),1):
            times_crossed.append( np.sum( zs[:i] == zs[i] ) )
        return np.asarray(times_crossed)

    def initializeTemperaturePropagation(self,scaler_func, T, k, J, idx = 0, delay_model = None):
        '''
        Initializing and returning temperatures from p.T_t[idx]
        '''
        if not self.T:
            self.T = T
            self.k = k
            self.J = J
        self.evaluation_mode_ = True
        self.timestep = 0

        points_with_neighbors = [p for p in self.Points if p._hasNeighbors_()]
        temps = []
        for p in points_with_neighbors:
            unnormalized_temp = p.T_t[::self.subsample][:idx+1]
            temp = scaler_func.transform( np.array( [unnormalized_temp, np.zeros_like(unnormalized_temp)]).T )[:,0]
            # temp = scaler_func.transform( np.array( [p.T_t[idx], 0])[:,None].T )[0,0]
            # p.T_t_use = [np.array(temp)[:,None]] 
            p.T_t_use = np.split(np.array(temp),len(temp)) # create multiple 1d arrays
            temps.append( temp )

                # calculate next features
        self.formRawInputs(T = self.T, timestep = idx , delay_model = delay_model ) #timestep not used yet
        if self.conv_filters is None:
            self.conv_filters = hankel.init_hankelfilter(shape = (self.T,1,1,self.k))

        out = []
        for p in points_with_neighbors:
            out.append(self.poolFeatures(p))
        
        # out = pool.map(self.poolFeatures,points_with_neighbors)
        for (data,p) in zip(out,points_with_neighbors):
            # p.T_t_use = [p.T_t_use]
            p._setFeatures(data[0])
            p._setFeaturesInputs(data[1])
        
        return np.vstack(temps)[:,-1]
    
    def propagateTemperatureFieldMultithread(self, models, timestep, pool, delay_model = None):

        self.timestep = timestep 

        next_temps = []
        # calculate next temperature for each node
        points_with_neighbors = [p for p in self.Points if p._hasNeighbors_()]
        for (p,m) in zip( points_with_neighbors , models ):
            
            # T_t_use = p.T_t_use
            dynamics = fromDynVecToDynamics(m,self.k).T
            update = dynamics.dot(p.features[:,-1])
            next_temps.append(update)
            p.T_t_use += [ update ] 

            
        # calculate next features
        self.formRawInputs(T = self.T, timestep = timestep, delay_model = delay_model ) #timestep not used yet
        out = []
        for p in points_with_neighbors:
            out.append( self.poolFeatures(p) )
        
        # out = pool.map(self.poolFeatures,points_with_neighbors)
        for (data,p) in zip(out,points_with_neighbors):
            p._setFeatures(data[0])
            p._setFeaturesInputs(data[1])

        return np.hstack(next_temps)
    
    def propagateTemperatureFieldDTMultithread(self, models, timestep, pool, delay_model = None):

        self.timestep = timestep 

        next_temps = []
        # calculate next temperature for each node
        points_with_neighbors = [p for p in self.Points if p._hasNeighbors_()]
        for (p,m) in zip( points_with_neighbors , models ):
            
            # T_t_use = p.T_t_use
            dynamics = fromDynVecToDynamics(m,self.k).T
            update = dynamics.dot(p.features[:,-1]) + p.T_t_use[-1]
            next_temps.append(update)
            p.T_t_use += [ update ] 

            
        # calculate next features
        self.formRawInputs(T = self.T, timestep = timestep, delay_model = delay_model ) #timestep not used yet
        out = []
        for p in points_with_neighbors:
            out.append( self.poolFeatures(p) )
        
        # out = pool.map(self.poolFeatures,points_with_neighbors)
        for (data,p) in zip(out,points_with_neighbors):
            p._setFeatures(data[0])
            p._setFeaturesInputs(data[1])

        return np.hstack(next_temps)

    def poolFeatures(self,p):

        # check if you will work with windows or with the whole array
        if self.evaluation_mode_:
            # first_index = min( len( p.T_t_use[0] ) , self.T) # len for lists
            # first_index = min( p.T_t_use.shape[0] , self.T) 
            if len(p.T_t_use) < self.T:
            # if p.T_t_use.shape[0] < self.T:
                first_index = 0
            else:
                first_index = -self.T
            # last_index = self.timestep + 1
        else:
            first_index =  - p.T_t_use.shape[0]
            last_index = -1
        last_index = -1
        # create an array with the convolutions of each DeltaT_i
        # first 
        # #print(f"\tProcessing node {p.node}")
        # if np.any(p.Delta_T):
        if p._hasNeighbors_():
            # compute features
            point_convolved_features_timeseries = []
            for DT_i in p.Delta_T:                
                #create windows for the convolutions
                # convolution_window = rolling_window(DT_i[  first_index:last_index ],self.T)
                convolution_window = rolling_window(DT_i[  first_index: ],self.T)
                # #print("conv window [0,-5:] : ",convolution_window[0,-5:])
                # #print("conv window [1,-5:] : ",convolution_window[1,-5:])
                # #print("conv window [2,-5:] : ",convolution_window[2,-5:])
                # convolved_features = np.dot(self.conv_filters.T,convolution_window.T) 
                convolved_features = np.matmul( self.conv_filters.T,convolution_window[:-1,:].T )
                # point_convolved_features_timeseries.append(convolved_features[:,:-1]) # discard lat element (I think it's pad)
                point_convolved_features_timeseries.append(convolved_features) # discard lat element (I think it's pad)
            
            # point_features_nodes = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T[ first_index:last_index ]))
            point_features_nodes = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T[ : , first_index: ]))
            # p._setFeatures(point_features)

            # compute boundary features
            point_convolved_features_timeseries = []
            for DTamb_i in p.Delta_T_ambient:
                # convolution_window = rolling_window(DTamb_i[ first_index:last_index ],self.T)
                convolution_window = rolling_window(DTamb_i[ first_index: ],self.T)
                # #print("conv window [0,-5:] : ",convolution_window[0,-5:])
                # #print("conv window [1,-5:] : ",convolution_window[1,-5:])
                # #print("conv window [2,-5:] : ",convolution_window[2,-5:])
                # convolved_features = np.dot(self.conv_filters.T,convolution_window.T) 
                convolved_features = np.matmul( self.conv_filters.T,convolution_window[:-1,:].T ) 
                # point_convolved_features_timeseries.append(convolved_features[:,:-1]) # discard lat element (I think it's pad)
                point_convolved_features_timeseries.append(convolved_features) # discard lat element (I think it's pad)
            
            # point_features_boundaries = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T_ambient[ first_index:last_index ])) 
            point_features_boundaries = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T_ambient[ :, first_index: ])) 
            # p._setDeltaTambient(point_features_boundaries)

            all_features = np.vstack((point_features_nodes,point_features_boundaries)) 
            # p._setFeatures(all_features)
            
            # compute input features
            point_convolved_features_timeseries = []
            if  len(p.input_raw_features)>0:
                for IF_i in p.input_raw_features:
                    # convolution_window = rolling_window(IF_i[ first_index:last_index ],self.T)
                    convolution_window = rolling_window(IF_i[ first_index: ],self.T)
                    # #print("conv window [0,-5:] : ",convolution_window[0,-5:])
                    # #print("conv window [1,-5:] : ",convolution_window[1,-5:])
                    # #print("conv window [2,-5:] : ",convolution_window[2,-5:])
                    # convolved_features = np.dot(self.conv_filters.T,convolution_window.T) 
                    convolved_features = np.matmul( self.conv_filters.T,convolution_window[:-1,:].T ) 
                    # point_convolved_features_timeseries.append(convolved_features[:,:-1]) # discard lat element (I think it's pad) 
                    point_convolved_features_timeseries.append(convolved_features) # discard lat element (I think it's pad) 
                
                # point_input_features = np.vstack((np.vstack(point_convolved_features_timeseries),p.input_raw_features[ first_index:last_index ])) 
                point_input_features = np.vstack((np.vstack(point_convolved_features_timeseries),p.input_raw_features[ :, first_index: ])) 
               
                all_features = np.vstack((all_features,point_input_features))

            # compute input features
            lengthscale = norm(p.coordinates-p.neighbor_nodes_coordinates[0])
            sqrt2 = np.sqrt(2)

            # find the relevance coefficients for every excitation 
            relevance_coefs = []
            energy_input = []
            # use only the indexes that agree with 
            ps_to_use = p.excitation_idxs[:len(p.input_temperatures)] # truncate peaks not visible in the usable data
            for idx in ps_to_use:
                
                d_i_xy = []
                energy_temp = 0
                # explore previous and next points' contribution to local power excitation
                for i in range(-self.J,self.J+1,1):
                    d_i_xy.append( np.exp(-((self.trajectory[idx+i,0] - p.coordinates[0])/lengthscale)**2/sqrt2) *
                                np.exp(-((self.trajectory[idx+i,1] - p.coordinates[1])/lengthscale)**2/sqrt2) * self.statuses[idx+i] )
                    energy_temp += d_i_xy[-1]/self.J

                energy_input.append(energy_temp)
                relevance_coefs.append(d_i_xy)
            
            zs = np.copy(ps_to_use) # height of welding tool)
            n = self.excitationsPerLayer(zs) # number of node crossings per layer 
            n_so_far = self.excitationsPerLayerSoFar(zs) # number of node crossings per layer so far
            
            # when welding tool crossed the node 
            # ts = p.excitation_times 
            ts = p.excitation_times[:len(p.input_temperatures)] # truncate peaks not visible in the usable data 
            delta_ts = [500] + list(np.diff(ts)) # difference in seconds between two subsequent crossings
            # delta_ts = list(np.max(np.diff(ts))) + list(np.diff(ts)) # difference in seconds between two subsequent crossings
            crossed_node = []
            for excitation_idx in ps_to_use:
                # check if the torch crosses the node or if it starts/stops there
                # #print("+1 ",self.statuses[excitation_idx+1])
                # #print("-1 ",self.statuses[excitation_idx-1])
                if self.statuses[excitation_idx+1] == 0 or self.statuses[excitation_idx-1] == 0:
                    crossed_node.append(False)
                else:
                    crossed_node.append(True)

            if zs.size>0:
                input_features = np.vstack((zs,delta_ts,crossed_node,energy_input,n,n_so_far))
            else:
                input_features = None
            # p._setFeaturesInputs(input_features)
        else:
            # p._setFeatures(None)
            # p._setFeaturesInputs(None)
            input_features = None
            all_features = None

        out = [all_features,input_features]
        # #print(f"DEBUG: exited node {p.node}")
        return out

    def poolActivatedFeatures(self,p):

        # check if you will work with windows or with the whole array
        if self.evaluation_mode_:
            # first_index = min( len( p.T_t_use[0] ) , self.T) # len for lists
            # first_index = min( p.T_t_use.shape[0] , self.T) 
            if len(p.T_t_use) < self.T:
            # if p.T_t_use.shape[0] < self.T:
                first_index = 0
            else:
                first_index = -self.T
            # last_index = self.timestep + 1
        else:
            first_index =  - p.T_t_use.shape[0]
            last_index = -1
        last_index = -1
        # create an array with the convolutions of each DeltaT_i
        # first 
        # #print(f"\tProcessing node {p.node}")
        # if np.any(p.Delta_T):
        if p._hasNeighbors_():

            # introduce feature relevance 
            # ip_relevance = p.relevance_coef[::self.subsample][ len(p.T_t_use) + first_index : len(p.T_t_use) ] # first_idx is negative
            
            # compute features
            point_convolved_features_timeseries = []
            for DT_i in p.Delta_T:                
                #create windows for the convolutions
                # convolution_window = rolling_window(DT_i[  first_index:last_index ],self.T)
                convolution_window = rolling_window(DT_i[  first_index: ],self.T)
                # #print("conv window [0,-5:] : ",convolution_window[0,-5:])
                # #print("conv window [1,-5:] : ",convolution_window[1,-5:])
                # #print("conv window [2,-5:] : ",convolution_window[2,-5:])
                # convolved_features = np.dot(self.conv_filters.T,convolution_window.T) 
                convolved_features = np.matmul( self.conv_filters.T,convolution_window[:-1,:].T )
                # point_convolved_features_timeseries.append(convolved_features[:,:-1]) # discard lat element (I think it's pad)
                point_convolved_features_timeseries.append(convolved_features) # discard lat element (I think it's pad)
            
            # point_features_nodes = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T[ first_index:last_index ]))
            point_features_nodes = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T[ : , first_index: ])) 
            # p._setFeatures(point_features)

            # compute boundary features
            point_convolved_features_timeseries = []
            for DTamb_i in p.Delta_T_ambient:
                # convolution_window = rolling_window(DTamb_i[ first_index:last_index ],self.T)
                convolution_window = rolling_window(DTamb_i[ first_index: ],self.T)
                # #print("conv window [0,-5:] : ",convolution_window[0,-5:])
                # #print("conv window [1,-5:] : ",convolution_window[1,-5:])
                # #print("conv window [2,-5:] : ",convolution_window[2,-5:])
                # convolved_features = np.dot(self.conv_filters.T,convolution_window.T) 
                convolved_features = np.matmul( self.conv_filters.T,convolution_window[:-1,:].T ) 
                # point_convolved_features_timeseries.append(convolved_features[:,:-1]) # discard lat element (I think it's pad)
                point_convolved_features_timeseries.append(convolved_features) # discard lat element (I think it's pad)
            
            # point_features_boundaries = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T_ambient[ first_index:last_index ])) 
            point_features_boundaries = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T_ambient[ :, first_index: ])) 
            # p._setDeltaTambient(point_features_boundaries)

            all_features = np.vstack((point_features_nodes,point_features_boundaries)) 
            # p._setFeatures(all_features)
            
            # compute input features
            point_convolved_features_timeseries = []
            if  len(p.input_raw_features)>0:
                for IF_i in p.input_raw_features:
                    # convolution_window = rolling_window(IF_i[ first_index:last_index ],self.T)
                    convolution_window = rolling_window(IF_i[ first_index: ],self.T)
                    # #print("conv window [0,-5:] : ",convolution_window[0,-5:])
                    # #print("conv window [1,-5:] : ",convolution_window[1,-5:])
                    # #print("conv window [2,-5:] : ",convolution_window[2,-5:])
                    # convolved_features = np.dot(self.conv_filters.T,convolution_window.T) 
                    convolved_features = np.matmul( self.conv_filters.T,convolution_window[:-1,:].T ) 
                    # point_convolved_features_timeseries.append(convolved_features[:,:-1]) # discard lat element (I think it's pad) 
                    point_convolved_features_timeseries.append(convolved_features) # discard lat element (I think it's pad) 
                
                # point_input_features = np.vstack((np.vstack(point_convolved_features_timeseries),p.input_raw_features[ first_index:last_index ])) 
                point_input_features = np.vstack((np.vstack(point_convolved_features_timeseries),p.input_raw_features[ :, first_index: ])) 
               
                all_features = np.vstack((all_features,point_input_features))

            # compute input features
            lengthscale = norm(p.coordinates-p.neighbor_nodes_coordinates[0])
            sqrt2 = np.sqrt(2)

            # find the relevance coefficients for every excitation 
            relevance_coefs = []
            energy_input = []
            # use only the indexes that agree with 
            ps_to_use = p.excitation_idxs[:len(p.input_temperatures)] # truncate peaks not visible in the usable data
            for idx in ps_to_use:
                
                d_i_xy = []
                energy_temp = 0
                # explore previous and next points' contribution to local power excitation
                for i in range(-self.J,self.J+1,1):
                    d_i_xy.append( np.exp(-((self.trajectory[idx+i,0] - p.coordinates[0])/lengthscale)**2/sqrt2) *
                                np.exp(-((self.trajectory[idx+i,1] - p.coordinates[1])/lengthscale)**2/sqrt2) * self.statuses[idx+i] )
                    energy_temp += d_i_xy[-1]/self.J

                energy_input.append(energy_temp)
                relevance_coefs.append(d_i_xy)
            
            zs = self.tool_deposition_depth[ps_to_use] # height of welding tool
            n = self.excitationsPerLayer(zs) # number of node crossings per layer 
            n_so_far = self.excitationsPerLayerSoFar(zs) # number of node crossings per layer so far
            
            # when welding tool crossed the node 
            # ts = p.excitation_times 
            ts = p.excitation_times[:len(p.input_temperatures)] # truncate peaks not visible in the usable data 
            delta_ts = [500] + list(np.diff(ts)) # difference in seconds between two subsequent crossings
            # delta_ts = list(np.max(np.diff(ts))) + list(np.diff(ts)) # difference in seconds between two subsequent crossings
            crossed_node = []
            for excitation_idx in ps_to_use:
                # check if the torch crosses the node or if it starts/stops there
                # #print("+1 ",self.statuses[excitation_idx+1])
                # #print("-1 ",self.statuses[excitation_idx-1])
                if self.statuses[excitation_idx+1] == 0 or self.statuses[excitation_idx-1] == 0:
                    crossed_node.append(False)
                else:
                    crossed_node.append(True)

            if zs.size>0:
                input_features = np.vstack((zs,delta_ts,crossed_node,energy_input,n,n_so_far))
            else:
                input_features = None
            # p._setFeaturesInputs(input_features)
        else:
            # p._setFeatures(None)
            # p._setFeaturesInputs(None)
            input_features = None
            all_features = None

        out = [all_features,input_features]
        # #print(f"DEBUG: exited node {p.node}")
        return out

    def formFeatureVectorsMultithreadWrapper(self,k,T, J = 2):

        self.conv_filters = hankel.init_hankelfilter(shape = (T,1,1,k))
        self.T = T
        self.k = k
        self.J = J

        points_with_neighbors = [p for p in self.Points if p.neighbor_nodes]
        with mp.Pool(processes = mp.cpu_count()) as pool:
            out = pool.map(self.poolFeatures,points_with_neighbors)
            
            # pool.imap_unordered(self.poolFeatures,self.Points)
        for (data,p) in zip(out,points_with_neighbors):
            p._setFeatures(data[0])
            p._setFeaturesInputs(data[1])

    def formActivatedFeatureVectorsMultithreadWrapper(self,k,T, J = 2):

        self.conv_filters = hankel.init_hankelfilter(shape = (T,1,1,k))
        self.T = T
        self.k = k
        self.J = J

        points_with_neighbors = [p for p in self.Points if p.neighbor_nodes]
        with mp.Pool(processes = mp.cpu_count()) as pool:
            out = pool.map(self.poolActivatedFeatures,points_with_neighbors)
            
            # pool.imap_unordered(self.poolFeatures,self.Points)
        for (data,p) in zip(out,points_with_neighbors):
            p._setFeatures(data[0])
            p._setFeaturesInputs(data[1])

    def formFeatureVectorsWrapper(self,k,T, J = 2):

        self.conv_filters = hankel.init_hankelfilter(shape = (T,1,1,k))
        self.T = T
        self.k = k
        self.J = J

        points_with_neighbors = [p for p in self.Points if p.neighbor_nodes]
        for p in points_with_neighbors:

            out = self.poolFeatures(p)  
            p._setFeatures(out[0])
            p._setFeaturesInputs(out[1])
                
    def formActivatedFeatureVectorsWrapper(self,k,T, J = 2):

        self.conv_filters = hankel.init_hankelfilter(shape = (T,1,1,k))
        self.T = T
        self.k = k
        self.J = J

        points_with_neighbors = [p for p in self.Points if p.neighbor_nodes]
        for p in points_with_neighbors:

            out = self.poolActivatedFeatures(p)  
            p._setFeatures(out[0])
            p._setFeaturesInputs(out[1])
                
    def formFeatureVectors(self,k,T,J = 2):
        """
        Obsolette: consider changing with formFeatureVectorWrapper
        k : Number of Filters
        T : Number of lags for convolutions
        J : Number of samples for path 
        """
        points_with_neighbors = [p for p in self.Points if p.neighbor_nodes]
        # first 
        #print(f"DEBUG: entered node { points_with_neighbors[-1].node }")
        self.conv_filters = hankel.init_hankelfilter(shape = (T,1,1,k))
        for p in points_with_neighbors:
            # create an array with the convolutions of each DeltaT_i

            if np.any(p.Delta_T):
                # compute features
                point_convolved_features_timeseries = []
                for DT_i in p.Delta_T:                
                    #create windows for the convolutions
                    convolution_window = rolling_window(DT_i,T)
                    # #print("conv window [0,-5:] : ",convolution_window[0,-5:])
                    # #print("conv window [1,-5:] : ",convolution_window[1,-5:])
                    # #print("conv window [2,-5:] : ",convolution_window[2,-5:])
                    # convolved_features = np.dot(self.conv_filters.T,convolution_window.T) 
                    convolved_features = np.matmul( self.conv_filters.T,convolution_window.T )
                    point_convolved_features_timeseries.append(convolved_features[:,:-1]) # discard lat element (I think it's pad)
                
                point_features_nodes = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T))
                # p._setFeatures(point_features)

                # # compute features for boundary models
                # compute boundary features
                point_convolved_features_timeseries = []
                for DTamb_i in p.Delta_T_ambient:
                    convolution_window = rolling_window(DTamb_i,T)
                    # #print("conv window [0,-5:] : ",convolution_window[0,-5:])
                    # #print("conv window [1,-5:] : ",convolution_window[1,-5:])
                    # #print("conv window [2,-5:] : ",convolution_window[2,-5:])
                    convolved_features = np.matmul( self.conv_filters.T,convolution_window.T ) 
                    point_convolved_features_timeseries.append(convolved_features[:,:-1]) # discard lat element (I think it's pad)
                
                point_features_boundaries = np.vstack((np.vstack(point_convolved_features_timeseries),p.Delta_T_ambient)) 
                p._setDeltaTambient(point_features_boundaries)
                all_features = np.vstack((point_features_nodes,point_features_boundaries)) 
                p._setFeatures(all_features)

                # compute input features
                lengthscale = norm(p.coordinates-p.neighbor_nodes_coordinates[0])
                sqrt2 = np.sqrt(2)

                # find the relevance coefficients for every excitation 
                relevance_coefs = []
                energy_input = []

                # use only the indexes that agree with 
                ps_to_use = p.excitation_idxs[:len(p.input_temperatures)] # truncate peaks not visible in the usable data
                for idx in ps_to_use:
                    
                    d_i_xy = []
                    energy_temp = 0
                    # explore previous and next points' contribution to local power excitation
                    for i in range(-J,J+1,1):
                        d_i_xy.append( np.exp(-((self.trajectory[idx+i,0] - p.coordinates[0])/lengthscale)**2/sqrt2) *
                                    np.exp(-((self.trajectory[idx+i,1] - p.coordinates[1])/lengthscale)**2/sqrt2) * self.statuses[idx+i] )
                        energy_temp += d_i_xy[-1]/J

                    energy_input.append(energy_temp)
                    relevance_coefs.append(d_i_xy)
                
                zs = self.tool_deposition_depth[ps_to_use] # height of welding tool
                n = self.excitationsPerLayer(zs) # number of node crossings per layer 
                n_so_far = self.excitationsPerLayerSoFar(zs) # number of node crossings per layer so far
                
                # when welding tool crossed the node 
                # ts = p.excitation_times 
                ts = p.excitation_times[:len(p.input_temperatures)] # truncate peaks not visible in the usable data 
                delta_ts = [500] + list(np.diff(ts)) # difference in seconds between two subsequent crossings
                # delta_ts = list(np.max(np.diff(ts))) + list(np.diff(ts)) # difference in seconds between two subsequent crossings
                crossed_node = []
                for excitation_idx in ps_to_use:
                    # check if the torch crosses the node or if it starts/stops there
                    # #print("+1 ",self.statuses[excitation_idx+1])
                    # #print("-1 ",self.statuses[excitation_idx-1])
                    if self.statuses[excitation_idx+1] == 0 or self.statuses[excitation_idx-1] == 0:
                        crossed_node.append(False)
                    else:
                        crossed_node.append(True)

                if zs.size>0:
                    input_features = np.vstack((zs,delta_ts,crossed_node,energy_input,n,n_so_far))
                    # relevance_coefs = np.hstack(relevance_coefs)
                else:
                    input_features = None
                    # relevance_coefs = None
                p._setFeaturesInputs(input_features)
                # p._setTrajectoryRelevanceCoefs(relevance_coefs)
            else:
                p._setFeatures(None)
                p._setFeaturesInputs(None)

    def hocusPocusMultithreadWrapper(self,inputs):
        try:
            node = inputs[0]
            # #print("\tprocessing node ", node)
            kwargs = inputs[1]
            return self.findNodeActuationIntervalsFromPeaks(node, **kwargs)    
        except Exception as e:
            #print(e)
            return e
   
    def findNodeActuationIntervalsFromPeaks(self, node,
                                                  coef_subdmn_length_1 = 0.95, 
                                                  coef_subdmn_length_2 = 0.6, 
                                                  dT_thress = 0.2,  # tolerance for jittering
                                                  peak_height = 400, # temperature thresshold value for finding peaks  
                                                  peak_distance = 20, # time thresshold value for finding peaks
                                                  half_local_window_length = 5 ,
                                                  plateau_size = 1,
                                                  prominence = 1,
                                                  relative_prominence_thress = 0,
                                                  peaks_to_keep = None,
                                                  save_folder = "../results/plots/ ",
                                                  long_short_interval_coef = 1000,
                                                  save_ = False
                                                  ): 

        """
        Hocus Pocus method for defining which parts of T_t are caused by external excitation. Check for peaks in a particular node and then manually tune what part of the peak is considered an input.
        @params node : scalar indicating node for index to use
        @params coef_subdmn_length_1 : what part of the indentified subdomain should be considered? (more than long_short_interval_coef timesteps)
        @params coef_subdmn_length_2 : what part of the indentified subdomain should be considered? (less than or long_short_interval_coef timesteps)
        @params peak_height : scalar indicating a temperature thresshold value for finding peaks  
        @params peak_distance : scalar indicating a time thresshold value for specifying distances between peaks
        @params half_local_window_length : half the window length around an identified peak. Used for discarding valleys from hills.  
        @params long_short_interval_coef : elements in interval to be considered long.  
        """
        output_folder = save_folder + f"Hocus_Pocus_plots_{self.Parent_plotter.time}/"
        os.makedirs(output_folder, exist_ok = True)
        # #print(f"node {node}")
        ##################################################################################################################
        if np.any(self.Points[node].excitation_times):
            peaks_to_keep = self.Points[node].excitation_times.shape[0] + 1 # for some reason, paths neglect 1 last excitation
        # #print(f"\tpeaks_to_keep {peaks_to_keep}")
        ##################################################################################################################

        Temp = self.Points[node].T_t_use # subsampled array
        excitation_times = self.Points[node].excitation_times # subsampled array
        excitation_idxs = self.Points[node].excitation_idxs # subsampled array

        # find main peaks - note: T_t_use is subsampled so the indexes you find refer to the subsampled array
        all_peaks,a = find_peaks(Temp, height = peak_height, distance = peak_distance,plateau_size = plateau_size, prominence = prominence ) 
        
        prom = a["prominences"]
        rel_prominences = prom/np.max(prom)
        if not peaks_to_keep:
            peaks_idxs_2 = all_peaks[np.where(rel_prominences>relative_prominence_thress)]
        else:
            peaks_idxs_2 = all_peaks[ (-rel_prominences).argsort()[:peaks_to_keep] ]

        # #print(f"\tlen(peaks_idxs_2) {len(peaks_idxs_2)} vs len(all_peaks) {len(all_peaks)}")
    
        time_peaks_2  = [self.Time[:: self.subsample] [pkidx] for pkidx in peaks_idxs_2] # T_t_use is subsampled
        # time_peaks_2  = [self.Time[pkidx] for pkidx in peaks_idxs_2]
        T_peaks_2 = [Temp[pkidx] for pkidx in peaks_idxs_2]

        # discard valleys
        peaks_neighborhoods = np.asarray([[Temp[pkidx + offset] for offset in range(-half_local_window_length,half_local_window_length,1)] for pkidx in peaks_idxs_2], dtype = "object") 
        second_diff = np.diff(peaks_neighborhoods,n = 2,axis = 1)
        signs = np.sum(np.sign(second_diff),axis=1)
        hills = np.where(signs<=0) # only consider the peaks that (mostly) have negative second der (concave)
        final_peaks_idxs = peaks_idxs_2[hills]
            
        final_peaks_idxs.sort()
        # select only the peaks that lie in the usable part of your data
        final_peaks_idxs_to_use = [idx for idx in final_peaks_idxs if idx<= self.nots_to_use] 
        time_peaks_final  = [self.Time[:: self.subsample] [pkidx] for pkidx in final_peaks_idxs_to_use] # T_t_use is subsampled
        T_peaks_final = [Temp[pkidx] for pkidx in final_peaks_idxs_to_use]

        # find observation delay (Dt for heat to reach part)
        number_of_peaks_available = min( len(time_peaks_final), len(excitation_times))
        # observation_delays = np.asarray( time_peaks_final[:number_of_peaks_available] - excitation_times[:number_of_peaks_available] )
        observation_delays = np.asarray( (final_peaks_idxs_to_use[:number_of_peaks_available] - excitation_idxs[:number_of_peaks_available]/self.subsample), dtype= np.int64 )
        torchHeight_at_excitation = np.asarray( self.tool_deposition_depth[ excitation_idxs[:number_of_peaks_available] ] )

        self.Points[node]._setInputs(T_peaks_final)
        self.Points[node]._setInputTimes(time_peaks_final)
        self.Points[node]._setInputIdxs(final_peaks_idxs_to_use)
        self.Points[node]._setExcitation_delay(observation_delays)
        self.Points[node]._setExcitation_excitationDelayTorchHeight(torchHeight_at_excitation)
        thermal_inputs = {
            "T_peaks_final": T_peaks_final,
            "time_peaks_final": time_peaks_final,
            "final_peaks_idxs": final_peaks_idxs_to_use,
            "observation_delays": observation_delays,
            "torchHeight_at_excitation": torchHeight_at_excitation
            }

        # (time_peaks_final,T_peaks_final,final_peaks_idxs) = sorted((time_peaks_2,T_peaks_2,list(final_peaks_idxs)))

        self.Points[node]._setInputs(T_peaks_final)
        self.Points[node]._setInputTimes(time_peaks_final)
        self.Points[node]._setInputIdxs(final_peaks_idxs)

        # (time_peaks_final,T_peaks_final,final_peaks_idxs) = sorted((time_peaks_2,T_peaks_2,list(final_peaks_idxs)))

        # find all values in peaks (plateaus)

        peak_width_idxs = [] # peak limits
        peaks_idxs = []
        for (peak,idx) in zip(T_peaks_final,final_peaks_idxs):
            #first find all points with the peak value on the left
            k_min = - 1 # idx offset
            while ( abs(Temp[idx + k_min] - peak) < dT_thress ):
                if idx + k_min - 1 >= 0:
                    k_min -= 1
                else:
                    break
            
            k_plus = 1
            while ( abs(Temp[idx + k_plus] - peak) < dT_thress ):
                if idx + k_plus + 1 < Temp.shape[0]:
                    k_plus += 1
                else:
                    break

            peak_width_idxs.append( ( idx+k_min , idx+k_plus ) )
            peaks_idxs.append([idx for idx in range(idx+k_min, idx+k_plus)])

        # find subdomain_idxs with no excitation
        subdomain_idxs = [] # idxs in each subdomain
        for i in range(len(peak_width_idxs)-1):

            peak = peak_width_idxs[i]
            next_peak = peak_width_idxs[i+1]

            start_idx = peak[1]
            next_start_idx = next_peak[0]

            # only include peaks on the usable part of the data
            if start_idx>self.nots_to_use:
                break

            # THIS IS SPECIFIC FOR THE TESTS, it's based on the mtm optimization
            diff = next_start_idx - start_idx
            if diff>long_short_interval_coef:
                final_idx = start_idx + round(diff*coef_subdmn_length_1)
            else:
                final_idx = start_idx + round(diff*coef_subdmn_length_2)

            # trauncate to usable elements
            if final_idx>self.nots_to_use:
                final_idx = self.nots_to_use

            subdomain_idxs.append([idx for idx in range(start_idx, final_idx)])

        # include last part as well
        if final_idx< Temp.shape[0]:
            subdomain_idxs.append([idx for idx in range(peak_width_idxs[-1][1] + round(diff*(1-coef_subdmn_length_1)), Temp.shape[0]-1)]) # refer to subsampled array
        # subdomain_idxs.append([idx for idx in range(peak_width_idxs[-1][1] + round(diff*(1-coef_subdmn_length_1)), len(Temp))])
        # self.Points[node]._setInputs(peaks_idxs)

        self.Points[node]._setSubdomains(subdomain_idxs)

        # plot to see if Hocus Pocus worked
        hocus_pocus_plot1 = plt.figure("Input search for node {}".format(node),figsize = (16,11))
        hocus_pocus_plot1.patch.set_facecolor('white')
        # hocus_pocus_plot1[-1].patch.set_alpha(1)
        hocus_pocus_ax1 = hocus_pocus_plot1.add_subplot(211)
        hocus_pocus_ax1 = self.Parent_plotter.plotNodeTimeEvolution(node, temps = [Temp], times = [self.Time[::self.subsample]], axes = hocus_pocus_ax1, file_numbers = [self.experiment_id], labels = ["T"])
        hocus_pocus_ax1.set_xlabel("")

        plt.scatter(x = time_peaks_2, y = T_peaks_2, marker='x', c ='black')
        plt.scatter(x = time_peaks_final, y = T_peaks_final, marker='x', c ='red')
        hocus_pocus_ax1.legend(["Temperature profile","Valleys","Peaks"])

        for interval in peak_width_idxs:
    
            # t_interval = ( time[interval[0]], time[interval[1]] )
            # T_interval = ( Temp[interval[0]], Temp[interval[1]] )

            t_interval = [self.Time[::self.subsample] [idx] for idx in range(interval[0], interval[1])]
            T_interval = [Temp[idx] for idx in range(interval[0], interval[1])]
            # plt.plot(t_interval,T_interval, '-bo',color = 'red', markersize = 2)
            plt.plot(t_interval,T_interval,color = 'red')

        hocus_pocus_ax1.set_xticklabels([])
        
        hocus_pocus_ax2 = hocus_pocus_plot1.add_subplot(212)
        hocus_pocus_ax2 = self.Parent_plotter.plotNodeTimeEvolution(node, temps = [Temp], times = [self.Time[::self.subsample]], axes = hocus_pocus_ax2, file_numbers = [self.experiment_id], labels = ["Temp"])
        for sub in subdomain_idxs:
                t_interval = np.asarray([self.Time[::self.subsample] [idx] for idx in sub], dtype = "object")
                T_interval = np.asarray([Temp[idx] for idx in sub], dtype = "object")
                # plt.plot(t_interval,T_interval, '-bo',color = 'red', markersize = 2)
                plt.plot(t_interval,T_interval,color = 'orange')
                plt.title("domains to consider in optimization")

        plt.plot( self.Time[::self.subsample], self.Points[node].relevance_coef[::self.subsample], alpha = 0.5, linestyle = '--')
        hocus_pocus_ax2.legend(["Temperature profile","Training domains", "Relevance"])
        if save_:
            plt.savefig( output_folder + "{}.jpg".format(node), bbox_inches='tight',)
        plt.close(hocus_pocus_plot1)

        # plot to see excitation dealy vs height
        hocus_pocus_plot2 = plt.figure("Excitation delay {}".format(node),figsize = (16,11))
        hocus_pocus_plot2.patch.set_facecolor('white')
        # hocus_pocus_plot2[-1].patch.set_alpha(1)
        hocus_pocus_ax21 = hocus_pocus_plot2.add_subplot(111)
        plt.plot( torchHeight_at_excitation, observation_delays, marker = 'x')
        hocus_pocus_ax21.set_title("Observation Delay")
        hocus_pocus_ax21.set_xlabel("Height")
        hocus_pocus_ax21.set_ylabel("Dt")
        if save_:
            plt.savefig( output_folder + "input_delay_{}.jpg".format(node), bbox_inches='tight',)
        plt.close(hocus_pocus_plot2)

        # return subdomain_idxs, peak_width_idxs, peaks_idxs, copy.deepcopy(hocus_pocus_plot1)
        return subdomain_idxs, peak_width_idxs, peaks_idxs, thermal_inputs
        
    def disectTemperatureRegions(self, nodes, T_t_custom = None, bounds = [], element_thresshold = 100):
        """
        Find idxs of domains with similar Temperatures.
        @params node : idx of a node for selecting its temperature.
        @params T_t_custom : array with temperatures. If not None then the node is neglected and the function operates on these data
        @params bounds : itterable with itterables including the domain edges. 
        @params element_thresshold : scalar giving the min number of elements in a domain. 
        If a domain has less elements than the amount that this parameters indicates, concatenate it with another domain.   
        @returns section_idxs : indexes of the designated array
        """
        if not bounds:
            bounds = [(-1e10,1e10)]

        if (T_t_custom is None):
            T_t_all = []
            for node in nodes: 
                T_t_all.append(self.Points[node].T_t_use)
            
            T_t_all = np.hstack(T_t_all)
        else:
            T_t_all = T_t_custom
        
        ## first create boundaries with enough elements and then disect node T_t vectors
        # #print("T_t ",T_t.shape)
        section_idxs = []
        section_lengths = []
        for bound in bounds:
            T_low = bound[0]
            T_high = bound[1]
            section_idxs.append( np.where((T_t_all>=T_low) & (T_t_all<=T_high))[0] )
            # #print("Section ",section_idxs[-1].shape)

            section_lengths.append(max(section_idxs[-1].shape))
        
        # find the sections with less indexes than the thresshold and concatenate them
        while np.any(np.hstack(section_lengths)<element_thresshold):
            for i,sec_len in enumerate(section_lengths):
                if sec_len<element_thresshold:
                    #Check the number of elements on the previous and the next (if exist) section. Concatenate your data to the list with
                    #the smallest number of elements (you want to reduce the inital boundaries as less as possible).
                    if i>0 and i<len(section_idxs)-1:
                        # find number of elements before and after
                        el_prev = (i-1,section_lengths[i-1])
                        el_next = (i+1,section_lengths[i+1])

                        # find index of entry with the least numer of elements
                        argmin = min((el_prev,el_next),key= lambda t: t[1])[0]
                    
                    elif i == 0:
                        argmin = 1
                    
                    else:
                        argmin = len(section_idxs)-2

                    # add the curremt element to the selected interval and delete the entry
                    section_idxs[argmin] = np.concatenate((section_idxs[argmin],section_idxs[i]))
                    # change bounds respectively
                    if argmin - i > 0 :
                        # extend lower bound
                        bounds[argmin] = (bounds[i][0],bounds[argmin][1])
                    else:
                        # extend upper bound
                        bounds[argmin] = (bounds[argmin][0],bounds[i][1])

                    section_idxs[i] = []
                    # section_idxs.remove(section_idxs[i])
                    section_lengths[argmin] += section_lengths[i] 
                    section_lengths[i] = np.inf # don't delete but mark as inf
                    # section_lengths[i] = 0 # don't delete but mark as 0

            #find empty slices and remove them
            new_bounds = []
            new_idxs = []
            new_lengths = []
            for (length,bound,idxs) in zip(section_lengths,bounds,section_idxs):
                if len(idxs) != 0:
                    new_bounds.append(bound)
                    new_idxs.append(idxs.astype(np.int64))
                    new_lengths.append(len(idxs))
            
            bounds = new_bounds
            section_idxs = new_idxs
            section_lengths = new_lengths

        # now that you computed your boundaries, disect T_t arrays
        if (T_t_custom is None):
            
            for node in nodes:
                section_idxs = []
                T_t = self.Points[node].T_t_use
                for bound in bounds:
                    T_low = bound[0]
                    T_high = bound[1]
                    section_idxs.append( np.where((T_t>=T_low) & (T_t<=T_high))[0] )
                self.Points[node]._setTemperatureIdxs(section_idxs)

        return bounds,section_idxs

    def _copy(self):
        return copy.deepcopy(self)   
