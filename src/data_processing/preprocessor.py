#%%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import utils.helper_func as hlp

import glob
import os
# import imageio # gif
# import cv2 # openCV for video
import copy

from data_processing._Experiment import _Experiment
from data_processing._Point import _Point
from datetime import datetime, time
from scipy.spatial import Delaunay
from numpy.linalg import norm
from math import floor
from sklearn.preprocessing import MinMaxScaler as scalingFunction

import torch

# from layers.predefined_filters import hankel
from scipy.optimize import least_squares

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 24}
mpl.rc('font', **font)
mpl.rcParams['lines.markersize'] = 20
# mpl.use('GTK3Cairo')

square_fig = (12,12)
long_fig = (16,9)
# %% [markdown]
# Data loader class.
# Preprocess data and export them in order.

class objective_fun:
    def __init__(self,Xs,Ys,band,nob,systemTransform):
        self.Xs = Xs
        self.Ys = Ys
        self.band = band # temperature band
        self.nob = nob # number of bands
        self.counter = 0
        self.systemTransform = systemTransform

    def __call__(self,dyn_vec):

        self.counter += 1
        # #print("dyn_vec ",dyn_vec.shape)
        dyn_vec_conv = dyn_vec[:-4]
        # #print("dyn_vec_conv ",dyn_vec_conv.shape)
        dynamics = np.hstack((dyn_vec_conv,dyn_vec_conv,dyn_vec_conv,dyn_vec)).T
        dynamics = self.systemTransform(dyn_vec)
        # #print("dynamics ",dynamics.shape)
        # rollout response
        Delta_responses = dynamics.dot(self.Xs).squeeze()
        # #print("Xs ",self.Xs.shape)
        # #print("Delta_resp ",Delta_responses.shape)
        # Delta_responses = Delta_responses[:-1] #discard the last one because you predict the next Delta_y

        # calculate error
        rel_error = (100*(Delta_responses-self.Ys))
        mre = np.mean(rel_error**2/(self.Ys+10))

        # report if needed
        if not (self.counter%50000):
            print("\tModel:{}\\{}\t|\tCalls:\t{}\tcost:\t{:.5f}".format(self.band+1,self.nob,self.counter,mre))

        return mre

class preProcessor:
    def __init__(self, files_folder, point_temperature_files, point_coordinates_files, tool_trajectory_files, results_folder = "./results/", subsample = 1):
        '''
        Data loader class. Preprocess data and export.
        @param files_folder : string with path to the data folder.
        @param point_temperature_files : string with names of files including temperature measurements.
        @param point_coordinates_files : string with names of files including coordinates of points.
        @param tool_trajectory_files : string with names of files including tool trajectory.
        @param files_folder : string with path to the designated result folder.
        '''
        self.FILES_FOLDER = files_folder
        self.POINT_TEMPERATURE_FILES = point_temperature_files
        self.POINT_COORDINATES = point_coordinates_files
        self.TRAJECTORY_FILES = tool_trajectory_files
        self.RESULTS_FOLDER = results_folder
        self.subsample = subsample
        self.plotter = _prePrcossesorPlotter(self)
        self.scaler = None
        self.d_grid = None
        # setattr(self.contourPlot,"_save",0)
        # setattr(self.contourPlot,"_count",0)
        # self._save_ = 0s

    def loadData(self,scaling_ = True, d_grid = None, debug_data = True, deposition_height_of_boundary = 0, deposition_height_of_non_excited = 0):
        '''
        Load files and save them into Experiments format.
        @params scaling_ : whether to scale or not.
        @params d_grid : distance between nodes on grid. If None the min distance found is taken
         
        @returns experiments : List of experiment objects.
        '''
        # get file names in the specified locations
        all_temp_files = sorted(glob.glob( os.path.join( self.FILES_FOLDER, self.POINT_TEMPERATURE_FILES) ))
        all_traj_files = sorted(glob.glob( os.path.join( self.FILES_FOLDER, self.TRAJECTORY_FILES) ))
        point_coordinates_files = glob.glob( os.path.join( self.FILES_FOLDER, self.POINT_COORDINATES) )

        # load temperature and trajectory files
        rep_temp = []
        rep_traj = []
        rep_timestamps = []
        #print("Loading measurements...")
        # for (temp_file,traj_file) in zip(all_temp_files,all_traj_files):
        experiment_ids = []
        for temp_file in all_temp_files:

            try:
                # images_file = pd.read_csv(temp_file, index_col=0, header = 0,dtype = np.float64,delimiter=';',decimal=',').to_numpy(dtype = np.float64, na_value= np.inf )

                # read data and make sure they are all floats
                images_file_pd = images_file = pd.read_csv(temp_file, index_col=0, header = 0,delimiter=';',decimal=',')
                types = images_file_pd.dtypes
                keys = images_file_pd.columns

                for (column_type,column_key) in zip(types,keys):
                    if column_type != "float64":
                        images_file_pd[column_key] = images_file_pd[column_key].apply(lambda x: hlp.tryconvert(x.replace(',','.'),np.nan,float) if type(x)==str else x)
                images_file = images_file_pd.to_numpy(dtype = np.float64, na_value= np.nan)
                
                # split measurement frames from thermal data
                timestamps = images_file[:,0]
                point_temperatures = images_file[:,1:]

                # find experiment id
                experiment_name = temp_file.split('/')[-1] 
                experiment_id = experiment_name.split('_')[0] 
                # for windows
                try:
                    experiment_id = experiment_id.split('\\')[-1]
                except:
                    print("DEBUG: the '\\' failed")
                    
                
                experiment_ids.append(experiment_id)
                # find corresponding trajectory by checking for the id tag
                for traj in all_traj_files:
                    if experiment_id in traj:
                        traj_file = traj
                        break
                
                if debug_data:

                    fig = plt.figure(figsize=(16,9))
                    fig.patch.set_facecolor('white')

                    ax1 = fig.add_subplot(211)
                    ax1.plot(timestamps,label="timestamps")
                    ax1.set_title(experiment_id)
                    ax1.set_ylim([0,len(timestamps)+1])
                    ax1.legend()

                    ax2 = fig.add_subplot(212)
                    node = 40
                    ax2.plot(point_temperatures[:,node],label=f"node {node}")
                    node = 80
                    ax2.plot(point_temperatures[:,node],label=f"node {node}")
                    ax2.set_title(experiment_id)
                    ax2.legend()

                    # plt.show()
                    plt.savefig(self.RESULTS_FOLDER + f"/DEBUG/dataset_experiment_{experiment_id}.png")
                    plt.close()

                treated_temperatures = self.treatNans(point_temperatures)
                treated_temperatures = self.treatInfs(treated_temperatures)
                
                rep_temp.append( treated_temperatures )
                rep_traj.append( pd.read_csv(traj_file, index_col=None, header=None).to_numpy() )
                rep_timestamps.append( timestamps )
            except Exception as e:
                print(e)

        #print("Done")
        if scaling_:
            self.scaler = scalingFunction()

        # load and order points
        #print("Loading points...")
        points = pd.read_csv(point_coordinates_files[0], index_col=None, header=0,delimiter=';').to_numpy()

        self.d_grid = d_grid
        N_grid_sorted, N_coord_sorted, self.d_grid = self.orderPoints( points, d_grid = self.d_grid )
        #print("Done")

        # create _Point objects. Assign them their spatial attributes.
        #print("Ordering points...")
        points_temp = []
        for j,(coordinates, N_tuple, N_coord_tuple) in enumerate(zip(points, N_grid_sorted, N_coord_sorted)):
            curr_point = _Point(j,coordinates)

            curr_point._setNeighbors(N_tuple,N_coord_tuple)
            curr_point._setBoundaryStatus_()
            points_temp.append(curr_point)

        self.points_spatial_info = points_temp # save spatial point information - you need it 
        #print("Done")

        # build _Experiment objects and assign _Point temporal attributes.
        #print("Form experiment data...")
        self.experiments = []
        for j,(temp,traj,ts) in enumerate(zip(rep_temp,rep_traj,rep_timestamps)):
            ambient_temp = np.mean(temp[0,:]) 
            point_list = [] 
            for i,curr_point in enumerate(points_temp):
                curr_point._set_T_t_node( temp[:,i]) 
                point_list.append(curr_point._copy())
            
            curr_exp = _Experiment(Points = point_list, Parent_plotter = self.plotter, T_ambient = ambient_temp,Timestamps = ts, id = experiment_ids[j], subsample = self.subsample, results_folder = self.RESULTS_FOLDER, deposition_height_of_boundary = deposition_height_of_boundary, deposition_height_of_non_excited = deposition_height_of_non_excited)

            curr_exp._setTraj(traj)
            # curr_exp._calculateAmbientTempeature(nodes = [40,41,42,43])
            curr_exp._calculateAmbientTempeature()
            self.experiments.append(curr_exp._copy())
        #print("Done")
        ## keep each experiment separate
        ### i-indexing : 0-32 -> points
        # self.temp_np = np.asarray( [temp_exp.to_numpy() for temp_exp in rep_temp],dtype=object )
        # ### i-indexing : 0-2 -> tool position, 3 -> time, 4 -> weld bool
        # self.traj_np = np.asarray( [traj_exp.to_numpy() for traj_exp in rep_traj],dtype=object )
        return self.experiments

    @staticmethod
    def treatNans(array):
        """
        only works if single rows are nan
        """

        length = np.max(array.shape)
        new_array = copy.copy(array)

        for i,point_data in enumerate(array.T): 
            nan_idxs = np.where(np.isnan(point_data))[0]
            for idx in nan_idxs:
                if idx == 0:
                    new_array[idx,i] = point_data[1]
                elif idx == length-1:
                    new_array[idx,i] = point_data[-2]
                else:
                    new_array[idx,i] = 0.5*(point_data[idx+1] + point_data[idx+2])
        
        return new_array
                

    @staticmethod
    def treatInfs(array, max_lim = 1000):
        """
        only works if single rows are infs
        """

        length = np.max(array.shape)
        new_array = copy.copy(array)

        for i,point_data in enumerate(array.T): 
            nan_idxs = np.where(point_data>max_lim)[0]
            for idx in nan_idxs:
                if idx == 0:
                    new_array[idx,i] = point_data[1]
                elif idx == length-1:
                    new_array[idx,i] = point_data[-2]
                else:
                    new_array[idx,i] = 0.5*(point_data[idx+1] + point_data[idx+2])
        
        return new_array
                
    @staticmethod
    def orderPoints(points,d_grid = None):
        '''
        Order points in a list wrt to their position on plane. So put neighboors in the same order to take under consideration anisotropy in your model.

        @returns N_grid_sorted : List of lists with the indexes of the sorted neighbors. Array indexing:[a][b][c]: a -> point, b -> neighboring tupple in the form (current point, neighboring point), c: 0 -> point index, 1 -> neighbor index.
        @returns N_coord_sorted : List of lists with the coordinates of the sorted neighbors. Array indexing:[a][b][c]: a -> point, b -> neighboring tupple in the form (current point, neighboring point), c: 0 -> x, 1 -> y, 2 -> previous index in list of neighbors.
        '''
        # Create a distance Matrix and find neighbours

        ## Distance matrix: distances between nodes
        nop = len(points)
        D = np.zeros((nop,nop))
        ### Build upper half
        for i,pi in enumerate(points):
            for j,pj in enumerate(points[i::]):
                D[i,i+j] = norm(pj - pi)

        ### Build lower symmetric half
        D = D + D.T - np.diag(D.diagonal())

        # ## find most frequent distance
        D_no_0 = np.nonzero(D)
        if d_grid == None:
            d_grid_int = np.min(D[D_no_0]).astype(np.int64)
        else:
            d_grid_int = np.asarray(d_grid).astype(np.int64)

        # d_grid_int = np.bincount(D[D_no_0].astype(np.int64)).argmax()
        d_grid = D.reshape(-1)[np.argmin(np.abs( d_grid_int - D.reshape(-1)))]

        ## find neighboring nodes
        N_grid = [] # tupples with indexes of neighbors
        for i in range(nop):
            neighboring_nodes = np.where(D[i,:] == d_grid)
            if np.any(neighboring_nodes[0]):
                N_grid.append([(i,n_i) for n in neighboring_nodes  for n_i in n])
            else:
                N_grid.append([])

        # Sort neighbors in the same order (wrt to the plane)

        # N_coord = [(points[p[0]],points[p[1]]) for n in N_grid for p in n] # for empty rows in N you don't get any rows

        N_coord_sorted = []
        N_grid_sorted = []
        # #print(N_grid)
        for n in N_grid:
            # #print("N: {}".format(n))
            if n:
                tmp = []
                for i,pi in enumerate(n):
                    # #print("n:{},p:{}".format(n,p))
                    # tmp.append(( points[pi[0]] , points[pi[1]] ,i)) # cannot sort after
                    tmp.append((tuple(points[pi[0]]),tuple(points[pi[1]]),i)) # add indexes
                    # tmp.append((points[pi[0]][0],points[pi[1]][0],i)) # add indexes - these are not exactly the points

                # #print("Appending:",tmp)
                # n_coord_sorted = sorted(tmp)
                n_coord_sorted = sorted(tmp, key=lambda tup: (tup[1]) ) # (xMin,yMin,yMax,xMax)
                N_coord_sorted.append([(n_coord[0],n_coord[1]) for n_coord in n_coord_sorted])
                # #print("Sorted:",n_coord_sorted)
                # #print("\n")
                # temp_sorted = [n[n_coord[2]] for n_coord in n_coord_sorted]
                N_grid_sorted.append([n[n_coord[2]] for n_coord in n_coord_sorted]) # reorder points based on indexes
                # #print("n_coord_sorted:",n_coord_sorted)
                # #print("N_grid_sorted:",N_grid_sorted[-1])
                # N_grid_sorted.append(n[sorted_idxs])
            else:
                N_coord_sorted.append(())
                N_grid_sorted.append(())

        return N_grid_sorted, N_coord_sorted,d_grid

    @staticmethod
    def findDomainLimits(coord_list):
        all_x = np.asarray([coord[0] for point in coord_list for tup in point for coord in tup])
        all_y = np.asarray([coord[1] for point in coord_list for tup in point for coord in tup])

        xMax = np.max(all_x)
        xMin = np.min(all_x)

        yMax = np.max(all_y)
        yMin = np.min(all_y)

        return [xMax,xMin,yMax,yMin]

    def orthogonalBoundarySet(self, points, only_outer = True):
        """
        Compute set of boundary points for orthogonal geometry.
        @returns set with node indexes
        """
        # find boundaries (of points with neighbors)
        [xMax,xMin,yMax,yMin] = self.findDomainLimits(points)

        # find points in boundaries

        xMax_list_idxs = [i for i,pi in enumerate(points) if pi[0]==xMax] # initial indexes
        xMin_list_idxs = [i for i,pi in enumerate(points) if pi[0]==xMin]

        yMax_list_idxs = [i for i,pi in enumerate(points) if pi[1]==yMax]
        yMin_list_idxs = [i for i,pi in enumerate(points) if pi[1]==yMin]

        # xMax_list_coord = [pi for i,pi in enumerate(points) if pi[0]==xMax] # initial indexes
        # xMin_list_coord = [pi for i,pi in enumerate(points) if pi[0]==xMin]

        # yMax_list_coord = [pi for i,pi in enumerate(points) if pi[1]==yMax]
        # yMin_list_coord = [pi for i,pi in enumerate(points) if pi[1]==yMin]


        # #print("xMax_list,",xMax_list_idxs)
        # # map to clean indexes

        # xMax_list_idxs = self._goToCleanIndexes(xMax_list_idxs)
        # xMin_list_idxs = self._goToCleanIndexes(xMin_list_idxs)

        # yMax_list_idxs = self._goToCleanIndexes(yMax_list_idxs)
        # yMin_list_idxs = self._goToCleanIndexes(yMin_list_idxs)

        #print("Mapped xMax_list,",xMax_list_idxs) # should be 2 indexes back

        self.boundaries_nodes = set()
        self.boundaries_nodes |= set(xMin_list_idxs)
        self.boundaries_nodes |= set(xMax_list_idxs)
        self.boundaries_nodes |= set(yMin_list_idxs)
        self.boundaries_nodes |= set(yMax_list_idxs)

        # check
        # xMin_points = []
        # plt.figure()
        # plt.scatter(x = [x[0] for x in xMax_list_coord], y = [x[1] for x in xMax_list_coord])
        # plt.scatter(x = [x[0] for x in xMin_list_coord], y = [x[1] for x in xMin_list_coord])
        # plt.scatter(x = [y[0] for y in yMax_list_coord], y = [y[1] for y in yMax_list_coord])
        # plt.scatter(x = [y[0] for y in yMin_list_coord], y = [y[1] for y in yMin_list_coord])

        #find edges -> tupples of neighboring nodes on boundaries

        return self.boundaries_nodes

    @staticmethod
    def _addEdge(edges, i, j, only_outer):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        return edges.add((i, j))

    def alphaShape(self, points, alpha, only_outer=True):
        """
        Compute the alpha shape (concave hull) of a set of points. Nice way for finding boundaries, but it does not work for sparse sets. Source: https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
        :param alpha: alpha value.
        :param only_outer: boolean value to specify if we keep only the outer border
        or also inner edges.
        :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
        the indices in the points array.
        """
        # assert points.shape[0] > 3, "Need at least four points"


        tri = Delaunay(points)
        edges = set()
        # Loop over triangles:
        # ia, ib, ic = indices of corner points of the triangle
        for ia, ib, ic in tri.vertices:
            pa = points[ia]
            pb = points[ib]
            pc = points[ic]
            # Computing radius of triangle circumcircle
            # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
            a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
            b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
            c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_r = a * b * c / (4.0 * area)
            if circum_r < alpha:
                self._addEdge(edges, ia, ib, only_outer)
                self._addEdge(edges, ib, ic, only_outer)
                self._addEdge(edges, ic, ia, only_outer)
        return edges # need to return boundaries

    def spawnHallucinatedNodes(self):
        """
        Create the Hallucinated nodes around the boundary. Boundary nodes should be a set of nodes.
        """
        self.hallucinated_nodes_coordinates = []
        # self.all_neighbors_clean = self.N_coord_sorted_clean.copy() # neighbors -> List of tuples with coordinates of neighbors and node label
        point_supplement = [np.array([self.d_grid,0]),np.array([-self.d_grid,0]),np.array([0,self.d_grid]),np.array([0,-self.d_grid])]
        self.first_example_unobservable_nodes_coord = []
        for i in self.boundaries_nodes:
            # j = self.map_clean_initial[i]
            neighbors = set(coord[1] for coord in self.N_coord_sorted_clean[i])
            pi = self.N_coord_sorted_clean[i][0][0]
            # #print("neighbors ",neighbors)
            # #print("pi ",pi)
            # pi = self.points[j]
            for sup in point_supplement:
                new_point = tuple(pi + sup)
                if new_point not in neighbors:
                    # well, apparently it's not like that
                    # if not self.checkIfInOrthogonalDomain(new_point):
                    #     self.hallucinated_nodes_coordinates.append(new_point)
                    # else:
                    #     self.first_example_unobservable_nodes_coord.append(new_point)
                    self.hallucinated_nodes_coordinates.append(new_point)
        # # check
        plt.figure()
        plt.scatter(x = [x[0] for x in self.hallucinated_nodes_coordinates], y = [x[1] for x in self.hallucinated_nodes_coordinates])
        plt.scatter(x = [x[0] for x in self.first_example_unobservable_nodes_coord], y = [x[1] for x in self.first_example_unobservable_nodes_coord])
        # plt.title("Hallucinated and unobservable nodes")
        return self.hallucinated_nodes_coordinates

    def checkIfInOrthogonalDomain(self,point):
        """
        @param point : subscriptable object with coordinates
        @return True if point is in the box
        """
        [xMax,xMin,yMax,yMin] = self.findDomainLimits(self.N_coord_sorted_clean)
        if ((point[0]<xMin) |
            (point[0]>xMax) |
            (point[1]>yMax) |
            (point[1]<yMin) ):
            return False

        return True

    def trajectorySynchronization_old(self):
        """
        Populate trajectories. After this function is called, each trajectory should have as many elements as the corresponding temperatures
        creates more points than it should
        """

        # compute timesteps and time for each experiment
        Ts = []
        Time = []
        for (traj,temps) in zip(self.traj_np,self.temp_np):
            Ts.append(traj[-1,3]/temps.shape[0]) # timestep at each experiment different (but ~0.09)
            Time.append(np.linspace(0,Ts[-1]*temps.shape[0]-Ts[-1],temps.shape[0])) 

        # iterate over all the edges of the tool movement and break them in small pieces
        self.all_tool_trajectories = []
        self.all_tool_statuses = []

        traj = None
        for (traj,dt) in zip(self.traj_np,Ts):
            experiment_trajectory = []
            experiment_statuses = []

            for (point1,point2) in zip(traj[0:-3,:],traj[1:-2,:]):
                DeltaX_i = point2[0] - point1[0]
                DeltaY_i = point2[1] - point1[1]
                DeltaT_i = point2[3] - point1[3]
                status_i = point1[4]

                n_i = floor(DeltaT_i/dt)
                x_step = DeltaX_i/n_i
                y_step = DeltaY_i/n_i

                for k in range(n_i):
                    experiment_trajectory.append((k*x_step + point1[0],k*y_step + point1[1]))
                    experiment_statuses.append(status_i)

            self.all_tool_trajectories.append(experiment_trajectory)
            self.all_tool_statuses.append(experiment_statuses)

        #print("len(temp_np[0]) = {}, len(all_tool_trajectories[0]) = {}".format(len(self.all_tool_statuses[0]),len(self.temp_np[0])))
        return self.all_tool_trajectories,self.all_tool_statuses
   
    def trajectorySynchronization(self):
        """
        Populate trajectories. After this function is called, each trajectory should have as many elements as the corresponding temperatures
        """
        # Predifine number of trajectory points and then assign them to tool trajectory edges.
        # compute timesteps and time for each experiment
        self.Ts = []
        self.Time = []
        for (traj,temps) in zip(self.traj_np,self.temp_np):
            self.Ts.append(traj[-1,3]/temps.shape[0]) # timestep at each experiment different (but ~0.09)
            self.Time.append(np.linspace(0,self.Ts[-1]*temps.shape[0]-self.Ts[-1],temps.shape[0])) 

        # iterate over all the edges of the tool movement and break them in small pieces
        self.all_tool_trajectories = []
        self.all_tool_statuses = []

        traj = None
        # for i,(traj,dt) in enumerate(zip(self.traj_np,Ts)):
        for (traj,t,dt) in zip(self.traj_np,self.Time,self.Ts):
            experiment_trajectory = []
            experiment_statuses = []

            #define number of points per edge

            # noe = self.temp_np[i].shape[0]
            # Delta_time = np.diff(traj[:,3])
            # relative_time_interval = Delta_time/traj[-1,3] # what part of the total trajectory each edge takes
            # points_per_edge = np.round(relative_time_interval*noe)
            # check = np.sum(points_per_edge)

            i = 0 # counter for traj
            traj_elem = len(traj)
            for t_step in t:
                # #print("t_step",t_step)
                if t_step > traj[i+1,3]:
                    if i != traj_elem-1:
                        i = i+1
                    
                point1 = traj[i,:]
                point2 = traj[i+1,:]
                # #print( "\tpoint1 and 2 : {}\t{}".format(point1,point2))

                DeltaX_i = point2[0] - point1[0]
                DeltaY_i = point2[1] - point1[1]
                DeltaT_i = point2[3] - point1[3]
                status_i = point1[4]

                tk_ti1 = t_step - point1[3]
                mean_vel_x = DeltaX_i/DeltaT_i
                mean_vel_y = DeltaY_i/DeltaT_i
                # #print("\tmean_vel_x and y: {}\t{}".format(mean_vel_x,mean_vel_y))

                experiment_trajectory.append(( tk_ti1*mean_vel_x + point1[0], tk_ti1*mean_vel_y + point1[1]) )
                # #print("\tcurrent position",experiment_trajectory[-1])
                experiment_statuses.append(status_i)
            # for (point1,point2) in zip(traj[0:-3,:],traj[1:-2,:]):
            #     DeltaX_i = point2[0] - point1[0]
            #     DeltaY_i = point2[1] - point1[1]
            #     DeltaT_i = point2[3] - point1[3]
            #     status_i = point1[4]

            #     n_i = floor(DeltaT_i/dt)
            #     x_step = DeltaX_i/n_i
            #     y_step = DeltaY_i/n_i

            #     for k in range(n_i):
            #         experiment_trajectory.append((k*x_step + point1[0],k*y_step + point1[1]))
            #         experiment_statuses.append(status_i)

            self.all_tool_trajectories.append(experiment_trajectory)
            self.all_tool_statuses.append(experiment_statuses)

        self.all_tool_trajectories = np.asarray(self.all_tool_trajectories)
        self.all_tool_statuses = np.asarray(self.all_tool_statuses)


        #print("len(temp_np[0]) = {}, len(all_tool_trajectories[0]) = {}".format(len(self.all_tool_statuses[0]),len(self.temp_np[0])))
        return self.all_tool_trajectories,self.all_tool_statuses

class _prePrcossesorPlotter(preProcessor):
    def __init__(self,parent):
        self._save_ = 0
        self._contourPlot_count_ = 0
        # self.parent = parent
        self.time = '{}'.format(datetime.now().strftime("%m%d_%H%M%S"))

    def get_continuous_cmap(self,hex_list, float_list=None):
        ''' creates and returns a color map that can be used in heat map figures.
            If float_list is not provided, colour map graduates linearly between each color in hex_list.
            If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
            
            Parameters
            ----------
            hex_list: list of hex code strings
            float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
            
            Returns
            ----------
            colour map'''
        rgb_list = [self.rgb_to_dec(self.hex_to_rgb(i)) for i in hex_list]
        if float_list:
            pass
        else:
            float_list = list(np.linspace(0,1,len(rgb_list)))
            
        cdict = dict()
        for num, col in enumerate(['red', 'green', 'blue']):
            col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
            cdict[col] = col_list
        cmp = mpl.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
        return cmp

    @staticmethod
    def hex_to_rgb(value):
        '''
        Converts hex to rgb colours
        value: string of 6 characters representing a hex colour.
        Returns: list length 3 of RGB values'''
        value = value.strip("#") # removes hash symbol if present
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    @staticmethod
    def rgb_to_dec(value):
        '''
        Converts rgb to decimal colours (i.e. divides each value by 256)
        value: list (length 3) of RGB values
        Returns: list (length 3) of decimal values'''
        return [v/256 for v in value]
    
    def contourPlot(self,plot_data, axes = [], colour_property="temp", vmin = 0, vmax = 600,path="./"):
        '''
        Render gif with colour annotation of a property.
        @param plot_data : dictionary with the plot data. It includes the spatial information for plotting a time instance. Necessary keys "x","y" with the coordinates of the points on the plain, one key with the colour_property (default "temp"), and one key "time" for the current time step.
        @axes : pyplot axes to draw on.
        @colour_property : Key indicating property to create the colour annotation.
        @path : Path to results.
        '''
        # fig = plt.figure(figsize=square_fig)
        if not axes:
            fig_contour = plt.figure(figsize=square_fig)
            ax = fig_contour.add_subplot(111)
        else:
            ax = axes

        ## Plot the contour
        ### This potato library does not fill the fucking plot
        # sns.kdeplot(data = plot_data, hue = colour_property, fill = True, thresh=0, levels=25, vmin = 25, vmax = 1000 , cmap="mako")
        # sns.kdeplot(data = plot_data, x = plot_data["x"], y = plot_data["y"], hue = plot_data[colour_property], fill = True, thresh=0, levels=50)
        ## Simple stuff
        drawing = plt.scatter(x = plot_data['x'], y = plot_data['y'], c = plot_data[colour_property], vmin = vmin, vmax = vmax, cmap = 'plasma')

        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        ax.set_title("Temperature profile, t: {:.2f}(s)".format(plot_data['time'][0]))
        ax.set_xlim(-105,105)
        ax.set_ylim(-105,105)
        ax.set_aspect('equal', adjustable='box')
        cbar = fig_contour.colorbar(drawing, ax=ax)
        cbar.ax.set_ylabel("Temperature")

        self._contourPlot_count_+=1


        fig_contour.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig_contour.canvas.tostring_rgb(), dtype='uint8')
        # TODO: this 4 there makes things work but, wtf, does not make any sense. It's like you create 4 plots instead of 1
        image = image.reshape(fig_contour.canvas.get_width_height()[::-1] + (3,))
        # image = image.reshape(fig_contour.canvas.get_width_height()[::-1] + (3,4,))[:,:,:,0]
        if self._save_:
            plt.savefig( path + "{}.jpg".format(self._contourPlot_count_), bbox_inches='tight')
        else:
            plt.close(fig_contour)

        return image,ax

    def renderContourGif(self,to_plot,data_timestep,path,**kwargs):
        '''
        Render gif with colour annotation of a property. A wrapper for self.contourPlot
        @param to_plot : dictionary with the time and property instances to be plotted. It includes the time data. The first key should be "time", the second key is a string with the name of the variable that will create the colour annotations.
        @data_timestep : dictionary like the one used in self.contourPlot.
        @**kwargs : refer to contour plot
        '''
        ## Unpack
        keys = list(to_plot.keys())
        key_time = keys[0]
        key_colour_prop = keys[1]
        time_to_plot = to_plot[key_time]
        val_to_plot = to_plot[key_colour_prop]

        ## create saving folder
        os.makedirs(path+self.time,exist_ok=True)

        ## create images
        imgs_rep = []
        self._contourPlot_count_ = 0
        self._save_ = 0

        if not kwargs:
            kwargs = { 'colour_property' : "temp", 
            'path' : path+self.time,
            "vmin" : np.min(val_to_plot),
            "vmax" : np.max(val_to_plot)}
        else:
            if "vmin" not in kwargs:
                kwargs["vmin"] = np.min(val_to_plot)
        
            if "vmax" not in kwargs:
                kwargs["vmax"] = np.max(val_to_plot)
        
            if "path" not in kwargs:
                kwargs["path"] =  path+self.time
                
            if "colour_property" not in kwargs:
                kwargs["colour_property"] =  "temp"

        for (t,val) in zip(time_to_plot,val_to_plot):
            data_timestep[key_colour_prop] = val
            data_timestep["time"] = t
            # imgs_rep.append( self.contourPlot(plot_data = data_timestep, colour_property=key_colour_prop, path = path+time,**kwargs) )
            im,_ = self.contourPlot(plot_data = data_timestep,**kwargs)
            imgs_rep.append( im )

        ## render gif
        imageio.mimsave(path + self.time +'/gif.gif',imgs_rep, fps=10)
        return None

    def contourPlotTool(self,plot_data,colour_property="temp", vmin = 0, vmax = 600,path="./"):
        '''
        Render gif with colour annotation of a property.
        @param plot_data : dictionary with the plot data. It includes the spatial information for plotting a time instance. Necessary keys "x","y" with the coordinates of the points on the plain, one key with the colour_property (default "temp"), and one key "time" for the current time step.
        @colour_property : Key indicating property to create the colour annotation.
        @path : Path to results.
        '''
        # fig = plt.figure(figsize=square_fig)
        fig_contour = plt.figure(figsize=square_fig)
        ax = fig_contour.add_subplot(111)

        ## Plot the contour
        ### This potato library does not fill the fucking plot
        # sns.kdeplot(data = plot_data, hue = colour_property, fill = True, thresh=0, levels=25, vmin = 25, vmax = 1000 , cmap="mako")
        # sns.kdeplot(data = plot_data, x = plot_data["x"], y = plot_data["y"], hue = plot_data[colour_property], fill = True, thresh=0, levels=50)
        ## Simple stuff

        divnorm = mpl.colors.TwoSlopeNorm(vmin = 0, vcenter = 0.5, vmax = 1)

        drawing = plt.scatter(x = plot_data['x'], y = plot_data['y'], c = plot_data[colour_property],  vmin = vmin, vmax = vmax, cmap = 'plasma')
        plt.scatter(x = plot_data['x_tool'], y = plot_data['y_tool'], c = plot_data["tool_status"], cmap = 'rainbow', norm = divnorm, marker="x")
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        ax.set_title("Temperature profile, t: {:.2f}(s)".format(plot_data['time'][0]))
        ax.set_xlim(-120,120)
        ax.set_ylim(-120,120)
        ax.set_aspect('equal', adjustable='box')
        cbar = fig_contour.colorbar(drawing, ax=ax)
        cbar.ax.set_ylabel("Temperature")

        self._contourPlot_count_+=1


        fig_contour.canvas.draw()       # draw the canvas, cache the renderer
        image = np.frombuffer(fig_contour.canvas.tostring_rgb(), dtype='uint8')
        # TODO: this 4 there makes things work but, wtf, does not make any sense. It's like you create 4 plots instead of 1
        image = image.reshape(fig_contour.canvas.get_width_height()[::-1] + (3,))
        # image = image.reshape(fig_contour.canvas.get_width_height()[::-1] + (3,4,))[:,:,:,0]
        if self._save_:
            plt.savefig( path + "{}.jpg".format(self._contourPlot_count_), bbox_inches='tight')
        else:
            plt.close(fig_contour)

        return image
    
    def renderContourVideoTool(self, to_plot, data_timestep, fps= 1, video_type = "mp4v", path = "../results/plots/video/",**kwargs):
        '''
        Render video with colour annotation of a property. A wrapper for self.contourPlot
        @param to_plot : dictionary with the time and property instances to be plotted. It includes the time data. The first key should be "time", the second key is a string with the name of the variable that will create the colour annotations.
        @data_timestep : dictionary like the one used in self.contourPlot.
        @**kwargs : refer to contour plot
        '''
        fps = max(1, fps)
        out = None

        ## Unpack
        keys = list(to_plot.keys())
        key_time = keys[0]
        key_colour_prop = keys[1]

        time_to_plot = to_plot[key_time]
        val_to_plot = to_plot[key_colour_prop]

        try:
            key_traj = keys[2]
            key_status = keys[3]

            traj_to_plot = to_plot[key_traj]
            status_to_plot = to_plot[key_status]
        except:
            key_traj = ""
            key_status = ""

            traj_to_plot = time_to_plot
            status_to_plot = val_to_plot

        ## create saving folder
        os.makedirs(path + self.time,exist_ok=True)

        ## create images
        imgs_rep = []
        self._contourPlot_count_ = 0
        self._save_ = 0

        if not kwargs:
            kwargs = { 'colour_property' : "temp", 
            'path' : path+self.time,
            "vmin" : np.min(val_to_plot),
            "vmax" : np.max(val_to_plot)}
        else:
            if "vmin" not in kwargs:
                kwargs["vmin"] = np.min(val_to_plot)
        
            if "vmax" not in kwargs:
                kwargs["vmax"] = np.max(val_to_plot)
        
            if "path" not in kwargs:
                kwargs["path"] =  path+self.time
                
            if "colour_property" not in kwargs:
                kwargs["colour_property"] =  "temp"


        for (t,val,tool_point,tool_status) in zip(time_to_plot,val_to_plot,traj_to_plot,status_to_plot):
            data_timestep[key_colour_prop] = val
            data_timestep["time"] = t
            data_timestep["x_tool"] = tool_point[0]
            data_timestep["y_tool"] = tool_point[1]
            data_timestep["tool_status"] = tool_status

            # imgs_rep.append( self.contourPlot(plot_data = data_timestep, colour_property=key_colour_prop, path = results_folder+time,**kwargs) )
            imgs_rep.append( self.contourPlotTool(plot_data = data_timestep,**kwargs) )

        ## render video
        vid = self.writeVideo( imgs_rep, results_folder = path +"/"+ self.time+"/", file_name= 'video_tool.' + video_type, fps=fps )
        return vid

    def writeVideo(self,frames, show = False, fps= 1, results_folder = "./results/plots/video/", file_name = "video"):
               
        video_type = file_name.split(".")[-1]
        fps = max(1, fps)
        out = None
        file = results_folder + file_name
        try:
            for image in frames:
                # frame = cv2.imread(image)
                frame = image
                if show:
                    cv2.imshow('video', frame)

                if not out:
                    height, width, channels = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*f"{video_type}")
                    out = cv2.VideoWriter(file, fourcc, fps, (width, height))

                out.write(frame)
        except Exception as e:
            print(e)
        
        out.release()
        cv2.destroyAllWindows()
        return out

    def renderContourGifTool(self, to_plot, data_timestep, fps = 10, path = "../results/plots/gif/",**kwargs):
        '''
        Render gif with colour annotation of a property. A wrapper for self.contourPlot
        @param to_plot : dictionary with the time and property instances to be plotted. It includes the time data. The first key should be "time", the second key is a string with the name of the variable that will create the colour annotations.
        @data_timestep : dictionary like the one used in self.contourPlot.
        @**kwargs : refer to contour plot
        '''
        fps = max(1, fps)

        ## Unpack
        keys = list(to_plot.keys())
        key_time = keys[0]
        key_colour_prop = keys[1]

        time_to_plot = to_plot[key_time]
        val_to_plot = to_plot[key_colour_prop]

        try:
            key_traj = keys[2]
            key_status = keys[3]

            traj_to_plot = to_plot[key_traj]
            status_to_plot = to_plot[key_status]
        except:
            key_traj = ""
            key_status = ""

            traj_to_plot = time_to_plot
            status_to_plot = val_to_plot

        ## create saving folder
        os.makedirs(path + self.time,exist_ok=True)

        ## create images
        imgs_rep = []
        self._contourPlot_count_ = 0
        self._save_ = 0

        if not kwargs:
            kwargs = { 'colour_property' : "temp", 
            'path' : path+self.time,
            "vmin" : np.min(val_to_plot),
            "vmax" : np.max(val_to_plot)}
        else:
            if "vmin" not in kwargs:
                kwargs["vmin"] = np.min(val_to_plot)
        
            if "vmax" not in kwargs:
                kwargs["vmax"] = np.max(val_to_plot)
        
            if "path" not in kwargs:
                kwargs["path"] =  path+self.time
                
            if "colour_property" not in kwargs:
                kwargs["colour_property"] =  "temp"


        for (t,val,tool_point,tool_status) in zip(time_to_plot,val_to_plot,traj_to_plot,status_to_plot):
            data_timestep[key_colour_prop] = val
            data_timestep["time"] = t
            data_timestep["x_tool"] = tool_point[0]
            data_timestep["y_tool"] = tool_point[1]
            data_timestep["tool_status"] = tool_status

            # imgs_rep.append( self.contourPlot(plot_data = data_timestep, colour_property=key_colour_prop, path = path+time,**kwargs) )
            imgs_rep.append( self.contourPlotTool(plot_data = data_timestep,**kwargs) )

        ## render gif
        imageio.mimsave(path + self.time +'/gif_tool.gif',imgs_rep, fps=fps)
        return None

    def plotNodeTimeEvolution(self,node, temps = None, times = None, axes = [], file_numbers=[], labels = [], path="./"):

        #€ Experiments to plot
        if not file_numbers:
            warnings.warn("File numbers are not compatible any more. Try passing temps and times")
            # # file_numbers = [i for i in range(len(self.parent.experiments))]         

        # temps_to_plot = self.parent.temp_np[file_numbers]
        if temps is None:
            # temps_to_plot = [self.parent.experiments[file_number].Points[node].T_t for file_number in file_numbers]
            assert temps is None, "Need to pass temps"
        else:
            temps_to_plot = temps

        if times is None:
            # times_to_plot = [self.parent.experiments[file_number].Time for file_number in file_numbers]
            assert temps is None, "Need to pass temps"
        else:
            times_to_plot = times

        if not labels:
            labels = [i for i in range(len(file_numbers))]

        ## Node to plot
        i_plot = node

        ## Plot
        if not axes:
            temps_i_plot = plt.figure("Temperature profiles of point {}".format(i_plot),figsize = long_fig)
            temp_ax = temps_i_plot.add_subplot(111)
        else:
            temp_ax = axes

        for (times,temps,l) in zip(times_to_plot,temps_to_plot,labels):

            plt.plot(times,temps,label = "{}".format(l))

        temp_ax.set_title("Temperature profiles, point {}".format(i_plot))
        temp_ax.set_xlabel("Time [s]")
        temp_ax.set_ylabel("Temperature")
        temp_ax.legend(title="File:")

        return temp_ax

def plot_video():
    _ = prc.plotter.renderContourVideoTool(to_plot,data_timestep,fps=10, path = RESULTS_FOLDER + "video/")
    return None

#%% [markdown]
# Test everything
if __name__ == "__main__":
    
    # mpl.use('GTK3Agg')

    PARENT = "./../"
    FILES_FOLDER = PARENT + "my_extracted_points/27052021/"
    # FILES_FOLDER = "../Points_Extraction" # use them when running as normal script
    POINT_TEMPERATURE_FILES = "temperatures/T*.csv" # no time information on thermal imaging camera
    TRAJECTORY_FILES = "Coordinate_Time/Coordinates_T*.csv"
    POINT_COORDINATES = "Coordinate_Time/point_coordinates.csv"

    RESULTS_FOLDER =  PARENT + "results/plots/"
    RESULTS_FOLDER_HOCUSPOCUS = RESULTS_FOLDER + "/HocusPocus/" 

    RESULTS_FOLDER_MODEL = PARENT + "results/models/linear/"
    BACKUP_FOLDER = PARENT + "back_up/"
    PRC_BACKUP_FILE = "prepocessor_file_"
    os.makedirs(BACKUP_FOLDER, exist_ok = True)
    os.makedirs(RESULTS_FOLDER, exist_ok = True)
    os.makedirs(RESULTS_FOLDER_MODEL, exist_ok = True)

    #%% 
    # Load data
    prc = preProcessor(FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER)
    _ = prc.loadData()
    prc2 = preProcessor(FILES_FOLDER,POINT_TEMPERATURE_FILES,POINT_COORDINATES,TRAJECTORY_FILES,RESULTS_FOLDER, subsample= 5)
    _ = prc2.loadData()

    #%%
    # interpolate trajectory and form Delta_T
    experiment = prc.experiments[0]
    _ = experiment.trajectorySynchronization()
    _ = experiment.torchCrossingNodes()
    experiment2 = prc2.experiments[0]
    _ = experiment2.trajectorySynchronization()
    _ = experiment2.torchCrossingNodes()
    
    # prepare scaler
    measurements_for_scaler = []
    points_used_for_training = [p for p in experiment.Points] + [p for p in experiment2.Points]
    T_t, relevance_coef = [],[]
    for p in points_used_for_training:
        T_t.append(p.T_t_use)
        relevance_coef.append(p.relevance_coef)
    
    T_t_np = np.hstack( T_t )
    relevance_coef_np = np.hstack( relevance_coef )
    feature_data_rep = [T_t_np, relevance_coef_np]
    feature_data = np.vstack(feature_data_rep).T

    prc.scaler.fit(feature_data) 
    prc2.scaler.fit(feature_data) 

    _ = experiment.scaleMeasurements(prc.scaler)
    _ = experiment2.scaleMeasurements(prc2.scaler)
    _ = experiment.formDeltaT()
    _ = experiment2.formDeltaT()

    #%%
    prc.plotter._contourPlot_count_ = 0
    prc.plotter._save_ = 1
    experiment_to_plot = 0
    timestep_to_plot = 5000
    xs = [p.coordinates[0] for p in prc.points_spatial_info ]
    ys = [p.coordinates[1] for p in prc.points_spatial_info ]
    data_timestep = pd.DataFrame({"x":xs,
                                "y":ys,
                                "temp":experiment.exportAllTemperatures()[timestep_to_plot,:],
                                "time":timestep_to_plot*0.09,
                                "x_tool": 0,
                                "y_tool": 0,
                                "tool_status": 1 })
    _ = prc.plotter.contourPlot(plot_data = data_timestep, colour_property="temp", path = RESULTS_FOLDER)
    # _ = prc.plotter.contourPlotTool(plot_data = data_timestep, colour_property="temp", path = RESULTS_FOLDER)
    _ = prc.plotter.contourPlotTool(plot_data = data_timestep, colour_property="temp", path = RESULTS_FOLDER)


    # %%
    subsmpl_coef = 10
    ## Temperatures to plot
    temps_to_plot = experiment.exportAllTemperatures()[0:3000:subsmpl_coef,:]
    trajectories_to_plot = experiment.trajectory[0:3000:subsmpl_coef,:]
    statuses_to_plot = experiment.statuses[0:3000:subsmpl_coef]
    time_to_plot = experiment.Time[0:3000:subsmpl_coef]
    ## Corresponding time to plot
    # temps =experiment.exportAllTemperatures()
    # Dt = prc.traj_np[experiment_to_plot][-1,3]/temps.shape[0] # timestep at each experiment (different but ~0.09)
    # Time = np.linspace(0,Dt*temps.shape[0]-Dt,temps.shape[0])
    # time_to_plot = Time[::subsmpl_coef]

    ## Pack data
    to_plot = {"time": time_to_plot,"temp": temps_to_plot, "tool_pos" : trajectories_to_plot, "statuses" : statuses_to_plot}

    ## Uncomment to render gif
    # prc.plotter.renderContourGif(to_plot,data_timestep, path = RESULTS_FOLDER + "gif/")
    # prc.plotter.renderContourGifTool(to_plot,data_timestep, path = RESULTS_FOLDER + "gif/")
    # _ = prc.plotter.renderContourVideoTool(to_plot,data_timestep,fps=10, path = RESULTS_FOLDER + "video/")
    # plot_video()

    # %%
    group_pointy = [2,4,6,8,9,10,11,12,13,15,17,19,20,21,22,23,24,26,28,30]
    group_curvy = [3,5,7,14,16,18,25,27,29] # 25 not sure
    # nodes_for_training = [1,13,15,17,19] # the only ones that do not miss information
    nodes_for_training = [i for i in range(0,len(experiment.Points),1) if experiment.Points[i]._hasNeighbors_()] # all nodes should be fine
 # the only ones that do not miss information

    node_to_train = 13
    hocus_pocus_kwargs = {
                            "coef_subdmn_length_1" : 0.9,
                            "coef_subdmn_length_2" : 0.7,
                            "dT_thress" : 0.2,  # tolerance for jittering
                            "peak_height" : 200/prc.scaler.scale_[0]/2, # temperature thresshold value for finding peaks
                            "peak_distance" : 50, # time thresshold value for finding peaks
                            "half_local_window_length" : 1 ,
                            "plateau_size" : 1,
                            "prominence" : 0.5,
                            "relative_prominence_thress" : 0,
                            "long_short_interval_coef" : 1800,
                            "save_folder" : RESULTS_FOLDER_HOCUSPOCUS}
    #print("Finding peaks...")
    # for point in experiment.Points:
    # for point in [experiment.Points[group_pointy]]:
    thermal_ins = []
    for point in [experiment2.Points[idx] for idx in nodes_for_training]:
        if point._hasNeighbors_():
            #print("Node ", point.node)
            *_,thermal_inputs  = experiment2.findNodeActuationIntervalsFromPeaks(point.node, **hocus_pocus_kwargs)
            thermal_ins.append( thermal_inputs )
            # plt.show()
            # plt.close(curr_plot)


    # for point in experiment.Points:
    # point = experiment.Points[0]
    # re = experiment.hocusPocusMultithreadWrapper({"node":point.node,"kwargs":hocus_pocus_kwargs})
    with mp.Pool(mp.cpu_count()) as pool:
        # experiment.hocusPocusKwargs = hocus_pocus_kwargs
        # for point in nodes_for_training:
        #     #print(f"DEBUG: entered {point}")
        #     if experiment.Points[point]._hasNeighbors_():
        inputs = [[node,kwargs] for (node,kwargs) in zip(nodes_for_training,[hocus_pocus_kwargs]*len(nodes_for_training))]
        results = pool.map( experiment.hocusPocusMultithreadWrapper, inputs)
        results2 = pool.map( experiment2.hocusPocusMultithreadWrapper, inputs)
    
    for i,(res,node) in enumerate(zip(results,nodes_for_training)):
        subdomain_idxs, _, peaks_idxs, thermal_inputs = res  
        p = experiment.Points[node]
        experiment.Points[node]._setInputs(thermal_inputs["T_peaks_final"])
        experiment.Points[node]._setInputTimes(thermal_inputs["time_peaks_final"])
        experiment.Points[node]._setInputIdxs(thermal_inputs["final_peaks_idxs"])
        experiment.Points[node]._setInputs(peaks_idxs)
        experiment.Points[node]._setSubdomains(subdomain_idxs) 
        
        p = experiment2.Points[node]
        experiment2.Points[node]._setInputs(thermal_inputs["T_peaks_final"])
        experiment2.Points[node]._setInputTimes(thermal_inputs["time_peaks_final"])
        experiment2.Points[node]._setInputIdxs(thermal_inputs["final_peaks_idxs"])
        experiment2.Points[node]._setInputs(peaks_idxs)
        experiment2.Points[node]._setSubdomains(subdomain_idxs) 
        
    # plt.show()
    # %%
    k = 10
    T = 360
    # experiment.formFeatureVectorsMultithreadWrapper(k,T)
    #print("Forming Feature vectors...")
    # experiment2.formFeatureVectors(k,T,J = 200)
    experiment.formFeatureVectorsMultithreadWrapper(k,T,J = 20)
    experiment2.formFeatureVectorsMultithreadWrapper(k,T,J = 20)

    # subsmampling debug 
    p = experiment.Points[0]
    new_p = experiment2.Points[0]
   
    features = p.features[40:44]
    new_features = new_p.features[40:44]
   
    features2 = p.features[:5]
    new_features2 = new_p.features[:5]

    features_time = np.linspace(0,features.shape[1]-1,features.shape[1])
    new_features_time = np.linspace(0,features.shape[1]-1,new_features.shape[1])

    fig = plt.figure('Compare features',figsize=(16,9))
    # for (feat,new_feat) in zip(features)
    plt.plot(features_time,features.T, label = 'normal')
    plt.plot(new_features_time,new_features.T,linestyle = '--', label = 'subsampled')
    plt.title("subsampled vs normal")
    plt.legend()

    fig2 = plt.figure('Compare conv features',figsize=(16,9))
    # for (feat,new_feat) in zip(features)
    plt.plot(features_time,features2.T, label = 'normal')
    plt.plot(new_features_time,new_features2.T,linestyle = '--', label = 'subsampled')
    plt.title("CONVOLUTIONAL subsampled vs normal")
    plt.legend()

    plt.show()

    #### debug ####
    points_with_neighbors = [p for p in experiment.Points if p.neighbor_nodes]
    for p in points_with_neighbors:
        print("Node: ",p.node)
        print("\tZs: ",p.features_inputs[0,:])
        print("\tNs: ",p.features_inputs[5,:])

    input_features_repository = []
    input_targets_repository = []
    for p in experiment.Points:
        input_features_repository.append([p.features_inputs[0,:], # select torch depth
                                            p.features_inputs[1,:], # Delta T 
                                            p.features_inputs[3,:], # this cumulative relevance thing 
                                            1/(p.features_inputs[0,:]+1)*p.features_inputs[3,:]]),# and their product
        input_targets_repository.append(p.input_temperatures[:-1]) # last peak is not my data!

    input_features = np.hstack(input_features_repository)
    input_features_normalizers = input_features.max(axis=1)
    input_features /= input_features_normalizers

    input_targets = np.hstack(input_targets_repository)
    M_0 = torch.randn(input_features.shape[1])

    # %%
    # model = np.array([1.0624,6.4177,2.7967])
    # predictions = model.dot(input_features)

    # %%
    # M_0 = torch.randn(4)

    batch = 8
    kwargs_opt = {"lr"  : 1e-6, "weight_decay" :1e0, "hypergrad_lr" : 1e-5}
    cost_scaler = 0.01
    max_epochs = int(775)
    no_decrease_thress = 10

    # model_weights = linearInputModel(input_features,input_targets,M_0,batch = 8, cost_scaler = cost_scaler, max_Epochs = max_epochs, no_decrease_thress = no_decrease_thress, kwargs_opt = kwargs_opt)
    model = model_weights.detach().numpy()

    # loss on training data
    input_predictions = model.dot(input_features)

    rel_error = (input_predictions - input_targets)/input_predictions * 100
    mre = np.mean(np.abs(rel_error))
    stdre = np.std(np.abs(rel_error))

    #print(f"mre : {mre}%\tstdre: {stdre}%")
    #print("model ",model)
    # %%
    # define the bounds of the Temperature zones
    min_temperature = 300
    max_temperature = 550
    step = 25
    band_length = 125
    bounds = [(0,min_temperature-step)]
    for ub in range(min_temperature,max_temperature+1,step):
        bounds.append((ub-band_length,ub))
    bounds.append((ub-75,1000)) # set the last temperature limit high to include all the rest temps 

    # bands of measurements for training models in different temperatures
    section_idxs = experiment.disectTemperatureRegions(node_to_train, bounds = bounds, element_thresshold = 1200)

    #%%
    group_pointy2 = [13,15,17,19] # the only ones that do not miss information
    node_to_train = group_pointy2[0]

    model_on_point = []
    for node_to_train in group_pointy2:
        model_on_point.append(experiment.Points[node_to_train])
    # Node specific features that will be used in the temperature band models
    # train_idxs = model_on_point.domain_idxs[[0,1,2,3]]
    train_idxs = [p.domain_idxs[1:] for p in model_on_point]
    # #print(len(train_idxs))
    x_conv = [np.hstack([p.features[:,idxs] for idxs in train_idxs_set]) for (p,train_idxs_set) in zip(model_on_point,train_idxs)]
    # #print(len(train_idxs))

    y_all = [m.T_t for m in model_on_point ]
    delta_y_all = [np.append(np.diff(y),0) for y in y_all]

    y_idxs = [np.hstack([y[idxs] for idxs in train_idxs_set]) for (y,train_idxs_set) in zip(y_all,train_idxs)] # for banding
    delta_y_idxs = [np.hstack([delta_y[idxs] for idxs in train_idxs_set]) for (delta_y,train_idxs_set) in zip(delta_y_all,train_idxs)] # training targets

    x_train = np.hstack([x[:,:-2] for x in x_conv]) # get rid of the last element bcs you predict the Delta T between this step and the next
    normalizer = np.max(np.abs(x_train),axis = 1)
    normalizer[np.where(normalizer == 0)] = 1
    x_train = x_train/normalizer[:,None]


    y_target = np.hstack([dy[:-2] for dy in delta_y_idxs]) # get rid of the last element bcs you added it artificially before
    y_bands = np.hstack([y[:-2] for y in y_idxs])
    # #print(y_bands.shape)
    # #print(y_idxs[0].shape)
    # #print(y_idxs[1].shape)

    # bands of measurements for training models in different temperatures although here you only select the training data
    section_idxs = experiment.disectTemperatureRegions(node_to_train, T_t_custom = y_bands, bounds = bounds, element_thresshold = 400)
    # _ = [#print("section : {} shape : {}".format(i,s.shape)) for i,s in enumerate(section_idxs)]
    # xs_train = [x_train[:,idxs] for idxs in section_idxs]
    xs_train = [x_train[:,idxs]for idxs in section_idxs]
    ys_train = [y_target[idxs] for idxs in section_idxs]
    Ts_in_bands = [y_bands[idxs] for idxs in section_idxs]

    n = xs_train[0].shape[0]
    M_0 = np.ones((1,2*k+2))*0.001

    #%%
    #print("Linear model optimization starting:")
    models_repository = []
    temperatures = []
    final_costs = []
    nob = len(Ts_in_bands)

    for i,(x_tr,y_tr,Ts_in_band) in enumerate(zip(xs_train,ys_train,Ts_in_bands)):
        # obj = objective_fun(x_conv,y_target)
        #print("Model : {}".format(i+1))
        obj = objective_fun(x_tr,y_tr,i,nob)
        # mse = obj(M_0)
        # #print(mse)

        res = least_squares( obj, x0 = M_0.squeeze(), verbose = 1, ftol = 5e-6, max_nfev = 5e5)

        M_0 = res.x # use current solution as next initializer
        final_costs.append(res.cost)
        # models_repository.append(fromDynVecToDynamics(res.x))
        models_repository.append(res.x)
        temperatures.append(np.mean(Ts_in_band))
       
    cost = res.cost
    M_opt = res.x
    # #print("MRE : ",res.cost)
    #print("\nSystem : ",res.x)
    # iterate over dataset
    responses = M_opt.dot(x_conv).squeeze()

    df_out = pd.DataFrame(M_opt)    
    export_name = RESULTS_FOLDER_MODEL + "T_{}_k_{}_id_{}".format(T,k,datetime.now().strftime("%m%d%H%M"))
    df = pd.DataFrame(M_opt)
    df.to_csv(export_name + '.csv', header = False, index = False)
    #%%
    dbg = plt.figure(figsize = long_fig)
    plt.plot(np.linspace(0,1,y_target.shape[0]),y_target,label = "real Delta Ys")
    plt.plot(np.linspace(0,1,responses.shape[0]),responses,label = "Predicted Delta Ys")
    # plt.plot(np.linspace(0,1,x_conv.shape[0]),responses,label = "simulated")
    plt.legend()
    dbg.savefig(export_name + '.png')
    plt.show()
# %%
