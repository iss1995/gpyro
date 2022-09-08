#%%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils.generic_utils as gnu

import glob
import os
import imageio # gif
import cv2 # openCV for video
import copy

from data_processing._Experiment import _Experiment
from data_processing._Point import _Point
from datetime import datetime
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler as scalingFunction

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
class preProcessor:
    def __init__(self, files_folder : str, point_temperature_files : str, point_coordinates_files : str, tool_trajectory_files :str, results_folder :str = "./results/", subsample : int = 1):
        '''
        Data loader class. Preprocess data and export.
        @param files_folder : string with path to the data folder.
        @param point_temperature_files : string with names of files including temperature measurements.
        @param point_coordinates_files : string with names of files including coordinates of points.
        @param tool_trajectory_files : string with names of files including tool trajectory.
        @param files_folder : string with path to the designated result folder.
        @param subsample : int with subsampling factor.
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

    def loadData(self,scaling_ :bool = True, d_grid : float or None = None, debug_data : bool = True, deposition_height_of_boundary : float = 0, deposition_height_of_non_excited : float = 0):
        '''
        Load files and save them into Experiments format.
        @params scaling_ : whether to scale or not.
        @params d_grid : distance between nodes on grid. If None the min distance found is taken
        @params debug_data : whether to load debug data or not.
        @params deposition_height_of_boundary : hallucinated height of deposition of boundary points.
        @params deposition_height_of_non_excited : hallucinated height of deposition of non excited points.
         
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
        experiment_ids = []
        for temp_file in all_temp_files:

            try:
                # read data and make sure they are all floats
                images_file_pd = images_file = pd.read_csv(temp_file, index_col=0, header = 0,delimiter=';',decimal=',')
                types = images_file_pd.dtypes
                keys = images_file_pd.columns

                for (column_type,column_key) in zip(types,keys):
                    if column_type != "float64":
                        images_file_pd[column_key] = images_file_pd[column_key].apply(lambda x: gnu.tryconvert(x.replace(',','.'),np.nan,float) if type(x)==str else x)
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
        points = pd.read_csv(point_coordinates_files[0], index_col=None, header=0,delimiter=';').to_numpy()

        self.d_grid = d_grid
        N_grid_sorted, N_coord_sorted, self.d_grid = self.orderPoints( points, d_grid = self.d_grid )

        # create _Point objects. Assign them their spatial attributes.
        points_temp = []
        for j,(coordinates, N_tuple, N_coord_tuple) in enumerate(zip(points, N_grid_sorted, N_coord_sorted)):
            curr_point = _Point(j,coordinates)

            curr_point._setNeighbors(N_tuple,N_coord_tuple)
            curr_point._setBoundaryStatus_()
            points_temp.append(curr_point)

        self.points_spatial_info = points_temp # save spatial point information - you need it 

        # build _Experiment objects and assign _Point temporal attributes.
        self.experiments = []
        for j,(temp,traj,ts) in enumerate(zip(rep_temp,rep_traj,rep_timestamps)):
            ambient_temp = np.mean(temp[0,:]) 
            point_list = [] 
            for i,curr_point in enumerate(points_temp):
                curr_point._set_T_t_node( temp[:,i]) 
                point_list.append(curr_point._copy())
            
            curr_exp = _Experiment(Points = point_list, Parent_plotter = self.plotter, T_ambient = ambient_temp,Timestamps = ts, id = experiment_ids[j], subsample = self.subsample, results_folder = self.RESULTS_FOLDER, deposition_height_of_boundary = deposition_height_of_boundary, deposition_height_of_non_excited = deposition_height_of_non_excited)

            curr_exp._setTraj(traj)
            curr_exp._calculateAmbientTempeature()
            self.experiments.append(curr_exp._copy())
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
        N_coord_sorted = []
        N_grid_sorted = []
        for n in N_grid:
            if n:
                tmp = []
                for i,pi in enumerate(n):
                    tmp.append((tuple(points[pi[0]]),tuple(points[pi[1]]),i)) # add indexes
                
                n_coord_sorted = sorted(tmp, key=lambda tup: (tup[1]) ) # (xMin,yMin,yMax,xMax)
                N_coord_sorted.append([(n_coord[0],n_coord[1]) for n_coord in n_coord_sorted])
                N_grid_sorted.append([n[n_coord[2]] for n_coord in n_coord_sorted]) # reorder points based on indexes
            else:
                N_coord_sorted.append(())
                N_grid_sorted.append(())

        return N_grid_sorted, N_coord_sorted,d_grid


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

def plot_video( prc, to_plot, data_timestep, fps = 10, RESULTS_FOLDER = "./"):
    _ = prc.plotter.renderContourVideoTool(to_plot, data_timestep, fps=fps, path = RESULTS_FOLDER + "video/")
    return None
