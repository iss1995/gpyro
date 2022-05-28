
import numpy as np
import matplotlib as mpl
import copy

from numpy.linalg import norm

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 24}
mpl.rc('font', **font)
mpl.rcParams['lines.markersize'] = 20
# mpl.use('GTK3Cairo')

square_fig = (12,12)
long_fig = (16,9)

# %% 
class _Point:
    def __init__(self,node,coordinates):
        self.node = node
        self.coordinates = np.atleast_1d(coordinates)
        self.isHallucinated_ = False
        self.isBoundary_ = False
        self.neighbor_nodes = [] # order of neighboring nodes
        self.neighbor_nodes_coordinates = []
        self.hallucinated_nodes = [] # index of hallucinated nodes
        self.hallucinated_nodes_coordinates = []
        self.features = []
        self.features_inputs = [] # 0: z of torch, 1: Dt between peaks, 2: crossed or not (bool),3: Relevance metric, 4: number of crossings per layer, 5: number of crossings per layer so far
        self.Delta_T = []
        self.Delta_T_ambient = []
        self.input_raw_features = []
        self.T_t = []
        self.T_t_use = []
        self.input_temperatures = [] # temperature peaks values
        self.domain_idxs = []
        self.temperature_idxs = []
        self.excitation_times = [] # times in seconds, torch crosses point
        self.excitation_idxs = [] # torch crosses point
        self.excitation_temperatures = [] # temperature on plate when torch crosses point
        self.input_times = [] # times in seconds, observe input peak
        self.input_idxs = [] # observe input peak
        self.T_at_the_start_of_input = [] # temperature on plate when you start observing the input peak
        self.relevance_coef = [] # torch position relevance coefficient
        self.delayed_relevance_coef = [] # torch position relevance coefficient
        self.relevance_coef_sharp = [] # torch position relevance coefficient but with small lengthscale
        self.excitation_delay = [] # sample difference between excitation time and temperature peak
        self.excitation_delay_torch_height = [] # torch height during excitation
        self.excitation_delay_torch_height_trajectory = [] # (interpolated) trajectory of torch heights during excitation 

    def _setExcitation_delay(self,excitation_delay):
        """
        indexing : 0-> experiment, 1-> idxstep
        """
        self.excitation_delay = np.atleast_1d(excitation_delay)

    def _setExcitation_excitationDelayTorchHeight(self,excitation_delay_torch_height):
        """
        indexing : 0-> experiment, 1-> idxstep
        """
        self.excitation_delay_torch_height = np.atleast_1d(excitation_delay_torch_height)

    def _setExcitation_excitationDelayTorchHeightTrajectory(self,excitation_delay_torch_height_trajectory):
        """
        indexing : 0-> experiment, 1-> idxstep
        """
        self.excitation_delay_torch_height_trajectory = np.atleast_1d(excitation_delay_torch_height_trajectory)

    def _setExcitation_TatTheStartOfInput(self,T_at_the_start_of_input):
        """
        indexing : 0-> experiment, 1-> idxstep
        """
        self.T_at_the_start_of_input = np.atleast_1d(T_at_the_start_of_input)

    def _setExcitation_idxs(self,excitation_idxs):
        """
        indexing : 0-> experiment, 1-> idxstep
        """
        self.excitation_idxs = np.atleast_1d(excitation_idxs)

    def _setExcitation_temperatures(self,excitation_temperatures):
        """
        indexing : 0-> experiment, 1-> idxstep
        """
        self.excitation_temperatures = np.atleast_1d(excitation_temperatures)

    def _setExcitation_times(self,excitation_times):
        """
        indexing : 0-> experiment, 1-> timestep
        """
        self.excitation_times = np.atleast_1d(excitation_times)
        # ("\tnode: {}, T_t shape {}".format(self.node,self.T_t.shape))#debug

    def _set_T_t_node(self,T_t):
        """
        indexing : 0-> experiment, 1-> timestep
        """
        self.T_t = np.atleast_1d(T_t)
        self.T_t_use = self.T_t 

    def _set_T_t_use_node(self,T_t_use):
        """
        indexing : 0-> experiment, 1-> timestep
        """
        self.T_t_use = np.atleast_1d(T_t_use) 


    def _setDeltaT(self,DeltaT):
        self.Delta_T = DeltaT

    def _setDeltaTambient(self,DeltaTamb):
        self.Delta_T_ambient = DeltaTamb

    def _setInputRawFeatures(self,input_raw_features):
        self.input_raw_features = input_raw_features

    def _setTemperatureIdxs(self,idxs):
        """
        Idxs refer to T_t
        """
        self.temperature_idxs = idxs

    def _setFeatures(self,features):
        self.features = features

    def _setFeaturesInputs(self,features_inputs):
        self.features_inputs = features_inputs

    def _setSubdomains(self,domain_idxs):
        self.domain_idxs = domain_idxs

    def _setInputTimes(self,input_times):
        """
        Find the parts of T_t that are excited by the Torch. Ideally this should be computed with the trajectory of the tool but, for now,
        do your hocus pocus things with the peaks. Be cautius though, this is a MTM optimization process.  
        @params input_idxs : List with idxs of diferent Temperature Peaks.
        """
        self.input_times = input_times

    def _setInputIdxs(self,input_idxs):
        """
        Find the parts of T_t that are excited by the Torch. Ideally this should be computed with the trajectory of the tool but, for now,
        do your hocus pocus things with the peaks. Be cautius though, this is a MTM optimization process.  
        @params input_idxs : List with idxs of diferent Temperature Peaks.
        """
        self.input_idxs = input_idxs

    def _setRelevanceCoef(self,relevance_coef):
        """
        Find the parts of T_t that are excited by the Torch. Ideally this should be computed with the trajectory of the tool but, for now,
        do your hocus pocus things with the peaks. Be cautius though, this is a MTM optimization process.  
        @params relevance_coef : List with idxs of diferent Temperature Peaks.
        """
        self.relevance_coef = relevance_coef

    def _setDelayedRelevanceCoef(self,delayed_relevance_coef):
        """
        Find the parts of T_t that are excited by the Torch. Ideally this should be computed with the trajectory of the tool but, for now,
        do your hocus pocus things with the peaks. Be cautius though, this is a MTM optimization process.  
        @params delayed_relevance_coef : List with idxs of diferent Temperature Peaks.
        """
        self.delayed_relevance_coef = delayed_relevance_coef

    def _setRelevanceCoef_sharp(self,relevance_coef):
        """
        Find the parts of T_t that are excited by the Torch. Ideally this should be computed with the trajectory of the tool but, for now,
        do your hocus pocus things with the peaks. Be cautius though, this is a MTM optimization process.  
        @params relevance_coef : List with idxs of diferent Temperature Peaks.
        """
        self.relevance_coef_sharp = relevance_coef

    def _setInputs(self,input_temperatures):
        """
        Find the parts of T_t that are excited by the Torch. Ideally this should be computed with the trajectory of the tool but, for now,
        do your hocus pocus things with the peaks. Be cautius though, this is a MTM optimization process.  
        @params input_idxs : List with idxs of diferent Temperature Peaks.
        """
        self.input_temperatures = input_temperatures

    def _setNeighbors(self,neighbor_tuples,neighbor_coord_tuples):
        """
        indexing : 0-> experiment, 1-> timestep
        """
        for (node,coord) in zip(neighbor_tuples,neighbor_coord_tuples):
            self.neighbor_nodes.append(node[1])
            self.neighbor_nodes_coordinates.append(coord[1])
        # ("\tnode: {}, T_t shape {}".format(self.node,self.T_t.shape))#debug

    def _setBoundaryStatus_(self):
        """
        Check if Node is in boundary.
        """
        if len(self.neighbor_nodes)==4:
            self.isBoundary_ =  False
        else:
            self._spawnHallucinatedNodes()
            self.isBoundary_ = True
        
        return self.isBoundary_ 

    def _spawnHallucinatedNodes(self):
        """
        Hallucinated nodes are NODE specific. It thus makes sense to spawn them here.
        """
        if self.neighbor_nodes_coordinates:
            pj = self.neighbor_nodes_coordinates[0]
            self.d_grid = norm(pj - self.coordinates)
            point_supplement = [np.array([self.d_grid,0]),np.array([-self.d_grid,0]),np.array([0,self.d_grid]),np.array([0,-self.d_grid])]
            neighbors = set (self.neighbor_nodes_coordinates)
            j = 0
            for sup in point_supplement:
                new_point = tuple(self.coordinates + sup)
                if new_point not in neighbors:
                    # print("new_point ",new_point)
                    self.hallucinated_nodes.append(j)
                    j += 1
                    self.hallucinated_nodes_coordinates.append(new_point)
            
            self.hallucinated_nodes_coordinates = sorted(self.hallucinated_nodes_coordinates)
        else:
            self.hallucinated_nodes_coordinates = None

        return self.hallucinated_nodes_coordinates

    def _hasNeighbors_(self):
        """
        Check if Node has neighbors.
        """
        if self.neighbor_nodes:
            return True
        else:
            return False

    def _copy(self):
        return copy.deepcopy(self)
