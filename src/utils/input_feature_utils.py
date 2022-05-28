import numpy as np
import matplotlib.pyplot as plt
import utils.helper_func as hlp
from scipy.optimize import minimize,Bounds
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from copy import deepcopy as copy


# %%¨
def optimizeInputFeature_(points_used_for_training, debug = True, underestimate_lengthscales = 0, lengthscale_regularizer = 0.001, parameters_regularizer = 0.1):
    """
    Optimize bell lengthscale for each height level.
    """
    # find all deposition height levels -> you have to look at points that get ecited 
    excited_points = [ p for p in points_used_for_training if len(p.input_idxs)>0]
    height_levels = np.unique(excited_points[0].excitation_delay_torch_height) # consider that all points get excited on every layer 

    # create a map telling you which peaks belong to different excitation levels
    map_repository = []
    for height_level in height_levels:
        point_excitations_on_level_map = [] # 1 array for each point telling you which peaks to consider
        for p in excited_points:
            point_excitations_on_level_map.append( p.input_idxs[ p.excitation_delay_torch_height == height_level ] )
        
        map_repository.append(point_excitations_on_level_map)

    # for each height level find the input model parameters that create look-alike bells
    print("lengthscales...")
    lengthscaleModel = optimizeLengthscale(map_repository, excited_points, height_levels, underestimate_lengthscales = 0,lengthscale_regularizer = lengthscale_regularizer)

    # Patchwork
    #################################################################################################################################
    # lengthscaleModel = halfBellLegthscaleModel([3,5,8,10,12,14], height_levels)
    # print("Overwriting lengthscales with ",[3,5,8,10,12,14])
    #################################################################################################################################
    print("Done!")

    print("parameters...")
    # learn the rest parameters
    parameters_per_height_level =  optimizeParameters(height_levels,map_repository,lengthscaleModel,excited_points,parameter_regularizer = parameters_regularizer)
    print("Done!")
    
    coefficients = parameters_per_height_level[:,:5]
    ipCoefModel = modelsForInputCoefs(coefficients, height_levels)
    if debug:
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        for i,coef in enumerate(coefficients.T): 
            ax.plot(height_levels,coef,label = i)
            plt.legend(title = "g coef:")
        plt.title("debug g optimization")
    return ipCoefModel, lengthscaleModel

def optimizeLengthscale(map_repository, excited_points, height_levels, underestimate_lengthscales = 0, lengthscale_regularizer = 0.1):
    # constraint problem to only have positive lengthscales and only scale the bell in a positive way
    parameters_per_height_level = []
    bounds = Bounds([-np.inf, -np.inf, -np.inf, -np.inf, 0.2, 0, -np.inf], 
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf ]) 

    # learn the lengthscale
    param_0 = np.zeros((7))
    param_0[-3] = 1 # lengthscale
    for height_level_idxs in map_repository:
        optimizer = optimizeLayerInputParameters(height_level_idxs,excited_points, regularizer=lengthscale_regularizer)
        res = minimize( optimizer , param_0, bounds= bounds, method= "L-BFGS-B", tol = 1e-12 )
        param_0 = res.x
        parameters_per_height_level.append(res.x[:-1]) # omit the last one

        # for testing
        check = optimizeLayerInputParameters(height_level_idxs,excited_points)
        resp = check(param_0)
    
    # find lengthscale model
    lengthscales = np.abs( [x[4] for x in parameters_per_height_level] )
    lengthscaleModel = halfBellLegthscaleModel(10*lengthscales * (1-underestimate_lengthscales),height_levels) # underestimate 

    return lengthscaleModel

def optimizeParameters(height_levels,map_repository,lengthscaleModel,excited_points, parameter_regularizer = 0.1):
    parameters_per_height_level = []
    bounds = Bounds([-np.inf, -np.inf, -np.inf, -np.inf, 0.2, 0, -np.inf], 
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf ]) 
    param_0 = np.ones((7))*0.1
    # TODO: 1 optimization for all heights
    for (height_lvl,height_level_idxs) in zip(height_levels,map_repository):

        height_lengthscale = lengthscaleModel(height_lvl)
        optimizer = optimizeLayerInputParameters(height_level_idxs,excited_points,height_lengthscale,regularizer= parameter_regularizer)
        res = minimize( optimizer , param_0, bounds= bounds, method= "L-BFGS-B", tol = 1e-12 )
        # param_0 = res.x
        parameters_per_height_level.append(res.x[:-1]) # omit the last one

    print(np.vstack( parameters_per_height_level ))
    return np.vstack( parameters_per_height_level )

def optimizeInputFeature(points_used_for_training, debug = True, underestimate_lengthscales = 0, lengthscale_regularizer = 1e-3, G_regularizer = 1e-4):
    """
    Optimize bell lengthscale for each height level.
    """
    # find all deposition height levels -> you have to look at points that get ecited 
    excited_points = [ p for p in points_used_for_training if len(p.input_idxs)>0]
    height_levels = np.unique(excited_points[0].excitation_delay_torch_height) # consider that all points get excited on every layer 

    # create a map telling you which peaks belong to different excitation levels
    map_repository = []
    for height_level in height_levels:
        point_excitations_on_level_map = [] # 1 array for each point telling you which peaks to consider
        for p in excited_points:
            point_excitations_on_level_map.append( p.input_idxs[ p.excitation_delay_torch_height == height_level ] )
        
        map_repository.append(point_excitations_on_level_map)

    # for each height level find the input model parameters that create look-alike bells
    print("lengthscales...")
    lengthscaleModel = initializeLengthscale(map_repository, excited_points, height_levels, underestimate_lengthscales, lengthscale_regularizer = lengthscale_regularizer )
    print("Done!")

    print("parameters...")
    # learn the rest parameters
    coefficients, lengths = optimizeInputModel( height_levels, map_repository, lengthscaleModel, excited_points, G_regularizer = G_regularizer, lengthscale_regularizer = lengthscale_regularizer )
    print("Done!")
    
    lengthscales = np.squeeze( 10*lengths*(1-underestimate_lengthscales) )

    # fit models on parameters
    normalize = np.max(np.sum(np.abs(coefficients),axis = 1))
    ipCoefModel = modelsForInputCoefs(coefficients/normalize, height_levels)
    lengthscaleModel = halfBellLegthscaleModel(lengthscales, height_levels)

    return ipCoefModel, lengthscaleModel

def initializeLengthscale(map_repository, excited_points, height_levels, underestimate_lengthscales = 0, lengthscale_regularizer = 0.1):
    # constraint problem to only have positive lengthscales and only scale the bell in a positive way
    parameters_per_height_level = []
    bounds = Bounds([ 0.2, -np.inf], 
                [ np.inf, np.inf ]) 

    # learn the lengthscale
    param_0 = np.zeros((2))
    param_0[0] = 0.2 # lengthscale
    for height_level_idxs in map_repository:
        optimizer = initialGuessLengthscale(height_level_idxs,excited_points, regularizer= lengthscale_regularizer)
        res = minimize( optimizer , param_0, bounds= bounds, method= "L-BFGS-B", tol = 1e-12 )
        param_0 = res.x
        parameters_per_height_level.append(res.x[:-1]) # omit the last one
    
    # find lengthscale model
    lengthscales = np.abs( [x[0] for x in parameters_per_height_level] )
    lengthscaleModel = halfBellLegthscaleModel(lengthscales * (1-underestimate_lengthscales), height_levels) # underestimate 

    return lengthscaleModel

def optimizeInputModel(height_levels,map_repository,lengthscaleModel,excited_points, G_regularizer = 1e-4, lengthscale_regularizer = 1e-3):
    
    parameters_per_height_level = []
    lengthscales_per_height_level = []

    bounds_G = Bounds([-np.inf, -np.inf, -np.inf, -np.inf,  0, -np.inf], 
                [np.inf, np.inf, np.inf, np.inf, np.inf,  np.inf ]) 
    bounds_F = Bounds([ 0.2, -np.inf], 
                [ np.inf, np.inf ])
    
    params_G = np.ones((6))*0.1
    params_F = np.ones((2))*0.1

    for (height_lvl,height_level_idxs) in zip(height_levels,map_repository):

        # initialize lengthscale with calculated guess
        height_lengthscale = lengthscaleModel(height_lvl)

        optimizer = optimizeG( height_level_idxs, excited_points, lengthscale = height_lengthscale,regularizer= G_regularizer)
        res = minimize( optimizer , params_G, bounds= bounds_G, method= "L-BFGS-B", tol = 1e-12 )

        parameters_G = res.x[:-1] # omit the last one (it's f, you will learn it later with m)
        params_G = res.x

        params_F[0] = height_lengthscale
        optimizer = optimizeF( height_level_idxs, excited_points, parameters = parameters_G,regularizer= lengthscale_regularizer)
        res = minimize( optimizer , params_F, bounds= bounds_F, method= "L-BFGS-B", tol = 1e-12 )
        parameters_F = res.x[0]
        
        parameters_per_height_level.append(parameters_G) 
        lengthscales_per_height_level.append(parameters_F)
    
    return np.vstack(parameters_per_height_level), np.vstack( lengthscales_per_height_level  )

def calculateGFeatures(idx,noe,p,bell,lengthscale,resolution):
    if idx>noe:
        final_idx = np.min([ idx + noe + 1 , len(p.T_t_use) - 1 ])
        bell_final_idx = resolution + np.min([ (len(p.T_t_use) - 1) - (idx + noe + 1) , 0  ])
        T = p.T_t_use[ idx-noe+1 : final_idx ] 
        dTdt = T - p.T_t_use[ idx - noe : final_idx - 1 ]
        excitation_heights = p.excitation_delay_torch_height_trajectory[ idx - noe + 1 : final_idx]
        calculated_input = bell [:bell_final_idx]
        T_start_idx = noe - int(lengthscale) * 3
        # T_start_idx,_ = hlp.findDeactivationStartingIdxs(feature_activation = calculated_input)
        T_start = np.ones_like(calculated_input)*T[T_start_idx]
        dummy = 1
    else:
        T = p.T_t_use[ 1 : idx + noe +1 ]
        dTdt = T - p.T_t_use[ : idx + noe ]
        excitation_heights = p.excitation_delay_torch_height_trajectory[ 1 : idx + noe + 1 ]
        calculated_input = bell[- (noe+idx):]
        T_start_idx = 0
        T_start = np.ones_like(calculated_input)*T[T_start_idx]
        dummy = 1
    
    return calculated_input,excitation_heights,T,T_start
    
class initialGuessLengthscale:
    def __init__(self,height_level_idxs,excited_points, regularizer = 0.1):
        self.height_level_idxs = height_level_idxs 
        self.excited_points = excited_points 
        self.regularizer = regularizer

    def __call__(self,params):

        (lengthscale,f) = params
        # upscale, for optimization well poseness
        lengthscale *=10

        # define a bell with current lengthscale
        resolution = 100
        bell = unitBellFunction( lengthscale, resolution = resolution)
        bellDer = unitBellFunctionDerivative( lengthscale, resolution = resolution)
        bell[bellDer<0] = 0

        # define the rest of the paraketrs
        a,b,c,d,e = 1, 0, 0, 0, 0        
        noe = int( len(bell)/2 )

        residuals = 0
        for (idxs,p) in zip( self.height_level_idxs, self.excited_points):
            # in this loop level you have all the idxs mapping to peaks in the thermal data for one node and height level
            new_residuals = 0
            for idx in idxs:
                # calculate necessary things for g
                calculated_input,excitation_heights,T,T_start = calculateGFeatures(idx,noe,p,bell,lengthscale,resolution)
                
                # reconstruct g response
                input_model = trainIpFeature(a,b,c,d,e,calculated_input,excitation_heights,T,T_start)
                
                peak_T = T[-1]
                # learned = np.ones_like(T) * (peak_T + T[0]) * 0.5 + f*np.asarray(input_model)
                learned = np.ones_like(T) * T[0] + f*np.asarray(input_model)
                
                # calculate cost
                error = learned - T
                new_residuals += np.mean(error**2)  
           
            # normalize, regularize and integrate cost
            new_residuals /= len(idxs)
            residuals += new_residuals + self.regularizer*np.linalg.norm(params) 

        # regularize
        # residuals += self.regularizer*np.linalg.norm(params) 

        return residuals        
    
class optimizeG:
    def __init__(self,height_level_idxs,excited_points, lengthscale, regularizer = 0.1):
        self.height_level_idxs = height_level_idxs 
        self.excited_points = excited_points 
        self.regularizer = regularizer
        self.lengthscale = lengthscale

    def __call__(self,params):

        (a,b,c,d,e,f_sc) = params
        
        # upscale, for optimization well poseness
        lengthscale = self.lengthscale * 10

        # define a bell with current lengthscale
        resolution = 100
        bellDer = unitBellFunctionDerivative( lengthscale, resolution = resolution)
        f = unitBellFunctionDerivative( lengthscale, resolution = resolution)
        
        # keep only positive part for input, let negative be handled by the rest
        f[bellDer<0] = 0

        # define the rest of the paraketrs
        noe = int( len(f)/2 )

        residuals = 0
        for (idxs,p) in zip( self.height_level_idxs, self.excited_points):
            # in this loop level you have all the idxs mapping to peaks in the thermal data for one node and height level
            new_residuals = 0
            for idx in idxs:
                # calculate necessary things for g
                calculated_input,excitation_heights,T,T_start = calculateGFeatures(idx,noe,p,f,lengthscale,resolution)
                
                # reconstruct g response
                input_model = []
                state = T[0]
                for (excitation,height) in zip(calculated_input,excitation_heights):    
                    state += trainIpFeature(a,b,c,d,e,excitation,height,state,T_start)
                    input_model.append(state)
                
                peak_T = T[-1]
                learned = np.ones_like(T)* (peak_T + T[0]) * 0.5 + f_sc*np.asarray(input_model)
                
                # calculate cost
                error = learned - T
                new_residuals += np.mean(error**2)  
           
            # normalize and integrate cost
            new_residuals /= len(idxs)
            residuals += new_residuals
        
        #  regularize
        residuals +=  + self.regularizer*np.linalg.norm(params)  

        return residuals        
class optimizeF:
    def __init__(self,height_level_idxs,excited_points, parameters, regularizer = 0.1):
        self.height_level_idxs = height_level_idxs 
        self.excited_points = excited_points 
        self.regularizer = regularizer
        self.parameters = parameters

    def __call__(self,params):

        (lengthscale,f_sc) = params
        (a,b,c,d,e) = self.parameters

        # upscale, for optimization well poseness
        lengthscale = lengthscale * 10

        # define a bell with current lengthscale
        resolution = 100
        bellDer = unitBellFunctionDerivative( lengthscale, resolution = resolution)
        f = unitBellFunctionDerivative( lengthscale, resolution = resolution)
        
        # keep only positive part for input, let negative be handled by the rest
        f[bellDer<0] = 0

        # define the rest of the paraketrs
        # a,b,c,d,e = 1, 0, 0, 0, 0        
        noe = int( len(f)/2 )

        residuals = 0
        for (idxs,p) in zip( self.height_level_idxs, self.excited_points):
            # in this loop level you have all the idxs mapping to peaks in the thermal data for one node and height level
            new_residuals = 0
            for idx in idxs:
                # calculate necessary things for g
                calculated_input,excitation_heights,T,T_start = calculateGFeatures(idx,noe,p,f,lengthscale,resolution)
                
                # reconstruct g response
                input_model = []
                state = T[0]
                for (excitation,height) in zip(calculated_input,excitation_heights):    
                    state += trainIpFeature(a,b,c,d,e,excitation,height,state,T_start)
                    input_model.append(state)
                
                peak_T = T[-1]
                learned = np.ones_like(T)* (peak_T + T[0]) * 0.5 + f_sc*np.asarray(input_model)
                
                # calculate cost
                error = learned - T
                new_residuals += np.mean(error**2)  
           
            # normalize and integrate cost
            new_residuals /= len(idxs)
            residuals += new_residuals
        
        #  regularize
        residuals +=  + self.regularizer*np.linalg.norm(params)  

        return residuals        

class optimizeLayerInputParameters:
    
    def __init__(self,height_level_idxs,excited_points,lengthscale = None, regularizer = 0.001):
        self.height_level_idxs = height_level_idxs 
        self.excited_points = excited_points 
        self.lengthscale = lengthscale
        self.regularizer = regularizer

    def __call__(self,params):
        """
        return the sum over all points of the residuals that the given parameter set resulted in. When you want to define the lengthscale, fit the bell
        to the temperature peaks while for tuning the input model parameters fit the dTdt by extrapolation. 
        """
        (a,b,c,d,lengthscale,e,f) = params
        resolution = 100
        if self.lengthscale is not None:
            lengthscale = self.lengthscale
            bellDer = unitBellFunctionDerivative( lengthscale, resolution = resolution)
            # bell = unitBellFunction( lengthscale, resolution = resolution)
            bell = unitBellFunctionDerivative( lengthscale, resolution = resolution)
            # keep only positive part for input, let negative be handled by the rest
            bell[bellDer<0] = 0

            # works well for short term predictions
            # bell = unitBellFunctionDerivative( lengthscale, resolution = resolution)
            # bell[bell<0] = 0
        else:
            lengthscale *=10
            bell = unitBellFunction( lengthscale, resolution = resolution)
            bellDer = unitBellFunctionDerivative( lengthscale, resolution = resolution)
            bell[bellDer<0] = 0
            a,b,c,d,e = 1, 0, 0, 0, 0
        # e = 1
        # a,b,c,d = 0.1, 0.1, 0.1, 0.1
        # a,b,c,d = 0, 1, 1, 0
        # int_lengthscale = int(lengthscale) # in time samples
        # points_in_lengthscale = int_lengthscale
        noe = int( len(bell)/2 )
        # half_bell = bell[:noe]

        residuals = 0
        for (idxs,p) in zip(self.height_level_idxs,self.excited_points):
            if len(idxs) == 0:
                continue
            # in this loop level you have all the idxs mapping to peaks in the thermal data for one node and height level
            new_residuals = 0
            for idx in idxs:
                calculated_input,excitation_heights,T,T_start = calculateGFeatures(idx,noe,p,bell,lengthscale,resolution)
                
                # propagate T to see how does your input model work
                if self.lengthscale is not None:
                    input_model = []
                    state = T[0]
                    for (excitation,height) in zip(calculated_input,excitation_heights):    
                        # state += trainIpFeature(a,b,c,d,0,excitation,height,state,dTdt)
                        state += trainIpFeature(a,b,c,d,e,excitation,height,state,T_start)
                        input_model.append(state)
                    # input_model = trainIpFeature(a,b,c,d,0,calculated_input,excitation_heights,T,dTdt)
                else:
                    # input_model = trainIpFeature(a,b,c,d,0,calculated_input,excitation_he ights,T,dTdt)
                    input_model = trainIpFeature(a,b,c,d,e,calculated_input,excitation_heights,T,T_start)


                # evaluate residuals
                # error = e*input_model - T
                peak_T = T[-1]
                # error = ( np.ones_like(T)* (T[0]) + e*input_model) - T
                # learned = np.ones_like(T)* (peak_T + T[0])   + f*np.asarray(input_model)
                learned = np.ones_like(T)* T[0]   + f*np.asarray(input_model)
                error = learned - T
                # self.plotResponse(learned,T)
                # error = ( e*input_model) - dTdt
                # new_residuals += 0.8*np.mean( error**2 ) + 0.2*np.max( error**2 ) + np.linalg.norm(params ) #+ np.linalg.norm(params[-1])  
                # new_residuals += np.mean((( np.ones_like(T)* T[0] + e*input_model) - T)**2) 
                new_residuals += np.mean(error**2)  
                # new_residuals += np.mean((( e*input_model) - T)**2) + (a**2+b**2+c**2+d**2) 
            
            weights = np.array([a,b,c,d,e])
            new_residuals /= len(idxs)
            # residuals += new_residuals + self.regularizer * np.linalg.norm(weights)
            residuals += new_residuals + self.regularizer * np.linalg.norm(params)
        
        # regularize
        # residuals += self.regularizer*np.linalg.norm(params) 

        return residuals

    def plotResponse(self,learned,T):
        time = np.arange(len(learned))
        plt.plot(time,learned,time,T)
        plt.show()

def halfBellLegthscaleModel( lengthscale_models, torch_heights_at_excitation ):
    a = np.polyfit(torch_heights_at_excitation,lengthscale_models,1)
    model = np.poly1d(a)
    return model

# useless
def lengthscalesForAllHeights(heights,model):
    lengthscales = np.zeros_like(heights)
    for i,height in enumerate(heights):
        lengthscales[i] = model(height)
    
    return lengthscales
        
def transfromDealyedRelevanceToBellSequence(peak_idxs,lengthscales,array_length,bell_resolution = 100):
    """
    TODO: can you introduce a bell with half of the initial lengthscale to the first part of the signal?
    """
    out = np.zeros((array_length,))
    for (peak_idx,lengthscale) in zip(peak_idxs,lengthscales):

        # form the right half-bell
        input_feature_span = 3*int(lengthscale) # how many non 0 elements to expect in the half bell 
        bell_top_idx = int(bell_resolution/2)  # where the bell top is
        # bell = unitBellFunction(lengthscale, resolution = bell_resolution)   
        bell = unitBellFunction(lengthscale, resolution = bell_resolution)  
        # bell[bell<0] = 0 
        # half_bell_shape = bell[ np.max( [bell_top_idx-input_feature_span,0] ) : bell_top_idx] # keep the non-0 bell elements

        # indexes to insert new half bell
        first_bell_idx_on_input_trajectory = np.max( [peak_idx - input_feature_span, 0] ) 
        last_bell_idx_on_input_trajectory = np.min( [peak_idx + input_feature_span, array_length -1] ) 

        first_bell_idx_on_bell = bell_top_idx - input_feature_span
        last_bell_idx_on_bell = bell_top_idx + input_feature_span

        # check if you are about to over-write a previous excitaion
        more_than_0 = out[first_bell_idx_on_input_trajectory : peak_idx] > 0
        if np.sum(more_than_0)>0:
            final_idx_with_non0_value = len(more_than_0) - np.argmax(more_than_0) - 1

            # move first idx to a non-written idx
            first_bell_idx_on_input_trajectory += final_idx_with_non0_value + 1 
            first_bell_idx_on_bell += final_idx_with_non0_value + 1
        
        # check if your first peak in bell trajectory is 0
        if first_bell_idx_on_input_trajectory==0 :
            first_bell_idx_on_bell = bell_top_idx - peak_idx

        # check if your last peak in bell trajectory is array_length
        if last_bell_idx_on_input_trajectory == array_length-1 :
            last_bell_idx_on_bell = bell_top_idx + array_length - peak_idx -1   
        
        # insert the half-bell to the input sequence
        out[ first_bell_idx_on_input_trajectory : last_bell_idx_on_input_trajectory ] = bell [ first_bell_idx_on_bell : last_bell_idx_on_bell ]

    return out
        
def transfromDealyedRelevanceToHalfBellSequence(peak_idxs,lengthscales,array_length,bell_resolution = 101):

    out = np.zeros((array_length,))
    for (peak_idx,lengthscale) in zip(peak_idxs,lengthscales):

        # form the right half-bell
        input_feature_span = 3*int(lengthscale) # how many non 0 elements to expect in the half bell 
        bell_top_idx = int(bell_resolution/2) + 1 # where the bell top is
        bell = unitBellFunctionDerivative(lengthscale, resolution = bell_resolution)   
        half_bell_shape = bell[ np.max( [bell_top_idx-input_feature_span,0] ) : bell_top_idx] # keep the non-0 bell elements

        # indexes to insert new half bell
        first_half_bell_idx_on_input_trajectory = np.max( [peak_idx - input_feature_span, 0] ) 
        first_half_bell_idx_on_bell = 0

        # check if you are about to over-write a previous excitaion
        more_than_0 = out[first_half_bell_idx_on_input_trajectory : peak_idx] > 0
        if np.sum(more_than_0)>0:
            final_idx_with_non0_value = len(more_than_0) - np.argmax(more_than_0) - 1

            # move first idx to a non-written idx
            first_half_bell_idx_on_input_trajectory += final_idx_with_non0_value + 1 
            first_half_bell_idx_on_bell = final_idx_with_non0_value
        
        # check if your first peak in bell trajectory is 0
        elif first_half_bell_idx_on_input_trajectory==0 :
            first_half_bell_idx_on_bell = input_feature_span - peak_idx    
        
        # insert the half-bell to the input sequence
        out[ first_half_bell_idx_on_input_trajectory : peak_idx ] = half_bell_shape [ first_half_bell_idx_on_bell :  ]

    return out
        
def transfromDealyedRelevanceToHalfBellSequence(peak_idxs,lengthscales,array_length,bell_resolution = 101):

    out = np.zeros((array_length,))
    for (peak_idx,lengthscale) in zip(peak_idxs,lengthscales):

        # form the right half-bell
        input_feature_span = 3*int(lengthscale) # how many non 0 elements to expect in the half bell 
        bell_top_idx = int(bell_resolution/2) + 1 # where the bell top is
        bell = unitBellFunction(lengthscale, resolution = bell_resolution)   
        half_bell_shape = bell[ np.max( [bell_top_idx-input_feature_span,0] ) : bell_top_idx] # keep the non-0 bell elements

        # indexes to insert new half bell
        first_half_bell_idx_on_input_trajectory = np.max( [peak_idx - input_feature_span, 0] ) 
        first_half_bell_idx_on_bell = 0

        # check if you are about to over-write a previous excitaion
        more_than_0 = out[first_half_bell_idx_on_input_trajectory : peak_idx] > 0
        if np.sum(more_than_0)>0:
            final_idx_with_non0_value = len(more_than_0) - np.argmax(more_than_0) - 1

            # move first idx to a non-written idx
            first_half_bell_idx_on_input_trajectory += final_idx_with_non0_value + 1 
            first_half_bell_idx_on_bell = final_idx_with_non0_value
        
        # check if your first peak in bell trajectory is 0
        elif first_half_bell_idx_on_input_trajectory==0 :
            first_half_bell_idx_on_bell = input_feature_span - peak_idx    
        
        # insert the half-bell to the input sequence
        out[ first_half_bell_idx_on_input_trajectory : peak_idx ] = half_bell_shape [ first_half_bell_idx_on_bell :  ]

    return out
        
def transfromDealyedRelevanceToBellDerivativeSequence(peak_idxs,lengthscales,array_length,bell_resolution = 101):

    out = np.zeros((array_length,))
    for (peak_idx,lengthscale) in zip(peak_idxs,lengthscales):

        # form the right half-bell
        input_feature_span = 3*int(lengthscale) # how many non 0 elements to expect in the half bell 
        bell_top_idx = int(bell_resolution/2) + 1 # where the bell top is
        bell = unitBellFunctionDerivative(lengthscale, resolution = bell_resolution) 
        bell[ bell<0 ] = 0  
        half_bell_shape = bell[ np.max( [bell_top_idx-input_feature_span,0] ) : bell_top_idx] # keep the non-0 bell elements

        # indexes to insert new half bell
        first_half_bell_idx_on_input_trajectory = np.max( [peak_idx - input_feature_span, 0] ) 
        first_half_bell_idx_on_bell = 0

        # check if you are about to over-write a previous excitaion
        more_than_0 = out[first_half_bell_idx_on_input_trajectory : peak_idx] > 0
        if np.sum(more_than_0)>0:
            final_idx_with_non0_value = len(more_than_0) - np.argmax(more_than_0) - 1

            # move first idx to a non-written idx
            # first_half_bell_idx_on_input_trajectory += final_idx_with_non0_value + 1
            first_half_bell_idx_on_input_trajectory += final_idx_with_non0_value 
            first_half_bell_idx_on_bell = final_idx_with_non0_value
        
        # check if your first peak in bell trajectory is 0
        elif first_half_bell_idx_on_input_trajectory==0 :
            first_half_bell_idx_on_bell = input_feature_span - peak_idx    
        
        # insert the half-bell to the input sequence
        out[ first_half_bell_idx_on_input_trajectory : peak_idx ] = half_bell_shape [ first_half_bell_idx_on_bell :  ]

    return out

def modelsForInputCoefs(coefs,heights):
    # return interp1d(heights,coefs,axis = 0)
    
    heights = np.atleast_2d(heights)
    if heights.shape[1]>heights.shape[0]:
        heights = heights.T
    
    coefs = np.atleast_2d(coefs)
    if coefs.shape[1]>coefs.shape[0]:
        coefs = coefs.T
    
    coef_model = MultiOutputRegressor(LinearRegression(fit_intercept=True, normalize= True))
    coef_model.fit( heights, coefs)
    
    return coef_model.predict


def trainIpFeature(a,b,c,d,e,calculated_inputs,excitation_heights,T,T_start):
    """
    @params a,b,c,d for model the coefficient
    """
    # coef = np.abs( b/(a*excitation_heights + c ) + d * T )
    # coef = np.abs( b/(a*excitation_heights + c )  )
    # coef = np.abs( a + b * T[0] )
    # coef =  a + b * T # they cancel out 
    # coef =  a 
    # coef = np.abs( a/(excitation_heights + b ) + c * T + d / T )
    # coef = np.abs( a/(excitation_heights + b )  + d / T )
    # coef = np.abs( a/(excitation_heights + b )  + c * T )

    # coef =  a + b * excitation_heights
    # coef =  a  + b *(1 - excitation_heights) + c *(1 - T_start) + d * excitation_heights *(1 - T_start)
    coef =  a + b *(1 - excitation_heights) + c * (1 - T) + d * excitation_heights * (1 - T)
    # coef =  a + b *(1 - excitation_heights) + c * (1 - T) + d * excitation_heights * (1 - T) + e * (1 - T_start)
    # coef =  a + b *(1 - excitation_heights) + c * (1 - T) + d * excitation_heights * (1 - T)
    # coef =  a + b * excitation_heights + c / (T+0.00001) 
    # coef =  a + b * excitation_heights + c * (1/(T + d) )
    out = calculated_inputs * coef
    
    return out


# unit bell used to structure the input sequence
def unitBellFunction( lengthscale, resolution = 101, center = 0.0):
   
    # x = np.linspace( -3*lengthscale, 3*lengthscale, 6*resolution_per_lengthscale )
    x = np.linspace( -50, 50,resolution)
    out = np.exp( -(( (x -  center) / lengthscale )**2) / (2**0.5) )

    return out 

def unitBellFunctionDerivative( lengthscale, resolution = 101, center = 0.0):
   
    # x = np.linspace( -3*lengthscale, 3*lengthscale, 6*resolution_per_lengthscale )
    # x = np.linspace( -50, 50,resolution)
    # coef = (( 1 / lengthscale )**2) / (2**0.5)
    # dx = (x -  center)
    # out = -2 * coef * dx * np.exp( -coef * dx**2 )

    bell = unitBellFunction(lengthscale, resolution, center)
    delta_bell = np.zeros_like(bell)
    delta_bell[1:] = np.diff(bell) 
    delta_bell[0] = delta_bell[1]

    return delta_bell/np.max(delta_bell) 

# interpolated parameters for the coefficients of the ip model  
def interpolateInputParameters(ipCoefModel, excitation_delay_torch_height_trajectory):
    
    excitation_delay_torch_height_trajectory = np.atleast_2d(excitation_delay_torch_height_trajectory)
    if excitation_delay_torch_height_trajectory.shape[1]>excitation_delay_torch_height_trajectory.shape[0]:
        excitation_delay_torch_height_trajectory = excitation_delay_torch_height_trajectory.T
        
    parameters = ipCoefModel(excitation_delay_torch_height_trajectory)
    
    a = parameters[:,0]
    b = parameters[:,1]
    c = parameters[:,2]
    d = parameters[:,3]
    e = np.zeros_like(a)
    
    return a, b, c, d, e

def debugPlotBellTraining(learned,T,bell):
    plt.figure(figsize=(16,9))
    plt.plot(learned, label = "learned")
    plt.plot(T, label = "T")
    plt.plot(bell, label = "bell")
    plt.legend()
    plt.show()
    return None
# %%
