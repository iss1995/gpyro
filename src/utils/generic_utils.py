 #!/usr/bin/env python3
# %%
import numpy as np

import torch
from copy import deepcopy 

# %%
def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    # pad = np.full((len(a.shape),window),a[0]) # iss
    pad[-1] = window-1
    pad = list(zip(pad, np.ones(len(a.shape), dtype=np.int32)))
    # pad = list(zip(pad, np.ones(len(a.shape)))) # iss
    # a = np.pad(a, pad,mode='constant')
    a = np.pad(a, pad,mode='edge')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    try:
        strides = a.strides + (a.strides[1],) # not sure why this fails sometimes
    except:
        strides = a.strides + (a.strides[0],) # but when the other fails this works
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides) 

def moving_average(x, w):
    extended = np.pad(x,(int(w/2),int(w/2)-1),'edge')
    return np.convolve(extended, np.ones(w), 'valid') / w

def movingAvg(x, w):
    """
    only pad towards one side.
    """
    if max(x.shape)>w:
        out = np.convolve(x, np.ones(w), 'valid') / w
    else:    
        extended = np.pad(x,(int(w)-1,0),'edge')
        out = np.convolve(extended, np.ones(w), 'valid') / w
    return out

def setDevice(choice = None):
    '''
    set device for torch. Default is cpu (if available)
    @param choice: string with choice of device
    @return torch device
    '''

    import gc

    if choice is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gc.collect()
            torch.cuda.empty_cache()
            
        else:
            device = torch.device('cpu')
    
    else:
        device = torch.device(choice)

    return deepcopy(device)

def checkConcave_(idx,vector):
    "Only locally"
    try:
        if vector[idx]>vector[idx+1] and vector[idx]>vector[idx-1]:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return None

def tryconvert(value, default, *types):
    """
    try to convert value to types and if you fail return default.
    """
    for t in types:
        try:
            return t(value)
        except (ValueError, TypeError):
            continue
    return default

def pointsOnBoundaries_(experiment):
    
    points_used_for_training = [p for p in experiment.Points if p._hasNeighbors_()]
    coordinate_list = [p.coordinates for p in points_used_for_training]
    coordinates = np.vstack(coordinate_list)
    boundaries = findDomainLimits(coordinates)
    on_boundary_ = [_isOnBoundary_(p,boundaries) for p in points_used_for_training]
    
    # find cornenrs as well
    # on_corner_ = [_isOnCorner_(p,boundaries) for p in points_used_for_training]
    # on_corner_ = [False for p in points_used_for_training]

    return on_boundary_

def splitDataByState( training_points, steps_to_look_ahead = 1):

    # find unique deposition heights. -1 for not being activated
    all_heights = np.vstack([p.excitation_delay_torch_height_trajectory for p in training_points]).T
    unique_heights = np.unique(all_heights)

    # for each point find the idxs of the every height level
    height_level_idxs_container = [ ] # list of lists -> every list corresponds to a point and includes the idxs for every height level  
    for p in training_points:
        tmp = []

        for height in unique_heights:
            idxs_on_layer = np.asarray(np.where(p.excitation_delay_torch_height_trajectory == height))
            tmp.append(idxs_on_layer)
        
        height_level_idxs_container.append(tmp)

    # create the training segments
    xs_train = []
    ys_train = []
    band_heights = []
    for state,height in enumerate(unique_heights):
        xs_current_state = []
        ys_current_state = []
        heights_current_state = []
        
        # assign only data that were collected when printing this state 
        for (p,point_idxs) in zip(training_points,height_level_idxs_container):
            idxs_to_pick = np.atleast_1d(point_idxs[state].squeeze())
            if len(idxs_to_pick)>0:
                xs_current_state_point = p.features[:,idxs_to_pick]
                ys_current_state_point = p.T_t_use[idxs_to_pick] - p.T_t_use[idxs_to_pick-1]

                xs_current_state.append( xs_current_state_point[:,:-steps_to_look_ahead] )
                ys_current_state.append( ys_current_state_point[steps_to_look_ahead:] )

                heights_current_state.append ( np.ones( ( len(idxs_to_pick)-1, ) )*height )
        
        xs_train.append(np.hstack( xs_current_state ))
        ys_train.append( np.hstack(ys_current_state ))
        band_heights.append( np.hstack(heights_current_state ))
    
    return xs_train, ys_train, unique_heights, height_level_idxs_container

def findDomainLimits(coordinates):
    """
    For rectangular domain.
    """
    
    xMax = np.max(coordinates[:,0])
    xMin = np.min(coordinates[:,0])
    yMax = np.max(coordinates[:,1])
    yMin = np.min(coordinates[:,1])
    return (xMax,xMin,yMax,yMin)

def _isOnBoundary_(p,boundaries):
    
    [xMax,xMin,yMax,yMin] = boundaries
    coordinates = p.coordinates 
    if coordinates[0] == xMin or coordinates[0] == xMax:
        periphery_ = True
    elif coordinates[1] == yMin or coordinates[1] == yMax:
        periphery_ = True
    else:
        periphery_ = False

    return periphery_

def _isOnCorner_(p,boundaries):
    
    [xMax,xMin,yMax,yMin] = boundaries
    coordinates = p.hallucinated_nodes_coordinates 

    count = 0
    for coordinate in coordinates:
        if (coordinate[0] < xMin or coordinate[0] > xMax) or (coordinate[1] < yMin or coordinate[1] > yMax):
            count += 1

    if count == 2:
        corner_ = True
    else:
        corner_ = False
    return corner_