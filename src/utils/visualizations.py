from cProfile import label
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import utils.extrapolation_utils as exu
import utils.online_optimization_utils as onopt

import os
import torch
import gpytorch

from itertools import product
from scipy.interpolate import griddata
from copy import deepcopy as copy


font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 11 }

mpl.rc('font', **font)

a4_width = 17   # cm
a4_height = 25.7   # cm
phi = 1.61803398875
cm = 1/2.54

def stdPlot(std_T, std_delta_T, validation_experiment, starting_point, steps, RESULTS_FOLDER_ERROR_EVOLUTION, file):
    fig = plt.figure( figsize=(a4_width*cm,a4_width/phi*cm) )
    fig.patch.set_facecolor("white")
    x_axis = [0.1*validation_experiment.subsample*i for i in range(starting_point, starting_point + steps + 1,1)]

    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot( x_axis, std_T, label = "Temperature")
    ax2.plot( x_axis, std_delta_T, label = "Temperature differences")

    ax1.set_title("Full plate predictions")
    ax2.set_xlabel("Time (s)")

    ax1.legend()
    ax2.legend()

    plt.savefig(RESULTS_FOLDER_ERROR_EVOLUTION + f"/stds_evolution_file_{file}.pdf")
    plt.close()

def plotNodeSubplot( T_state_np_central_plate, timestamps, T_state_nominal_np_central_plate, nodes_in_subplot, RESULTS_FOLDER ,file, title = None):
    y_max = np.max(T_state_np_central_plate)
    y_min = np.min(T_state_np_central_plate)
    dy = y_max - y_min
    y_max += dy * 0.01
    y_min -= dy * 0.01

    x_max = timestamps[-1]
    x_min = timestamps[0]

    folder_for_node_plots_for_experiment = RESULTS_FOLDER + f"/nodes_time/{file}"
    os.makedirs(folder_for_node_plots_for_experiment, exist_ok = True)
    fig = plt.figure( figsize=(a4_width*cm,a4_width/phi*cm) )

    fig.patch.set_facecolor("white")
    axs = fig.subplots(3,2,sharex="row",sharey="col")
    x_axis = timestamps

    j = 0
    for ax in axs:

        for ax1 in ax:
            i = nodes_in_subplot[j]
            for axis in ['top', 'bottom', 'left', 'right']:
                ax1.spines[axis].set_linewidth(2.5)

            ax1.plot( x_axis, T_state_nominal_np_central_plate[:,i], label = "Measured", linewidth = 3)
            ax1.plot( x_axis, T_state_np_central_plate[ :,i], label = "Modelled",linestyle = "-.", linewidth = 3, color = "darkorange")

            if title is None:
                ax1.set_title(f"Node {i}",pad = 2)
            else:
                ax1.set_title(f"{title} {i}",pad = 2)

            ax1.set_xlim([x_min, x_max])
            ax1.set_ylim([y_min, y_max])

            # ax1.grid(True,which = "both",axis = "both",alpha = 0.5)
            if j == 5 or j == 4:
                # ax1.legend(loc = 4, bbox_to_anchor = (0.2,-0.22,0.82,0.5))
                ax1.set_xlabel("Time (s)")
            else:
                ax1.set_xticklabels([])

            if j%2>0 :
                ax1.set_yticklabels([])
            else:
                ax1.set_yticks([250,500,750])

            if j == 2:
                ax1.set_ylabel("Temperature")

            j += 1

    for ax in axs:
        for ax1 in ax:
            ax1.grid(True,alpha = 0.5)

    fig.tight_layout(h_pad = 0.3)
    # plt.show()
    plt.savefig(folder_for_node_plots_for_experiment + f"/subplots.pdf", bbox_inches='tight', dpi = 120)
    plt.close()
    return None

def  plotGPWeights( likelihoods, models, RESULTS_FOLDER, states = None, device = None, xticks = None, yticks = None, id = "",title = None):
    # plot GPs
    n = len(likelihoods.likelihoods)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if states is None:
            low = -0.5
            up = 0.5
        else:
            up = np.max(states) 
            low = np.min(states)
            up += 0.4* np.abs(up)  
            low -= 0.4* np.abs(low)  

        test_x = torch.linspace(low, up, np.abs(int((up-low)*100))).float().to(device = device).detach()
        # This contains predictions for both outcomes as a list
        ins = [copy(test_x) for i in range(n)]
        predictions = likelihoods(*models(*ins))
    
    path = RESULTS_FOLDER + 'GP_weights/'
    os.makedirs(path, exist_ok = True)
    x_axis = test_x.cpu().numpy()

    for i,(submodel, prediction) in enumerate(zip(models.models, predictions)):
        f, ax = plt.subplots(1,1, figsize=(a4_width*cm*0.32,a4_width*cm*0.32))
        mean = prediction.mean.squeeze()
        lower, upper = prediction.confidence_region()

        tr_x = submodel.train_inputs[0].cpu().detach().numpy().squeeze()
        tr_y = submodel.train_targets.cpu().detach().numpy().squeeze()

        # # Predictive mean as blue line
        tr_x = submodel.train_inputs[0].cpu().detach().numpy().squeeze()
        tr_y = submodel.train_targets.cpu().detach().numpy().squeeze()
        ax.plot( x_axis, mean, label = "Measured", linewidth = 3,zorder=5)
        ax.fill_between(test_x, lower.squeeze().cpu().detach().numpy(), upper.squeeze().cpu().detach().numpy(), alpha=0.5,zorder=0)
        ax.scatter(tr_x, tr_y, marker = '*', s = 30,color = "black",zorder=10)
        # # Shade in confidence
        # ax.legend(['Observed Data', 'Mean', 'Confidence'],loc = "upper left")
        # ax.set_title(f'Feature {i}')
        # ax.set_xlabel("State")
        # ax.set_ylabel("Norm Value")

        if title:
            ax.set_title(f"{title} {i}",pad = 2)

        if np.any(xticks):
            dx = np.abs(np.max(xticks) - np.min(xticks))
            ax.set_xlim( [xticks[0] - 0.01*dx,  xticks[-1] + 0.01*dx] )
        else:
            ax.set_xlim([low,up])

        if np.any(yticks):
            dy = np.abs(np.max(yticks) - np.min(yticks))
            ax.set_ylim( [yticks[0] - 0.01*dy,  yticks[-1] + 0.01*dy] )
            # if j%3>0 :
            #     ax.set_yticks(yticks)
            #     ax.set_yticklabels([])
            # else:
            ax.set_yticks(yticks)
            ax.set_yticklabels([])
        else:
            min_val = np.min( lower.squeeze().cpu().detach().numpy() )
            max_val = np.max( upper.squeeze().cpu().detach().numpy() )

            dy = np.abs( max_val - min_val )
            ax.set_ylim( [min_val - 0.01*dy,  max_val + 0.01*dy] )

        ax.grid(True,alpha = 0.5)

        f.tight_layout()

        plt.savefig( path + f"{i}_" + id + ".png", bbox_inches='tight',dpi = 180)
        plt.savefig( path + f"{i}_" + id + ".svg", bbox_inches='tight',dpi = 180)
        plt.close(f)

def plotWeightsSubplot(likelihoods, models, RESULTS_FOLDER, weights_in_subplot, states = None, device = None, xticks = None, yticks = None, id = "",title = None):
    # plot GPs
    n = len(likelihoods.likelihoods)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        if states is None:
            up = 0.5
            low = -0.1
        else:
            up = np.max(states) 
            low = np.min(states)
            up += 0.4* np.abs(up)  
            low -= 0.4* np.abs(low)  
        test_x = torch.linspace(low, up, np.abs( int((up-low)*100) ) ).float().to(device = device).detach()

        # This contains predictions for both outcomes as a list
        ins = [copy(test_x) for i in range(n)]
        predictions = likelihoods(*models(*ins))
        # predictions = likelihoods(*models(*[copy(test_x) for i in range(n)]))

    path = RESULTS_FOLDER + 'GP_weights/'
    os.makedirs(path, exist_ok = True)
    fig = plt.figure( figsize=(a4_width*cm,a4_width/phi*cm/2) )

    fig.patch.set_facecolor("white")
    big_ax = fig.add_subplot(111)

    axs = fig.subplots(1,3,sharex="row",sharey="col")
    x_axis = test_x.cpu().numpy()

    j = 0
    for ax1 in axs:

        # for ax1 in ax:
        i = weights_in_subplot[j]
        submodel, prediction = models.models[i], predictions[i]
        mean = prediction.mean.squeeze().cpu().numpy()
        lower, upper = prediction.confidence_region()
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(0.5)

        
        tr_x = submodel.train_inputs[0].cpu().detach().numpy().squeeze()
        tr_y = submodel.train_targets.cpu().detach().numpy().squeeze()
        ax1.plot( x_axis, mean, label = "Measured", linewidth = 3,zorder=5)
        ax1.fill_between(test_x, lower.squeeze().cpu().detach().numpy(), upper.squeeze().cpu().detach().numpy(), alpha=0.5,zorder=0)
        ax1.scatter(tr_x, tr_y, marker = '*', s = 30,color = "black",zorder=10)

        if title:
            ax1.set_title(f"{title} {i}",pad = 2)

        if np.any(xticks):
            dx = np.abs(np.max(xticks) - np.min(xticks))
            ax1.set_xlim( [xticks[0] - 0.01*dx,  xticks[-1] + 0.01*dx] )
        else:
            ax1.set_xlim([low,up])

        if np.any(yticks):
            dy = np.abs(np.max(yticks) - np.min(yticks))
            ax1.set_ylim( [yticks[0] - 0.01*dy,  yticks[-1] + 0.01*dy] )
            if j%3>0 :
                ax1.set_yticks(yticks)
                ax1.set_yticklabels([])
            else:
                ax1.set_yticks(yticks)

        # ax1.grid(True,which = "both",axis = "both",alpha = 0.5)
        # if j >2:
        #     # ax1.legend(loc = 4, bbox_to_anchor = (0.2,-0.22,0.82,0.5))
        #     ax1.set_xlabel("State")
        # else:
        #     ax1.set_xticklabels([])

        # if j == 2:
        #     ax1.set_ylabel("Temperature")

        j += 1

    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    big_ax.set_xlabel("State")
    big_ax.set_ylabel("Norm. Weights")

    for ax1 in axs:
        # for ax1 in ax:
        ax1.grid(True,alpha = 0.5)

    fig.tight_layout(h_pad = 0.)
    # plt.show()
    plt.savefig(path + f"/{id}_weight_subplots.pdf", bbox_inches='tight', dpi = 120)
    plt.savefig(path + f"/{id}_weight_subplots.svg", bbox_inches='tight', dpi = 120)
    plt.close()
    return None

def plotNodeEvolution( T_state_np_central_plate, timestamps, T_state_nominal_np_central_plate, RESULTS_FOLDER ,file, title = None, xlabel = None, ylabel = None, yticks = [200,400,600], xticks = [500,1000,1500,2000], ytikcs_label = None, xtikcs_label = None ):

    y_max = np.max(T_state_np_central_plate)
    y_min = np.min(T_state_np_central_plate)
    dy = y_max - y_min
    y_max += dy * 0.01
    y_min -= dy * 0.01

    x_max = timestamps[-1]
    x_min = timestamps[0]


    folder_for_node_plots_for_experiment = RESULTS_FOLDER + f"/nodes_time/{file}"
    os.makedirs(folder_for_node_plots_for_experiment, exist_ok = True)
    for i in range(T_state_nominal_np_central_plate.shape[-1]):

        fig = plt.figure( figsize=(a4_width*0.45*cm,a4_width*0.45/phi*cm/3) )
        fig.patch.set_facecolor("white")
        # fig.patch.set_alpha(0)
        x_axis = timestamps

        ax1 = fig.add_subplot(111)
        # ax1.patch.set_alpha(0)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(0.5)

        ax1.plot( x_axis, T_state_nominal_np_central_plate[:,i], label = "Measured", linewidth = 3)
        ax1.plot( x_axis, T_state_np_central_plate[ :,i], label = "Modelled",linestyle = "-.", linewidth = 3, color = "darkorange")

        if title is not None:
            ax1.set_title(f"{title} {i}")

        if xlabel is not None:
            ax1.set_xlabel(xlabel)
        if ylabel is not None:
            ax1.set_xlabel(ylabel)

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])

        plt.grid(True,alpha = 0.5)
        # plt.grid(True,axis = "y")
        # if i == 22 or i == 48:
        #     ax1.legend(loc = 4, bbox_to_anchor = (0.2,-0.22,0.82,0.5))
        #     plt.xticks([])
        plt.yticks(yticks)
        if ytikcs_label is None:
            ax1.set_yticklabels([])
        else:
            ax1.set_yticklabels(ytikcs_label)
        
        plt.xticks(xticks)
        if xtikcs_label is None:
            ax1.set_xticklabels([])
        else:
            ax1.set_xticklabels(xtikcs_label)

        plt.savefig(folder_for_node_plots_for_experiment + f"/node_{i}.svg", bbox_inches='tight', dpi = 120)
        plt.close()

def plotNodeUncertainEvolution( T_mean, l_b, u_b, timestamps, T_mean_nominal, RESULTS_FOLDER ,file, title = None, xlabel = None, ylabel = None, yticks = [200,400,600,800,1000], xticks = [500,1000,1500,2000], ytikcs_label = None, xtikcs_label = None ):

    y_max = np.max(u_b)
    y_min = np.min(l_b)
    dy = y_max - y_min
    y_max += dy * 0.01
    y_min -= dy * 0.01

    x_max = timestamps[-1]
    x_min = timestamps[0]


    folder_for_node_plots_for_experiment = RESULTS_FOLDER + f"/nodes_time/{file}"
    os.makedirs(folder_for_node_plots_for_experiment, exist_ok = True)
    for i in range(T_mean_nominal.shape[-1]):

        fig = plt.figure( figsize=(a4_width*cm,a4_width/phi*cm/3) )
        fig.patch.set_facecolor("white")
        # fig.patch.set_alpha(0)
        x_axis = timestamps

        ax1 = fig.add_subplot(111)
        # ax1.patch.set_alpha(0)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(0.5)

        ax1.fill_between(x_axis, u_b[:,i], l_b[:,i], alpha = 0.5, color = "darkorange" ,zorder=0)
        ax1.plot( x_axis, T_mean[ :,i], label = "Modelled",linestyle = "-.", linewidth = 2, color = "darkorange",zorder=15)
        ax1.plot( x_axis, l_b[:,i], linestyle = "-.", linewidth = 1, color = "darkorange",zorder=10)
        ax1.plot( x_axis, T_mean_nominal[:,i], label = "Measured", linewidth = 2,zorder=5)
        ax1.plot( x_axis, u_b[:,i], linestyle = "-.", linewidth = 1, color = "darkorange",zorder=10)

        if title is not None:
            ax1.set_title(f"{title} {i}")

        if xlabel is not None:
            ax1.set_xlabel(xlabel)
        if ylabel is not None:
            ax1.set_xlabel(ylabel)

        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])

        plt.grid(True,alpha = 0.5)
        # plt.grid(True,axis = "y")
        # if i == 22 or i == 48:
        #     ax1.legend(loc = 4, bbox_to_anchor = (0.2,-0.22,0.82,0.5))
        #     plt.xticks([])
        plt.yticks(yticks)
        if ytikcs_label is None:
            ax1.set_yticklabels([])
        else:
            ax1.set_yticklabels(ytikcs_label)
        
        plt.xticks(xticks)
        if xtikcs_label is None:
            ax1.set_xticklabels([])
        else:
            ax1.set_xticklabels(xtikcs_label)

        plt.savefig(folder_for_node_plots_for_experiment + f"/node_prob_{i}.svg", bbox_inches='tight', dpi = 120)
        plt.savefig(folder_for_node_plots_for_experiment + f"/node_prob_{i}.png", bbox_inches='tight', dpi = 120)
        plt.close()

def tempExtrapolationPlot(ins, plot = False):

    # unpack
    if len(ins)== 8:
        (validation_experiment, T_state_np, inputs, starting_idx, steps, layer, node, RESULTS_FOLDER ) = ins
    elif len(ins) == 7:
        (validation_experiment, T_state_np, inputs, starting_idx, steps, layer, node) = ins
        RESULTS_FOLDER = "results/general_dbging"
    else:
        raise AssertionError()

    if plot:
        fig = plt.figure(figsize = (a4_width*0.45*cm,a4_width*0.45/phi*cm))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(311)
        x_axis = np.linspace( starting_idx, starting_idx + steps, steps+1)*0.5
        ax.plot(x_axis, T_state_np[:,node], label = "simulated")
        ax.plot(x_axis[1:], inputs[  :, node], label = "in1", color = "indianred")
        # ax.plot(x_axis[1:], DEBUG_inputs[ :, node], label = "in1", color = "indianred")
        ax.plot(x_axis,validation_experiment.Points[node].T_t_use[starting_idx : starting_idx+steps+1], label = "real")
        ax.plot(x_axis,validation_experiment.Points[node].relevance_coef_sharp[starting_idx : starting_idx+steps+1], label = "torch")
        ax.legend(bbox_to_anchor=(1.1, 1))
        ax.set_title(f"node {validation_experiment.Points[node].node} Temperatures")
        ax.set_ylabel("Temperature [norm]")

        ax.set_ylim([-0.05, 1.05])

        # os.makedirs( RESULTS_FOLDER + f"/DEBUG/node{node}evolution", exist_ok = True)
        # plt.savefig( RESULTS_FOLDER + f"/DEBUG/node{node}evolution/temperature_extrapolation_node{node}_layer{layer}.pdf", bbox_inches = 'tight')
        plt.savefig( RESULTS_FOLDER + f"/temperature_extrapolation_node{validation_experiment.Points[node].node}_layer{layer}.pdf", bbox_inches = 'tight')

        plt.close(fig)
    # if node != 0 or node != 13:
    #     plt.close(fig)

    # calculate dtw distance
    *_, normalized_distance, std_step_cost_increase, max_step_cost_increase, min_step_cost_increase  = exu.synchronizeWithDtw( T_state_np[:,node], validation_experiment.Points[node].T_t_use[starting_idx : starting_idx+steps+1])

    return normalized_distance, std_step_cost_increase, min_step_cost_increase, max_step_cost_increase


def testBellSeq(excited_points,bell_sequences,ipCoefModel):
    fig = plt.figure(figsize = (a4_width*cm,a4_width*cm))

    node_to_plot = 0
    p = excited_points[node_to_plot]
    (a,b,c,d,e) = ipCoefModel(p.excitation_delay_torch_height_trajectory)
    inputs_new =  onopt.G(a,b,c,d,e,bell_sequences[node_to_plot],p.excitation_delay_torch_height_trajectory,p.T_t_use,p.T_at_the_start_of_input)

    ax1 = fig.add_subplot(211)
    ax1.plot(bell_sequences[node_to_plot], label = "excitation")
    ax1.plot(p.T_t_use, label = "Temperature")
    ax1.plot(inputs_new, label = "Input feature")
    ax1.plot(p.T_at_the_start_of_input, label = "T start", alpha = 0.5)
    ax1.legend(title =  f"node {p.node}")

    ax1.set_title("Half Bell-like structures")

    node_to_plot = 9
    p = excited_points[node_to_plot]
    (a,b,c,d,e) = ipCoefModel(p.excitation_delay_torch_height_trajectory)
    inputs_new = onopt.G(a,b,c,d,e,bell_sequences[node_to_plot],p.excitation_delay_torch_height_trajectory,p.T_t_use,p.T_at_the_start_of_input)

    ax2 = fig.add_subplot(212)
    ax2.plot(bell_sequences[node_to_plot], label = "excitation")
    ax2.plot(p.T_t_use, label = "Temperature")
    ax2.plot(inputs_new, label = "Input feature")
    ax2.plot(p.T_at_the_start_of_input, label = "T start", alpha = 0.5)
    ax2.legend(title = f"node {p.node}")


def plotContourIntrep(field_value,point_locations,steps_to_plot = None,d_grid = 27, result_folder = "results/general_dbging", field_value_name_id = "", title = None, colorbar_scaling = None, grid_dim = 100):
    """
    Plot a grid with element centers being the nodes, and colour it according to the field value at the corresponding node.
    @params field_value : field value to plot with dimensions n x num_of_points_on_grid
    @params point_locations : array with node coordinates (x,y) with dimensions 2 x num_of_points_on_grid
    @params steps_to_plot : itterable with the steps to plot the field_value
    @params d_grid : int with the distance between nodes
    @params results_folder
    @params field_value_name_id : string for naming the saved files and the colorbar
    @params colorbar_scaling : array including min and max values of the illustrated fields
    """
    destination_folder = result_folder + "/contour_plots"
    os.makedirs( destination_folder, exist_ok = True)

    if steps_to_plot is None:
        steps_to_plot = np.arange(0,field_value.shape[0]-1,5000)

    # build grid locations (coordinates of each tile's corners)
    xs_contour = point_locations[0,:].reshape(int(np.sqrt(len(point_locations))),-1) - d_grid/2
    ys_contour = point_locations[1,:].reshape(int(np.sqrt(len(point_locations))),-1) - d_grid/2
    xs_contour = np.column_stack( (xs_contour, xs_contour[:,-1] + d_grid)) # add extra element on every row
    xs_contour = np.row_stack( (xs_contour, xs_contour[-1,:] )) # add extra row for the extra y element
    ys_contour = np.row_stack( ( ys_contour[0,:] + d_grid, ys_contour)) # add extra element on every column
    ys_contour = np.column_stack( (ys_contour, ys_contour[:,-1] )) # add extra column for the extra x element

    # fill dimensions with interpolated values
    # x_axis = np.linspace(np.max(xs_contour) - 2*d_grid/3,np.min(xs_contour) + 2*d_grid/3,grid_dim)
    # y_axis = np.linspace(np.min(ys_contour) + 2*d_grid/3,np.max(ys_contour) - 2*d_grid/3,grid_dim)
    x_axis = np.linspace(np.max(xs_contour),np.min(xs_contour) ,grid_dim)
    y_axis = np.linspace(np.min(ys_contour),np.max(ys_contour) ,grid_dim)
    # x_axis = np.linspace(np.min(point_locations[0,:]),np.max(point_locations[0,:]),grid_dim)
    # y_axis = np.linspace(np.min(point_locations[1,:]),np.max(point_locations[1,:]),grid_dim)

    all_points = np.vstack(list(product(x_axis,y_axis))).T
    xs_contour_interp = all_points[1,:].reshape(grid_dim,grid_dim)
    ys_contour_interp = all_points[0,:].reshape(grid_dim,grid_dim)
    # all_points = np.array(list(product(x_axis,y_axis))).T
    # xs_contour_interp = all_points[0,:].reshape(grid_dim,-1) - d_grid/2/grid_dim
    # ys_contour_interp = all_points[1,:].reshape(grid_dim,-1) - d_grid/2/grid_dim
    # xs_contour_interp = np.column_stack( (xs_contour_interp, xs_contour_interp[:,-1] + d_grid/grid_dim)) # add extra element on every row
    # xs_contour_interp = np.row_stack( (xs_contour_interp, xs_contour_interp[-1,:] )) # add extra row for the extra y element
    # ys_contour_interp = np.row_stack( ( ys_contour_interp[0,:] + d_grid/grid_dim, ys_contour_interp)) # add extra element on every column
    # ys_contour_interp = np.column_stack( (ys_contour_interp, ys_contour_interp[:,-1] )) # add extra column for the extra x element


    # scale colorbar
    if colorbar_scaling is None:
        min_val = field_value.min()
        max_val = field_value.max()
    else:
        min_val = np.min(colorbar_scaling)
        max_val = np.max(colorbar_scaling)

    # plot for each step
    for step in steps_to_plot:
        field_values_on_step = field_value[step,:]
        # field_interp_step = interp2d( point_locations[0,:],  point_locations[1,:],field_values_on_step, kind='linear' )
        # field_values_on_step_interp = field_interp_step(x_axis,y_axis)
        field_values_on_step_interp = griddata(point_locations.T,field_values_on_step,all_points.T, method = "cubic")
        # print(field_values_on_step_interp.shape)
        field_values_plane_on_step = np.squeeze(field_values_on_step_interp.reshape((grid_dim,-1)))

        fig = plt.figure(figsize=(a4_width*cm*0.45,a4_width*cm*0.45))
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(0)
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.5)
        # plot contours
        the_plot = ax.pcolormesh(xs_contour_interp,ys_contour_interp, field_values_plane_on_step, vmin = min_val, vmax = max_val, cmap = "hot", shading = "nearest")

        # plot structure walls
        line_coord = [-81,-27,27,81]
        for coord in line_coord:
            # parallel_lines
            ax.plot( [-81,81], [coord,coord], color = "grey", linewidth = 2)
            # vertical_lines
            ax.plot( [coord,coord], [-81,81], color = "grey", linewidth = 2)

        # plot nodes
        for i in range(point_locations.shape[-1]):
            if (np.abs(point_locations[0,i]) == 108) or (np.abs(point_locations[1,i]) == 108):
                ax.scatter(x = point_locations[0,i],y = point_locations[1,i], marker = "o", color = "black", linewidth = 2)
            else:
                ax.scatter(x = point_locations[0,i],y = point_locations[1,i], marker = "x", color = "grey", linewidth = 2, s = 30)

        # ax.set_title(f"Step {step}")
        # ax.set_xlabel("x (mm)")
        # ax.set_ylabel("y (mm)")
        cbar = fig.colorbar(the_plot,ax = ax)
        # cbar.ax.get_yaxis().labelpad = 15
        # cbar.ax.set_ylabel("", rotation=270)
        plt.xticks([])
        plt.yticks([])

        if title is not None:
            ax.set_title(label=title)

        plt.savefig(destination_folder + "/" + field_value_name_id + f"_contour2_step_{step}.pdf", bbox_inches = 'tight' , dpi = 120)
        plt.close()

def plotAllContours(field_value,target_field_value,point_locations,steps_to_plot = None,d_grid = 27, result_folder = "results/general_dbging", field_value_name_id = "", title = None, colorbar_scaling = None):
    """
    Plot a grid with element centers being the nodes, and colour it according to the field value at the corresponding node.
    @params field_value : field value to plot with dimensions n x num_of_points_on_grid
    @params point_locations : array with node coordinates (x,y) with dimensions 2 x num_of_points_on_grid
    @params steps_to_plot : itterable with the steps to plot the field_value
    @params d_grid : int with the distance between nodes
    @params results_folder
    @params field_value_name_id : string for naming the saved files and the colorbar
    @params colorbar_scaling : array including min and max values of the illustrated fields
    """

    destination_folder = result_folder + "/contour_plots"
    os.makedirs( destination_folder, exist_ok = True)

    if steps_to_plot is None:
        steps_to_plot = np.arange(0,field_value.shape[0]-1,5000)

    # build grid locations (coordinates of each tile's corners)
    xs_contour = point_locations[0,:].reshape(int(np.sqrt(field_value.shape[-1] )),-1) - d_grid/2
    ys_contour = point_locations[1,:].reshape(int(np.sqrt(field_value.shape[-1] )),-1) - d_grid/2
    # xs_contour = np.column_stack( (xs_contour, xs_contour[:,-1] + d_grid)) # add extra element on every row
    # xs_contour = np.row_stack( (xs_contour, xs_contour[-1,:] )) # add extra row for the extra y element
    # ys_contour = np.row_stack( ( ys_contour[0,:] + d_grid, ys_contour)) # add extra element on every column
    # ys_contour = np.column_stack( (ys_contour, ys_contour[:,-1] )) # add extra column for the extra x element

    # scale colorbar
    if colorbar_scaling is None:
        min_val = field_value.min()
        max_val = field_value.max()
    else:
        min_val = np.min(colorbar_scaling)
        max_val = np.max(colorbar_scaling)

    # plot for each step
    for step in steps_to_plot:
        field_values_on_step = field_value[step,:]
        field_values_plane_on_step = field_values_on_step.reshape((int(np.sqrt(len(field_values_on_step))),-1))

        fig1 = plt.figure(figsize=(a4_width*cm*0.45,a4_width*cm*0.45))
        fig1.patch.set_facecolor('white')
        ax1 = fig1.add_subplot(111)
        # plot contours
        the_plot = ax1.pcolormesh(xs_contour,ys_contour, field_values_plane_on_step, vmin = min_val, vmax1 = max1_val, cmap = "hot", shading="nearest")

        # plot structure walls
        line_coord = [-81,-27,27,81]
        offset = - d_grid/2
        for coord in line_coord:
            # parallel_lines
            ax1.plot( [-81 + offset ,81 + offset], [ coord + offset, coord + offset ], color = "grey", linewidth = 2)
            # vertical_lines
            ax1.plot( [coord + offset,coord + offset], [-81 + offset,81 + offset], color = "grey", linewidth = 2)

        # plot nodes
        ax1.scatter(x = point_locations[0,:] + offset,y = point_locations[1,:] + offset, marker = "x", color = "grey",linewidth = 2, s = 30)

        cbar = fig1.colorbar(the_plot,ax = ax1)
        # cbar.ax.tick_params(labelsize=80)

        plt.xticks([])
        plt.yticks([])

        if title is not None:
            ax1.set_title(label=title)

        plt.savefig(destination_folder + "/" + field_value_name_id + f"_contour2_step_{step}.pdf", bbox_inches = 'tight' , dpi = 120)
        plt.close()


def plotContour2(field_value,point_locations,steps_to_plot = None,d_grid = 27, result_folder = "results/general_dbging", field_value_name_id = "", title = None, colorbar_scaling = None):
    """
    Plot a grid with element centers being the nodes, and colour it according to the field value at the corresponding node.
    @params field_value : field value to plot with dimensions n x num_of_points_on_grid
    @params point_locations : array with node coordinates (x,y) with dimensions 2 x num_of_points_on_grid
    @params steps_to_plot : itterable with the steps to plot the field_value
    @params d_grid : int with the distance between nodes
    @params results_folder
    @params field_value_name_id : string for naming the saved files and the colorbar
    @params colorbar_scaling : array including min and max values of the illustrated fields
    """

    destination_folder = result_folder + "/contour_plots"
    os.makedirs( destination_folder, exist_ok = True)

    if steps_to_plot is None:
        steps_to_plot = np.arange(0,field_value.shape[0]-1,5000)

    # build grid locations (coordinates of each tile's corners)
    xs_contour = point_locations[0,:].reshape(int(np.sqrt(field_value.shape[-1] )),-1) - d_grid/2
    ys_contour = point_locations[1,:].reshape(int(np.sqrt(field_value.shape[-1] )),-1) - d_grid/2
    # xs_contour = np.column_stack( (xs_contour, xs_contour[:,-1] + d_grid)) # add extra element on every row
    # xs_contour = np.row_stack( (xs_contour, xs_contour[-1,:] )) # add extra row for the extra y element
    # ys_contour = np.row_stack( ( ys_contour[0,:] + d_grid, ys_contour)) # add extra element on every column
    # ys_contour = np.column_stack( (ys_contour, ys_contour[:,-1] )) # add extra column for the extra x element

    # scale colorbar
    if colorbar_scaling is None:
        min_val = field_value.min()
        max_val = field_value.max()
    else:
        min_val = np.min(colorbar_scaling)
        max_val = np.max(colorbar_scaling)

    # plot for each step
    for step in steps_to_plot:
        field_values_on_step = field_value[step,:]
        field_values_plane_on_step = field_values_on_step.reshape((int(np.sqrt(len(field_values_on_step))),-1))

        fig = plt.figure(figsize=(a4_width*cm*0.32,a4_width*cm*0.32))
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(111)
        # plot contours
        the_plot = ax.pcolormesh(xs_contour,ys_contour, field_values_plane_on_step, vmin = min_val, vmax = max_val, cmap = "hot", shading="nearest")

        # plot structure walls
        line_coord = [-81,-27,27,81]
        offset = - d_grid/2
        for coord in line_coord:
            # parallel_lines
            ax.plot( [-81 + offset ,81 + offset], [ coord + offset, coord + offset ], color = "grey", linewidth = 2)
            # vertical_lines
            ax.plot( [coord + offset,coord + offset], [-81 + offset,81 + offset], color = "grey", linewidth = 2)

        # plot nodes
        ax.scatter(x = point_locations[0,:] + offset,y = point_locations[1,:] + offset, marker = "x", color = "grey",linewidth = 2, s = 30)

        cbar = fig.colorbar(the_plot,ax = ax)
        # cbar.ax.tick_params(labelsize=80)

        plt.xticks([])
        plt.yticks([])

        if title is not None:
            ax.set_title(label=title)

        plt.savefig(destination_folder + "/" + field_value_name_id + f"_contour2_step_{step}.pdf", bbox_inches = 'tight' , dpi = 120)
        plt.savefig(destination_folder + "/" + field_value_name_id + f"_contour2_step_{step}.svg", bbox_inches = 'tight' , dpi = 120)
        plt.close()


def relErrorsBarPlot(rel_errors, RESULTS_FOLDER = "./resutls/general_dbging/", field_name="", title = None, legend = None, x_lims = None, bins = 5, y_label = "Instances", x_label = "Relative error"):

    fig = plt.figure( figsize = (a4_width*cm*0.45,a4_width*cm*0.45) )
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    _ = ax.hist(rel_errors, bins, alpha = 1, label=["Data-Driven"], color = ["#34495e"],  lw=6)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    # _ = ax.hist([rel_errors,Fem_rel_errors], bins, alpha = 1, label=["Data-Driven", "FVM"], color = ["#34495e", "#2ecc71"], linestyle=["-", "-.", ":", "--"], lw=3)

    if x_lims is None:
        max_rel_error = np.min ( [np.max(rel_errors),1] )
        min_rel_error = np.max ( [np.min(rel_errors),-1] )
        offset = (abs(min_rel_error) + abs(max_rel_error))*0.5
        x_lims = [ min_rel_error - 1.1*offset , max_rel_error + 1.1*offset]
    ax .set_xlim(x_lims)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(label=title)

    if legend is not None:
        ax.legend( labels = legend )

    plt.savefig(RESULTS_FOLDER + f"{field_name}.svg", dpi = 120,bbox_inches='tight')
    plt.savefig(RESULTS_FOLDER + f"{field_name}.png", dpi = 120,bbox_inches='tight')
    plt.close(fig)

    return None

def plotComponents(T,M_val,G_val,F_val,slot_min = 2250, slot_max = 2750, folder = "./"):

    slot = np.arange(slot_min,slot_max)
    timesteps = 0.5*slot

    fig = plt.figure(figsize=(a4_width*cm*0.45,a4_width*cm*0.45))
    fig.patch.set_facecolor('white')

    ax1 = fig.add_subplot(111)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(0.5)
    
    ax1.plot(timesteps,500*M_val[slot,10],linewidth = 2, label = "(1-f)m")
    ax1.plot(timesteps,G_val[slot,10],linewidth = 2, linestyle = "-.", label = "f g")
    ax1.plot(timesteps,F_val[slot,10],linewidth = 2, linestyle = "--", label = "f")
    ax1.plot(timesteps,T[slot,10],linewidth = 2, linestyle = ":", label = "T")

    ax1.legend(loc = 3)
    plt.savefig(folder + "components.svg", dpi = 120,bbox_inches='tight')
    plt.savefig(folder + "components.png", dpi = 120,bbox_inches='tight')
    plt.close(fig)
    plotFs(folder)

def plotFs(folder,lengths = [5,10,15],resolution = 101):

    input_feature_span = 3*int(lengths[0]) # how many non 0 elements to expect in the half bell 
    if input_feature_span>resolution/2:
        resolution = 2*input_feature_span
    bell_1 = onopt.unitBellFunctionDerivative(lengths[0], resolution = resolution) 
    bell_1[ bell_1<0 ] = 0  

    input_feature_span = 3*int(lengths[1]) # how many non 0 elements to expect in the half bell 
    if input_feature_span>resolution/2:
        resolution = 2*input_feature_span
    bell_2 = onopt.unitBellFunctionDerivative(lengths[1], resolution = resolution) 
    bell_2[ bell_2<0 ] = 0  

    input_feature_span = 3*int(lengths[2]) # how many non 0 elements to expect in the half bell 
    if input_feature_span>resolution/2:
        resolution = 2*input_feature_span
    bell_3 = onopt.unitBellFunctionDerivative(lengths[2], resolution = resolution) 
    bell_3[ bell_3<0 ] = 0  

    fig2 = plt.figure(figsize=(a4_width*cm*0.45,a4_width*cm*0.45))
    fig2.patch.set_facecolor('white')

    ax2 = fig2.add_subplot(111)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(0.5)
    x_axis = np.linspace(-5,5,len(bell_1))
    ax2.plot(x_axis,bell_1,linewidth = 2, label = "5")
    x_axis = np.linspace(-5,5,len(bell_2))
    ax2.plot(x_axis,bell_2,linewidth = 2, label = "10", linestyle = "-.")
    x_axis = np.linspace(-5,5,len(bell_3))
    ax2.plot(x_axis,bell_3,linewidth = 2, label = "15", linestyle = "--")
    ax2.legend(title = "Lengthscale",loc = 0)

    plt.savefig(folder + "f_sequences.svg", dpi = 120,bbox_inches='tight')
    plt.savefig(folder + "f_sequences.png", dpi = 120,bbox_inches='tight')
    plt.close(fig2)