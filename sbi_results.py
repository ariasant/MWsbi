import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from my_utils.plotting_utils import call_plotting_formatting
from scipy import stats


colors = cc.glasbey_light  # Color map

# Set-up plot formatting
call_plotting_formatting()

def cross_validation_plot(samples_dict, plot_ranges, plot_labels, percentile_range):

    
    # Set up subplots with shared x-axis for each column
    fig, axs = plt.subplots(2, len(plot_labels), 
                            figsize=(3*len(plot_labels), 4), 
                            gridspec_kw={'height_ratios': [3, 1], 
                                         'hspace':0}, sharex='col')

    # Top row will be axs[0, :], bottom row will be axs[1, :]
    axs_top = axs[0, :]
    axs_bottom = axs[1, :]

    # Set labels and plot ranges for top panels (actual vs predicted values)
    for ax, p_range, label in zip(axs_top, plot_ranges, plot_labels):
        #ax.set_xlabel('True')  # This won't show up because x-axis is shared, but keeps labels consistent
        ax.set_title(label, fontsize=12)
        ax.set_xlim(p_range)
        ax.set_ylim(p_range)
        ax.set_aspect(1)
        ax.plot(np.linspace(p_range[0], p_range[1], 10),
                np.linspace(p_range[0], p_range[1], 10), 'k:', alpha=0.5)

    axs_top[0].set_ylabel('Predicted')

    # Set labels for the bottom panels (fractional error histograms)
    for ax, label in zip(axs_bottom, plot_labels):
        ax.set_xlabel('True')
    axs_bottom[0].set_ylabel('$\\frac{\\theta-\\theta_{true}}{\\sigma_{\\theta}}$')

    # Initialize lists to store true values and fractional errors for each parameter
    true_values = {i: [] for i in range(len(plot_labels))}
    fractional_errors = {i: [] for i in range(len(plot_labels))}
    
    merger_IDs = list(samples_dict.keys())

    for merger_ID in merger_IDs:

        # Get the median, 34th, and 68th percentiles
        mean_parameters = np.percentile(samples_dict[merger_ID][0], 50, axis=0)
        bottom_errorbar = np.percentile(samples_dict[merger_ID][0], percentile_range[0], axis=0)
        up_errorbar = np.percentile(samples_dict[merger_ID][0], percentile_range[1], axis=0)

        # Loop over the parameters to plot (top panels)
        for i in range(len(plot_labels)):
            true_value = samples_dict[merger_ID][2][i]  # True value
            pred_value = mean_parameters[i]  # Predicted value
            std = up_errorbar[i]-bottom_errorbar[i]

            # Plot error bars in the top panel
            axs_top[i].errorbar(true_value,
                                pred_value,
                                np.array([[pred_value - bottom_errorbar[i]], 
                                        [up_errorbar[i] - pred_value]]),
                                elinewidth=0.5, 
                                marker='.',
                                markersize=1, 
                                capsize=2, 
                                alpha=0.5, 
                                color="k"
                                )

            # Calculate fractional error
            if true_value != 0:  # Avoid division by zero
                fractional_error = (pred_value-true_value)/std
                true_values[i].append(true_value)
                fractional_errors[i].append(fractional_error)
                # Plot fractional error in bottom pannel
                axs_bottom[i].scatter(true_value, 
                                    fractional_error, 
                                    marker='.',
                                    s=2,
                                    alpha=0.5,
                                    c="k")
                    
    # After looping through galaxies, plot the median binned fractional errors using histograms
    for i in range(len(plot_labels)):
        # Bin the data and compute the mean fractional error in each bin
        num_bins = 20  # Number of bins for true values
        bin_means, bin_edges, binnumber = stats.binned_statistic(true_values[i], 
                                                                fractional_errors[i], 
                                                                statistic='median', 
                                                                bins=num_bins)
        
        # Calculate the bin widths and centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Plot the binned fractional errors as a bar plot
        axs_bottom[i].plot(bin_centers, 
                        bin_means, 
                        color='black', 
                        linewidth=1
                        )
        axs_bottom[i].set_ylim([-3,3])
        axs_bottom[i].plot(np.linspace(plot_ranges[i][0], plot_ranges[i][1],10),
                        np.tile(0,10),
                        color='k',
                        linewidth=0.5,
                        linestyle=':')
        axs_bottom[i].fill_between(np.linspace(plot_ranges[i][0], plot_ranges[i][1],10),
                                np.tile(-1,10),
                                np.tile(1,10),
                                color='green',
                                alpha=0.15
                                )
    axs_bottom[1].tick_params(axis='y', labelleft=False)
    axs_bottom[2].tick_params(axis='y', labelleft=False)

    return fig



def rms_table_per_galaxy(samples_dict, parameters, filename):

    table = open(f'{filename}','w+')
    header = 'Galaxy,'+",".join(parameters)+"\n"
    _=table.write(header)

    rms_list = []


    # Get number of merging events for which the inference is run
    merger_IDs = list(samples_dict.keys())

    rms_list_galaxy = []
    for ID in merger_IDs:
        # Calculate model predictions
        predictions = samples_dict[ID][0]

        fiducial = samples_dict[ID][2]

        if fiducial[0]>5:
            continue

        rms = np.sqrt(np.mean((predictions-fiducial)**2,axis=0))

        rms_list_galaxy.append(rms)
    
    if len(rms_list_galaxy)<1:

        line = 'All,' + ",".join([f'' for i in range(len(parameters))]) + '\n'
        table.write(line)

    else:            
        rms_galaxy = np.mean(np.vstack(rms_list_galaxy),axis=0)
        # Add line to table of results
        line = f'All,' + ",".join([f'{rms:.2f}' for rms in rms_galaxy]) + '\n'
        table.write(line)

    rms_list += rms_list_galaxy
    print(len(rms_list))

    # Add row to define the average rms
    rms_list = np.vstack(rms_list)
    rms_all = np.mean(rms_list,axis=0)
    rms_str = ["{:.2f}".format(rms) for rms in rms_all]
    table.write("\nAVG (all),"+",".join(rms_str))

    table.close()




