import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
from my_utils.plotting_utils import call_plotting_formatting, generate_color_list
from scipy import stats


colors = cc.glasbey_light  # Color map

# Set-up plot formatting
call_plotting_formatting()

def cross_validation_plot(samples, plot_ranges, plot_labels, percentile_range, filename):

    # Create color dictionary; assign a colour to a galaxy
    if len(samples)==1:
        color_dict = {0: 'k'}
    else:
        color_dict = {i: colors[i % len(colors)] 
                      for i in np.arange(len(samples))}
    
    
    
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
    for galaxy,samples_dict in enumerate(samples):

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
                                    markersize=2, 
                                    capsize=2, 
                                    alpha=0.5, 
                                    color=color_dict[galaxy]
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
                                        c=color_dict[galaxy])
                    
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
        axs_bottom[i].set_ylim([-5,5])
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


    # Save the figure
    fig.savefig(f'{filename}', dpi=400)

    return

def cross_validation_plot_colorcoded(samples, 
                                     galaxies,
                                     color_value,
                                     cbar_label,
                                     cbar_ranges,
                                     plot_ranges, 
                                     plot_labels, 
                                     percentile_range, 
                                     filename):

    # Create color map based on color values
    cmap = cc.cm.CET_L20
    norm = mpl.colors.Normalize(vmin=cbar_ranges[0], 
                                vmax=cbar_ranges[1])
    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for ScalarMappable, as no actual data is tied to it
    
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

    for galaxy,samples_dict in zip(galaxies, samples):

        merger_IDs = list(samples_dict.keys())

        for merger_ID in merger_IDs:

            # Define color associate to progenitor property
            merger_df = galaxy[galaxy['progID']==merger_ID[:-2]]
            color_mapped = cmap(norm(merger_df[color_value].values[0]))

            # Get the median, 34th, and 68th percentiles
            mean_parameters = np.percentile(samples_dict[merger_ID][0], 50, axis=0)
            bottom_errorbar = np.percentile(samples_dict[merger_ID][0], percentile_range[0], axis=0)
            up_errorbar = np.percentile(samples_dict[merger_ID][0], percentile_range[1], axis=0)

            # Loop over the three parameters to plot (top panels)
            for i in range(len(plot_labels)):
                true_value = samples_dict[merger_ID][2][i]  # True value
                pred_value = mean_parameters[i]  # Predicted value
                std = up_errorbar[i]-bottom_errorbar[i]

                # Plot error bars in the top panel
                plot = axs_top[i].errorbar(true_value,
                                    pred_value,
                                    np.array([[pred_value - bottom_errorbar[i]], 
                                            [up_errorbar[i] - pred_value]]),
                                    elinewidth=0.5, 
                                    marker='.',
                                    markersize=2, 
                                    capsize=2, 
                                    alpha=0.5, 
                                    c=color_mapped
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
                                        c=color_mapped)
                    
    # Plot colorbar
    colorbar = fig.colorbar(sm, ax=axs, orientation='vertical')
    colorbar.set_label(cbar_label)
                    
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
        axs_bottom[i].set_ylim([-5,5])
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


    # Save the figure
    fig.savefig(f'{filename}', dpi=400)

    return



def cross_validation_plot_satellites(samples, 
                                     galaxies,
                                     plot_ranges, 
                                     plot_labels, 
                                     percentile_range, 
                                     filename):

    # Create color map based on categorical values
    color_dict = {0: 'k', 1:'r'}

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
        ax.set_title(label, fontsize=12)
        ax.set_xlim(p_range)
        ax.set_ylim(p_range)
        ax.set_aspect(1)
        ax.plot(np.linspace(p_range[0], p_range[1], 10),
                np.linspace(p_range[0], p_range[1], 10), 'k:', alpha=0.5)

    axs_top[0].set_ylabel('Predicted')

    # Set labels for the bottom panels (fractional error histograms)
    for ax in axs_bottom:
        ax.set_xlabel('True')
    axs_bottom[0].set_ylabel('$\\frac{\\theta-\\theta_{true}}{\\sigma_{\\theta}}$')

    # Initialize lists to store true values and fractional errors for each parameter
    true_values = {i: [] for i in range(len(plot_labels))}
    fractional_errors = {i: [] for i in range(len(plot_labels))}
    true_values_sat = {i: [] for i in range(len(plot_labels))}
    fractional_errors_sat = {i: [] for i in range(len(plot_labels))}

    for galaxy, samples_dict in zip(galaxies, samples):

        merger_IDs = list(samples_dict.keys())

        for merger_ID in merger_IDs:

            # Define category and associated color
            merger_df = galaxy[galaxy['progID'] == merger_ID[:-2]]
            category = merger_df['satellite_flag'].values[0]
            color = color_dict[category]

            # Get the median, 34th, and 68th percentiles
            mean_parameters = np.percentile(samples_dict[merger_ID][0], 50, axis=0)
            bottom_errorbar = np.percentile(samples_dict[merger_ID][0], percentile_range[0], axis=0)
            up_errorbar = np.percentile(samples_dict[merger_ID][0], percentile_range[1], axis=0)

            # Loop over the three parameters to plot (top panels)
            for i in range(len(plot_labels)):
                true_value = samples_dict[merger_ID][2][i]  # True value
                pred_value = mean_parameters[i]  # Predicted value
                std = up_errorbar[i] - bottom_errorbar[i]

                # Plot error bars in the top panel
                axs_top[i].errorbar(true_value,
                                    pred_value,
                                    np.array([[pred_value - bottom_errorbar[i]], 
                                              [up_errorbar[i] - pred_value]]),
                                    elinewidth=0.5, 
                                    marker='.',
                                    markersize=2, 
                                    capsize=2, 
                                    alpha=0.5, 
                                    c=color)

                # Calculate fractional error
                if true_value != 0:  # Avoid division by zero
                    fractional_error = (pred_value - true_value) / std
                    if category==0:
                        true_values[i].append(true_value)
                        fractional_errors[i].append(fractional_error)
                    elif category==1:
                        true_values_sat[i].append(true_value)
                        fractional_errors_sat[i].append(fractional_error)
                    # Plot fractional error in bottom panel
                    axs_bottom[i].scatter(true_value, 
                                          fractional_error, 
                                          marker='.',
                                          s=2,
                                          alpha=0.5,
                                          c=color)

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

        # Plot the binned fractional errors as a line plot
        axs_bottom[i].plot(bin_centers, 
                           bin_means, 
                           color='black', 
                           linewidth=1)
        
        # Repeat for satellites
        bin_means, bin_edges, binnumber = stats.binned_statistic(true_values_sat[i], 
                                                                 fractional_errors_sat[i], 
                                                                 statistic='median', 
                                                                 bins=num_bins)
        
        # Calculate the bin widths and centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot the binned fractional errors as a line plot
        axs_bottom[i].plot(bin_centers, 
                           bin_means, 
                           color='r', 
                           linewidth=1)


        axs_bottom[i].set_ylim([-5, 5])
        axs_bottom[i].plot(np.linspace(plot_ranges[i][0], plot_ranges[i][1], 10),
                           np.zeros(10),
                           color='k',
                           linewidth=0.5,
                           linestyle=':')
        axs_bottom[i].fill_between(np.linspace(plot_ranges[i][0], plot_ranges[i][1], 10),
                                   np.full(10, -1),
                                   np.full(10, 1),
                                   color='green',
                                   alpha=0.15)
    axs_bottom[1].tick_params(axis='y', labelleft=False)
    axs_bottom[2].tick_params(axis='y', labelleft=False)

    # Save the figure
    fig.savefig(f'{filename}', dpi=400)

    return



def rms_table_per_galaxy(samples, parameters, filename):

    table = open(f'{filename}','w+')
    header = 'Galaxy,'+",".join(parameters)+"\n"
    _=table.write(header)

    rms_list = []

    for test_galaxy,samples_dict in samples.items():

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

            line = f'{test_galaxy},' + ",".join([f'' for i in range(len(parameters))]) + '\n'
            table.write(line)

        else:            
            rms_galaxy = np.mean(np.vstack(rms_list_galaxy),axis=0)
            # Add line to table of results
            line = f'{test_galaxy},' + ",".join([f'{rms:.2f}' for rms in rms_galaxy]) + '\n'
            table.write(line)

        rms_list += rms_list_galaxy
        print(len(rms_list))

    # Add row to define the average rms
    rms_list = np.vstack(rms_list)
    rms_all = np.mean(rms_list,axis=0)
    rms_str = ["{:.2f}".format(rms) for rms in rms_all]
    table.write("\nAVG (all),"+",".join(rms_str))

    table.close()

def chi2_table_per_galaxy(samples, parameters, percentile_range, filename):

    table = open(f'{filename}','w+')
    header = 'Galaxy,'+",".join(parameters)+"\n"
    _=table.write(header)

    chi2_list = []

    for test_galaxy,samples_dict in samples.items():

        # Get number of merging events for which the inference is run
        merger_IDs = list(samples_dict.keys())

        chi2_list_galaxy = []
        for ID in merger_IDs:
            # Calculate model predictions
            predictions = samples_dict[ID][0]

            fiducial = samples_dict[ID][2]

            std = np.percentile(predictions, percentile_range[1], axis=0) - \
                  np.percentile(predictions, percentile_range[0], axis=0)

            chi2 = np.mean(((predictions-fiducial)/std)**2,axis=0) 

            chi2_list_galaxy.append(chi2)
        
        chi2_galaxy = np.mean(np.vstack(chi2_list_galaxy),axis=0)
        # Add line to table of results
        line = f'{test_galaxy},' + ",".join([f'{chi2:.2f}' for chi2 in chi2_galaxy]) + '\n'
        table.write(line)

        chi2_list += chi2_list_galaxy
        #print(len(chi2_list))

    # Add row to define the average rms
    chi2_list = np.vstack(chi2_list)
    chi2_all = np.mean(chi2_list,axis=0)
    chi2_str = ["{:.2f}".format(chi2) for chi2 in chi2_all]
    table.write("\nAVG (all),"+",".join(chi2_str))

    table.close()

def coefficient_of_determination_table_per_galaxy(samples, parameters,filename):

    table = open(f'{filename}','w+')
    header = 'Galaxy,'+",".join(parameters)+"\n"
    _=table.write(header)

    R_list = []

    for test_galaxy,samples_dict in samples.items():

        # Get number of merging events for which the inference is run
        merger_IDs = list(samples_dict.keys())

        R_list_galaxy = []
        for ID in merger_IDs:
            # Calculate model predictions
            predictions = samples_dict[ID][0]
            avg_predictions = np.median(predictions,axis=0)

            fiducial = samples_dict[ID][2]

            R = 1 - (np.sum((predictions-fiducial)**2,axis=0)) / \
                    (np.sum((predictions-avg_predictions)**2,axis=0))

            R_list_galaxy.append(R)
        
        R_galaxy = np.mean(np.vstack(R_list_galaxy),axis=0)
        # Add line to table of results
        line = f'{test_galaxy},' + ",".join([f'{R:.2f}' for R in R_galaxy]) + '\n'
        table.write(line)

        R_list += R_list_galaxy
        #print(len(chi2_list))

    # Add row to define the average rms
    R_list = np.vstack(R_list)
    R_all = np.mean(R_list,axis=0)
    R_str = ["{:.2f}".format(R) for R in R_all]
    table.write("\nAVG (all),"+",".join(R_str))

    table.close()

def mru_table_per_galaxy(samples, parameters, percentile_range, ranges, filename):

    table = open(f'{filename}','w+')
    header = 'Galaxy,'+",".join(parameters)+"\n"
    _=table.write(header)

    mru_list = []

    for test_galaxy,samples_dict in samples.items():

        mru_list_galaxy = []
        # Get number of merging events for which the inference is run
        merger_IDs = list(samples_dict.keys())

        for ID in merger_IDs:

            # Get range extents
            parameter_extent = np.array([r[1]-r[0] for r in ranges]) 

            # Calculate model standard deviation
            std = np.percentile(samples_dict[ID][0], percentile_range[1], axis=0) - \
                  np.percentile(samples_dict[ID][0], percentile_range[0], axis=0)

            mru = np.abs(std) / np.abs(parameter_extent)
            if (np.isnan(mru).any()) | (np.isinf(mru).any()):
                continue    

            mru_list.append(mru) 
            mru_list_galaxy.append(mru)                        
    
        mru_list_galaxy = np.mean(np.vstack(mru_list_galaxy), axis=0)

        # Add line to table of results
        line = f'{test_galaxy},' + ",".join([f'{mru:.2f}' for mru in mru_list_galaxy]) + '\n'
        table.write(line)

    # Add row to define the average mru
    mru_list = np.vstack(mru_list)
    avg_mru = ["{:.2f}".format(mru) for mru in np.mean(mru_list,axis=0)]
    table.write("AVG,"+",".join(avg_mru))

    table.close()


def count_predictions_within_range(samples, parameters, percentile_range, filename):

    table = open(f'{filename}','w+')
    header = 'Galaxy,'+",".join(parameters)+"\n"
    _=table.write(header)

    counter = []

    for test_galaxy,samples_dict in samples.items():

        # Get ID of merging events for which the inference is run
        merger_IDs = list(samples_dict.keys())

        counter_per_galaxy = []

        for ID in merger_IDs:
            # Get true values
            fiducial_values = samples_dict[ID][2]
            
            # Calculate parameter range inferred by model
            upper_limit = np.percentile(samples_dict[ID][0], percentile_range[1], axis=0)
            bottom_limit = np.percentile(samples_dict[ID][0], percentile_range[0], axis=0)
            
            is_within_range = (fiducial_values<=upper_limit) & (fiducial_values>=bottom_limit)

            # Convert bool to int
            counter_per_galaxy.append([int(i) for i in is_within_range])

        counter.append(counter_per_galaxy)
        counter_per_galaxy = np.vstack(counter_per_galaxy)
        fraction_of_mergers_within_range = np.sum(counter_per_galaxy,axis=0) / len(counter_per_galaxy)

        # Add line to table of results
        line = f'{test_galaxy},' + ",".join([f'{fraction_of_mergers_within_range[i]:.2f}' 
                                             for i in range(len(fraction_of_mergers_within_range))]) + '\n'
        table.write(line)

    # Add row for average
    counter = np.vstack(counter)
    avg_fraction = np.sum(counter,axis=0) / len(counter)
    avg_string = ["{:.2f}".format(avg) for avg in avg_fraction]
    table.write("AVG,"+",".join(avg_string))

    table.close()



def plot_model_comparison(test_galaxy, data_dir, model_names, labels_dict, samples_dict, filename):

    color_dict = dict(zip(model_names,generate_color_list(num_colors=len(model_names), cmap='tab10')))

    fig, axs = plt.subplots(1,3, figsize=(9,3.5))
    plot_ranges = [[0.5,13.5],[8.5,11.5],[-2.8,-0.1]]
    plot_labels = ['$\\tau_{infall}$', 
                'log($M_{prog}/M_{\odot}$)', 
                'Merger Mass Ratio (log)']
    for (ax,p_range,label) in zip(axs, plot_ranges, plot_labels):
        ax.set_xlabel('True')
        ax.set_title(label, fontsize=12)
        ax.set_xlim(p_range)
        ax.set_ylim(p_range)
        ax.set_aspect(1)
        ax.plot(np.linspace(p_range[0],p_range[1],10),
                np.linspace(p_range[0],p_range[1],10), 'k:', alpha=0.5 )
    axs[0].set_ylabel('Predicted')


    for model in model_names:
        
        if model=="GFlow":
            samples_dict = pickle.load(open('/mnt/aridata1/users/ariasant/auriga-sbi/samples/100+/G05_Gflow+NPE_test_samples.pkl','rb'))
        else:
            samples_dict = pickle.load(open(f'{data_dir}Test_{test_galaxy}__{model}/test_samples.pkl', 'rb'))


        # Get number of merging events for which the inference is run
        merger_IDs = list(samples_dict.keys())

        for merger_ID in merger_IDs:

            mean_parameters = np.percentile(samples_dict[merger_ID][0], 50, axis=0)
            bottom_errorbar = np.percentile(samples_dict[merger_ID][0], 34, axis=0)
            up_errorbar = np.percentile(samples_dict[merger_ID][0], 68, axis=0)

            for i in range(3):

                axs[i].errorbar(samples_dict[merger_ID][2][i], 
                                mean_parameters[i],
                                np.array([[mean_parameters[i]-bottom_errorbar[i]], 
                                        [up_errorbar[i]-mean_parameters[i]]]),
                                elinewidth=1,    
                                marker='.',
                                markersize=5,
                                capsize=2,
                                color=color_dict[model],
                                label=labels_dict[model],
                                alpha=0.5
                                )
            

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[::len(merger_IDs)], 
            labels[::len(merger_IDs)], 
            loc='upper center', 
            ncols=len(model_names),
            fontsize=12,
            handlelength=0)
    fig.savefig(f'/mnt/aridata1/users/ariasant/auriga-sbi/plots/model_comparison_single_galaxy/{filename}.png',dpi=400)

    plt.close('all')



def find_major_mergers(scalers_list, samples_list):


    TP_major = 0
    FP_major = 0
    TP_minor = 0
    FP_minor = 0
    N_major = 0
    N_minor = 0
    threshold=-1.3
    for scaler,samples in zip(scalers_list,samples_list):
        for ID in samples.keys():
            inf_par = np.median(scaler.inverse_transform(samples[ID][0]),axis=0)
            true_par = scaler.inverse_transform(samples[ID][2][None,:])
            MMR_true = true_par[0][2]
            MMR_inf = inf_par[2]
            if MMR_true>threshold:
                N_major+=1  
                if (MMR_inf>threshold):
                    TP_major+=1
            elif (MMR_inf>threshold) & (MMR_true<threshold):
                FP_major+=1
            if MMR_true<threshold:
                N_minor+=1
                if MMR_inf<threshold:
                    TP_minor+=1
            elif (MMR_inf<threshold) & (MMR_true>threshold):
                FP_minor+=1
                

    precision_major = TP_major/(TP_major+FP_major)
    recall_major = TP_major/N_major

    precision_minor = TP_minor/(TP_minor+FP_minor)
    recall_minor = TP_minor/N_minor

    print(f"N major: {N_major}")
    print(f"Precision: {precision_major:.2f}")
    print(f"Recall: {recall_major:.2f}")

    print(f"\nN minor: {N_minor}")
    print(f"Precision: {precision_minor:.2f}")
    print(f"Recall: {recall_minor:.2f}")

    return


