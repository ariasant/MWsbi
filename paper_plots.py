import colorcet as cc
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
from scipy.stats import binned_statistic



output_dir = "/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/coral/paper_plots/"
posterior_samples_dir ="/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/coral/"


# Declare plot formatting
def call_plotting_formatting():

    font = {'family' : 'sans-serif',
    'weight' : 'medium',
    'size'   : 15,
    'variant' : 'normal',
    'style' : 'normal',
    'stretch' : 'normal',
    }

    xtick = {'top' : True,
            'bottom' : True,
            'major.size' : 7,
            'minor.size' : 4,
            'major.width' : 0.5,
            'minor.width' : 0.35,
            'direction' : 'in',
            'minor.visible' : True,
            'color' : 'black',
            'labelcolor' : 'black'
            }

    ytick = {'left' : True,
            'right' : True,
            'major.size' : 7,
            'minor.size' : 4,
            'major.width' : 0.5,
            'minor.width' : 0.35,
            'direction' : 'in',
            'minor.visible' : True,
            'color' : 'black',
            'labelcolor' : 'black'
            }

    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['figure.figsize'] = (6.973848069738481, 4.310075139476229)
    mpl.rcParams['figure.subplot.hspace'] = 0.01

    mpl.rc('font', **font)
    mpl.rc('xtick', **xtick)
    mpl.rc('ytick', **ytick)
    mpl.rcParams['legend.fontsize'] = 18
    mpl.rcParams["font.sans-serif"] = ["DejaVu Serif"]
    mpl.rcParams['mathtext.fontset']='dejavuserif'
    mpl.rcParams["text.usetex"] = False

call_plotting_formatting()

# Define plotting functions
def plot_ax(x,y,
            ax,
            bin_number=500,
            extent=None,
            cmap='cividis'):
            

    counts,x_edges,y_edges = np.histogram2d(x, y, 
                                            bins=(bin_number,bin_number), 
                                            range=extent)
    
    norm = mpl.colors.LogNorm(vmin=1, vmax=200)

    plot = ax.imshow(counts.T,
              cmap=cmap,
              norm=norm,
              origin='lower',
              extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
              interpolation='nearest',
              aspect='equal')


    return plot

def plot_2D_heatmap(x,y,xlabel,ylabel,filename,
                    title=None,
                    weights=None,
                    weighted=False,
                    density=False,
                    cbar_label=r"Number of Stars",
                    bin_number=500,
                    extent=None,
                    cmap='cividis',
                    aspect_ratio=None,
                    norm=None,
                    return_figure=False):
            
    fig,ax = plt.subplots(figsize=(4,4))

    counts,x_edges,y_edges = np.histogram2d(x, y, 
                                            bins=(bin_number,bin_number), 
                                            range=extent,
                                            weights=weights,
                                            density=density)
    
    if weighted:
        nw_counts, _, __ = np.histogram2d(x, y, 
                                          bins=(bin_number,bin_number), 
                                          range=extent,
                                          weights=None)
        nw_counts[nw_counts==0]=-1
        counts = counts/nw_counts
        counts[nw_counts==-1] = np.nan

    if norm is None:
        norm = mpl.colors.LogNorm(vmin=1, vmax=np.percentile(counts, 99.9))

    plot = ax.imshow(counts.T,
                     cmap=cmap,
                     norm=norm,
                     origin='lower',
                     extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
                     interpolation='nearest',
                     aspect='equal')

    cbar = plt.colorbar(plot, orientation='vertical', shrink=0.8)
    cbar.set_label(cbar_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not aspect_ratio:
        if extent:
            ax.set_aspect((extent[0][1]-extent[0][0])/(extent[1][1]-extent[1][0]))

    if title:
        ax.set_title(title, fontsize=14)

    if return_figure:
        return fig,ax
    else:
        fig.savefig(filename+'.png')
        plt.close('all')

    return 

# Define markers for the different substructures
def generate_markers(n):
    """Generates a list of n unique markers for Matplotlib scatter plots."""
    marker_options = ['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'X', 'h', 'H', '+', 'x', '|', '_']
    
    # If n is larger than available markers, cycle through them
    markers = list(itertools.islice(itertools.cycle(marker_options), n))
    
    return markers

substructures = ['Arjuna', 'GES', 'Sagittarius', 'Helmi',
       'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
       'LMS', 'Heracles']

substructure_labels = {'Arjuna': 'Arjuna',
                      'GES': 'GES',
                      'Sagittarius': 'Sagittarius',
                      'Helmi': 'Helmi streams',
                      'Sequoia_K19': 'K19',
                      'Sequoia_M19': 'M19',
                      'Sequoia_N20': 'N20',
                      'Iitoi': "I'itoi",
                      'Thamnos': "Thamnos",
                      'LMS': "LMS-1",
                      'Heracles': "Heracles"}

substructure_markers = dict(zip(substructures, generate_markers(len(substructures))))

# No Aleph: because it might be a spurious detection (linked to the high-alpha disk)
# No Arjuna: because it significantly overlaps chemically with GES
# No Nyx: because it significantly overlaps chemically with the high-alpha disk
# Order substructure list alphabetically
substructures = sorted(substructures)
# Associate a color to each substructure
colors = cc.glasbey_light[:len(substructures)]
colors_dict = dict(zip(substructures,colors))

#########################################################################################
#########################################################################################
#
#               DATASET
#
#########################################################################################
#########################################################################################


# LOAD DATA
df = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")
df.dropna(inplace=True, subset=["FeH","MgFe","E","L"])

#########################################################################################
# Plot IoM 
LMAX = 4
EMIN = -2.8

fig_IoM,ax_IoM = plot_2D_heatmap(x=df['Lz']*1e-3,
                        y=df['E']*1e-5,
                        xlabel="$L_{z} \; [\\times10^{3} \, \mathrm{kpc}\, \mathrm{kms}^{-1}]$",
                        ylabel="$E \; [\\times10^{5} \, \mathrm{kpc}^{2}\, \mathrm{s}^{-2}]$",
                        filename="/mnt/aridata1/users/ariasant/MW-sbi/data/IoM",
                        extent=[[-LMAX,LMAX],[EMIN,-0.5]],
                        cmap="gist_gray",
                        return_figure=True)

fig_IoM.set_figheight(5)
fig_IoM.set_figwidth(5)

# Fix the axis limits after plotting the heatmaps
ax_IoM.set_xlim([-LMAX, LMAX])
ax_IoM.set_ylim([EMIN, -0.5])


for substructure in substructures:
    # Load position of stars 
    substructure_flag = f"{substructure}_flag"
    df_sub = df[df[substructure_flag]==1]
    
    if substructure=="GES":
        alpha=0.15
    else:
        alpha=0.5
    
    if "Sequoia" in substructure:
        label = f"Sequoia ({substructure_labels[substructure]})"
    else:
        label = substructure_labels[substructure]

    # Plot points on top of IoM figure
    ax_IoM.scatter(x=df_sub['Lz']*1e-3,
                   y=df_sub['E']*1e-5, 
                   s=10,
                   marker=substructure_markers[substructure],
                   edgecolor="k",
                   linewidth=0.01,
                   color=colors_dict[substructure])
    
    ax_IoM.scatter([],[], 
                   s=35,
                   marker=substructure_markers[substructure],
                   edgecolor="k",
                   linewidth=0.5,
                   color=colors_dict[substructure],
                   label=label)
    
    
# Plot legend outside of figure
ax_IoM.legend(loc="upper center", 
          bbox_to_anchor=(0.6, 1.4), 
          ncols=3,
          frameon=True, edgecolor='black',
          fontsize=11)

fig_IoM.savefig(f"{output_dir}IoM.pdf",dpi=300)

# ALPHA-IRON PLANE
xlim = [-2.5,1]
ylim = [-0.25,0.6]

fig,axs = plt.subplots(ncols=5, 
                       nrows=2, 
                       gridspec_kw={"wspace":0, 
                                   "hspace":0.3, 
                                   "width_ratios": [1,1,1,1,1],
                                   "height_ratios": [1,1]},
                       sharey=True,
                       sharex=True,
                       figsize=(12,8))

axs  = axs.flatten()
i=0

for substructure in substructures:
    
    substructure_flag = f"{substructure}_flag"
    df_sub = df[df[substructure_flag]==1]
    
    ax_chem=axs[i]
    plot = plot_ax(x=df['FeH'],
            y=df['MgFe'],
            ax=ax_chem,
            extent=[xlim, ylim],
            cmap="gist_gray")
    
    if substructure=="GES":
        alpha=0.15
    else:
        alpha=1
    
    ax_chem.scatter(x=df_sub['FeH'],
                    y=df_sub['MgFe'], 
                    s=10,
                    marker=substructure_markers[substructure],
                    color=colors_dict[substructure],
                    edgecolor='k',
                    linewidth=0.01,
                    alpha=alpha)
    ax_chem.scatter([],[], 
                   s=35,
                   marker=substructure_markers[substructure],
                   edgecolor="k",
                   linewidth=0.5,
                   color=colors_dict[substructure],
                   label=substructure_labels[substructure])
    
    ax_chem.set_xlim(xlim)
    ax_chem.set_ylim(ylim)
    ax_chem.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
    
    ax_chem.set_title(substructure_labels[substructure], fontsize=11)
    
    if i%5==0:
        ax_chem.set_ylabel("$[\\alpha/\mathrm{Fe}]$")
    if i//5==1:
        ax_chem.set_xlabel("$[\mathrm{Fe}/\mathrm{H}]$")
    
    
    if substructure not in ['Sequoia_K19','Sequoia_M19']:
        i+=1
        
    if substructure=="Sequoia_N20":
        ax_chem.legend(loc="upper right",
                      fontsize=9,
                      handletextpad=0,
                      borderpad=0,
                      frameon=False)
        ax_chem.set_title("Sequoia", fontsize=11)
        
axs[-1].set_visible(False)

cbar = fig.colorbar(plot, ax=axs.ravel().tolist(), shrink=0.8, orientation="horizontal")
cbar.set_label('Number of Stars')

fig.savefig(f"{output_dir}alphaIron.pdf", dpi=300)

#########################################################################################
#########################################################################################
#
#               COMPARISON BETWEEN PREDICTIONS AND VALUES FROM LITERATURE
#
#########################################################################################
#########################################################################################

infall_times = {
    "GES": [(8,11), 10, 10, (10.2,0.2,0.1), 9.1, (7,9), 10, 10.5, 10.5, (10.2,0.2,0.2)],
    "Helmi": [(6,9), (5,8), 10.1, 7.9, 8, 9.4],
    "Heracles": [(10.5,11.6)],
    "Iitoi": [10.4],
    "LMS": [8, 8.3, 12.9],
    "Sagittarius": [(5,7), 6.8, 5.5, (3,4), 5.9,],
    "Sequoia_K19": [9, 9.4, (8,11), 11.6],
    "Sequoia_M19": [9, 9.4, (8,11), 11.6],
    "Sequoia_N20": [9, 9.4, (8,11), 11.6],
    "Thamnos": [(13,14), 13.4]
}

stellar_masses = {
    "GES": [8.8, (8.5,9), 9.7, (8.5,0.1,0.2), (8.85,9.85), (8.43,0.15,0.16), 9.5, 8.7, 8.8, (8.16,0.21,0.17)],
    "Helmi": [8.3, 8, (7.96,0.19,0.18), 8],
    "Heracles": [8.7, 8.86],
    "Iitoi": [6.3],
    "LMS": [(6,7), 7.1],
    "Sagittarius": [8, 7.3, (8.44,0.22,0.21), 9.3, 8.8],
    "Sequoia_K19": [7.7, (7.9,0.11,0.11), 7.2],
    "Sequoia_M19": [7.7, (7.9,0.11,0.11), 7.2],
    "Sequoia_N20": [7.7, (7.9,0.11,0.11), 7.2],
    "Thamnos": [6.7, (5,6.7)]
}


fig, axs = plt.subplots(2, 1, 
                        figsize=(6,6),
                       gridspec_kw={"hspace":0})


axs[0].set_ylim([0.2, 13.8])
#ax.set_xlim([0, 34])

axs[0].set_ylabel("Lookback \nInfall Time [Gyr]")
axs[0].set_xticks([])
axs[0].grid(alpha=0.5)

axs[1].set_ylabel("$\log(M_{*}/\mathrm{M}_{\odot})$")
axs[1].set_ylim([5.2,10.4])
axs[1].set_xticks([])
axs[1].grid(alpha=0.5)

def plot_comparison_with_literature(ax,
                                    ind_samples,
                                    substructures, 
                                    literature_dict):
                                    
    n = 0
                                    
    for substructure in substructures:
        # Get estimate of infall time from GalactiKit
        samples = pickle.load(open(f"{posterior_samples_dir}{substructure}.pkl", "rb"))

        median = np.percentile(samples[:, ind_samples], 50)
        _16 = np.percentile(samples[:, ind_samples], 16)
        _84 = np.percentile(samples[:, ind_samples], 84)
    

        if substructure in ["Sequoia_K19", "Sequoia_M19"]:
                continue

        start = n
        
        if substructure in literature_dict.keys():
            for estimate in literature_dict[substructure]:
                if isinstance(estimate, tuple): 
                    if len(estimate)==2:
                        ax.errorbar(n, [estimate[0]], yerr=[[0], [estimate[1]-estimate[0]]],
                                    fmt='none', 
                                    color=colors_dict[substructure],
                                    capsize=2,
                                    linewidth=1)
                    else:
                        ax.errorbar(n, [estimate[0]], 
                                    yerr=[[estimate[2]], [estimate[1]]],
                                    marker=substructure_markers[substructure],
                                    markersize=8,
                                    markeredgecolor='k',
                                    markeredgewidth=0.5,
                                    color=colors_dict[substructure],
                                    capsize=2,
                                    linewidth=1)

                else:
                    ax.errorbar(n, 
                               estimate, 
                                [[1e-3],[1e-3]],
                               color=colors_dict[substructure],
                               marker=substructure_markers[substructure],
                               markersize=8,
                               markeredgecolor='k',
                               markeredgewidth=0.5,
                              )

                n += 1

        x = list(range(start-1, n+1))  # Ensure correct range


        ax.fill_between(x,
                        [_16] * len(x),  # Use list comprehension
                        [_84] * len(x),
                        color=colors_dict[substructure],
                        alpha=0.35)
        ax.plot(x,
                [median] * len(x),
                color=colors_dict[substructure])
        
        n+=1
        if substructure == "Sequoia_N20":

            for s in ["Sequoia_K19", "Sequoia_M19"]:

                samples = pickle.load(open(f"{posterior_samples_dir}{s}.pkl", "rb"))
                median = np.percentile(samples[:, ind_samples], 50)
                _16 = np.percentile(samples[:, ind_samples], 16)
                _84 = np.percentile(samples[:, ind_samples], 84)
                ax.fill_between(x,
                            [_16] * len(x),  # Use list comprehension
                            [_84] * len(x),
                            color=colors_dict[s],
                            alpha=0.35)
                ax.plot(x,
                        [median] * len(x),
                        color=colors_dict[s])


    return
            
plot_comparison_with_literature(ax=axs[0],
                               ind_samples=0,
                               substructures=substructures,
                               literature_dict=infall_times)


plot_comparison_with_literature(ax=axs[1],
                               ind_samples=1,
                               substructures=substructures,
                               literature_dict=stellar_masses)

handles = []
for substructure, color in colors_dict.items():
    if "Sequoia" in substructure:
        label = f"Sequoia ({substructure_labels[substructure]})"
    else:
        label = substructure_labels[substructure]
        
    line = plt.scatter([],[], 
                   s=50,
                   marker=substructure_markers[substructure],
                   edgecolor="k",
                   linewidth=0.5,
                   color=colors_dict[substructure],
                   label=label) 
    handles.append(line)


fig.legend(handles=handles,
           loc="upper center", 
           bbox_to_anchor=(0.45, 1.06), 
           ncols=3, 
           frameon=True, 
           edgecolor='black', 
           fontsize=11)

fig.savefig(f"{output_dir}Comparison_with_literature.pdf", dpi=200)


########################################################################
def plot_feature_relation(feature,
                          label,
                          xlims,
                          filename
                         ):


    fig,axs = plt.subplots(nrows=1,
                           ncols=2,
                           sharex=True,
                           figsize=(7.5,3)
                          )

    axs[0].set_ylim([0.2, 13.8])
    axs[0].set_ylabel("Lookback \nInfall Time [Gyr]")
    axs[0].grid(alpha=0.5)
    axs[1].set_ylabel("$\log(M_{*}/\mathrm{M}_{\odot})$")
    axs[1].set_ylim([5.8,9.4])
    axs[1].grid(alpha=0.5)


    axs[0].set_xlabel(label)
    axs[1].set_xlabel(label)
    axs[0].set_xlim(xlims)
    axs[1].set_xlim(xlims)

    axs[0].set_aspect((xlims[1]-xlims[0])/(13.6))
    axs[1].set_aspect((xlims[1]-xlims[0])/(3.6))

    for substructure in substructures:

        # Read data from observations
        substructure_flag = f"{substructure}_flag"
        df_sub = df[df[substructure_flag]==1]

        median_obs = np.percentile(df_sub[feature].values,50)
        obs_16 = np.percentile(df_sub[feature].values,16)
        obs_84 = np.percentile(df_sub[feature].values,84)

        # Get estimate of infall time and stellar mass from GalactiKit
        samples = pickle.load(open(f"{posterior_samples_dir}{substructure}.pkl", "rb"))

        median_time = np.percentile(samples[:, 0], 50)
        time_16 = np.percentile(samples[:, 0], 16)
        time_84 = np.percentile(samples[:, 0], 84)

        median_mass = np.percentile(samples[:, 1], 50)
        mass_16 = np.percentile(samples[:, 1], 16)
        mass_84 = np.percentile(samples[:, 1], 84)

        axs[0].errorbar(median_obs,
                        median_time,
                        [[median_time-time_16],[time_84-median_time]],
                        [[median_obs-obs_16],[obs_84-median_obs]],
                        markersize=8,
                        markeredgecolor='k',
                        markeredgewidth=0.5,
                        marker=substructure_markers[substructure],
                        capsize=2,
                        linewidth=1,
                       label=substructure_labels[substructure],
                       color=colors_dict[substructure]
                  )

        axs[1].errorbar(median_obs,
                        median_mass,
                        [[median_mass-mass_16],[mass_84-median_mass]],
                        [[median_obs-obs_16],[obs_84-median_obs]],
                        markersize=8,
                        markeredgecolor='k',
                        markeredgewidth=0.5,
                        marker=substructure_markers[substructure],
                        capsize=2,
                        linewidth=1,
                        label=substructure_labels[substructure],
                        color=colors_dict[substructure]
                  )
        
        fig.savefig(f"{filename}.pdf",dpi=400)
    

plot_feature_relation(feature="FeH",
                      label="$[\mathrm{Fe}/\mathrm{H}]$",
                      xlims = [-2.5,0],
                      filename=f"{output_dir}PropVSFeH"
                     )

plot_feature_relation(feature="MgFe",
                      label="$[\mathrm{Mg}/\mathrm{Fe}]$",
                      xlims = [-0.1,0.5],
                      filename=f"{output_dir}PropVSMgFe"
                     )

plot_feature_relation(feature="r",
                      label="$r \, [\mathrm{kpc}]$",
                      xlims = [0,30],
                      filename=f"{output_dir}PropVSr"
                     )

df["E_plot"] = df["E"].values*1e-5
plot_feature_relation(feature="E_plot",
                      label="$E \, [\mathrm{kpc}^{2}\, \mathrm{kms}^{-2}]$",
                      xlims = [EMIN,-0.5],
                      filename=f"{output_dir}PropVSE"
                     )

##############################################################################

# Load dataframes with FeH and MgFe abundances for the progenitors in the simulations
prog_FeH_dict = pickle.load(open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_prog_FeH.pkl", "rb"))
prog_MgFe_dict = pickle.load(open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_prog_MgFe.pkl", "rb"))
prog_Mstar_dict = pickle.load(open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_prog_stellar_mass.pkl", "rb"))


fig, ax = plt.subplots()

for progID,Mstar in prog_Mstar_dict.items():
    
    median_FeH = np.percentile(prog_FeH_dict[progID]-0.2, 50)  
    ax.scatter(Mstar, 
               median_FeH,
               s=1,
               c="k",
               alpha=0.5)
    
# MW satellites prediction
for substructure in substructures:

    # Read data from observations
    substructure_flag = f"{substructure}_flag"
    df_sub = df[df[substructure_flag]==1]

    # Get estimate of infall time and stellar mass from GalactiKit
    samples = pickle.load(open(f"{posterior_samples_dir}{substructure}.pkl", "rb"))

    median_obs = np.percentile(df_sub["FeH"].values,50)
    obs_16 = np.percentile(df_sub["FeH"].values,16)
    obs_84 = np.percentile(df_sub["FeH"].values,84)

    median_mass = np.percentile(samples[:, 1], 50)
    mass_16 = np.percentile(samples[:, 1], 16)
    mass_84 = np.percentile(samples[:, 1], 84)

    
    ax.errorbar(median_mass,
                median_obs,
                [[median_mass-mass_16],[mass_84-median_mass]],
                [[median_obs-obs_16],[obs_84-median_obs]],
                markersize=8,
                markeredgecolor='k',
                markeredgewidth=0.5,
                marker=substructure_markers[substructure],
                capsize=2,
                linewidth=1,
                label=substructure_labels[substructure],
                color=colors_dict[substructure]
                )
    
ax.set_xlabel('log($M_{*}/M_{\odot}$)')
ax.set_ylabel('[Fe/H]')

ax.set_ylim([-2.5,0])
ax.set_xlim([6,10.4])

ax.set_aspect((4.4/2.5)*0.5)
fig.savefig(f"{output_dir}MZR.pdf", dpi=400)


#########################################################################################
#########################################################################################
#
#               MASS ASSEMBLY OF THE MILKY WAY
#
#########################################################################################
#########################################################################################

assembly_dict = {}
for substructure in substructures:
    
    samples = pickle.load(open(f"{posterior_samples_dir}{substructure}.pkl", "rb"))
    infall_times = samples[:,0]
    halo_mass = samples[:,2]
    mmr = samples[:,3]
    
    proto_MW_mass = halo_mass-mmr
    
    assembly_dict[substructure] = (infall_times, proto_MW_mass)

    


fig,ax = plt.subplots(1,1)

xlim = [0.2,13.8]
ylim = [8.8,12.5]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*0.5)

ax.set_ylabel("$\log(M_{\mathrm{MW}}/\mathrm{M}_{\odot})$")
ax.set_xlabel("Lookback Infall Time [Gyr]")



for substructure in substructures:
    
    x = assembly_dict[substructure][0]
    y = assembly_dict[substructure][1]
    
    output = binned_statistic(x=x, 
                              values=y, 
                              bins=20, 
                              statistic="mean",
                              range=(np.percentile(x,16),
                                     np.percentile(x,84))
                             )
    
    ax.plot((output[1][1:]+output[1][:-1])/2,
            output[0],
            color=colors_dict[substructure]
           )
    
    ax.bar(x=np.percentile(x,16),
           align="edge",
           height=np.mean(output[0]),
           color=colors_dict[substructure],
           width=np.percentile(x,84)-np.percentile(x,16),
           alpha=0.5
          )

handles = []
for substructure, color in colors_dict.items():
    if "Sequoia" in substructure:
        label = f"Sequoia ({substructure_labels[substructure]})"
    else:
        label = substructure_labels[substructure]
        
    line = plt.Line2D([],[], 
                   linestyle='-', 
                   linewidth=5,
                   color=colors_dict[substructure],
                   label=label) 
    handles.append(line)


fig.legend(handles=handles,
           loc="upper center", 
           bbox_to_anchor=(0.51, 1.06), 
           ncols=3, 
           frameon=True, 
           edgecolor='black', 
           fontsize=11)

fig.savefig(f"{output_dir}MW_mass_assembly_all_substructures.pdf", dpi=200)

############################################################################

# Comparison between MW inferred assembly history and Auriga galaxies
auriga_assemblies = pickle.load(open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_galaxies_mass_assembly.pkl","rb"))

fig, ax = plt.subplots(1,1)
xlim = [0.2,13.8]
ylim = [8.8,12.5]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*0.5)

ax.set_ylabel("$\log(M_{\mathrm{MW}}/\mathrm{M}_{\odot})$")
ax.set_xlabel("Lookback Infall Time [Gyr]")

# Plot the assembly histories of the Auriga halos
for halo,(time,mass) in auriga_assemblies.items():
    ax.plot(time, 
            mass,
            lw=1,
            c='k',
            alpha=0.5
           )
    
# Plot assembly history inferred for the MW
for substructure, (time,mass) in assembly_dict.items():
    
    median_time = np.percentile(time,50)
    time_16 = np.percentile(time,16)
    time_84 = np.percentile(time,84)
    
    median_mass = np.percentile(mass,50)
    mass_16 = np.percentile(mass,16)
    mass_84 = np.percentile(mass,84)
    
    ax.fill_between([time_16, time_84],
                    [mass_16, mass_16],
                    [mass_84, mass_84],
                    color="red",
                    alpha=0.5,
                    zorder=10,
                   )
fig.savefig(f"{output_dir}MW_mass_assembly_vs_auriga_all_substructures.pdf", dpi=200)

######################################################################################

fig, ax = plt.subplots(1,1, figsize=(8,6))
xlim = [0.2,13.8]
ylim = [-0.5,20]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*0.5)

ax.set_ylabel("Mass $[\\times \, 10^{8} \, \mathrm{M}_{\odot}]$")
ax.set_xlabel("Lookback Infall Time [Gyr]") 


# Plot cumulative distribution of accreted stellar mass
median_infall_time, median_accreted_stellar_mass = [], []

for substructure in substructures:
    samples = pickle.load(open(f"{posterior_samples_dir}{substructure}.pkl", "rb"))
    infall_times = samples[:,0]
    stellar_mass = samples[:,1]
    
    median_infall_time.append(np.percentile(infall_times,50))
    median_accreted_stellar_mass.append(np.percentile(stellar_mass,50))
    
    # Plot bands corresponding to accretion times
    time_16 = np.percentile(assembly_dict[substructure][0], 16)
    time_84 = np.percentile(assembly_dict[substructure][0], 84)
    ax.fill_betweenx([ylim[0], ylim[1]],
                     time_16,
                     time_84,
                     alpha=0.5,
                     color=colors_dict[substructure]
                    )
    

# Order events from earliest to latest
idx = np.argsort(median_infall_time)[::-1]
accreted_stellar_mass = [10**median_accreted_stellar_mass[i] for i in idx]
infall_time = [median_infall_time[i] for i in idx]

cdf = [sum(accreted_stellar_mass[:i+1])*1e-8 for i in range(len(accreted_stellar_mass))]

ax.plot(infall_time,
        cdf,
        lw=3,
        c="k",
        label="MW accreted mass"
       )
ax.scatter(infall_time,
        cdf,
        s=40,
        c="k",
        zorder=20
       )
####
ax.fill_between([0,14],
                10,
                18,
                alpha=0.1,
                color="k"
               )
ax.plot([0,14],
        [10,10],
        lw=0.5,
        ls="-",
        color="k"
       )
ax.plot([0,14],
        [18,18],
        lw=0.5,
        ls="-",
        color="k"
       )

ax.text(0.5,16,s="MW stellar halo (Deason+19)", fontsize=12)
####

ax.fill_between([0,14],
                8,
                11.1,
                alpha=0.1,
                color="b"
               )
ax.plot([0,14],
        [8,8],
        lw=0.5,
        ls="-",
        color="b"
       )
ax.plot([0,14],
        [11.1,11.1],
        lw=0.5,
        ls="-",
        color="b"
       )

ax.text(0.5,5.,s="MW accreted stellar halo\n (Mackereth&Bovy23)", fontsize=12, color="b")
ax.legend(bbox_to_anchor=(0.38,0.6), fontsize=12, frameon=False)

fig.savefig(f"{output_dir}MW_mass_accreted_mass_all_substructures.pdf", dpi=400)

###########################################################################################
###########################################################################################
#       NAIDU SPLIT OF THE RETROGRADE HALO
###########################################################################################
###########################################################################################

substructures.remove("Sequoia_K19")
substructures.remove("Sequoia_M19")
substructures.sort()

substructure_labels["Sequoia_N20"] = "Sequoia N20"

colors_dict.pop("Sequoia_K19", None)
colors_dict.pop("Sequoia_M19", None)

assembly_dict = {}
for substructure in substructures:
    
    samples = pickle.load(open(f"{posterior_samples_dir}{substructure}.pkl", "rb"))
    infall_times = samples[:,0]
    halo_mass = samples[:,2]
    mmr = samples[:,3]
    
    proto_MW_mass = halo_mass-mmr
    
    assembly_dict[substructure] = (infall_times, proto_MW_mass)

    


fig,ax = plt.subplots(1,1)

xlim = [0.2,13.8]
ylim = [8.8,12.5]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*0.5)

ax.set_ylabel("$\log(M_{\mathrm{MW}}/\mathrm{M}_{\odot})$")
ax.set_xlabel("Lookback Infall Time [Gyr]")



for substructure in substructures:
    
    x = assembly_dict[substructure][0]
    y = assembly_dict[substructure][1]
    
    output = binned_statistic(x=x, 
                              values=y, 
                              bins=20, 
                              statistic="mean",
                              range=(np.percentile(x,16),
                                     np.percentile(x,84))
                             )
    
    ax.plot((output[1][1:]+output[1][:-1])/2,
            output[0],
            color=colors_dict[substructure]
           )
    
    ax.bar(x=np.percentile(x,16),
           align="edge",
           height=np.mean(output[0]),
           color=colors_dict[substructure],
           width=np.percentile(x,84)-np.percentile(x,16),
           alpha=0.5
          )

handles = []
for substructure in substructures:
    label = substructure_labels[substructure]
        
    line = plt.Line2D([],[], 
                   linestyle='-', 
                   linewidth=5,
                   color=colors_dict[substructure],
                   label=label) 
    handles.append(line)


fig.legend(handles=handles,
           loc="upper center", 
           bbox_to_anchor=(0.51, 1.06), 
           ncols=3, 
           frameon=True, 
           edgecolor='black', 
           fontsize=11)

fig.savefig(f"{output_dir}MW_mass_assembly_naidu_split.pdf", dpi=400)

############################################################
############################################################
############################################################
############################################################

# Comparison between MW inferred assembly history and Auriga galaxies
auriga_assemblies = pickle.load(open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_galaxies_mass_assembly.pkl","rb"))

fig, ax = plt.subplots(1,1)
xlim = [0.2,13.8]
ylim = [8.8,12.5]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*0.5)

ax.set_ylabel("$\log(M_{\mathrm{MW}}/\mathrm{M}_{\odot})$")
ax.set_xlabel("Lookback Infall Time [Gyr]")

# Plot the assembly histories of the Auriga halos
for halo,(time,mass) in auriga_assemblies.items():
    ax.plot(time, 
            mass,
            lw=1,
            c='k',
            alpha=0.5
           )
    
# Plot assembly history inferred for the MW
for substructure, (time,mass) in assembly_dict.items():
    
    median_time = np.percentile(time,50)
    time_16 = np.percentile(time,16)
    time_84 = np.percentile(time,84)
    
    median_mass = np.percentile(mass,50)
    mass_16 = np.percentile(mass,16)
    mass_84 = np.percentile(mass,84)
    
    ax.fill_between([time_16, time_84],
                    [mass_16, mass_16],
                    [mass_84, mass_84],
                    color="red",
                    alpha=0.5,
                    zorder=10
                   )
    
fig.savefig(f"{output_dir}MW_mass_assembly_vs_auriga_naidu_split.pdf", dpi=400)
    
############################################################
############################################################
############################################################
############################################################

fig, ax = plt.subplots(1,1, figsize=(8,6))
xlim = [0.2,13.8]
ylim = [-0.5,20]
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0])*0.5)

ax.set_ylabel("Mass $[\\times \, 10^{8} \, \mathrm{M}_{\odot}]$")
ax.set_xlabel("Lookback Infall Time [Gyr]") 


# Plot cumulative distribution of accreted stellar mass
median_infall_time, median_accreted_stellar_mass = [], []

for substructure in substructures:
    samples = pickle.load(open(f"{posterior_samples_dir}{substructure}.pkl", "rb"))
    infall_times = samples[:,0]
    stellar_mass = samples[:,1]
    
    median_infall_time.append(np.percentile(infall_times,50))
    median_accreted_stellar_mass.append(np.percentile(stellar_mass,50))
    
    # Plot bands corresponding to accretion times
    time_16 = np.percentile(assembly_dict[substructure][0], 16)
    time_84 = np.percentile(assembly_dict[substructure][0], 84)
    ax.fill_betweenx([ylim[0], ylim[1]],
                     time_16,
                     time_84,
                     alpha=0.5,
                     color=colors_dict[substructure]
                    )
    

# Order events from earliest to latest
idx = np.argsort(median_infall_time)[::-1]
accreted_stellar_mass = [10**median_accreted_stellar_mass[i] for i in idx]
infall_time = [median_infall_time[i] for i in idx]

cdf = [sum(accreted_stellar_mass[:i+1])*1e-8 for i in range(len(accreted_stellar_mass))]

ax.plot(infall_time,
        cdf,
        lw=3,
        c="k",
        label="MW accreted mass"
       )
ax.scatter(infall_time,
        cdf,
        s=40,
        c="k",
        zorder=20
       )
####
ax.fill_between([0,14],
                10,
                18,
                alpha=0.1,
                color="k"
               )
ax.plot([0,14],
        [10,10],
        lw=0.5,
        ls="-",
        color="k"
       )
ax.plot([0,14],
        [18,18],
        lw=0.5,
        ls="-",
        color="k"
       )

ax.text(0.5,16,s="MW stellar halo (Deason+19)", fontsize=12)
####

ax.fill_between([0,14],
                8,
                11.1,
                alpha=0.1,
                color="b"
               )
ax.plot([0,14],
        [8,8],
        lw=0.5,
        ls="-",
        color="b"
       )
ax.plot([0,14],
        [11.1,11.1],
        lw=0.5,
        ls="-",
        color="b"
       )

ax.text(0.5,5.,s="MW accreted stellar halo\n (Mackereth&Bovy23)", fontsize=12, color="b")
ax.legend(bbox_to_anchor=(0.88,0.6), fontsize=12, frameon=False)

fig.savefig(f"{output_dir}MW_mass_accreted_mass_naidu_split.pdf", dpi=400)