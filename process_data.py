from adapt.feature_based import CORAL
import corner
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

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


def plot_stars_data(dfs: list, RANGE=None):

    # Initialize color list for the different dataframes
    colors = [mpl.cm.tab10(i/len(dfs)) for i in range(len(dfs))]

    for i, df in enumerate(dfs):

        if i==0:
            fig = corner.corner(df[features].values,
                                color=colors[i],
                                labels=["$E$", "$L$", "[Fe/H]", "[Mg/Fe]"],
                                bins=20,
                                plot_contours=True,
                                plot_datapoints=False,
                                fill_contours=True,
                                hist_kwargs={"density": True},
                                alpha=0.5,
                                range=RANGE,
                                label_kwargs={'fontsize': 18},
                                )
        else:
            corner.corner(df[features].values,
                            color=colors[i],
                            bins=20,
                            plot_contours=True,
                            plot_datapoints=False,
                            fill_contours=True,
                            hist_kwargs={"density": True},
                            alpha=0.5,
                            range=RANGE,
                            label_kwargs={'fontsize': 18},
                            fig=fig)
            
    return fig


features = ['E', 'L', 'FeH', 'MgFe']
parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']

output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/coral/'

print(f"output_dir: {output_dir}", flush=True)

filename = f"Suite_"+"".join(features)

substructures = ['Arjuna','GES', 'Sagittarius', 'Helmi',
                 'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
                 'LMS', 'Heracles']

###########################################################################################
# Data preparation
###########################################################################################

# Load Milky Way (target) data
obs_data = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")
apogee_ds_satellites = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_satellites_ds.pkl")
obs_data = pd.concat([obs_data, apogee_ds_satellites])
obs_data.dropna(subset=features, inplace=True)
obs_data = obs_data[(obs_data["E"]<0)&(obs_data["L"]>0)]
# Select accreted stars
obs_accreted = ((obs_data.AlFe<-0.07) & (obs_data.MgMn>=0.25)) | \
               ((obs_data.AlFe>=-0.07) & (obs_data.MgMn>=4.25*obs_data.AlFe+0.5475))
obs_accreted = np.logical_or.reduce([obs_accreted]+[obs_data[f"{substructure}_flag"]==1 
                                    for substructure in substructures])


# Load simulation (source) data
data_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/data/with_satellites/"
sim_data = []

for file in os.listdir(data_dir):

    df = pd.read_pickle(f"{data_dir}{file}")
    df.rename(columns={"aFe":"MgFe"}, inplace=True)

    # Get rid of stars with numerical issues
    df = df[(df["E"]<0) & (df["L"]>0) & 
            (df["MgFe"]<0.5) & (df["MgFe"]>-0.5) &
            (df["FeH"]<1) & (df["FeH"]>-3)]

    sim_data.append(df)

sim_data = pd.concat(sim_data, ignore_index=True)

# Load sample of accreted stars similarly distributed in simulations
sim_df_accreted = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/disc_plots/sim_df_accreted.pkl")
sim_df_accreted = sim_df_accreted[(sim_df_accreted["L"]>0)]

# Rank matching simulations to MW stars
#sim_data = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/sim_match2_obs/sim_data.pkl")


# Plot initial data
fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]])
fig.savefig(f"{output_dir}initial_data_{filename}.pdf", dpi=300, bbox_inches='tight')


# Preprocess data
sim_data["E"] = np.log(-sim_data["E"].values)
sim_data["L"] = np.log(sim_data["L"].values)

obs_data["E"] = np.log(-obs_data["E"].values)
obs_data["L"] = np.log(obs_data["L"].values)

sim_df_accreted["E"] = np.log(-sim_df_accreted["E"].values)
sim_df_accreted["L"] = np.log(sim_df_accreted["L"].values)

# Match the mean of feature distribution to the target ones
mean_shifts=sim_df_accreted[features].mean().values - obs_data.loc[obs_accreted,features].mean().values
#mean_shifts[2]=0.4
#mean_shifts[3]=-0.4

sim_data[features] = sim_data[features] - mean_shifts


# Scale data to 0 mean and 1 std
data_scaler = RobustScaler()
data_scaler.fit(obs_data.loc[obs_accreted,features].values)
sim_data[features] = data_scaler.transform(sim_data[features].values)
obs_data[features] = data_scaler.transform(obs_data[features].values)

# Remove outliers
sim_data = sim_data[np.logical_and.reduce([sim_data[feature]**2 < 5**2 for feature in features])]
obs_data = obs_data[np.logical_and.reduce([obs_data[feature]**2 < 5**2 for feature in features])]

obs_accreted = ((obs_data.AlFe<-0.07) & (obs_data.MgMn>=0.25)) | \
               ((obs_data.AlFe>=-0.07) & (obs_data.MgMn>=4.25*obs_data.AlFe+0.5475))
obs_accreted = np.logical_or.reduce([obs_accreted]+[obs_data[f"{substructure}_flag"]==1 
                                    for substructure in substructures])

# Match covariance matrices with coral
coral_model = CORAL()
sim_data[features] = coral_model.fit_transform(Xs=sim_data[features].values, 
                                               Xt=obs_data.loc[obs_accreted,features].values)

# Plot data after processing
fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]], 
                      RANGE=[(-4,4), (-4,4), (-4,4), (-4,4)])
# Add legend
labels = ["Auriga", "APOGEE", "APOGEE (accreted)"]
colors = [mpl.cm.tab10(i/3) for i in range(3)]
mpl.pyplot.legend(
        handles=[
            mpl.lines.Line2D([], [], 
                             linewidth=5,
                             color=colors[i], 
                             label=labels[i])
            for i in range(3)
        ],
        fontsize=20, frameon=False,
        bbox_to_anchor=(1, 4), loc="upper right"
        )
fig.savefig(f"{output_dir}transformed_data_{filename}.pdf", dpi=300, bbox_inches='tight')

"""# Plot the stars in each substructure
print("Counting stars in each substructure of the MW before and after removing outliers:", flush=True)
for substructure in substructures:
    # Plot the stars in each substructure
    fig = plot_stars_data([sim_data, obs_data[obs_data[f"{substructure}_flag"]==1]],
                          RANGE=[(-3,3), (-3, 3), (-3, 3), (-3, 3)])
    fig.savefig(f"{output_dir}transformed_data_{filename}_shifted_{substructure}.pdf", dpi=300, bbox_inches='tight')"""

# Save FeH and stellar mass of processed data for plots
FeH_dict, stellar_mass_dict = {}, {}
for ID in sim_data["progID"].unique():
    df = sim_data[sim_data["progID"]==ID]
    df[features] = data_scaler.inverse_transform(df[features].values)
    FeH_dict[ID] = df["FeH"].median()
    stellar_mass_dict[ID] = df["log_Mprog_stellar"].median()

pickle.dump(FeH_dict, open(f"{output_dir}FeH_dict.pkl", "wb"))
pickle.dump(stellar_mass_dict, open(f"{output_dir}stellar_mass_dict.pkl", "wb"))

# Plot merger parameters
fig = corner.corner(sim_data[parameters].values,
                    color='k',
                    labels=parameters,
                    bins=20,
                    plot_contours=False,
                    plot_datapoints=False,
                    fill_contours=False,
                    hist_kwargs={"density": True})
fig.savefig(f"{output_dir}merger_parameters_{filename}.pdf", dpi=300, bbox_inches='tight')


# Initialize the scaler for the merger parameters
scaler_params = RobustScaler()
# Scale the merger parameters
sim_data[parameters] = scaler_params.fit_transform(sim_data[parameters].values)


# Save scalers for future analysis
pickle.dump(data_scaler,open(f"{output_dir}/data/data_scaler_{filename}.pkl","wb"))
pickle.dump(scaler_params,open(f"{output_dir}/data/theta_scaler_{filename}.pkl","wb"))
# Save processed simulation data
sim_data.to_pickle(f"{output_dir}/data/sim_ds_processed_{filename}.pkl")

# Save processed Milky Way data
obs_data.to_pickle(f"{output_dir}/data/apogee_ds_processed_{filename}.pkl")

