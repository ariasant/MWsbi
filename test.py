import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
import pymc as pm
import torch
from torch.utils.data import Dataset

import corner

import pickle
import torch
import jax


class NoisyDataset(Dataset):
    def __init__(self, initial_data, noise, compression_model):

        # Store features and labels as PyTorch tensors directly
        self.features = initial_data[0].float()
        self.labels = initial_data[1].float()
        self.compression_model = compression_model

        # Add noise to data
        noise = torch.from_numpy(noise).float() if isinstance(noise, np.ndarray) else noise
        noisy_data = self.features + noise

        # Convert PyTorch tensor to JAX array using DLPack (zero-copy if on same device)
        noisy_data_jax = jax.dlpack.from_dlpack(noisy_data, copy=False)

        # Apply JAX compression model
        compressed_data_jax = self.compression_model(noisy_data_jax)[0]

        # Convert JAX array back to PyTorch tensor using DLPack
        self.compressed_data_torch = torch.from_dlpack(compressed_data_jax)


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.compressed_data_torch[idx]
        return data, label

    def update_noise_matrix(self, new_noise):
        
        # Add new noise to data
        noise = torch.from_numpy(new_noise).float() if isinstance(new_noise, np.ndarray) else new_noise
        noisy_data = self.features + noise

        # Convert PyTorch tensor to JAX array using DLPack (zero-copy if on same device)
        noisy_data_jax = jax.dlpack.from_dlpack(noisy_data, copy=False)

        # Apply JAX compression model
        compressed_data_jax = self.compression_model(noisy_data_jax)[0]

        # Convert JAX array back to PyTorch tensor using DLPack
        self.compressed_data_torch = torch.from_dlpack(compressed_data_jax)

def generate_mean_cov_model(features, 
                            n_stars_per_prog,
                            n_progenitors):

    n_features = len(features)
    #shifts = [abs(s) for s in shifts]
    coords = {"features": features, 
              "features_bis": features, 
              "star_id": np.arange(n_stars_per_prog),
              "prog_id": np.arange(n_progenitors)}

    with pm.Model(coords=coords) as model:
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", 
            n=n_features, 
            eta=1.0, 
            sd_dist=pm.HalfNormal.dist(sigma=1,shape=n_features)
        )
        mu_components = pm.math.stack([pm.Normal("mu_E", mu=0, sigma=0.5),
                                       pm.Normal("mu_L", mu=0, sigma=0.5),
                                       pm.Uniform("mu_FeH", lower=-0.5, upper=-0.1),
                                       pm.Normal("mu_MgFe", mu=0.4, sigma=0.1)
                                       ])
        mu = pm.Deterministic("shifts", mu_components, dims="features")

        noise_stars_in_single_prog = pm.MvNormal("noise", 
                                                  mu, 
                                                  chol=chol, 
                                                  dims=("prog_id", "star_id", "features"))
        
        return model
        
def sample_noise_training(model, 
                          n_epochs,
                          random_seed):

    with model:
        prior_samples = pm.sample_prior_predictive(samples=n_epochs,
                                                   random_seed=random_seed)
        
    return prior_samples.prior["noise"].values[0]

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

    if RANGE is None:
        dfs_all = pd.concat(dfs)
        RANGE = [(dfs_all[f].quantile(0.01), dfs_all[f].quantile(0.99))
                 for f in features]

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



# Load training and test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set numpy random key
rng = np.random.default_rng(17)



features = ["E","L","FeH","MgFe"]
parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']
filename = f"Suite_"+"".join(features)

output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/shifts_marg/'

"""sim_data = pd.read_pickle(f"{output_dir}/data/sim_ds_processed_{filename}.pkl")
obs_data = pd.read_pickle(f"{output_dir}/data/apogee_ds_processed_{filename}.pkl")
scaler_params = pickle.load(open(f"{output_dir}/data/theta_scaler_Suite_ELFeHMgFe.pkl","rb")) 
data_scaler = pickle.load(open(f"{output_dir}/data/data_scaler_{filename}.pkl","rb"))
"""

import os
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

obs_data = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")
apogee_ds_satellites = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_satellites_ds.pkl")
obs_data = pd.concat([obs_data, apogee_ds_satellites])
obs_data.dropna(subset=features, inplace=True)
obs_data = obs_data[(obs_data["E"]<0)&(obs_data["L"]>0)]

sim_data["E"] = np.log(-sim_data["E"].values)
sim_data["L"] = np.log(sim_data["L"].values)

obs_data["E"] = np.log(-obs_data["E"].values)
obs_data["L"] = np.log(obs_data["L"].values)


shifts = [sim_data[f].median() - obs_data[f].median() for f in features]


substructures = ['Arjuna','GES', 'Sagittarius', 'Helmi',
                 'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
                 'LMS', 'Heracles']
obs_accreted = ((obs_data.AlFe<-0.07) & (obs_data.MgMn>=0.25)) | \
               ((obs_data.AlFe>=-0.07) & (obs_data.MgMn>=4.25*obs_data.AlFe+0.5475))
obs_accreted = np.logical_or.reduce([obs_accreted]+[obs_data[f"{substructure}_flag"]==1 
                                    for substructure in substructures])

fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]])
fig.savefig(f"{output_dir}initial_data.pdf", dpi=300, bbox_inches='tight')

## Generate noise

def add_noise(group):

    n_stars = min(1000, group.shape[0])

    train_noise_model = generate_mean_cov_model(features=features,
                                            n_stars_per_prog=n_stars,
                                            n_progenitors=1) 
    train_noise_epochs = sample_noise_training(model=train_noise_model,
                                            n_epochs=1,
                                            random_seed=len(group))
    
    return group.sample(n_stars)[features] + train_noise_epochs[0,0]



sim_data = sim_data.groupby("progID").apply(add_noise)

fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]])
fig.savefig(f"{output_dir}initial_data_plus_noise2.pdf", dpi=300, bbox_inches='tight')