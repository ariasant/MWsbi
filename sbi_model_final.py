import argparse
import corner
import math
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import torch

import sys
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import multitask_model as mt
import sbi_results


class CombinedDataset(torch.utils.data.TensorDataset):

    def __init__(self, 
                 datasetA: torch.utils.data.TensorDataset,
                 datasetB: torch.utils.data.TensorDataset):
    
        self.datasetA = datasetA
        self.datasetB = datasetB
        
    def __getitem__(self, index):
        dataA, conditionA = self.datasetA[index]
        dataB, conditionB = self.datasetB[index]
        
        data = torch.vstack((dataA, dataB))
        conditions = torch.vstack((conditionA, conditionB))
        return data, conditions
    
    def __len__(self):
        return len(self.datasetA)


def plot_stars_data(dfs: list, RANGE=None):

    # Initialize color list for the different dataframes
    colors = [mpl.cm.tab10(i/len(dfs)) for i in range(len(dfs))]

    for i, df in enumerate(dfs):

        if i==0:
            fig = corner.corner(df[features].values,
                                color=colors[i],
                                labels=features,
                                bins=20,
                                plot_contours=True,
                                plot_datapoints=False,
                                fill_contours=True,
                                hist_kwargs={"density": True},
                                alpha=0.5,
                                range=RANGE
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
                            fig=fig)
    return fig



CLI = argparse.ArgumentParser()
CLI.add_argument(
        "--features",
        nargs="*",
        type=str,
        default=['E', 'L', 'FeH', 'MgFe']
    )

args = CLI.parse_args()
features = args.features
parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']

dataframes_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observations/data/"
output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/multitask_results/'

filename = f"Suite_"+"".join(features)

substructures = ['GES', 'Sagittarius', 'Helmi',
       'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
       'LMS', 'Heracles']


BATCH_SIZE = 64

# Define device where the model will be trained
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###########################################################################################
# Data preparation
###########################################################################################


f = np.load("/mnt/aridata1/users/ariasant/MW-sbi/multitask_results/train_data.npz")
sim_X_train, sim_Y_train, obs_X_train, obs_Y_train = f["sim_X"], f["sim_Y"], f["obs_X"], f["obs_Y"]

f = np.load("/mnt/aridata1/users/ariasant/MW-sbi/multitask_results/val_data.npz")
sim_X_val, sim_Y_val, obs_X_val, obs_Y_val = f["sim_X"], f["sim_Y"], f["obs_X"], f["obs_Y"]



"""# Load simulation (source) data
data_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/data/with_satellites/"
sim_data = []

for file in os.listdir(data_dir):

    df = pd.read_pickle(f"{data_dir}{file}")
    df.rename(columns={"aFe":"MgFe"}, inplace=True)

    # Get rid of stars with numerical issues
    df = df[(df["E"]<0) & (df["L"]>0)
            & (df["FeH"]>-3) & (df["FeH"]<1)
            & (df["MgFe"]>-1) & (df["MgFe"]<1)]

    sim_data.append(df)

sim_data = pd.concat(sim_data, ignore_index=True)

# Load Milky Way (target) data
apogee_ds = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")
apogee_ds.dropna(subset=features, inplace=True)
# Select accreted stars
obs_accreted = ((apogee_ds.AlFe<-0.07) & (apogee_ds.MgMn>=0.25)) | \
               ((apogee_ds.AlFe>=-0.07) & (apogee_ds.MgMn>=4.25*apogee_ds.AlFe+0.5475))
obs_accreted = np.logical_or.reduce([obs_accreted]+[apogee_ds[f"{substructure}_flag"]==1 
                                    for substructure in ['GES', 'Sagittarius', 'Helmi',
                                                         'Sequoia_K19','Sequoia_M19','Sequoia_N20',
                                                         'Iitoi', 'Thamnos','LMS', 'Heracles']])

obs_data = apogee_ds

# Plot initial data
fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]],
                      RANGE=[(-3e5, 0), (0, 1e4), (-3, 1), (-0.6, 0.6)])
fig.savefig(f"{output_dir}initial_data_{filename}.pdf", dpi=300, bbox_inches='tight')


# Scale data
scaler = RobustScaler()
scaler.fit(np.vstack([sim_data[features].values, obs_data[features].values]))
sim_data[features] = scaler.transform(sim_data[features].values)
obs_data[features] = scaler.transform(obs_data[features].values)

# Save scaler
pickle.dump(scaler, open(f"{output_dir}/data_scaler_{filename}.pkl", "wb"))


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


print(f"N progID: {len(sim_data['progID'].unique())}", flush=True)

# Create datasets for training 
X_train, Y_train = [], []

for progID in sim_data["progID"].unique():
    # Get the data for the current progenitor
    prog_data = sim_data[sim_data["progID"] == progID]
    if len(prog_data) < 100:
        continue

    # Precompute the reshaped feature values for the progenitor
    prog_features = prog_data[features].values
    prog_parameters = prog_data[parameters].values

    # Sample the data n times
    n = min(10, math.ceil(len(prog_data)/100)) # number of samples per progenitor
    idx_samples = np.random.randint(0, len(prog_data), size=(n, 100))
    X_train.extend(prog_features[idx_samples].reshape(n, -1))
    Y_train.extend(prog_parameters[idx_samples[:, 0]])

sim_X = np.stack(X_train)
sim_Y = np.stack(Y_train)

# Precompute observational data indices
obs_idx_samples = np.random.randint(0, len(obs_data), size=(len(X_train), 100))
obs_X = obs_data[features].values[obs_idx_samples].reshape(len(X_train), -1)

obs_Y = np.vstack([np.array([np.nan] * len(parameters))] * len(obs_X))

# Split data for training and test
sim_X_train, sim_X_val, sim_Y_train, sim_Y_val = train_test_split(sim_X, sim_Y, test_size=0.2)
obs_X_train, obs_X_val, obs_Y_train, obs_Y_val = train_test_split(obs_X, obs_Y, test_size=0.2)
print(f"Sim train examples: {sim_X_train.shape[0]}", flush=True)
print(f"Obs train examples: {obs_X_train.shape[0]}", flush=True)
print(f"Sim test examples: {sim_X_val.shape[0]}", flush=True)
print(f"Obs test examples: {obs_X_val.shape[0]}", flush=True)

np.savez(f"{output_dir}train_data", sim_X=sim_X_train, sim_Y=sim_Y_train, obs_X=obs_X_train, obs_Y=obs_Y_train)
np.savez(f"{output_dir}val_data", sim_X=sim_X_val, sim_Y=sim_Y_val, obs_X=obs_X_val, obs_Y=obs_Y_val)"""


# Create dataloaders for training
sim_train_dataset = torch.utils.data.TensorDataset(torch.Tensor(sim_X_train).to(device), 
                                                   torch.Tensor(sim_Y_train).to(device))
obs_train_dataset = torch.utils.data.TensorDataset(torch.Tensor(obs_X_train).to(device), 
                                                   torch.Tensor(obs_Y_train).to(device))
# Combine simulation and observations in a single dataset
train_dataset = CombinedDataset(datasetA=sim_train_dataset,
                                datasetB=obs_train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH_SIZE//2,
                                           shuffle=False,
                                           pin_memory=False)

# Repeat for validation set
sim_val_dataset = torch.utils.data.TensorDataset(torch.Tensor(sim_X_val).to(device), 
                                                 torch.Tensor(sim_Y_val).to(device))
obs_val_dataset = torch.utils.data.TensorDataset(torch.Tensor(obs_X_val).to(device), 
                                                 torch.Tensor(obs_Y_val).to(device))
test_dataset = CombinedDataset(datasetA=sim_val_dataset,
                               datasetB=obs_val_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=BATCH_SIZE//2,
                                         shuffle=False,
                                         pin_memory=False)

# Save processed Milky Way data
#pickle.dump(obs_data, open(f"{output_dir}/apogee_ds_processed_{filename}.pkl", "wb"))

####################################################################################
####################################################################################
# Training
####################################################################################
####################################################################################

# Initialize the model
model = mt.MultiTask(theta_dim=len(parameters), # Dimensions of the probability distribution approximated by the flow
                     n_conditions=len(features)*100, 
                     n_layers_enc = 3,
                     latent_dim_enc = 50,
                     n_transforms = 10,
                     n_layers_per_transform = 2,
                     n_neurons_flow = 50)

# Move model to training device
model = model.to(device)

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), 
                             lr=1e-4, 
                             weight_decay=1e-5)

# Train the model
training_results = model.train_model(train_dataloader=train_loader,
                                     val_dataloader=test_loader,
                                     optimizer=optimizer,
                                     epochs=1000,
                                     n_warmup_epochs=20,
                                     weights_path=f"{output_dir}model_weights/"
                                     )

# Plot the training losses
fig = mt.plot_training_losses(training_results)
fig.savefig(f"{output_dir}training.pdf")
fig = mt.plot_distances(training_results)
fig.savefig(f"{output_dir}training_distances.pdf")

# Save the training results
pickle.dump(training_results, open(f"{output_dir}training_results_{filename}.pkl", "wb"))

#training_results = pickle.load(open(f"{output_dir}training_results_{filename}.pkl","rb"))

# Load weights of the best model (lower validation loss) from training
best_epoch = np.argmin(training_results["val_loss"])
best_weights = torch.load(f"{output_dir}model_weights/model_{best_epoch}.pt", map_location=torch.device(device)) 

model.load_state_dict(best_weights)

####################################################################################
####################################################################################
# Validation
####################################################################################
####################################################################################

# Set the model for evaluation
model.eval()

# Sample parameters for test galaxy
n_samples = 1000
prior_ranges = [[0,14],[6,11],[7,12],[-3,0]]

samples_dict = {}

for i,(X,fiducial) in enumerate(sim_val_dataset):

    fiducial = fiducial.cpu().numpy()
    samples = model.sample(X[None,:], 
                           n_samples=n_samples, 
                           prior_ranges=prior_ranges).cpu().numpy()
    
    samples_dict[f"test{i}"] = (samples, 0, fiducial)

# Save samples 
pickle.dump(samples_dict, 
            open(f'{output_dir}{filename}_test_samples.pkl', 'wb'))

filename = "Suite_"+"".join(features)+"".join(parameters)
# Make plot of cross-validated parameters inference
plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']
plot_ranges=[[0.1,13.9],[5.9,10.9],[8.1,11.9],[-3.2,-0.1]]

fig = sbi_results.cross_validation_plot(samples_dict=samples_dict,
                                        percentile_range=[16,84],
                                        plot_labels=plot_labels,
                                        plot_ranges=plot_ranges)
fig.savefig(f"{output_dir}cross_validation_1684_{filename}.pdf", dpi=400)


sbi_results.rms_table_per_galaxy(samples_dict=samples_dict,
                                 parameters=parameters,
                                 filename=f'{output_dir}rms_table_{filename}.csv')
