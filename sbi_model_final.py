import argparse
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import pickle
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import time
import torch
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater

import sys
sys.path.append("/mnt/aridata1/users/ariasant/auriga-sbi/")
from domain_shift import DataProcessor
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import sbi_training
import sbi_results



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

dataframes_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/data/gnn/"
output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/gnn/'

filename = f"Suite_"+"".join(features)

substructures = ['GES', 'Sagittarius', 'Helmi',
       'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
       'LMS', 'Heracles']

###########################################################################################
# Data preparation
###########################################################################################

# Load simulation (source) data
data_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/data/with_satellites/"
sim_data = []

for file in os.listdir(data_dir):

    df = pd.read_pickle(f"{data_dir}{file}")
    df.rename(columns={"aFe":"MgFe"}, inplace=True)

    # Get rid of stars with numerical issues
    df = df[(df["E"]<0) & (df["L"]>0)]

    # Shift chemical abundances
    df["FeH"] = df["FeH"]-0.2
    df["MgFe"] = df["MgFe"]+0.4

    sim_data.append(df)

df = pd.concat(sim_data, ignore_index=True)

# Load Milky Way (target) data
apogee_ds = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")
apogee_ds_satellites = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_satellites_ds.pkl")
apogee_ds = pd.concat([apogee_ds, apogee_ds_satellites])
apogee_ds.dropna(subset=features, inplace=True)
# Select accreted stars
obs_accreted = ((apogee_ds.AlFe<-0.07) & (apogee_ds.MgMn>=0.25)) | \
               ((apogee_ds.AlFe>=-0.07) & (apogee_ds.MgMn>=4.25*apogee_ds.AlFe+0.5475))
obs_accreted = np.logical_or.reduce([obs_accreted]+[apogee_ds[f"{substructure}_flag"]==1 
                                    for substructure in ['GES', 'Sagittarius', 'Helmi',
                                                         'Sequoia_K19','Sequoia_M19','Sequoia_N20',
                                                         'Iitoi', 'Thamnos','LMS', 'Heracles']])


obs_data = apogee_ds

# Preprocess data
sim_data, obs_data, pt, FeH_min, MgFe_min = DataProcessor(features=features,
                                                          sim_data=df,
                                                          obs_data=obs_data)

# Repeat accreted stars selection because of the transformation
obs_accreted = ((obs_data.AlFe<-0.07) & (obs_data.MgMn>=0.25)) | \
               ((obs_data.AlFe>=-0.07) & (obs_data.MgMn>=4.25*obs_data.AlFe+0.5475))
obs_accreted = np.logical_or.reduce([obs_accreted]+[obs_data[f"{substructure}_flag"]==1 
                                    for substructure in ['GES', 'Sagittarius', 'Helmi',
                                                         'Sequoia_K19','Sequoia_M19','Sequoia_N20',
                                                         'Iitoi', 'Thamnos','LMS', 'Heracles']])
apogee_ds_processed = obs_data[obs_accreted]

# Initialize the scaler for the merger parameters
scaler_params = RobustScaler()
# Scale the merger parameters
df[parameters] = scaler_params.fit_transform(df[parameters].values)

print(f"N progID: {len(df['progID'].unique())}", flush=True)

# Create datasets for training 
graphs = []

max_nodes = 1000  # maximum number of nodes per graph

for progID in df["progID"].unique():
    # Get the data for the current progenitor
    prog_data = df[df["progID"] == progID]
    n_nodes = prog_data.shape[0]
    if n_nodes < 100:
        continue  # skip very small progenitors

    # Sample up to max_nodes
    if n_nodes > max_nodes:
        n_samples = min(n_nodes // max_nodes, 10)
        prog_data_list = [prog_data.sample(n=max_nodes, random_state=42) for i in range(n_samples)]
        n_nodes = max_nodes
    else:
        prog_data_list = [prog_data]

    for prog_data in prog_data_list:
        # Node features: shape (n_nodes, n_features)
        node_features = torch.tensor(prog_data[features].values, dtype=torch.float)

        # Compute pairwise distances in feature space
        edge_weights = cdist(node_features, node_features, metric='euclidean')  # shape (n_nodes, n_nodes)

        # Create edge index and edge attributes for a fully connected graph
        row_idx, col_idx = np.where(np.ones((n_nodes, n_nodes)) - np.eye(n_nodes))
        edge_index = torch.tensor(np.vstack([row_idx, col_idx]), dtype=torch.int64)
        edge_attr = torch.tensor(edge_weights[row_idx, col_idx][:, None], dtype=torch.float)  # shape (num_edges, 1)

        # Create torch_geometric Data object
        data = Data(
            x=node_features,           # Node features
            edge_index=edge_index,     # Edge indices (2, num_edges)
            edge_attr=edge_attr,       # Edge attributes (num_edges, 1)
            y=torch.tensor(prog_data[parameters].values[0], dtype=torch.float)[None,:], # merger parameters to infer
            progID=progID,             # Store progID for reference
        )
        graphs.append(data)

data = graphs
# output (input, output) pairs
# use pyg's collater
collater = Collater(data)

# output (input, output) pairs
def collate_fn(batch):
    batch = collater(batch)
    return batch, batch.y

# Save a fraction of data examples for validation
train_data, val_data = train_test_split(data, test_size=0.2)

print(f"Number of training examples: {len(train_data):,}", flush=True)
print(f"Number of validation examples: {len(val_data):,}", flush=True)


# Save scalers for future analysis
pickle.dump(pt,open(f"{output_dir}/X_scaler_{filename}.pkl","wb"))
np.savez(f"{output_dir}/min_values_{filename}", FeH=FeH_min, MgFe=MgFe_min)
pickle.dump(scaler_params,open(f"{output_dir}/theta_scaler_{filename}.pkl","wb"))
# Save processed Milky Way data
pickle.dump(apogee_ds_processed, open(f"{output_dir}/apogee_ds_processed_{filename}.pkl", "wb"))

pickle.dump(data, open(f"{output_dir}/data.pkl", "wb"))


####################################################################################
####################################################################################
# Training
####################################################################################
####################################################################################

start = time.time()
print("Training NPE model...", flush=True)
posterior_model = sbi_training.NPE_training(train_data=train_data,
                                            val_data=val_data,
                                            collate_fn=collate_fn,
                                            prior_ranges=(scaler_params.transform(np.array([0,6,8,-3])[None,:] )[0],
                                                          scaler_params.transform(np.array([14,11,12,0])[None,:] )[0]),
                                            batch_size=32,
                                            filename=filename,
                                            output_dir=output_dir
                                            )
end = time.time()
print("Model trained in {:.0f} minutes".format((end-start)/60), flush=True)


####################################################################################
####################################################################################
# Validation
####################################################################################
####################################################################################

samples = sbi_training.validation(posterior_ensemble=posterior_model,
                            val_data=val_data,
                            filename=filename,
                            output_dir=output_dir)


    
# Scale back the merger parameters into the original representation
for progID in samples.keys():
    theta_pred, log_p, theta_fid = samples[progID]
    samples[progID] = (scaler_params.inverse_transform(theta_pred),
                        log_p,
                        scaler_params.inverse_transform(theta_fid[None,:])[0])


pickle.dump(samples, 
            open(f'{output_dir}{filename}_test_samples.pkl', 'wb'))
    

filename = "Suite_"+"".join(features)+"".join(parameters)
# Make plot of cross-validated parameters inference
plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']
plot_ranges=[[0.1,13.9],[5.9,10.9],[7.1,11.9],[-3.2,-0.1]]

sbi_results.cross_validation_plot(samples=[samples],
                                  percentile_range=[16,84],
                                  plot_labels=plot_labels,
                                  plot_ranges=plot_ranges,
                                  filename=f'{output_dir}cross_validation_1684_{filename}.png')

# Save table with quantitative results 
sbi_results.rms_table_per_galaxy(samples={"SUITE":samples},
                                 parameters=parameters,
                                 filename=f'{output_dir}rms_table_{filename}.csv')

sbi_results.count_predictions_within_range(samples={"SUITE":samples},
                                           parameters=parameters,
                                           percentile_range=[16,84],
                                           filename=f'{output_dir}range_table_{filename}_1684.csv')

