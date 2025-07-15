from ili.dataloaders import TorchLoader
from ili.utils import Uniform, load_nde_lampe
from ili.inference import InferenceRunner
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import numpy as np
import optuna
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import time
import torch

import sys
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import fishnets
import optuna_opt
import sbi_results
import sbi_training


# Load training and test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


features = ["E","L","FeH","MgFe"]
parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']
filename = f"Suite_"+"".join(features)

output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/coral/'

sim_data = pd.read_pickle(f"{output_dir}/data/sim_ds_processed_{filename}.pkl")
obs_data = pd.read_pickle(f"{output_dir}/data/apogee_ds_processed_{filename}.pkl")
scaler_params = pickle.load(open(f"{output_dir}/data/theta_scaler_Suite_ELFeHMgFe.pkl","rb")) 



print(f"N progID: {len(sim_data['progID'].unique())}", flush=True)

# Create datasets for training 
X_train, Y_train = [], []

for progID in sim_data["progID"].unique():
    # Get the data for the current progenitor
    prog_data = sim_data[sim_data["progID"]==progID]
    if len(prog_data) < 100:
        continue
    # Sample the data n times
    n = min(100, math.ceil(len(prog_data)//100))
    for i in range(n):
        idx_sample = np.random.randint(0, len(prog_data), size=100)

        X_train.append(prog_data[features].values[idx_sample])
        Y_train.append(prog_data[parameters].values[idx_sample][0])

X_train = np.stack(X_train)
Y_train = np.stack(Y_train)


# Split the data into training and test(validation) sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1)
test_dictionary = {"X": X_test,
                   "Y": Y_test,
                   "ID": [f"{i:05}" for i in range(len(Y_test))]}

print(f"X_train shape: {X_train.shape}", flush=True)
print(f"Y_train shape: {Y_train.shape}", flush=True)
print(f"X_test shape: {X_test.shape}", flush=True)
print(f"Y_test shape: {Y_test.shape}", flush=True)

    

####################################################################################
####################################################################################
# Training
####################################################################################
####################################################################################

if os.path.exists("/mnt/aridata1/users/ariasant/MW-sbi/optuna_study/hyperparameters_search.db"):
    # Load exististing study
    study = optuna.load_study(study_name="ltu_ili_npe_tarp_study",
                              storage="sqlite:////mnt/aridata1/users/ariasant/MW-sbi/optuna_study/hyperparameters_search.db")
    params = study.best_trials[0].params

else:
    # Run hyperparameter tuning 
    params = optuna_opt.hyperparameter_search(X_train=X_train,
                                              Y_train=Y_train,
                                              X_test=X_test,
                                              Y_test=Y_test,
                                              scaler_params=scaler_params)
    
fishnet_params = {
        "n_hidden_layers": params["hidden_layers_fish"],
        "n_nodes_per_layer": params["nodes_per_layer_fish"]
    }
npe_params = {
    "model": params["model"],
    "hidden_features": params["hidden_features"],
    "num_transforms": params["num_transforms"]
}

# Learn data compression model with fishnet
compression_model = fishnets.FISHNET(n_params=4,
                                     n_d=100,
                                     n_features=len(features),
                                     **fishnet_params)

# Train the compression model
print("Training compression model...", flush=True)
start = time.time()
training_results = compression_model.train(data_=X_train,
                                           theta_=Y_train,
                                           val_data_=X_test,
                                           val_theta_=Y_test,
                                           batch_size=256,
                                           lr=1e-4,
                                           epochs=3000,
                                           weights_dir="/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/coral/weights/")
end = time.time()
print(f"Compression model trained in {end-start:.2f} seconds", flush=True)

# Load weights from best epoch
best_epoch = np.argmin(training_results["val_losses"])
best_loss = training_results["val_losses"][best_epoch]
print(f"Loading weights from epoch {best_epoch} with val loss {best_loss:.2f}", flush=True)
best_weights = pickle.load(open(f"/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/coral/weights/epoch_{best_epoch}.pkl","rb"))

compression_model.w = best_weights

# Save the compression model weights
pickle.dump(compression_model.w, open(f"{output_dir}{filename}_compression_model_w.pkl", "wb"))

# Plot training 
fig, ax = mpl.pyplot.subplots()
ax.plot(training_results['losses'], label="Training Loss")
ax.plot(training_results['val_losses'], label="Validation Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss (log)")
ax.set_ylim([10, -10])
ax.legend()
fig.savefig(f"{output_dir}{filename}_compression_model_training.pdf", dpi=300, bbox_inches='tight')



# Compress data
print("Compressing data...", flush=True)
summary_stats, _, __ = compression_model(X_train)

summary_stats_test, _, __ = compression_model(X_test)
# Update the test dictionary with the compressed data
test_dictionary["X"] = summary_stats_test



# Define prior
prior = Uniform(low=scaler_params.transform(np.array([0,6,8,-3])[None,:] )[0],
                high=scaler_params.transform(np.array([14,11,12,0])[None,:] )[0],
                device=device)


train_args = dict(
    training_batch_size=256,
    learning_rate=1e-4,
)

# Define NPE model
nets = [load_nde_lampe(**npe_params,
        x_normalize=False,
        device=device,
) for i in range(3)]


runner = InferenceRunner.load(
    backend="lampe",
    engine="NPE",
    prior=prior,
    nets=nets,
    device=device,
    train_args=train_args,
)

# Train NPE
# Use fixed train/test split defined outside objective
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(summary_stats).float(), torch.from_numpy(Y_train).float()),
    batch_size=256, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(summary_stats_test).float(), torch.from_numpy(Y_test).float()),
    batch_size=256, shuffle=False
)
loader = TorchLoader(train_loader=train_loader, val_loader=val_loader)


posterior_model, summaries = runner(loader=loader)

# Plot train/validation loss
fig, ax = plt.subplots(1, 1, figsize=(6,4))
c = list(mcolors.TABLEAU_COLORS)
for i, m in enumerate(summaries):
    ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i])
    ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
ax.set_xlim(0)
ax.set_xlabel('Epoch')
ax.set_ylabel('Log probability')
ax.set_ylim([-10,10])
ax.legend()
fig.savefig(output_dir+f'{filename}_training_plot.png', dpi=400)

# Save posterior
pickle.dump(posterior_model, 
            open(f'{output_dir}{filename}.pkl', 'wb'))


####################################################################################
####################################################################################
# Validation
####################################################################################
####################################################################################


# Sample parameters for test galaxy
samples = sbi_training.validation(posterior_ensemble=posterior_model,
                              test_dictionary=test_dictionary,
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

