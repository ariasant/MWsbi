from ili.dataloaders import TorchLoader
from ili.utils import Uniform, load_nde_lampe, LampeEnsemble
from ili.inference import LampeRunner
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import numpy as np
import optuna
import os
import pandas as pd
import pickle
import pymc as pm
from sklearn.model_selection import train_test_split
import time
import torch
from torch.utils.data import Dataset

import sys
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import fishnets
import optuna_opt
import sbi_results
import sbi_training


import time
import logging
import pickle
from copy import deepcopy
from tqdm import tqdm
import torch
import lampe
import jax
import jax.numpy as jnp

import corner


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

        # Scale data
        noisy_data_jax = ( noisy_data_jax - jnp.mean(noisy_data_jax) ) / jnp.std(noisy_data_jax)

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

        # Scale data
        noisy_data_jax = ( noisy_data_jax - jnp.mean(noisy_data_jax) ) / jnp.std(noisy_data_jax)

        # Apply JAX compression model
        compressed_data_jax = self.compression_model(noisy_data_jax)[0]

        # Convert JAX array back to PyTorch tensor using DLPack
        self.compressed_data_torch = torch.from_dlpack(compressed_data_jax)

def generate_mean_cov_model(features, 
                            n_stars_per_prog,
                            n_progenitors):

    n_features = len(features)

    coords = {"features": features, 
              "features_bis": features, 
              "star_id": np.arange(n_stars_per_prog),
              "prog_id": np.arange(n_progenitors)}

    with pm.Model(coords=coords) as model:
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", 
            n=n_features, 
            eta=1.0, 
            sd_dist=pm.HalfNormal.dist(sigma=0.1,shape=n_features)
        )
        mu_components = pm.math.stack([pm.Normal("mu_E", mu=0.8, sigma=0.1),
                                       pm.Normal("mu_L", mu=-1., sigma=0.2),
                                       pm.Normal("mu_FeH", mu=-0.4, sigma=0.2),
                                       pm.Normal("mu_MgFe", mu=0.4, sigma=0.02)
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

class my_runner(LampeRunner):

    def __init__(self, 
                 train_noise_list, 
                 val_noise_list,
                 **kwargs):

        super().__init__(**kwargs)
        self.train_noise_list = train_noise_list
        self.val_noise_list = val_noise_list


    def _train_round(self, models,
                     train_loader, val_loader):
        """Train a single round of inference for an ensemble of models."""

        # initialize models
        x_, y_ = next(iter(train_loader))
        models_rnd = [
            model(x_, y_, self.prior).to(self.device)
            for model in models
        ]

        posteriors, summaries = [], []
        for i, model in enumerate(models_rnd):
            logging.info(f"Training model {i+1} / {len(models_rnd)}.")

            # define optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.train_args["learning_rate"]
            )
            stepper = lampe.utils.GDStep(
                optimizer, clip=self.train_args["clip_max_norm"])

            # train model
            best_val = float('inf')
            wait = 0
            summary = {'training_log_probs': [], 'validation_log_probs': []}
            with tqdm(iter(range(self.train_args["max_epochs"])),
                      unit=' epochs') as tq:
                for epoch in tq:
                    # Update noise for training and val data 
                    n_noise_realisations = self.train_noise_list.shape[0]
                    idx_noise = epoch%n_noise_realisations

                    train_loader.dataset.update_noise_matrix(torch.from_numpy(self.train_noise_list[idx_noise]).to(device))
                    val_loader.dataset.update_noise_matrix(torch.from_numpy(self.val_noise_list[idx_noise]).to(device))

                    # Define loader again
                    loader = TorchLoader(train_loader=train_loader, val_loader=val_loader)
                    train_loader, val_loader = self._prepare_loader(loader)

                    loss_train, loss_val = self._train_epoch(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        stepper=stepper,
                    )
                    tq.set_postfix(
                        loss=loss_train,
                        loss_val=loss_val,
                    )
                    summary['training_log_probs'].append(-loss_train)
                    summary['validation_log_probs'].append(-loss_val)

                    # check for convergence
                    if loss_val < best_val:
                        best_val = loss_val
                        best_model = deepcopy(model.state_dict())
                        wait = 0
                    elif wait > self.train_args["stop_after_epochs"]:
                        break
                    else:
                        wait += 1
                else:
                    logging.warning(
                        "Training did not converge in "
                        f"{self.train_args['max_epochs']} epochs.")
                summary['best_validation_log_prob'] = -best_val
                summary['epochs_trained'] = epoch

            # save model
            model.load_state_dict(best_model)
            posteriors.append(model)
            summaries.append(summary)

        # ensemble all trained models, weighted by validation loss
        val_logprob = torch.tensor(
            [float(x["best_validation_log_prob"]) for x in summaries]
        ).to(self.device)
        # Exponentiate with numerical stability
        weights = torch.exp(val_logprob - val_logprob.max())
        weights /= weights.sum()

        posterior_ensemble = LampeEnsemble(posteriors, weights)

        # record the name of the ensemble
        posterior_ensemble.name = self.name
        posterior_ensemble.signatures = self.signatures

        return posterior_ensemble, summaries


# Load training and test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set numpy random key
rng = np.random.default_rng(17)


# Trainning hyperparameters
batch_size = 256
lr = 1e-4
n_epochs = 1000


# Initialise datasets
features = ["E","L","FeH","MgFe"]
parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']
filename = f"Suite_"+"".join(features)

output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/shifts_marg/'

sim_data = pd.read_pickle(f"{output_dir}/data/sim_ds_processed_{filename}.pkl")
obs_data = pd.read_pickle(f"{output_dir}/data/apogee_ds_processed_{filename}.pkl")
scaler_params = pickle.load(open(f"{output_dir}/data/theta_scaler_Suite_ELFeHMgFe.pkl","rb")) 

print(f"N progID: {len(sim_data['progID'].unique())}", flush=True)


data_file = f"{output_dir}/data/training_data.npz"

if os.path.exists(data_file):
    print("Loading pre-saved training data...", flush=True)
    data = np.load(data_file)
    X_train = data['X_train']
    Y_train = data['Y_train']
else:
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
    print("Saving training data for future use...", flush=True)
    np.savez(data_file, X_train=X_train, Y_train=Y_train)


# Split the data into training and test(validation) sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1)

test_dictionary = {"X": X_test,
                   "Y": Y_test,
                   "ID": [f"{i:05}" for i in range(len(Y_test))]}

print(f"X_train shape: {X_train.shape}", flush=True)
print(f"Y_train shape: {Y_train.shape}", flush=True)
print(f"X_test shape: {X_test.shape}", flush=True)
print(f"Y_test shape: {Y_test.shape}", flush=True)

print("Generating noise realisations", flush=True)
print("Generating noise for validation and training data...", flush=True)
 
train_noise_model = generate_mean_cov_model(features=features,
                                            n_stars_per_prog=100,
                                            n_progenitors=X_train.shape[0])
val_noise_model = generate_mean_cov_model(features=features,
                                          n_stars_per_prog=100,
                                          n_progenitors=X_test.shape[0])

train_noise_epochs = sample_noise_training(model=train_noise_model,
                                            n_epochs=min(n_epochs,200),
                                            random_seed=16)
val_noise_epochs = sample_noise_training(model=val_noise_model,
                                            n_epochs=min(n_epochs,200),
                                            random_seed=17)

# Visualize noisy data
obs_accreted = ((obs_data.AlFe<-0.07) & (obs_data.MgMn>=0.25)) | \
                   ((obs_data.AlFe>=-0.07) & (obs_data.MgMn>=4.25*obs_data.AlFe+0.5475))
noise_data = X_train+train_noise_epochs[0]
train_df = pd.DataFrame(data=noise_data.reshape(-1,4), columns=features)
fig = plot_stars_data([train_df, obs_data, obs_data[obs_accreted]])
fig.savefig(f"{output_dir}train_data_with_noise_{filename}.pdf", dpi=300, bbox_inches='tight')


### now for NPE
#train_noise_list = sample_noise_training(model=train_noise_model,
#                                         n_epochs=300,
#                                        random_seed=23)
#val_noise_list = sample_noise_training(model=val_noise_model,
#                                       n_epochs=300,
#                                       random_seed=24)
train_noise_list, val_noise_list = train_noise_epochs, val_noise_epochs

prior = Uniform(low=scaler_params.transform(np.array([0,6,8,-3])[None,:] )[0],
                high=scaler_params.transform(np.array([14,11,12,0])[None,:] )[0],
                device=device)

# Default normalising flows architectures
nets = [load_nde_lampe(model="nsf",
                       num_transforms=5,
                       hidden_features=50,
                       x_normalize=False,
                       theta_normalize=False,
                       device=device)]

runner = my_runner(train_noise_list=train_noise_list,
                   val_noise_list=val_noise_list,
                   prior=prior,
                   nets=nets,
                   device=device)

# Default compression model
compression_model = fishnets.FISHNET(n_params=4,
                                     n_d=100,
                                     n_features=len(features),
                                     n_hidden_layers=4,
                                     n_nodes_per_layer=100
)

# Use fixed train/test split defined outside objective
train_ds = NoisyDataset(initial_data=(torch.from_numpy(X_train).float().to(device), torch.from_numpy(Y_train).float().to(device)),
                        noise=torch.from_numpy(train_noise_list[0]).to(device),
                        compression_model=compression_model)
val_ds = NoisyDataset(initial_data=(torch.from_numpy(X_test).float().to(device), torch.from_numpy(Y_test).float().to(device)),
                      noise=torch.from_numpy(val_noise_list[0]).to(device),
                      compression_model=compression_model)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

loader = TorchLoader(train_loader=train_loader, val_loader=val_loader)

####################################################################################
####################################################################################
# Training
####################################################################################
####################################################################################
study_path = f"{output_dir}optuna_study/hyperparameters_search.db"

"""
if os.path.exists(study_path):
    # Load exististing study
    study = optuna.load_study(study_name="ltu_ili_npe_tarp_study",
                              storage=f"sqlite:///{study_path}")
    params = study.best_trials[0].params

else:
    # Run hyperparameter tuning 
    params = optuna_opt.hyperparameter_search(X_train=X_train,
                                              Y_train=Y_train,
                                              X_test=X_test,
                                              Y_test=Y_test,
                                              scaler_params=scaler_params,
                                              study_dir=study_path,
                                              runner=runner,
                                              compression_model=compression_model,
                                              train_noise_epochs=train_noise_epochs,
                                              val_noise_epochs=val_noise_epochs,
                                              train_noise_list=train_noise_list,
                                              val_noise_list=val_noise_list)
    
fishnet_params = {
        "n_hidden_layers": params["hidden_layers_fish"],
        "n_nodes_per_layer": params["nodes_per_layer_fish"]
    }
npe_params = {
    "model": params["model"],
    "hidden_features": params["hidden_features"],
    "num_transforms": params["num_transforms"]
}
"""
fishnet_params = {"n_hidden_layers": 4,
                  "n_nodes_per_layer": 128}

npe_params = {"model": "maf",
              "hidden_features": 128,
              "num_transforms": 8}

# Learn data compression model with fishnet
compression_model = fishnets.FISHNET(n_params=4,
                                     n_d=100,
                                     n_features=len(features),
                                     **fishnet_params)

# Train the compression model
n_epochs = 3000
print("Training compression model...", flush=True)
start = time.time()
training_results = compression_model.train(data_=X_train,
                                           theta_=Y_train,
                                           val_data_=X_test,
                                           val_theta_=Y_test,
                                           train_noise_epochs=train_noise_epochs,
                                           val_noise_epochs=val_noise_epochs,
                                           batch_size=batch_size,
                                           lr=lr,
                                           epochs=n_epochs,
                                           weights_dir=f"{output_dir}/weights/")
end = time.time()
print(f"Compression model trained in {end-start:.2f} seconds", flush=True)

# Load weights from best epoch
best_epoch = np.argmin(training_results["val_losses"])
best_loss = training_results["val_losses"][best_epoch]
print(f"Loading weights from epoch {best_epoch} with val loss {best_loss:.2f}", flush=True)
best_weights = pickle.load(open(f"{output_dir}/weights/epoch_{best_epoch}.pkl","rb"))

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

# Train NPE
train_args = dict(
    training_batch_size=batch_size,
    learning_rate=lr*0.1,
    stop_after_epochs=10000,
    max_epochs=1000
)

# Update NPE model with results of optimisation
nets = [load_nde_lampe(**npe_params,
        x_normalize=False,
        theta_normalize=False,
        device=device,
) for i in range(3)]

# Re-define runner with optimised flows
runner = my_runner(train_noise_list=train_noise_list,
                   val_noise_list=val_noise_list,
                   prior=prior,
                   nets=nets,
                   device=device)

# Update compression model used to transform the data before passing them to the NPE
for data_loader in [train_loader, val_loader]:
    data_loader.dataset.compression_model = compression_model

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

test_dictionary["X"] = np.array(compression_model(jnp.array(X_test+val_noise_list[42]))[0])

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

