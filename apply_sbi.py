import corner
import math
import numpy as np
import optuna
import pandas as pd
import pickle
import torch

import sys
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import fishnets

model_dir = "/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/coral/"
output_dir = "/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/coral/"

features=["E","L","FeH","MgFe"]

# Posterior model
posterior = pickle.load(open(f"{model_dir}Suite_ELFeHMgFe.pkl","rb"))

# Data processing tools
theta_scaler = pickle.load(open(f"{model_dir}data/theta_scaler_Suite_ELFeHMgFe.pkl","rb")) # progenitor properties scaler

plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']

# Load the hyperaparameters for the compression model
# Load exististing study
study = optuna.load_study(study_name="ltu_ili_npe_tarp_study",
                            storage="sqlite:////mnt/aridata1/users/ariasant/MW-sbi/optuna_study/hyperparameters_search.db")
params = study.best_trials[0].params
fishnet_params = {
        "n_hidden_layers": params["hidden_layers_fish"],
        "n_nodes_per_layer": params["nodes_per_layer_fish"]
    }

# Compression model
compression_model = fishnets.FISHNET(n_params=4,
                                     n_d=100,
                                     n_features=len(features),
                                     **fishnet_params)
# Load trained weights
w = pickle.load(open(f"{output_dir}Suite_ELFeHMgFe_compression_model_w.pkl","rb")) 
compression_model.w = w

# Load pre-processed apogee sample
df = pd.read_pickle(f"{model_dir}data/apogee_ds_processed_Suite_ELFeHMgFe.pkl")

substructures = ['Arjuna', 'GES', 'Sagittarius', 'Helmi',
       'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
       'LMS', 'Heracles', 'Sequoia_ALL']

for substructure in substructures:
    
    if substructure!="Heracles":
        df_sub = df[df[substructure+"_flag"]==1]
        # Remove stars from the GES sample that are too close to the Galaxy centre
        #df_sub = df_sub[(df_sub.GAL_LAT**2>20**2) & (df_sub.GAL_LON**2>20**2)]

    if substructure=="GES":
        df_sub = df[df[substructure+"_flag"]==1]
        # Improve purity of GES sample
        #df_sub = df_sub[df_sub["FeH"]<-0.6]
        print(f"N stars in {substructure}: {df_sub.shape[0]:,}", flush=True)

    elif substructure=="Sagittarius":
        # Consider the stars that match with the selection of Hernquist
        #satellites_data = pd.read_csv("/mnt/aridata1/users/ariasant/MW-sbi/data/member_list_fe_mg.txt")
        #SGR_star_IDs = satellites_data[satellites_data["System"]=="Sgr"]["APOGEE_ID"].values
        #df_sub = df[df["APOGEE_ID"].isin(SGR_star_IDs)]
        df_sub = df[df[substructure+"_flag"]==1]
        print(f"N stars in {substructure}: {df_sub.shape[0]:,}", flush=True)

    elif substructure=="Sequoia_ALL":
        # Consider all the sequoia samples together
        df_sub = df[(df["Sequoia_M19_flag"]==1) | 
                    (df["Sequoia_K19_flag"]==1) | 
                    (df["Sequoia_N20_flag"]==1)]
        print(f"N stars in {substructure}: {df_sub.shape[0]:,}", flush=True)

    
    data = df_sub[features].values

    # Select 10 samples of 100 stars each from data
    # get how many times you can sample from the progenitor
    n_samples = math.ceil(len(data)/100)*10
    data_samples = [data[np.random.randint(0,len(data),size=100)] for i in range(n_samples)]

    # Sample the posterior of the progenitor properties as conditioned by each data sample
    posterior_samples = []
    for data_sample in data_samples:

        # Compress data features
        data_sample, _, __ = compression_model(data_sample)

        # Decide how many samples to get from the posterior
        n_samples = 100

        # Get posterior samples
        theta_samples = posterior.sample((n_samples,), 
                                torch.Tensor(data_sample).to(device="cuda"))

        theta_samples = theta_samples.cpu().numpy()
        theta_samples = theta_scaler.inverse_transform(theta_samples)

        posterior_samples.append(theta_samples)

    # Concatenate all posterior samples
    posterior_samples = np.concatenate(posterior_samples, axis=0)
    
    # Save posterior samples
    pickle.dump(posterior_samples, open(f"{output_dir}{substructure}.pkl","wb"))
    
    # Plot posterior samples
    fig = corner.corner(posterior_samples, 
                        bins=20, 
                        labels=plot_labels,
                        quantiles=[.16,.50,.84],
                        plot_contours=False,
                        show_titles=True,
                        title_kwargs={'fontsize':8},
                        verbose=True)
    fig.savefig(f"{output_dir}{substructure}.pdf",dpi=400)
    fig.clf()

