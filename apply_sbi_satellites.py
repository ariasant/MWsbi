import corner
import numpy as np
import pandas as pd
import pickle
import torch

model_dir = "/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/"
output_dir = "/mnt/aridata1/users/ariasant/MW-sbi/fishnet_results/"

features=["E","L","FeH","MgFe"]

# Posterior model
posterior = pickle.load(open(f"{model_dir}Suite_ELFeHMgFe.pkl","rb"))

# Data processing tools
chem_min_values = np.load(f"{model_dir}min_values_Suite_ELFeHMgFe.npz")
FeH_min = chem_min_values["FeH"]
MgFe_min = chem_min_values["MgFe"]
X_scaler = pickle.load(open(f"{model_dir}X_scaler_Suite_ELFeHMgFe.pkl","rb")) # star properties scaler
theta_scaler = pickle.load(open(f"{model_dir}theta_scaler_Suite_ELFeHMgFe.pkl","rb")) # progenitor properties scaler

plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']

# Load satellites data
df = pd.read_pickle(f"/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_satellites_ds.pkl")
df = df[(df["E"]<0)&(df["satelliteID"]!="GSE")]

# Preprocess dataframe
df["E"] *= -1
df["FeH"] -= FeH_min
df["MgFe"] -= MgFe_min

df[features] = X_scaler.transform(df[features].values)


for satellite in df["satelliteID"].unique():
    
    df_sub = df[df["satelliteID"]==satellite]
    print(f"N stars in {satellite}: {df_sub.shape[0]:,}", flush=True)
    
    data = df_sub[features].values

    if len(data)>100:
        # Select 10 samples of 100 stars each from data
        # get how many times you can sample from the progenitor
        n_samples = int(len(data)/100)*10
        data_samples = [data[np.random.randint(0,len(data),size=100)].flatten() for i in range(n_samples)]
    
    elif len(data)>25:
        data_samples = [data[np.random.randint(0,len(data),size=100)].flatten()]
    
    else:
        print(f"Not enough data for {satellite}.", flush=True)
        continue

    # Sample the posterior of the progenitor properties as conditioned by each data sample
    posterior_samples = []
    for data_sample in data_samples:

        # Decide how many samples to get from the posterior
        if len(data_samples)>1:
            n_samples = 100
        else:
            n_samples = 1000
        # Get posterior samples
        theta_samples = posterior.sample((n_samples,), 
                                torch.Tensor(data_sample).to(device="cuda"))

        theta_samples = theta_samples.cpu().numpy()
        theta_samples = theta_scaler.inverse_transform(theta_samples)

        posterior_samples.append(theta_samples)

    # Concatenate all posterior samples
    posterior_samples = np.concatenate(posterior_samples, axis=0)
    
    # Save posterior samples
    pickle.dump(posterior_samples, open(f"{output_dir}{satellite}_sat.pkl","wb"))
    
    # Plot posterior samples
    fig = corner.corner(posterior_samples, 
                        bins=20, 
                        labels=plot_labels,
                        quantiles=[.16,.50,.84],
                        plot_contours=False,
                        show_titles=True,
                        title_kwargs={'fontsize':8},
                        verbose=True)
    fig.savefig(f"{output_dir}{satellite}_sat.pdf",dpi=400)
    fig.clf()

