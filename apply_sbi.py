import corner
import numpy as np
import pandas as pd
import pickle
import torch

model_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observation_shifted/"
output_dir = "/mnt/aridata1/users/ariasant/MW-sbi/results_shifted/"

features=["E","L","FeH","MgFe"]

# Posterior model
posterior = pickle.load(open(f"{model_dir}Suite_ELFeHMgFe.pkl","rb"))

# Data processing tools
theta_scaler = pickle.load(open(f"{model_dir}theta_scaler_Suite_ELFeHMgFe.pkl","rb")) # progenitor properties scaler

plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']

# Load pre-processed apogee sample
df = pd.read_pickle(f"{model_dir}apogee_ds_processed_Suite_ELFeHMgFe.pkl")

substructures = ['GES', 'Sagittarius', 'Helmi',
       'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
       'LMS', 'Heracles']

for substructure in substructures:
    
    df_sub = df[df[substructure+"_flag"]==1]
    print(f"N stars in {substructure}: {df_sub.shape[0]:,}", flush=True)
    
    data = df_sub[features].values

    if len(data)>100:
        # Select 10 samples of 100 stars each from data
        # get how many times you can sample from the progenitor
        n_samples = int(len(data)/100)*10
        data_samples = [data[np.random.randint(0,len(data),size=100)].flatten() for i in range(n_samples)]
    else:
        data_samples = [data[np.random.randint(0,len(data),size=100)].flatten()]

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

