import corner
import math
import numpy as np
import pandas as pd
import pickle
import torch

import sys
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import multitask_model as mt

model_dir = "/mnt/aridata1/users/ariasant/MW-sbi/multitask_results/"
output_dir = "/mnt/aridata1/users/ariasant/MW-sbi/multitask_results/"

features=["E","L","FeH","MgFe"]

# Define device where the model will be trained
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Posterior model
posterior =  mt.MultiTask(theta_dim=4, # Dimensions of the probability distribution approximated by the flow
                     n_conditions=len(features)*100, 
                     n_layers_enc = 3,
                     latent_dim_enc = 50,
                     n_transforms = 10,
                     n_layers_per_transform = 2,
                     n_neurons_flow = 50)
# Load model weights
training_results = pickle.load(open(f"{output_dir}training_results_Suite_ELFeHMgFe.pkl","rb"))
# Load weights of the best model (lower validation loss) from training
best_epoch = np.argmin(training_results["val_loss"])
best_weights = torch.load(f"{output_dir}model_weights/model_{best_epoch}.pt", map_location=torch.device(device)) 

posterior.load_state_dict(best_weights)

# Move model to device and set evaluation mode
posterior.to(device)
posterior.eval()


plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']

# Load pre-processed apogee sample
df = pd.read_pickle(f"{model_dir}apogee_ds_processed_Suite_ELFeHMgFe.pkl")

substructures = ['Arjuna','GES', 'Sagittarius', 'Helmi',
       'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
       'LMS', 'Heracles']

for substructure in substructures:
    
    df_sub = df[df[substructure+"_flag"]==1]
    print(f"N stars in {substructure}: {df_sub.shape[0]:,}", flush=True)
    
    data = df_sub[features].values


    # Select 10 samples of 100 stars each from data
    # get how many times you can sample from the progenitor
    n_samples = math.ceil( len(data)/100 ) * 10
    
    data_samples = [data[np.random.randint(0,len(data),size=100)].flatten() for i in range(n_samples)]

    # Sample the posterior of the progenitor properties as conditioned by each data sample
    posterior_samples = []
    prior_ranges = [[0,14],[6,11],[7,12],[-3,0]]
    for data_sample in data_samples:

        # Decide how many samples to get from the posterior
        if len(data_samples)>1:
            n_samples = 1000
        else:
            n_samples = 1000
        # Get posterior samples
        theta_samples = posterior.sample(torch.Tensor(data_sample).to(device=device)[None,:], 
                                         n_samples,
                                         prior_ranges=prior_ranges)

        theta_samples = theta_samples.cpu().numpy()
        #theta_samples = theta_scaler.inverse_transform(theta_samples)

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

