import math
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/mnt/aridata1/users/ariasant/auriga-sbi/")
import get_results
import training



###########################################################################################
# Data preparation
###########################################################################################

# Load training and test data
features = ["E","L","FeH","MgFe"]
parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']
filename = f"Suite_"+"".join(features)

output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/coral/'

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

        X_train.append(prog_data[features].values[idx_sample].reshape(-1))
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

# Train NDE model
posterior_model = training.NPE_training(X_train=X_train,
                                        Y_train=Y_train,
                                        prior_ranges=[scaler_params.transform(np.array([0,6,8,-3])[None,:] )[0],
                                                      scaler_params.transform(np.array([14,11,12,0])[None,:] )[0]],
                                        filename=filename,
                                        output_dir=output_dir)


####################################################################################
####################################################################################
# Validation
####################################################################################
####################################################################################


# Sample parameters for test galaxy
samples = training.validation(posterior_ensemble=posterior_model,
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

get_results.cross_validation_plot(samples=[samples],
                                  percentile_range=[16,84],
                                  plot_labels=plot_labels,
                                  plot_ranges=plot_ranges,
                                  filename=f'{output_dir}cross_validation_1684_{filename}.png')

# Save table with quantitative results 
get_results.rms_table_per_galaxy(samples={"SUITE":samples},
                                 parameters=parameters,
                                 filename=f'{output_dir}rms_table_{filename}.csv')

get_results.count_predictions_within_range(samples={"SUITE":samples},
                                           parameters=parameters,
                                           percentile_range=[16,84],
                                           filename=f'{output_dir}range_table_{filename}_1684.csv')

