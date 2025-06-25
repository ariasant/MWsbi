import matplotlib as mpl
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import time

import sys
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import fishnets
import sbi_results
import sbi_training


# Load training and test data



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

# Learn data compression model with fishnet
compression_model = fishnets.FISHNET(n_params=4,
                                     n_d=100,
                                     n_features=len(features),
                                     n_hidden_layers=5,
                                     n_nodes_per_layer=256)

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

# Train NDE model
posterior_model = sbi_training.NPE_training(X_train=summary_stats,
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

