import argparse
import corner
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/mnt/aridata1/users/ariasant/auriga-sbi/")
from domain_shift import DataProcessor
import get_results
import training


def plot_stars_data(dfs: list):

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
                                alpha=0.5)
        else:
            corner.corner(df[features].values,
                            color=colors[i],
                            bins=20,
                            plot_contours=True,
                            plot_datapoints=False,
                            fill_contours=True,
                            hist_kwargs={"density": True},
                            alpha=0.5,
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
output_dir = '/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observation_shifted/'

filename = f"Suite_"+"".join(features)

substructures = ['GES', 'Sagittarius', 'Helmi',
       'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
       'LMS', 'Heracles']

###########################################################################################
# Data preparation
###########################################################################################

# Load simulation (source) data
df = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/auriga_sbi_data.pkl")
df.rename(columns={"aFe":"MgFe"}, inplace=True)
# Load Milky Way (target) data
apogee_ds = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")
apogee_ds.dropna(subset=features, inplace=True)


# Plot initial data
fig = plot_stars_data([df, apogee_ds])
fig.savefig(f"{output_dir}initial_data_{filename}.pdf", dpi=300, bbox_inches='tight')

# Initialize the data processor
data_processor = DataProcessor(features=features)  

df = data_processor.process_data_sim(sim_df=df)
apogee_ds_processed = data_processor.process_data_obs(obs_df=apogee_ds)


# Plot data after processing
fig = plot_stars_data([df, apogee_ds_processed])
fig.savefig(f"{output_dir}transformed_data_{filename}_shifted_rm_outliers.pdf", dpi=300, bbox_inches='tight')

## Print the number of stars in each substructure of the MW before and after removing outliers
print("Counting stars in each substructure of the MW before and after removing outliers:", flush=True)
for substructure in substructures:
    n_before = sum(apogee_ds[f"{substructure}_flag"]==1)
    n_after = sum(apogee_ds_processed[f"{substructure}_flag"]==1)
    print("="*50, flush=True)
    print(f"{substructure}: {n_before} -> {n_after}", flush=True)
    print("="*50, flush=True)

    # Plot the stars in each substructure
    fig = plot_stars_data([df, apogee_ds_processed[apogee_ds_processed[f"{substructure}_flag"]==1]])
    fig.savefig(f"{output_dir}transformed_data_{filename}_shifted_{substructure}.pdf", dpi=300, bbox_inches='tight')


# Plot merger parameters
fig = corner.corner(df[parameters].values,
                    color='k',
                    labels=parameters,
                    bins=20,
                    plot_contours=False,
                    plot_datapoints=False,
                    fill_contours=False,
                    hist_kwargs={"density": True})
fig.savefig(f"{output_dir}merger_parameters_{filename}.pdf", dpi=300, bbox_inches='tight')


# Initialize the scaler for the merger parameters
scaler_params = RobustScaler()
# Scale the merger parameters
df[parameters] = scaler_params.fit_transform(df[parameters].values)

# Create datasets for training 
X_train, Y_train = [], []
n = 100 # number of samples per progenitor

for progID in df["progID"].unique():
    # Get the data for the current progenitor
    prog_data = df[df["progID"]==progID]
    if len(prog_data) < 100:
        continue
    # Sample the data n times
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
    

# Save scaler for future analysis
pickle.dump(scaler_params,open(f"{output_dir}/theta_scaler_{filename}.pkl","wb"))
pickle.dump(data_processor,open(f"{output_dir}/DataProcessor_{filename}.pkl","wb"))
# Save processed Milky Way data
pickle.dump(apogee_ds_processed, open(f"{output_dir}/apogee_ds_processed_{filename}.pkl", "wb"))



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


"""# Sample parameters for test galaxy
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
                                           filename=f'{output_dir}range_table_{filename}_3468.csv')"""



