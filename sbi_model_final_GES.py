import corner
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler

def call_plotting_formatting():

    font = {'family' : 'sans-serif',
    'weight' : 'medium',
    'size'   : 15,
    'variant' : 'normal',
    'style' : 'normal',
    'stretch' : 'normal',
    }

    xtick = {'top' : True,
            'bottom' : True,
            'major.size' : 7,
            'minor.size' : 4,
            'major.width' : 0.5,
            'minor.width' : 0.35,
            'direction' : 'in',
            'minor.visible' : True,
            'color' : 'black',
            'labelcolor' : 'black'
            }

    ytick = {'left' : True,
            'right' : True,
            'major.size' : 7,
            'minor.size' : 4,
            'major.width' : 0.5,
            'minor.width' : 0.35,
            'direction' : 'in',
            'minor.visible' : True,
            'color' : 'black',
            'labelcolor' : 'black'
            }

    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['figure.figsize'] = (6.973848069738481, 4.310075139476229)
    mpl.rcParams['figure.subplot.hspace'] = 0.01

    mpl.rc('font', **font)
    mpl.rc('xtick', **xtick)
    mpl.rc('ytick', **ytick)
    mpl.rcParams['legend.fontsize'] = 18
    mpl.rcParams["font.sans-serif"] = ["DejaVu Serif"]
    mpl.rcParams['mathtext.fontset']='dejavuserif'
    mpl.rcParams["text.usetex"] = False

call_plotting_formatting()

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
                                labels=["$\log(E)$", "$\log(L)$", "[Fe/H]", "[Mg/Fe]"],
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


features = ['E', 'L', 'FeH', 'MgFe']
parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']

output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/trials4/'

print(f"output_dir: {output_dir}", flush=True)

filename = f"Suite_"+"".join(features)

substructures = ['Arjuna','GES', 'Sagittarius', 'Helmi',
                 'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
                 'LMS', 'Heracles']

###########################################################################################
# Data preparation
###########################################################################################

# Load Milky Way (target) data
obs_data = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds_with_errors.pkl")
obs_data.dropna(subset=features, inplace=True)
obs_data = obs_data[(obs_data["E"]<0)&(obs_data["L"]>0)]
# Select accreted stars
obs_accreted = ((obs_data.AlFe<-0.07) & (obs_data.MgMn>=0.25)) | \
               ((obs_data.AlFe>=-0.07) & (obs_data.MgMn>=4.25*obs_data.AlFe+0.5475))
obs_accreted = np.logical_or.reduce([obs_accreted]+[obs_data[f"{substructure}_flag"]==1 
                                    for substructure in substructures])


# Load simulation (source) data
data_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/data/with_satellites/"
sim_data = []

for file in os.listdir(data_dir):

    df = pd.read_pickle(f"{data_dir}{file}")
    df.rename(columns={"aFe":"MgFe"}, inplace=True)

    # Get rid of stars with numerical issues
    df = df[(df["E"]<0) & (df["L"]>0) & 
            (df["MgFe"]<0.5) & (df["MgFe"]>-0.5) &
            (df["FeH"]<1) & (df["FeH"]>-3)]

    sim_data.append(df)

sim_data = pd.concat(sim_data, ignore_index=True)


# Plot initial data
fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]])
fig.savefig(f"{output_dir}initial_data_{filename}.pdf", dpi=300, bbox_inches='tight')


# Preprocess data
sim_data["E"] = np.log(-sim_data["E"].values)
sim_data["L"] = np.log(sim_data["L"].values)

obs_data["E_ERR"] /= -obs_data["E"]
obs_data["E_astronn_ERR"] /= -obs_data["E_astronn"]
obs_data["E"] = np.log(-obs_data["E"].values)
obs_data["L_ERR"] /= obs_data["L"]
obs_data["L"] = np.log(obs_data["L"].values)

obs_noise = obs_data[["E_ERR","L_ERR","FeH_ERR","MgFe_ERR"]].values

# Plot data after processing
fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]])
# Add legend
labels = ["Auriga", "APOGEE", "APOGEE (accreted)"]
colors = [mpl.cm.tab10(i/3) for i in range(3)]
mpl.pyplot.legend(
        handles=[
            mpl.lines.Line2D([], [], 
                             linewidth=5,
                             color=colors[i], 
                             label=labels[i])
            for i in range(3)
        ],
        fontsize=20, frameon=False,
        bbox_to_anchor=(1, 4), loc="upper right"
        )
fig.savefig(f"{output_dir}transformed_data_{filename}.pdf", dpi=300, bbox_inches='tight')


# Plot merger parameters
fig = corner.corner(sim_data[parameters].values,
                    color='k',
                    labels=parameters,
                    bins=20,
                    plot_contours=False,
                    plot_datapoints=False,
                    fill_contours=False,
                    hist_kwargs={"density": True})
fig.savefig(f"{output_dir}merger_parameters_{filename}.pdf", dpi=300, bbox_inches='tight')


# Save processed simulation data
sim_data.to_pickle(f"{output_dir}/data/sim_ds_processed_{filename}.pkl")

# Save processed Milky Way data
obs_data.to_pickle(f"{output_dir}/data/apogee_ds_processed_{filename}.pkl")


#######################################
#######################################
#######################################
#######################################

from ili.dataloaders import TorchLoader
from ili.utils import Uniform, load_nde_lampe
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

import sys
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import fishnets
import optuna_opt
import sbi_results
import sbi_training


import time
import pickle
import torch
import jax
import jax.numpy as jnp

import corner


# Load training and test data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set numpy random key
rng = np.random.default_rng(17)


# Trainning hyperparameters
batch_size = 256
lr = 1e-4
n_epochs = 1000

prog_IDs = sim_data['progID'].unique()

print(f"N progID: {len(prog_IDs)}", flush=True)

# Split progenitors into training and validation sets
training_IDs = np.random.choice(prog_IDs, size=int(len(prog_IDs)*0.8))
test_IDs = np.array([ID for ID in prog_IDs if ID not in training_IDs])

data_file = f"{output_dir}data/training_data.npz"


def generate_mean_cov_model(features, 
                            n_stars_per_prog):

    n_features = len(features)

    coords = {"features": features, 
              "features_bis": features, 
              "star_id": np.arange(n_stars_per_prog)}

    with pm.Model(coords=coords) as model:
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol", 
            n=n_features, 
            eta=1.0, 
            sd_dist=pm.Exponential.dist(25, shape=n_features)
        )
        mu_components = pm.math.stack([pm.Normal("mu_E", sigma=0.1),
                                       pm.Normal("mu_L", sigma=0.1),
                                       pm.Normal("mu_FeH", mu=-0.20, sigma=0.08),
                                       pm.Normal("mu_MgFe", mu=0.42, sigma=0.02)
                                       ])
        mu = pm.Deterministic("shifts", mu_components, dims="features")

        noise_stars_in_single_prog = pm.MvNormal("noise", 
                                                  mu, 
                                                  chol=chol, 
                                                  dims=("star_id", "features"))
        
        return model

def sample_noise_training(model, 
                          n_samples,
                          random_seed):


    with model:
        prior_samples = pm.sample_prior_predictive(samples=n_samples,
                                                   random_seed=random_seed)
        noise_matrix = prior_samples.prior["noise"].values[0]
        
    return noise_matrix

print("Generating noise realisations", flush=True)
print("Generating noise for validation and training data...", flush=True)
 
noise_model = generate_mean_cov_model(features=features,
                                      n_stars_per_prog=100)
noise_list = sample_noise_training(model=noise_model,
                                   n_samples=10000,
                                   random_seed=16)

if os.path.exists(data_file):
    print("Loading pre-saved training data...", flush=True)
    data = np.load(data_file)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

else:
    # Create datasets for training 
    X_train, Y_train = [], []
    X_test, Y_test = [], [] 

    for progID in prog_IDs:
        # Get the data for the current progenitor
        prog_data = sim_data[sim_data["progID"]==progID]
        if len(prog_data) < 100:
            continue
        # Sample the data n times
        n = min(10, math.ceil(len(prog_data)//100))
        for i in range(n):
            idx_sample = np.random.randint(0, len(prog_data), size=100)
            u = np.random.uniform()

            if u>0.2: #progID in training_IDs:
                X_train.append(prog_data[features].values[idx_sample])
                Y_train.append(prog_data[parameters].values[idx_sample][0])

            else:
                
                x = prog_data[features].values[idx_sample]

                # Draw calibration noise at random
                noise_prog = noise_list[np.random.randint(len(noise_list))]
                x += noise_prog

                # Add observational uncertainties
                obs_err = obs_noise[np.random.randint(0,obs_noise.shape[0], size=100)]
                x += obs_err

                # Save to lists
                X_test.append(x)
                Y_test.append(prog_data[parameters].values[idx_sample][0])
        

    X_train = np.stack(X_train)
    Y_train = np.stack(Y_train)
    X_test = np.stack(X_test)
    Y_test = np.stack(Y_test)

    print("Saving training data for future use...", flush=True)
    np.savez(data_file, 
             X_train=X_train, Y_train=Y_train,
             X_test=X_test, Y_test=Y_test)


test_dictionary = {"X": X_test,
                   "Y": Y_test,
                   "ID": [f"{i:05}" for i in range(len(Y_test))]}

print(f"X_train shape: {X_train.shape}", flush=True)
print(f"Y_train shape: {Y_train.shape}", flush=True)
print(f"X_test shape: {X_test.shape}", flush=True)
print(f"Y_test shape: {Y_test.shape}", flush=True)

#####################################################
#####################################################
#####################################################
#####################################################



data_scaler = RobustScaler() 
data_scaler.fit(obs_data[features].values)

# Visualize noisy data
noise_data = np.vstack([X_train+noise_list[np.random.randint(0, noise_list.shape[0], size=X_train.shape[0])] 
                        for i in range(100)])

train_df = pd.DataFrame(data=noise_data.reshape(-1, len(features)), columns=features)

fig = plot_stars_data([train_df, obs_data, obs_data[obs_accreted]])
# Add legend
labels = ["Auriga", "MW", "MW (accreted)"]
colors = [mpl.cm.tab10(i/3) for i in range(3)]
mpl.pyplot.legend(
        handles=[
            mpl.lines.Line2D([], [], 
                             linewidth=5,
                             color=colors[i], 
                             label=labels[i])
            for i in range(3)
        ],
        fontsize=20, frameon=False,
        bbox_to_anchor=(1, 4), loc="upper right"
        )
fig.savefig(f"{output_dir}train_data_with_noise_{filename}.pdf", dpi=300, bbox_inches='tight')

############################################################
############################################################
############################################################
############################################################

prior = Uniform(low=[0,6,8,-3],
                high=[14,11,12,0],
                device=device)


####################################################################################
####################################################################################
# Training
####################################################################################
####################################################################################

if not os.path.exists(f"{output_dir}Suite_ELFeHMgFe_compression_model_w.pkl"):
    fishnet_params = {"n_hidden_layers": 10,
                      "n_nodes_per_layer": 128}

    # Learn data compression model with fishnet
    compression_model = fishnets.FISHNET(n_params=4,
                                         n_d=100,
                                         n_features=len(features),
                                         **fishnet_params)

    # Train the compression model
    n_epochs = 10000
    print("Training compression model...", flush=True)
    start = time.time()
    training_results = compression_model.train(data_=X_train,
                                               theta_=Y_train,
                                               val_data_=X_test,
                                               val_theta_=Y_test,
                                               noise_list=noise_list,
                                               obs_noise_list=obs_noise,
                                               data_scaler=data_scaler,
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

else:

    fishnet_params = {"n_hidden_layers": 10,
                    "n_nodes_per_layer": 128}

    # Learn data compression model with fishnet
    compression_model = fishnets.FISHNET(n_params=4,
                                        n_d=100,
                                        n_features=len(features),
                                        **fishnet_params)
    
    best_weights = pickle.load(open(f"{output_dir}/weights/epoch_4999.pkl","rb"))
    compression_model.w = best_weights
    

##########################################################################################################
###############################################################
###############################################################
###############################################################


# Get n permutations of the noise realisations
n_permutations = 10
X_train_MAF = []
for n in range(n_permutations):
    # Add random calibration noise
    x = X_train + noise_list[np.random.randint(0, noise_list.shape[0], size=X_train.shape[0])]
    # Add observational uncertainties
    x_flat = x.reshape(-1, len(features))
    x_flat += obs_noise[np.random.randint(0, obs_noise.shape[0], size=x_flat.shape[0])]
    # Scale and reshape
    x = data_scaler.transform(x_flat).reshape(-1, 100, len(features))
    X_train_MAF.append(x)
    

train_ds = np.vstack(X_train_MAF)
train_ds = torch.from_numpy(train_ds).float().to(device)
train_ds = jax.dlpack.from_dlpack(train_ds, copy=False)
train_ds = compression_model(train_ds)[0]
train_ds = torch.from_dlpack(train_ds).float().to(device)

train_ds_labels = torch.from_numpy(np.vstack([Y_train for i in range(n_permutations)])).float().to(device)

val_ds = X_test
val_ds = data_scaler.transform(val_ds.reshape(-1, len(features))).reshape(-1, 100, len(features))
val_ds = torch.from_numpy(val_ds).float().to(device)
val_ds = jax.dlpack.from_dlpack(val_ds, copy=False)
val_ds = compression_model(val_ds)[0]
val_ds = torch.from_dlpack(val_ds).float().to(device)

val_ds_labels = torch.from_numpy(Y_test).float().to(device)

# Use fixed train/test split defined outside objective
train_loader = torch.utils.data.DataLoader((train_ds, train_ds_labels), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader((val_ds,val_ds_labels), batch_size=batch_size, shuffle=False)

loader = TorchLoader(train_loader=train_loader, val_loader=val_loader)


##########
# Use OPTUNA to find the best set of MAF parameters

from optuna_opt import hyperparameter_search

npe_params = hyperparameter_search(loader=loader,
                                   prior=prior,
                                   study_dir=f"{output_dir}optuna/",
                                   X_test=X_test,
                                   Y_test=Y_test)


from ili.inference import InferenceRunner


# Train NPE
train_args = dict(
    training_batch_size=10000,
    learning_rate=lr,
    stop_after_epochs=10000,
    max_epochs=5000
)

# Update NPE model with results of optimisation
nets = [load_nde_lampe(**npe_params,
                       model="maf",
                       x_normalize=True,
                       theta_normalize=True,
                       device=device,
) for i in range(3)]

# Re-define runner with optimised flows
runner = InferenceRunner.load(
        backend="lampe",
        engine="NPE",
        prior=prior,
        nets=nets,
        device=device,
        train_args=train_args,
    )

posterior_model, summaries = runner(loader=loader)

# Plot train/validation loss
fig, ax = plt.subplots(1, 1, figsize=(6,4))
c = list(mcolors.TABLEAU_COLORS)
for i, m in enumerate(summaries):
    ax.plot(m['training_log_probs'], ls='-', label=f"{i}_train", c=c[i], alpha=0.5)
    ax.plot(m['validation_log_probs'], ls='--', label=f"{i}_val", c=c[i])
ax.set_xlim(0)
ax.set_xlabel('Epoch')
ax.set_ylabel('Log probability')
ax.set_ylim([-10,5])
ax.legend(fontsize=8)
fig.savefig(output_dir+f'{filename}_training_plot.png', dpi=400)

# Save posterior
pickle.dump(posterior_model, 
            open(f'{output_dir}{filename}.pkl', 'wb'))


################################################################
################################################################
################################################################
################################################################

####################################################################################
####################################################################################
# Validation
####################################################################################
####################################################################################

test_dictionary["X"] = val_loader.dataset[0].cpu().numpy()


# Sample parameters for test galaxy
samples = sbi_training.validation(posterior_ensemble=posterior_model,
                                  test_dictionary=test_dictionary,
                                  filename=filename,
                                  output_dir=output_dir)


pickle.dump(samples, 
            open(f'{output_dir}{filename}_test_samples.pkl', 'wb'))
    

filename = "Suite_"+"".join(features)+"".join(parameters)
# Make plot of cross-validated parameters inference
plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']
plot_ranges=[[0.1,13.9],[5.9,10.9],[8.5,11.9],[-3.2,-0.1]]

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


############################################################
############################################################
############################################################
############################################################

df = obs_data.copy()

substructures = ['Arjuna', 'GES', 'Sagittarius', 'Helmi',
       'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
       'LMS', 'Heracles']

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

        # Scale data
        data_sample = data_scaler.transform(data_sample)

        # Compress data features
        data_sample, _, __ = compression_model(data_sample)

        # Decide how many samples to get from the posterior
        n_samples = 100

        # Get posterior samples
        theta_samples = posterior_model.sample((n_samples,), 
                                torch.Tensor(data_sample).to(device="cuda"))

        theta_samples = theta_samples.cpu().numpy()

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
    fig.suptitle(f"Posterior samples for {substructure}", fontsize=16)
    fig.savefig(f"{output_dir}{substructure}.pdf",dpi=400)


