import corner
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import RandomOverSampler


def oversample_data(X,Y):
    
    x_idx = np.arange(len(X))
    class_x = np.zeros_like(x_idx)
    bin_edges = [[6,8], [8,9], [9,11]] # stellar mass   
    masks = [((Y[:,1]>be[0])&(Y[:,1]<be[1])) for be in bin_edges]

    for i,mask in enumerate(masks):
        class_x[mask] = i
        
    for c in set(class_x):
        f = len(class_x[class_x==c]) / len(class_x)
        print(f"Class {c}: {f:.2f}")

    print("="*30)
    ros = RandomOverSampler(random_state=42)
    
    x_idx_new, new_class_x = ros.fit_resample(x_idx[:,None], class_x)
    #
    for c in set(new_class_x):
        f = len(new_class_x[new_class_x==c]) / len(new_class_x)
        print(f"Class {c}: {f:.2f}")
        
    return X[x_idx_new][:,0], Y[x_idx_new][:,0]

def shuffle_axis1_independently(array):
    """
    Shuffles the order of elements along axis 1 (the 100 dimension)
    independently for each slice along axis 0 (the N dimension).
    """
    N, D2, D3 = array.shape
    
    # Create an array of random permutations for the 100 positions.
    # The shape will be (N, D2).
    # Each row will contain a unique shuffle of [0, 1, ..., 99].
    # We use np.arange(D2) to get the base indices [0, 1, ..., 99].
    permutations = np.stack([
        np.random.permutation(D2) for _ in range(N)
    ])
    
    # Now, we use advanced indexing to apply these permutations.
    # 1. np.arange(N) creates the row indices [0, 1, ..., N-1].
    # 2. We expand it to (N, 1) and broadcast it to (N, D2).
    # 3. This gives us the index pairs (i, permutations[i, j]) for the new array.
    
    # We need a meshgrid to match the row index (Ni) to its corresponding permutation.
    # row_indices: [[0, 0, ..., 0], [1, 1, ..., 1], ...] (Shape N x 100)
    row_indices = np.arange(N)[:, np.newaxis]
    
    # Apply the advanced indexing:
    # array[row_indices, permutations, :]
    shuffled_array = array[row_indices, permutations, :]
    
    return shuffled_array

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

def plot_stars_data(dfs: list, features: list[str], RANGE=None):

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
                                labels=["$\log(E)$", "$\log(L)$", "[Fe/H]", "[Mg/Fe]", "E_ERR", "L_ERR", "FeH_ERR", "MgFe_ERR"],
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

output_dir = '/mnt/aridata1/users/ariasant/MW-sbi/trials5/'

print(f"output_dir: {output_dir}", flush=True)

filename = f"Suite_"+"".join(features)

substructures = ['Arjuna','GES', 'Sagittarius', 'Helmi',
                 'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
                 'LMS', 'Heracles', 'LMC', 'SMC']

####################################################################################
####################################################################################
# Load data from the Auiga simulations and the GAIA and APOGEE surveys
####################################################################################
####################################################################################

# Load Milky Way (target) data
obs_data = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")

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
fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]], features=features)
fig.savefig(f"{output_dir}initial_data_{filename}.pdf", dpi=300, bbox_inches='tight')


# Preprocess data
sim_data["E"] = np.log(-sim_data["E"].values)
sim_data["L"] = np.log(sim_data["L"].values)

obs_data["E"] = np.log(-obs_data["E"].values)
obs_data["L"] = np.log(obs_data["L"].values)


# Plot data after processing
fig = plot_stars_data([sim_data, obs_data, obs_data[obs_accreted]], features=features)
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

data_scaler = RobustScaler() 
data_scaler.fit(obs_data[features].values)


# Save processed simulation data
sim_data.to_pickle(f"{output_dir}/data/sim_ds_processed_{filename}.pkl")

# Save processed Milky Way data
obs_data.to_pickle(f"{output_dir}/data/apogee_ds_processed_{filename}.pkl")


####################################################################################
####################################################################################
# Generate training and validation data for SBI 
####################################################################################
####################################################################################

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
import time
import torch

import sys
sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi/")
import fishnets
import optuna_opt
import sbi_results
import sbi_training
from scipy.ndimage import gaussian_filter1d
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
batch_size = 1024 
lr = 1e-4
n_epochs = 1000

prog_IDs = sim_data['progID'].unique()

print(f"N progID: {len(prog_IDs)}", flush=True)

# Split progenitors into training and validation sets
training_IDs = rng.choice(prog_IDs, size=int(len(prog_IDs)*0.8))
test_IDs = np.array([ID for ID in prog_IDs if ID not in training_IDs])

data_file = f"{output_dir}data/training_data.npz"

# Generate noise realisation to mimic Milky Way observations

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
            eta=1, 
            sd_dist=pm.Exponential.dist(100, shape=n_features)
        )
        mu_components = pm.math.stack([pm.Uniform("mu_E", lower=-0.1, upper=0.1),
                                       pm.Uniform("mu_L", lower=-0.1, upper=0.1),
                                       pm.Uniform("mu_FeH", lower=-0.5, upper=0.0),
                                       pm.Uniform("mu_MgFe", lower=0.2, upper=0.6) 
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


if os.path.exists(data_file):
    print("Loading existing training and validation data...", flush=True)
    data = np.load(data_file)
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    noise_list = data["noise_list"]

    print(f"X_train shape: {X_train.shape}", flush=True)
    print(f"Y_train shape: {Y_train.shape}", flush=True)
    print(f"X_test shape: {X_test.shape}", flush=True)
    print(f"Y_test shape: {Y_test.shape}", flush=True)
    
else:

    # Generate noise realisations for training and validation data
    print("Generating noise for validation and training data...", flush=True)
    
    noise_model = generate_mean_cov_model(features=features,
                                          n_stars_per_prog=100)
    noise_list = sample_noise_training(model=noise_model,
                                       n_samples=10000,
                                       random_seed=16)

    print("Splitting merger-stars pairs in Auriga into training and validation sets...", flush=True)
    # Create datasets for training 
    X_train, Y_train = [], []
    X_test, Y_test = [], [] 

    for progID in prog_IDs:
        # Get the data for the current progenitor
        prog_data = sim_data[sim_data["progID"]==progID]
        if len(prog_data) < 100:
            continue
        # Sample the data n times (maximum 20 and minimum 10 times)
        n = max(min(100, math.ceil(len(prog_data)//100)),10)
        for i in range(n):
            idx_sample = rng.choice(np.arange(len(prog_data)), size=100, replace=False)
            u = rng.uniform()

            if u>0.2: #progID in training_IDs:
                X_train.append(prog_data[features].values[idx_sample])
                Y_train.append(prog_data[parameters].values[idx_sample][0])

            else:
                X_test.append(prog_data[features].values[idx_sample])
                Y_test.append(prog_data[parameters].values[idx_sample][0])
        
    X_train = np.stack(X_train)
    Y_train = np.stack(Y_train)
    X_test = np.stack(X_test)
    Y_test = np.stack(Y_test)

    # Oversample training data
    X_train, Y_train = oversample_data(X_train, Y_train)

    # Shuffle training examples
    X_train = shuffle_axis1_independently(X_train)


    print("Saving training data for future use...", flush=True)
    np.savez(data_file, 
             X_train=X_train, Y_train=Y_train,
             X_test=X_test, Y_test=Y_test,
             noise_list=noise_list)


    print(f"X_train shape: {X_train.shape}", flush=True)
    print(f"Y_train shape: {Y_train.shape}", flush=True)
    print(f"X_test shape: {X_test.shape}", flush=True)
    print(f"Y_test shape: {Y_test.shape}", flush=True)

#####################################################


# Visualize noisy data
x_plot = []
for i in range(10):
    noise = noise_list[np.random.randint(0, noise_list.shape[0], size=X_train.shape[0])]
    x_plot.append(X_train + noise)
    
x_plot = np.vstack(x_plot)

train_df = pd.DataFrame(data=x_plot.reshape(-1, len(features)), columns=features)

fig = plot_stars_data([train_df, obs_data, obs_data[obs_accreted]], features=features)
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

# Plot merger parameters after oversampling
new_sim_data = pd.DataFrame(data=Y_train, columns=parameters)
fig = corner.corner(new_sim_data[parameters].values,
                    color='k',
                    labels=parameters,
                    bins=20,
                    plot_contours=False,
                    plot_datapoints=False,
                    fill_contours=False,
                    hist_kwargs={"density": True})
fig.savefig(f"{output_dir}oversampled_merger_parameters_{filename}.pdf", dpi=300, bbox_inches='tight')


####################################################################################
####################################################################################
# Initialise and train the compression and NPE models
####################################################################################
####################################################################################

if not os.path.exists(f"{output_dir}/optuna/fishnets_study.db"):
    
    # Run hyperparameter search
    """print("Starting hyperparameter search for the fishnets model", flush=True)
    fishnet_params = optuna_opt.hyperparameter_search_fishnets(X_train=X_train,
                                                               Y_train=Y_train,
                                                               X_test=X_test,
                                                               Y_test=Y_test,
                                                               data_scaler=data_scaler,
                                                               noise_list=noise_list,
                                                               study_dir=f"{output_dir}optuna/",
                                                               n_trials=100)  """   
    fishnet_params = {'n_hidden_layers': 15,
                      'n_nodes_per_layer': 128,
                      'batch_size': 8,
                      'lr': 0.0001}

    compression_model = fishnets.FISHNET(n_params=4,
                                         n_d=100,
                                         n_features=len(features),
                                         n_hidden_layers=fishnet_params["n_hidden_layers"],
                                         n_nodes_per_layer=fishnet_params["n_nodes_per_layer"])                 

    # Train the compression model
    n_epochs = 5000
    print("Training compression model...", flush=True)
    start = time.time()
    training_results = compression_model.train(data_=X_train,
                                                theta_=Y_train,
                                                val_data_=X_test,
                                                val_theta_=Y_test,
                                                noise_list=noise_list,
                                                data_scaler=data_scaler,
                                                batch_size=pow(2,fishnet_params["batch_size"]),
                                                burn_in=0,
                                                lr=fishnet_params["lr"],
                                                epochs=n_epochs,
                                                weights_dir=f"{output_dir}/weights/")
    end = time.time()
    print(f"Compression model trained in {end-start:.2f} seconds", flush=True)

    # Load weights from best epoch
    val_losses = np.array(training_results["val_losses"])
    val_losses = gaussian_filter1d(val_losses, sigma=2)
    val_losses[np.isnan(val_losses)] = np.inf
    best_epoch = np.argmin(val_losses)
    best_loss = val_losses[best_epoch]
    print(f"Loading weights from epoch {best_epoch} with val loss {best_loss:.2f}", flush=True)
    best_weights = pickle.load(open(f"{output_dir}/weights/epoch_{best_epoch}.pkl","rb"))

    # Save the compression model weights
    pickle.dump(compression_model.w, open(f"{output_dir}{filename}_compression_model_w.pkl", "wb"))

    # Plot training 
    fig, ax = mpl.pyplot.subplots()
    ax.plot(training_results['losses'], label="Training Loss")
    ax.plot(training_results['val_losses'], label="Validation Loss")
    ax.plot(val_losses, label="Smoothed Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (log)")
    ax.set_ylim([10, -10])
    ax.legend(ncol=3)
    fig.savefig(f"{output_dir}{filename}_compression_model_training.pdf", dpi=300, bbox_inches='tight')

    

else:

    
    """# Load optuna study
    study_name = "fishnets_study"
    storage_name = f"sqlite:///{output_dir}optuna/fishnets_study.db"
    study = optuna.load_study(study_name=study_name,
                                storage=storage_name)

    fishnet_params = study.best_trial.params"""

    fishnet_params = {'n_hidden_layers': 10,
                      'n_nodes_per_layer': 128,
                      'batch_size': 8,
                      'lr': 0.0001}


    lr = fishnet_params["lr"]
    batch_size = pow(2,fishnet_params["batch_size"])

    print(f"Best fishnet parameters: {fishnet_params}", flush=True)
    print(f"Learning rate: {lr}", flush=True)
    print(f"Batch size: {batch_size}", flush=True)

    compression_model = fishnets.FISHNET(n_params=4,
                                         n_d=100,
                                         n_features=len(features),
                                         n_hidden_layers=fishnet_params["n_hidden_layers"],
                                         n_nodes_per_layer=fishnet_params["n_nodes_per_layer"])      

    # Train the compression model
    n_epochs = 100
    print("Training compression model...", flush=True)
    start = time.time()
    training_results = compression_model.train(data_=X_train,
                                                theta_=Y_train,
                                                val_data_=X_test,
                                                val_theta_=Y_test,
                                                noise_list=noise_list,
                                                data_scaler=data_scaler,
                                                batch_size=batch_size,
                                                burn_in=1,
                                                lr=lr,
                                                epochs=n_epochs,
                                                weights_dir=f"{output_dir}/weights/")
    end = time.time()
    print(f"Compression model trained in {end-start:.2f} seconds", flush=True)

    # Load weights from best epoch
    val_losses = np.array(training_results["val_losses"])
    val_losses = gaussian_filter1d(val_losses, sigma=2)
    val_losses[np.isnan(val_losses)] = np.inf
    best_epoch = np.argmin(val_losses)
    best_loss = val_losses[best_epoch]
    print(f"Loading weights from epoch {best_epoch} with val loss {best_loss:.2f}", flush=True)
    best_weights = pickle.load(open(f"{output_dir}/weights/epoch_{best_epoch}.pkl","rb"))

    # Save the compression model weights
    pickle.dump(compression_model.w, open(f"{output_dir}{filename}_compression_model_w.pkl", "wb"))

    # Plot training 
    fig, ax = mpl.pyplot.subplots()
    ax.plot(training_results['losses'], label="Training Loss")
    ax.plot(training_results['val_losses'], label="Validation Loss")
    ax.plot(val_losses, label="Smoothed Validation Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (log)")
    ax.set_ylim([10, -10])
    ax.legend(ncol=3, fontsize=10)
    fig.savefig(f"{output_dir}{filename}_compression_model_training.pdf", dpi=300, bbox_inches='tight')           
                 

    # Load model parameters from best trial training
    compression_model.w = pickle.load(open(f"{output_dir}{filename}_compression_model_w.pkl", "rb"))



##########################################################################################################
# Prepare compressed training and validation data for NPE
n_permutations = 10 # Number of noise realisations per merger-stars pair

class data_aggregator():

    def __init__(self, compression_model, data_scaler, noise_list):

        self.compression_model = compression_model
        self.data_scaler = data_scaler
        self.noise_list = noise_list

    def __call__(self, x, add_noise=True):

        if add_noise:
            # Add random calibration noise
            x += self.noise_list[rng.integers(0, self.noise_list.shape[0], size=x.shape[0])]

        # Scale and reshape
        x = self.data_scaler.transform(x.reshape(-1,4)).reshape(-1, 100, 4)
        # Data aggregation with fishnets
        x = self.compression_model(x)[0]
        # Cast to torch tensor
        x = torch.from_dlpack(x).float().to(device)

        return x

data_agg = data_aggregator(compression_model=compression_model,
                           data_scaler=data_scaler,
                           noise_list=noise_list)

# Get n permutations of the noise realisations
X_train_MAF = []
for n in range(n_permutations):
    
    x_agg = data_agg(X_train)
    X_train_MAF.append(x_agg)

train_ds = torch.cat(X_train_MAF, dim=0)
train_ds_labels = torch.from_numpy(np.vstack([Y_train for i in range(n_permutations)])).float().to(device)

val_data = data_agg(X_test)
val_labels = torch.from_numpy(Y_test).float().to(device)

# Save compressed training and validation data
np.savez(f"{output_dir}data/compressed_data_{filename}.npz",
         X_train=train_ds.cpu().numpy(),
         Y_train=train_ds_labels.cpu().numpy(),
         X_test=val_data.cpu().numpy(),
         Y_test=val_labels.cpu().numpy())

# Create a TensorDataset object
train_ds = torch.utils.data.TensorDataset(train_ds, train_ds_labels)
val_ds = torch.utils.data.TensorDataset(val_data, val_labels)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

loader = TorchLoader(train_loader=train_loader, val_loader=val_loader)


##########
# Use OPTUNA to find the best set of MAF parameters
import optuna
from optuna_opt import hyperparameter_search
from ili.inference import InferenceRunner
from scipy.ndimage import gaussian_filter1d


prior = Uniform(low=[0,6,8,-3],
                high=[14,11,12,0],
                device=device)



if not os.path.exists(f"{output_dir}optuna/ltu_ili_npe_tarp_study.db"):

    """npe_params = hyperparameter_search(loader=loader,
                                       prior=prior,
                                       study_dir=f"{output_dir}optuna/",
                                       X_test=val_data.cpu().numpy(),
                                       Y_test=val_labels.cpu().numpy(), 
                                       n_trials=100)"""
    
    npe_params = {'hidden_features': 64,
                      'num_transforms': 5,
                      'batch_size': 8,
                      'lr': 0.0001}

    # Train NPE
    train_args = dict(
        training_batch_size=pow(2,npe_params["batch_size"]),
        learning_rate=npe_params["lr"],
        stop_after_epochs=100,
        max_epochs=2000,
        clip_max_norm=5.
    )

    # Update NPE model with results of optimisation
    nets = [load_nde_lampe(hidden_features=npe_params["hidden_features"],
                           num_transforms=npe_params["num_transforms"],
                           model="nsf",
                           x_normalize=True,
                           theta_normalize=True,
                           device=device) for i in range(1)]

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
    
else:

    """# Load optuna study
    study_name = "ltu_ili_npe_tarp_study"
    storage_name = f"sqlite:///{output_dir}optuna/ltu_ili_npe_tarp_study.db"
    study = optuna.load_study(study_name=study_name,
                                storage=storage_name)

    npe_params = study.best_trials[0].params"""

    npe_params = {'hidden_features': 64,
                      'num_transforms': 5,
                      'batch_size': 8,
                      'lr': 0.0001}


    # Load model trained with best parameters
    posterior_model = pickle.load(open(f'{output_dir}{filename}.pkl', 'rb'))



####################################################################################
####################################################################################
# Validation tests
####################################################################################
####################################################################################

# Create test dictionary with data and labels for each Auriga merger in the val set
test_dictionary = {"X": val_data.cpu().numpy(),
                    "Y": val_labels.cpu().numpy(),
                    "ID": [f"{i:05}" for i in range(len(val_labels))]}


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
             'log($M_{*}/\\rm{M}_{\odot}$)',
             'log($M/\\rm{M}_{\odot}$)', 
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


####################################################################################
####################################################################################
# Application to Milky Way substructures
####################################################################################
####################################################################################

plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']

substructures = ['Arjuna', 'GES', 'Sagittarius', 'Helmi',
                 'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
                 'LMS', 'Heracles', 'LMC', 'SMC']
# Decide how many samples to get from the posterior
n_posterior_samples = 1000

for substructure in substructures:

    print(f"Sampling posterior for {substructure}...", flush=True)

    # Select chemo-dynamical properties for the stars in the substructure 
    data = obs_data.loc[obs_data[substructure+"_flag"]==1,features].values

    # Select 10 samples of 100 stars each from data
    # get how many times you can sample from the progenitor
    n_data_samples = math.ceil(len(data)/100)
    if len(data)>=100:
        data_samples = [data[rng.choice(np.arange(len(data)), 
                                            size=100, replace=False)] 
                        for i in range(n_data_samples)]

    else:
        data_samples = [data[rng.choice(np.arange(len(data)), 
                                        size=100, replace=True)] 
                        for i in range(n_data_samples)]

    # Sample the posterior of the progenitor properties as conditioned by each data sample
    posterior_samples = []
    for data_sample in data_samples:

        # Scale and compress data features
        data_sample = data_agg(data_sample, add_noise=False)

        # Get posterior samples
        theta_samples = posterior_model.sample((n_posterior_samples,), 
                                                data_sample)

        posterior_samples.append(theta_samples.cpu().numpy())

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


