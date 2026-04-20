# Code to replicate the analysis end to end

import corner
from ili.dataloaders import TorchLoader
from ili.inference import InferenceRunner
from ili.utils import Uniform, load_nde_lampe

import math
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import RobustScaler
import sys
import time
import torch

sys.path.append("/mnt/aridata1/users/ariasant/MW-sbi")

import codes.calibration_model as cm
import codes.fishnets as fishnets
import codes.plotting_utils as plots
import codes.optuna_opt as optuna_opt
import codes.sbi_results as sbi_results


class data_aggregator():

    def __init__(self, compression_model, data_scaler, noise_list):

        self.compression_model = compression_model
        self.data_scaler = data_scaler
        self.noise_list = noise_list

    def __call__(self, x, add_noise=True):

        if add_noise:
            # Add random calibration noise
            x += self.noise_list[self.rng.integers(0, self.noise_list.shape[0], size=x.shape[0])]

        # Scale and reshape
        x = self.data_scaler.transform(x.reshape(-1,4)).reshape(-1, 100, 4)
        # Data aggregation with fishnets
        x = self.compression_model(x)[0]
        # Cast to torch tensor
        x = torch.from_dlpack(x).float().to(self.device)

        return x


class pipeline():

    def __init__(self,
                 features: list[str],
                 parameters: list[str],
                 substructures: list[str],
                 data_dir: str,
                 output_dir: str,
                 plot_labels: list[str],
                 plot_ranges: list[tuple]
                 ):
        
        # List of stellar properties used to run the inference
        self.features = features

        # List of merger parameters to infer
        self.parameters = parameters

        # List of MW substructures to characterise
        self.substructures = substructures

        # Path where the pre-processed Auriga simulations are stored
        self.data_dir = data_dir

        # Define random number generator for reproducibility
        self.rng = np.random.default_rng(17)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define output directory and filename for results
        self.output_dir = output_dir
        self.filename = "MW_sbi_results_"+"".join(self.features)+"".join(self.parameters)

        # Create new folders to store results and plots
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/weights", exist_ok=True)
        os.makedirs(f"{self.output_dir}optuna", exist_ok=True)

        # Define analysis outputs
        self.data_scaler = None
        self.data_agg = None
        self.compression_model = None
        self.posterior_model = None

        self.plot_labels = plot_labels
        self.plot_ranges = plot_ranges


    def __load_obs_data__(self):

        # Load Milky Way (target) data
        obs_data = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")

        obs_data.dropna(subset=self.features, inplace=True)
        obs_data = obs_data[(obs_data["E"]<0)&(obs_data["L"]>0)]
        # Select accreted stars
        obs_accreted = ((obs_data.AlFe<-0.07) & (obs_data.MgMn>=0.25)) | \
                    ((obs_data.AlFe>=-0.07) & (obs_data.MgMn>=4.25*obs_data.AlFe+0.5475))
        obs_accreted = np.logical_or.reduce([obs_accreted]+[obs_data[f"{substructure}_flag"]==1 
                                            for substructure in self.substructures])
        
        return obs_data, obs_accreted
    
    def __load_sim_data__(self):

        # Load data from the Auriga simulations
        sim_data = []

        for file in os.listdir(self.data_dir):

            df = pd.read_pickle(f"{self.data_dir}{file}")
            df.rename(columns={"aFe":"MgFe"}, inplace=True)

            # Get rid of stars with numerical issues
            df = df[(df["E"]<0) & (df["L"]>0) & 
                    (df["MgFe"]<0.5) & (df["MgFe"]>-0.5) &
                    (df["FeH"]<1) & (df["FeH"]>-3)]

            sim_data.append(df)

        sim_data = pd.concat(sim_data, ignore_index=True)

        return sim_data
    
    def __extract_sim_data__(
        self,
        sim_data: pd.DataFrame,
        ):

        print("Splitting merger-stars pairs in Auriga into training and validation sets...", flush=True)
        # Create datasets for training 
        X_train, Y_train = [], []
        X_test, Y_test = [], [] 

        prog_IDs = sim_data['progID'].unique()

        for progID in prog_IDs:
            # Get the data for the current progenitor
            prog_data = sim_data[sim_data["progID"]==progID]
            if len(prog_data) < 100:
                continue
            # Sample the data n times (maximum 20 and minimum 10 times)
            n = max(min(100, math.ceil(len(prog_data)//100)),10)
            for i in range(n):
                idx_sample = self.rng.choice(np.arange(len(prog_data)), size=100, replace=False)
                u = self.rng.uniform()

                if u>0.2: #progID in training_IDs:
                    X_train.append(prog_data[self.features].values[idx_sample])
                    Y_train.append(prog_data[self.parameters].values[idx_sample][0])

                else:
                    X_test.append(prog_data[self.features].values[idx_sample])
                    Y_test.append(prog_data[self.parameters].values[idx_sample][0])
            
        X_train = np.stack(X_train)
        Y_train = np.stack(Y_train)
        X_test = np.stack(X_test)
        Y_test = np.stack(Y_test)

        # Oversample training data
        X_train, Y_train = cm.oversample_data(X_train, Y_train)

        # Shuffle training examples
        X_train = cm.shuffle_axis1_independently(X_train)

        print(f"X_train shape: {X_train.shape}", flush=True)
        print(f"Y_train shape: {Y_train.shape}", flush=True)
        print(f"X_test shape: {X_test.shape}", flush=True)
        print(f"Y_test shape: {Y_test.shape}", flush=True)

        return X_train, X_test, Y_train, Y_test
    
    def __train_compression_model__(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        noise_list,
        fishnet_params = None
        ):

        if fishnet_params is None:
            # Run hyperparameter search
            print("Starting hyperparameter search for the fishnets model", flush=True)
            fishnet_params = optuna_opt.hyperparameter_search_fishnets(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                data_scaler=self.data_scaler,
                noise_list=noise_list,
                study_dir=f"{self.output_dir}optuna/",
                n_trials=100
            ) 
        
        compression_model = fishnets.FISHNET(
            n_params=len(self.parameters),
            n_d=100,
            n_features=len(self.features),
            n_hidden_layers=fishnet_params["n_hidden_layers"],
            n_nodes_per_layer=fishnet_params["n_nodes_per_layer"]
        )   
        
        # Train the compression model
        n_epochs = 5000
        print("Training compression model...", flush=True)
        start = time.time()
        training_results = compression_model.train(
            data_=X_train,
            theta_=Y_train,
            val_data_=X_test,
            val_theta_=Y_test,
            noise_list=noise_list,
            data_scaler=self.data_scaler,
            batch_size=pow(2,fishnet_params["batch_size"]),
            burn_in=0,
            lr=fishnet_params["lr"],
            epochs=n_epochs,
            weights_dir=f"{self.output_dir}/weights/"
        )
        end = time.time()
        print(f"Compression model trained in {end-start:.2f} seconds", flush=True)

        # Load weights from best epoch
        val_losses = np.array(training_results["val_losses"])
        val_losses = gaussian_filter1d(val_losses, sigma=2)
        val_losses[np.isnan(val_losses)] = np.inf
        best_epoch = np.argmin(val_losses)
        best_loss = val_losses[best_epoch]
        print(f"Loading weights from epoch {best_epoch} with val loss {best_loss:.2f}", flush=True)
        best_weights = pickle.load(open(f"{self.output_dir}/weights/epoch_{best_epoch}.pkl","rb"))

        # Save the compression model weights
        pickle.dump(compression_model.w, open(f"{self.output_dir}{self.filename}_compression_model_w.pkl", "wb"))

        # Plot training 
        fig, ax = mpl.pyplot.subplots()
        ax.plot(training_results['losses'], label="Training Loss")
        ax.plot(training_results['val_losses'], label="Validation Loss")
        ax.plot(val_losses, label="Smoothed Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (log)")
        ax.set_ylim([10, -10])
        ax.legend(ncol=3)
        fig.savefig(f"{self.output_dir}/plots/{self.filename}_compression_model_training.pdf", dpi=300, bbox_inches='tight')

        compression_model.w = pickle.load(open(f"{self.output_dir}{self.filename}_compression_model_w.pkl", "rb"))

        return compression_model


    def __create_NPE_data__(self,
                            X_train,
                            X_test,
                            Y_train,
                            Y_test,
                            batch_size = 256,
                            n_permutations = 10
                            ):

        
        # Get n permutations of the noise realisations
        X_train_MAF = []
        for n in range(n_permutations):
            
            x_agg = self.data_agg(X_train)
            X_train_MAF.append(x_agg)

        train_ds = torch.cat(X_train_MAF, dim=0)
        train_ds_labels = torch.from_numpy(np.vstack([Y_train for i in range(n_permutations)])).float().to(self.device)

        val_data = self.data_agg(X_test)
        val_labels = torch.from_numpy(Y_test).float().to(self.device)

        # Create a TensorDataset object
        train_ds = torch.utils.data.TensorDataset(train_ds, train_ds_labels)
        val_ds = torch.utils.data.TensorDataset(val_data, val_labels)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        loader = TorchLoader(train_loader=train_loader, val_loader=val_loader)

        return loader
    
    def __train_NPE__(self,
                      prior,
                      train_args,
                      loader,
                      npe_params = None                
        ):


        if npe_params is None:

            npe_params = optuna_opt.hyperparameter_search(
                loader=loader,
                prior=prior,
                study_dir=f"{self.output_dir}optuna/",
                X_test=loader.val_loader.val_data.cpu().numpy(),
                Y_test=loader.val_loader.val_labels.cpu().numpy(), 
                n_trials=100)

        # Define NPE models
        nets = [load_nde_lampe(
                    hidden_features=npe_params["hidden_features"],
                    num_transforms=npe_params["num_transforms"],
                    model="maf",
                    x_normalize=True,
                    theta_normalize=True,
                    device=self.device) for i in range(3)
                ]

        # Run training
        runner = InferenceRunner.load(
                backend="lampe",
                engine="NPE",
                prior=prior,
                nets=nets,
                device=self.device,
                train_args=train_args,
        )

        posterior, summaries = runner(loader=loader)

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
        fig.savefig(f'{self.output_dir}/plots/{self.filename}_training_plot.png', dpi=400)

        return posterior

    def __make_validation_plots__(self,
                                  samples,
        ):

        # Make plot of cross-validated parameters inference
        sbi_results.cross_validation_plot(
            samples=[samples],
            percentile_range=[16,84],
            plot_labels=self.plot_labels,
            plot_ranges=self.plot_ranges,
            filename=f'{self.output_dir}/plots/cross_validation_1684_{self.filename}.png'
        )

        # Save table with quantitative results 
        sbi_results.rms_table_per_galaxy(
            samples={"SUITE":samples},
            parameters=self.parameters,
            filename=f'{self.output_dir}rms_table_{self.filename}.csv'
        )

        sbi_results.count_predictions_within_range(
            samples={"SUITE":samples},
            parameters=self.parameters,
            percentile_range=[16,84],
            filename=f'{self.output_dir}range_table_{self.filename}_1684.csv'
        )


    def __predict_posterior__(
        self,
        substructure: str,
        obs_data: pd.DataFrame,
        n_posterior_samples: int = 1000,
        ):

        print(f"Sampling posterior for {substructure}...", flush=True)

        # Select chemo-dynamical properties for the stars in the substructure 
        data = obs_data.loc[obs_data[substructure+"_flag"]==1,self.features].values

        # Select 10 samples of 100 stars each from data
        # get how many times you can sample from the progenitor
        n_data_samples = math.ceil(len(data)/100)
        if len(data)>=100:
            data_samples = [data[self.rng.choice(np.arange(len(data)), 
                                                    size=100, replace=False)] 
                            for i in range(n_data_samples)]

        else:
            data_samples = [data[self.rng.choice(np.arange(len(data)), 
                                                    size=100, replace=True)] 
                            for i in range(n_data_samples)]

        # Sample the posterior of the progenitor properties as conditioned by each data sample
        posterior_samples = []
        for data_sample in data_samples:

            # Scale and compress data features
            data_sample = self.data_agg(data_sample, add_noise=False)

            # Get posterior samples
            theta_samples = self.posterior_model.sample((n_posterior_samples,), 
                                                    data_sample)

            posterior_samples.append(theta_samples.cpu().numpy())

        # Concatenate all posterior samples
        posterior_samples = np.concatenate(posterior_samples, axis=0)
        
        # Save posterior samples
        pickle.dump(posterior_samples, open(f"{self.output_dir}{substructure}.pkl","wb"))
        
        # Plot posterior samples
        fig = corner.corner(
            posterior_samples, 
            bins=20, 
            labels=self.plot_labels,
            range=self.plot_ranges,
            quantiles=[.16,.50,.84],
            plot_contours=False,
            show_titles=True,
            title_kwargs={'fontsize':8},
            verbose=True)
        fig.suptitle(f"Posterior samples for {substructure}", fontsize=16)
        fig.savefig(f"{self.output_dir}/plots/{substructure}.pdf",dpi=400)

    
    def run(self):

        # Load obs data
        obs_data, obs_acc = self.__load_obs_data__()

        # Load sim  data
        sim_data = self.__load_sim_data__()

        # Plot initial data distributions
        fig = plots.plot_stars_data([sim_data, obs_data, obs_data[obs_acc]], features=features)
        fig.savefig(f"{self.output_dir}/plots/initial_data_{self.filename}.pdf", dpi=300, bbox_inches='tight')


        # Preprocess data
        sim_data["E"] = np.log(-sim_data["E"].values)
        sim_data["L"] = np.log(sim_data["L"].values)

        obs_data["E"] = np.log(-obs_data["E"].values)
        obs_data["L"] = np.log(obs_data["L"].values)


        # Plot data after processing
        fig = plots.plot_stars_data([sim_data, obs_data, obs_data[obs_acc]], features=features)
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
        fig.savefig(f"{self.output_dir}/plots/transformed_data_{self.filename}.pdf", dpi=300, bbox_inches='tight')


        # Plot merger parameters
        fig = corner.corner(
            sim_data[parameters].values,
            color='k',
            labels=parameters,
            bins=20,
            plot_contours=False,
            plot_datapoints=False,
            fill_contours=False,
            hist_kwargs={"density": True}
        )
        fig.savefig(f"{self.output_dir}/plots/merger_parameters_{self.filename}.pdf", 
                    dpi=300, bbox_inches='tight')

        self.data_scaler = RobustScaler() 
        self.data_scaler.fit(obs_data[self.features].values)

        # Calibrate the simulation data to be in-distribution 
        print("Generating noise for validation and training data...", flush=True)
        
        noise_model = cm.generate_mean_cov_model(features=self.features,
                                                 n_stars_per_prog=100)
        noise_list = cm.sample_noise_training(model=noise_model,
                                              n_samples=10000,
                                              random_seed=16)
        
        # Create data to train NPE model
        X_train, X_test, Y_train, Y_test = self.__extract_sim_data__(sim_data)

        X_train = X_train[:100]
        Y_train = Y_train[:100]
        X_test = X_test[:20]
        Y_test = Y_test[:20]

        # Visualise new distributions
        x_plot = []
        for i in range(10):
            noise = noise_list[np.random.randint(0, noise_list.shape[0], size=X_train.shape[0])]
            x_plot.append(X_train + noise)
            
        x_plot = np.vstack(x_plot)

        train_df = pd.DataFrame(data=x_plot.reshape(-1, len(features)), columns=features)

        fig = plots.plot_stars_data([train_df, obs_data, obs_data[obs_acc]], features=features)
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
        fig.savefig(f"{self.output_dir}/plots/train_data_with_noise_{self.filename}.pdf", 
                    dpi=300, 
                    bbox_inches='tight')

        ##########
        # TRAINING
        ##########

        self.compression_model = self.__train_compression_model__(
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            noise_list=noise_list,
            optuna=False
        )

        # Prepare NPE training data
        
        # Preprocess data to train NPE by:
        # 1. Calibrating sim data to match obs data
        # 2. Scaling the data
        # 3. Compressing the information with FishNets  
        self.data_agg = data_aggregator(
            compression_model=compression_model,
            data_scaler=data_scaler,
            noise_list=noise_list
        )

        
        NPE_data_loader = self.__create_NPE_data__(
            X_train,
            X_test,
            Y_train,
            Y_test,
            n_permutations = 10
        )
        
        # Train NPE model

        prior = Uniform(
            low=[0,6,8,-3],
            high=[14,11,12,0],
            device=device
        )

        train_args = dict(
            training_batch_size=pow(2,npe_params["batch_size"]),
            learning_rate=npe_params["lr"],
            stop_after_epochs=100,
            max_epochs=2000,
            clip_max_norm=5.
        )


        self.posterior = self.__train_NPE__(
            prior,
            train_args,
            NPE_data_loader
        )

        #########
        # Model Validation
        #########

        # Create test dictionary with data and labels for each Auriga merger in the val set
        test_dictionary = {
            "X": val_data.cpu().numpy(),
            "Y": val_labels.cpu().numpy(),
            "ID": [f"{i:05}" for i in range(len(val_labels))]
        }


        # Sample parameters for test galaxy
        samples = sbi_results.validation(
            posterior_ensemble=self.posterior,
            test_dictionary=test_dictionary,
            filename=filename,
            output_dir=output_dir
        )

        self.__make_validation_plots__(
            samples,
            test_dictionary
        )

        ############
        # Apply model to MW data
        ############


        plot_labels=[
            '$\\tau \, [\mathrm{Gyr}]$',
            'log($M_{*}/M_{\odot}$)',
            'log($M/M_{\odot}$)', 
            'MMR (log)'
        ]

        # Decide how many samples to get from the posterior
        for substructure in self.substructures:

            self.__predict_posterior__(
                substructure,
                obs_data,
                n_posterior_samples=1000
            )


if __name__ == "__main__":

    features = ['E', 'L', 'FeH', 'MgFe']
    parameters = ['infall_time','log_Mprog_stellar', 'log_Mprog', 'log_Mprog2host']
    substructures = [
        'GES', 
        'Sagittarius', 
        'Helmi',
        'Sequoia_K19',
        'Sequoia_M19',
        'Sequoia_N20',
        'Iitoi', 
        'Thamnos',
        'LMS', 
        'Heracles', 
    ]

    data_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/data/with_satellites/"
    output_dir = "/mnt/aridata1/users/ariasant/MW-sbi/results/"

    plot_labels=[
        '$\\tau \, [\mathrm{Gyr}]$',
        'log($M_{*}/\\rm{M}_{\odot}$)',
        'log($M/\\rm{M}_{\odot}$)', 
        'MMR (log)'
    ]

    plot_ranges=[
        [0.1,13.9], # infall time
        [5.9,10.9], # log stellar mass
        [8.5,11.9], # log halo mass
        [-3.2,-0.1] # log MMR
    ]

    # Call formatting for plots
    plots.call_plotting_formatting()

    mwsbi = pipeline(
        features=features,
        parameters=parameters,
        substructures=substructures,
        data_dir=data_dir,
        output_dir=output_dir,
        plot_labels=plot_labels,
        plot_ranges=plot_ranges
    )

    mwsbi.run()