import optuna
import numpy as np
import torch
import fishnets
from ili.utils import load_nde_lampe
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorSamples
import tarp
import torch
from my_func import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hyperparameter_search(loader,
                          prior,
                          study_dir,
                          X_test,
                          Y_test,
                          n_trials: int = 100):

                          
    train_args = dict(
        training_batch_size=256,
        learning_rate=1e-4,
        stop_after_epochs=100,
        max_epochs=100,
        clip_max_norm=1
    )


    # Create optuna study
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_dir}ltu_ili_npe_tarp_study.db", 
                                         engine_kwargs={"connect_args": {"timeout": 300}})
    
    study = optuna.create_study(directions=["maximize", "minimize"],
                                storage=storage,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42),
                                study_name="ltu_ili_npe_tarp_study")

    
    
    # Define objective function

    def objective(trial):

        # Sample hyperparameters NPE
        hidden_features = trial.suggest_int("hidden_features", 100, 500, step=20)
        num_transforms = trial.suggest_int("num_transforms", 10, 50)
        

        # Define NPE model
        nets = load_nde_lampe(
                model="maf",
                hidden_features=hidden_features,
                num_transforms=num_transforms,
                x_normalize=True,
                theta_normalize=True,
                device=device,
        )


        runner = InferenceRunner.load(
            backend="lampe",
            engine="NPE",
            prior=prior,
            nets=nets,
            device=device,
            train_args=train_args,
        )

        # Train NPE
        try:
            posterior, summaries = runner(loader=loader)
        except:
            return -100, 100

        # Evaluate log probability and TARP curve for validation data
        # TARP midpoint deviation
        sampler = PosteriorSamples(num_samples=200, sample_method='direct')
        samps = sampler(posterior, torch.tensor(X_test).to(device))

        ecp, _ = tarp.get_tarp_coverage(
            samps, Y_test,
            norm=True, bootstrap=True, num_bootstrap=100
        )
        tarp_val = torch.mean(torch.from_numpy(
            ecp[:, ecp.shape[1] // 2])).to(device)

        log_p = summaries[0]['validation_log_probs'][-1]

        log_p = -100 if np.isinf(log_p) else log_p

        return log_p,abs(tarp_val - 0.5)

    # Run the study
    study.optimize(objective, n_trials=n_trials, timeout=18000)

    # Save study

    print("\nBest trial:", flush=True)
    best = study.best_trials[0]
    print(f"  Log-prob: {best.values[0]:.4f}, TARP dev: {best.values[1]:.4f}", flush=True)
    print("  Params:", flush=True)
    for k, v in best.params.items():
        print(f"    {k}: {v}")


    return best.params

def hyperparameter_search_fishnets(X_train,
                                   Y_train,
                                   X_test,
                                   Y_test,
                                   data_scaler,
                                   noise_list,
                                   obs_err_list,
                                   study_dir,
                                   n_trials: int = 100
                                   ):
    
    # Create optuna study
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{study_dir}fishnets_study.db", 
                                         engine_kwargs={"connect_args": {"timeout": 300}})
    
    study = optuna.create_study(directions=["minimize"],
                                storage=storage,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42),
                                study_name="fishnets_study")

    def objective_fishnets(trial):
        
        n_hidden_layers = trial.suggest_int("n_hidden_layers", 10, 50)
        n_nodes_per_layer = trial.suggest_int("n_nodes_per_layer", 100, 500, step=20)

        fishnet_params = {"n_hidden_layers": n_hidden_layers,
                        "n_nodes_per_layer": n_nodes_per_layer}

        # Learn data compression model with fishnet
        compression_model = fishnets.FISHNET(n_params=Y_train.shape[1],
                                             n_d=X_train.shape[1],
                                             n_features=X_test.shape[2],
                                             **fishnet_params)

        # Train the compression model
        n_epochs = 100
        print("Training compression model...", flush=True)
        start = time.time()
        try:
            training_results = compression_model.train(data_=X_train,
                                                       theta_=Y_train,
                                                       val_data_=X_test,
                                                       val_theta_=Y_test,
                                                       noise_list=noise_list,
                                                       obs_noise_list=obs_err_list,
                                                       data_scaler=data_scaler,
                                                       burn_in=20,
                                                       epochs=n_epochs)
        except:
            return 1000
            
        
        return np.median(training_results["val_losses"][-10:])
    
    # Run the study
    study.optimize(objective_fishnets, n_trials=n_trials, timeout=18000)

    # Save study
    print("\nBest trial:", flush=True)
    best = study.best_trial
    print(f"  Log-prob: {best.values[0]:.4f}", flush=True)
    print("  Params:", flush=True)
    for k, v in best.params.items():
        print(f"{k}: {v}")


    return best.params