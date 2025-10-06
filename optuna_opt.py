import optuna
import numpy as np
import torch
from ili.dataloaders import TorchLoader
import fishnets
from ili.utils import Uniform, load_nde_lampe
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorSamples
import tarp
import torch
from torch.utils.data import Dataset
import jax
import jax.numpy as jnp
from my_func import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hyperparameter_search(loader,
                          prior,
                          study_dir,
                          X_test,
                          Y_test):
    

    train_args = dict(
        training_batch_size=256,
        learning_rate=1e-4,
        stop_after_epochs=100,
        max_epochs=500,
        clip_max_norm=5
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
        num_transforms = trial.suggest_int("num_transforms", 5, 20)

        
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
        posterior, summaries = runner(loader=loader)

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

        return 1,abs(tarp_val - 0.5)#log_p,abs(tarp_val - 0.5)

    # Run the study
    study.optimize(objective, n_trials=100, timeout=1800)

    # Save study

    print("\nBest trial:", flush=True)
    best = study.best_trials[0]
    print(f"  Log-prob: {best.values[0]:.4f}, TARP dev: {best.values[1]:.4f}", flush=True)
    print("  Params:", flush=True)
    for k, v in best.params.items():
        print(f"    {k}: {v}")


    return best.params



