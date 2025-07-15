import optuna
import numpy as np
import torch
from ili.dataloaders import TorchLoader
import fishnets
from ili.utils import Uniform, load_nde_lampe
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorSamples
import tarp



def hyperparameter_search(X_train: np.array,
                          Y_train: np.array,
                          X_test: np.array,
                          Y_test: np.array,
                          scaler_params):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Define prior
    prior = Uniform(low=scaler_params.transform(np.array([0,6,8,-3])[None,:] )[0],
                    high=scaler_params.transform(np.array([14,11,12,0])[None,:] )[0],
                    device=device)


    train_args = dict(
        training_batch_size=256,
        learning_rate=1e-4,
        stop_after_epochs=15,
        max_epochs=100
    )


    # Create optuna study
    storage = optuna.storages.RDBStorage(url="sqlite:////mnt/aridata1/users/ariasant/MW-sbi/optuna_study/hyperparameters_search.db", 
                                         engine_kwargs={"connect_args": {"timeout": 300}})
    
    study = optuna.create_study(directions=["maximize", "minimize"],
                                storage=storage,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42),
                                study_name="ltu_ili_npe_tarp_study")
    
    # Define objective function

    def objective(trial,
                  X_train=X_train,
                  Y_train=Y_train,
                  X_test=X_test,
                  Y_test=Y_test):
        # Sample hyperparameters for fishnet
        hidden_layers_fish = trial.suggest_int("hidden_layers_fish", 1, 10)
        nodes_per_layer_fish = trial.suggest_int("nodes_per_layer_fish", 10, 500)

        # Sample hyperparameters NPE
        model = trial.suggest_categorical("model", ["nsf", "maf", "gf"])
        hidden_features = trial.suggest_int("hidden_features", 10, 500)
        num_transforms = trial.suggest_int("num_transforms", 5, 20)

        # Define compression model and train it
        compression_model = fishnets.FISHNET(n_params=4,
                                            n_d=100,
                                            n_features=4,
                                            n_hidden_layers=hidden_layers_fish,
                                            n_nodes_per_layer=nodes_per_layer_fish)
        
        training_results = compression_model.train(data_=X_train,
                                            theta_=Y_train,
                                            batch_size=256,
                                            lr=1e-4,
                                            epochs=500)
        
        # Define NPE model
        nets = [load_nde_lampe(
                model=model,
                hidden_features=hidden_features,
                num_transforms=num_transforms,
                x_normalize=False,
                device=device,
        )]


        runner = InferenceRunner.load(
            backend="lampe",
            engine="NPE",
            prior=prior,
            nets=nets,
            device=device,
            train_args=train_args,
        )

        # Train NPE
        # Use fixed train/test split defined outside objective
        X_train, _, _ = compression_model(X_train)
        X_test, _, _ = compression_model(X_test)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()),
            batch_size=256, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float()),
            batch_size=256, shuffle=False
        )
        loader = TorchLoader(train_loader=train_loader, val_loader=val_loader)


        posterior, summaries = runner(loader=loader)

        # Evaluate log probability and TARP curve for validation data
        # TARP midpoint deviation
        sampler = PosteriorSamples(num_samples=500, sample_method='direct')
        samps = sampler(posterior, torch.tensor(X_test).to(device))

        ecp, _ = tarp.get_tarp_coverage(
            samps, Y_test,
            norm=True, bootstrap=True, num_bootstrap=100
        )
        tarp_val = torch.mean(torch.from_numpy(
            ecp[:, ecp.shape[1] // 2])).to(device)

        return summaries[0]['validation_log_probs'][-1], abs(tarp_val - 0.5)

    # Run the study
    study.optimize(objective, n_trials=300, timeout=1800)

    # Save study

    print("\nBest trial:", flush=True)
    best = study.best_trials[0]
    print(f"  Log-prob: {best.values[0]:.4f}, TARP dev: {best.values[1]:.4f}", flush=True)
    print("  Params:", flush=True)
    for k, v in best.params.items():
        print(f"    {k}: {v}")


    return best.params


