import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import time

def DataProcessor(features: list,
                  sim_data: pd.DataFrame,
                  obs_data: pd.DataFrame):
        
        features = features

        print("Initializing DataProcessor", flush=True)
        start = time.time()

        # Prepare data for Box-Cox transformation
        FeH_MIN = min(sim_data["FeH"].min(), obs_data["FeH"].min()) - 1e-3
        MgFe_MIN = min(sim_data["MgFe"].min(), obs_data["MgFe"].min()) - 1e-3

        for df in [sim_data, obs_data]:
                df.dropna(subset=features, inplace=True)
                df.query("E < 0 & L > 0", inplace=True)  # Use query for in-place filtering

                # Remove outliers
                df.query("L < 1e4 & -3 < FeH < 1 & -1 < MgFe < 1", inplace=True)

                df["E"] *= -1
                df["FeH"] -= FeH_MIN
                df["MgFe"] -= MgFe_MIN
                

        # Apply Box-Cox transformation
        pt = PowerTransformer(method="box-cox", standardize=True)
        sim_data[features] = pt.fit_transform(sim_data[features].values)
        obs_data[features] = pt.transform(obs_data[features].values)

        # Remove outliers
        sim_data = sim_data[np.logical_and.reduce([sim_data[feature]**2 < 5**2 for feature in features])]
        obs_data = obs_data[np.logical_and.reduce([obs_data[feature]**2 < 5**2 for feature in features])]

        end = time.time()
        print(f"DataProcessor initialized in {end-start:.2f} seconds", flush=True)

        return sim_data, obs_data


    
