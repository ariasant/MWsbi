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

        # Transform data        
        sim_data = prepare_data_for_Box_Cox(sim_data, features)
        obs_data = prepare_data_for_Box_Cox(obs_data, features)

        # Apply Box-Cox transformation
        pt_sim = PowerTransformer(method="box-cox", standardize=True)
        sim_data[features] = pt_sim.fit_transform(sim_data[features].values)

        pt_obs = PowerTransformer(method="box-cox", standardize=True)
        obs_data[features] = pt_obs.fit_transform(obs_data[features].values)

        # Remove outliers
        sim_data = sim_data[np.logical_and.reduce([sim_data[feature]**2 < 5**2 for feature in features])]
        obs_data = obs_data[np.logical_and.reduce([obs_data[feature]**2 < 5**2 for feature in features])]

        end = time.time()
        print(f"DataProcessor initialized in {end-start:.2f} seconds", flush=True)

        return sim_data, obs_data


    

def prepare_data_for_Box_Cox(df: pd.DataFrame,
                             features: list):
        

        # Prepare data for Box-Cox transformation
        df.dropna(subset=features, inplace=True)
        df = df[df["E"]<0]

        # Remove outliers
        df = df[(df["L"]<1e4) &
                (df["FeH"]>-3) & (df["FeH"]<1) &
                (df["MgFe"]>-1) & (df["MgFe"]<1)]
        
        # Ensure all features are positive
        df["E"] *= -1
        df["FeH"] -= (df["FeH"].min()-1e-3)
        df["MgFe"] -= (df["MgFe"].min()-1e-3)
        
        return df
