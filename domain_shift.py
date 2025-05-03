import corner
import matplotlib as mpl
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import time

def plot_stars_data(dfs: list, features: list):

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


class DataProcessor:

    def __init__(self, 
                 features: list,
                 ):
        
        self.features = features
        self.mu_obs = {}
        self.std_obs = {}
        self.mu_sim = {}
        self.std_sim = {}


        print("Initializing DataProcessor", flush=True)
        start = time.time()
        # 1 create sample with accreted stars from the Auriga simulation
        data_dir = "/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observation_shifted/data/"
        sim_data = []

        for i in range(1,31):

            df = pd.read_pickle(f"{data_dir}G{i:02}.pkl")
            df.rename(columns={"aFe":"MgFe"}, inplace=True)

            # Get rid of stars with numerical issues
            df = df[(df["E"]<0) & (df["L"]>0)]
            
            # Select stars bound to main halo in the Solar area
            df = df[(df["in_satellite"]==0) & 
                    (df["insitu_flag"]==0) &
                    (df["r"]<15) & (df["r"]>5)]

            sim_data.append(df)

        sim_data = pd.concat(sim_data, ignore_index=True)


        # 2 Select accreted stars from the apogee sample
        df = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")
        df = df.dropna(subset=features)
        obs_data = df[(df["E"]<0)]
        obs_accreted = ((obs_data.AlFe<-0.07) & (obs_data.MgMn>=0.25)) | \
                       ((obs_data.AlFe>=-0.07) & (obs_data.MgMn>=4.25*obs_data.AlFe+0.5475))
        obs_accreted = np.logical_or.reduce([obs_accreted]+[obs_data[f"{substructure}_flag"]==1 
                                            for substructure in ['GES', 'Sagittarius', 'Helmi',
                                                                 'Sequoia_K19','Sequoia_M19','Sequoia_N20',
                                                                 'Iitoi', 'Thamnos','LMS', 'Heracles']])
        
        obs_data = obs_data[obs_accreted]

        fig = plot_stars_data([sim_data, obs_data], features=features)
        fig.savefig(f"/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observation_shifted/DataProcessor_before.pdf", 
                    dpi=300, bbox_inches='tight')

        # 3 Transform data        
        sim_data = self.prepare_data_for_Box_Cox(sim_data)
        obs_data = self.prepare_data_for_Box_Cox(obs_data)

        # Apply Box-Cox transformation
        self.pt_sim = PowerTransformer(method="box-cox", standardize=True)
        sim_data[features] = self.pt_sim.fit_transform(sim_data[features].values)

        self.pt_obs = PowerTransformer(method="box-cox", standardize=True)
        obs_data[features] = self.pt_obs.fit_transform(obs_data[features].values)

        # Remove outliers
        sim_data = sim_data[np.logical_and.reduce([sim_data[feature]**2 < 5**2 for feature in features])]
        obs_data = obs_data[np.logical_and.reduce([obs_data[feature]**2 < 5**2 for feature in features])]

        fig = plot_stars_data([sim_data, obs_data], features=features)
        fig.savefig(f"/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observation_shifted/DataProcessor_after.pdf", 
                    dpi=300, bbox_inches='tight')

        end = time.time()
        print(f"DataProcessor initialized in {end-start:.2f} seconds", flush=True)


    
    def process_data_sim(self, sim_df: pd.DataFrame):

        # Prepare data for Box-Cox transformation
        sim_df = self.prepare_data_for_Box_Cox(sim_df)

        # Apply Box-Cox transformation
        sim_df[self.features] = self.pt_sim.transform(sim_df[self.features].values)
        
        return sim_df
    

    def process_data_obs(self, obs_df: pd.DataFrame):

        # Prepare data for Box-Cox transformation
        obs_df = self.prepare_data_for_Box_Cox(obs_df)

        # Apply Box-Cox transformation
        obs_df[self.features] = self.pt_obs.transform(obs_df[self.features].values)

        return obs_df

    def prepare_data_for_Box_Cox(self, df: pd.DataFrame):

        # Prepare data for Box-Cox transformation
        df.dropna(subset=self.features, inplace=True)
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
