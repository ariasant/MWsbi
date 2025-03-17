import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
import torch


def iqr_range(df, variable, factor=1.5):
    """
    Calculate the IQR range for a variable in a dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame - The dataframe containing the variable.
    variable : str - The name of the variable.
    factor : float - The factor to multiply the IQR by to get the range.
        
    Returns
    -------
    lower_bound : float - The lower bound of the range.
    upper_bound : float - The upper bound of the range.
    
    """
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return lower_bound, upper_bound


def transform_variable(feature, transform):

    if transform=="log":
        
        feature -= feature.min()
        feature = np.clip(feature, a_min=1e-2, a_max=np.inf)
        

        return np.log10(feature)
    
    elif transform=="sqrt":

        feature = np.sqrt(feature**2)

        return np.sqrt(feature)
    
    elif transform=="None":

        return feature

    elif transform=="reverse":

        return -feature
    
    else:

        print('Invalid transformation specified ("log", "sqrt", "None", "reverse"). Returning feature')
        return feature
   

class DataProcessor:

    def __init__(self, 
                 features,
                 scaler, 
                 transformation_list, 
                 rm_outliers=False):

        self.scaler = scaler
        self.features_to_scale = features
        self.transformation_list = transformation_list

        self.rm_outliers = rm_outliers
        self.iqr_dict = {}
    
    def apply_transformations(self, 
                              df, 
                              transformations_list):

        df_adjusted = df.copy()

        for (feature, transformation) in transformations_list:

            df_adjusted[feature] = transform_variable(df_adjusted[feature].values, transformation)

        return df_adjusted
        
    def transform_dataframe(self, 
                            df):

        variables_to_transform = [feature for (feature,_) in self.transformation_list]
        other_variables = [feature for feature in df.columns 
                        if feature not in variables_to_transform]

        t = self.transformation_list + [(variable,"None") for variable in other_variables]
        df = self.apply_transformations(df, t)

        return df
    
    def fit(self, 
            df):

        # Transform dataframe
        df = self.transform_dataframe(df)

        if self.rm_outliers:
            # Get inter-quantile ranges per feature
            for feature in self.features_to_scale:
                lower_bound, upper_bound = iqr_range(df, feature, factor=3.0)
                df = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]
                # Save iqr ranges for inference
                self.iqr_dict[feature] = (lower_bound,upper_bound)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(df[self.features_to_scale].values)

        return scaled_data
    
    def transform(self, 
                  df):

        # Transform dataframe
        df = self.transform_dataframe(df)

        if self.rm_outliers:
            for feature in self.features_to_scale:
                lower_bound, upper_bound = self.iqr_dict[feature]
                df = df[(df[feature] > lower_bound) & (df[feature] < upper_bound)]

        if len(df)==0:
            return np.zeros((1,len(self.features_to_scale)))
        
        # Scale data
        scaled_data = self.scaler.transform(df[self.features_to_scale].values)

        return scaled_data



data_dir = "/mnt/aridata1/users/ariasant/MW-sbi/data/"


posterior = pickle.load(open("/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observations/Suite_ELFeHaFe.pkl","rb"))
X_scaler = pickle.load(open("/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observations/X_scaler_Suite_ELFeHaFe.pkl","rb"))
theta_scaler = pickle.load(open("/mnt/aridata1/users/ariasant/auriga-sbi/model_for_observations/theta_scaler_Suite_ELFeHaFe.pkl","rb"))
plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']

features=["E","L","FeH","aFe"]

for substructure in ["GES", "Sagittarius", "Helmi"]:

    # Get dataset
    data = pd.read_pickle(f"{data_dir}{substructure}.pkl")
    data = X_scaler.transform(data)

    data_sample = data[np.random.randint(0,len(data),size=100)].flatten()

    theta_samples = posterior.sample((1000,), 
                            torch.Tensor(data_sample).to(device="cuda"))

    theta_samples = theta_samples.cpu().numpy()
    theta_samples = theta_scaler.inverse_transform(theta_samples)

    fig = corner.corner(theta_samples, 
                        bins=20, 
                        labels=plot_labels,
                        quantiles=[.34,.50,.68],
                        plot_contours=False,
                        show_titles=True,
                        title_kwargs={'fontsize':8},
                        verbose=True)
    fig.set_size_inches(6,6)
    fig.savefig(f"/mnt/aridata1/users/ariasant/MW-sbi/posterior_plots/{substructure}.png",dpi=400)


