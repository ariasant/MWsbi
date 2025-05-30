import corner
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors
import torch

def plot_ax(x,y,
            ax,
            bin_number=500,
            extent=None,
            cmap='cividis'):
            

    counts,x_edges,y_edges = np.histogram2d(x, y, 
                                            bins=(bin_number,bin_number), 
                                            range=extent)
    

    norm = mpl.colors.LogNorm(vmin=1, vmax=200)
        

    xbins_c = 0.5 * (x_edges[:-1]+x_edges[1:])
    ybins_c = 0.5 * (y_edges[:-1]+y_edges[1:])

    xc, yc = np.meshgrid(xbins_c, ybins_c)

    plot = ax.imshow(counts.T,
              cmap=cmap,
              norm=norm,
              origin='lower',
              extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
              interpolation='nearest',
              aspect='equal')


    return plot



model_dir = "/mnt/aridata1/users/ariasant/MW-sbi/simple_shift/with_satellites/"
output_dir = "/mnt/aridata1/users/ariasant/MW-sbi/simple_shift/with_satellites/"

features=["E","L","FeH","MgFe"]

n_samples = 1000

# Posterior model
posterior = pickle.load(open(f"{model_dir}Suite_ELFeHMgFe.pkl","rb"))

# Data processing tools
theta_scaler = pickle.load(open(f"{model_dir}theta_scaler_Suite_ELFeHMgFe.pkl","rb")) # progenitor properties scaler

plot_labels=['$\\tau \, [\mathrm{Gyr}]$',
             'log($M_{*}/M_{\odot}$)',
             'log($M/M_{\odot}$)', 
             'MMR (log)']

# Load pre-processed apogee sample
df = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")
print(f"N stars in APOGEE ds: {len(df)}", flush=True)

# Select accreted stars
substructures = [k for k in df.keys() if "_flag" in k]
substructures.remove("Nyx_flag")
substructures.remove("Aleph_flag")
chem_cuts = ((df.AlFe<-0.07) & (df.MgMn>=0.25)) | ((df.AlFe>=-0.07) & (df.MgMn>=4.25*df.AlFe+0.5475))
is_accreted = np.logical_or.reduce([df[f"{s}"]==1 for s in substructures])
df_accreted = df[chem_cuts | is_accreted].copy()
print(f"N accreted stars in APOGEE ds: {len(df_accreted)}", flush=True)


xlim = [-2.5,1]
ylim = [-0.25,0.6]


# Plot the accreted stars in the [Fe/H] vs [Mg/Fe] plane
fig,ax = plt.subplots(ncols=1, 
                       nrows=1, 
                       figsize=(8,8))


plot = plot_ax(x=df['FeH'],
               y=df['MgFe'],
               ax=ax,
               extent=[xlim, ylim],
               cmap="gist_gray")
    

ax.scatter(x=df_accreted['FeH'],
           y=df_accreted['MgFe'], 
           s=10,
           marker="o",
           color="white",
           edgecolor='k',
           linewidth=0.3,
           alpha=0.5)

    
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect((xlim[1]-xlim[0])/(ylim[1]-ylim[0]))
    
ax.set_ylabel("$[\\alpha/\mathrm{Fe}]$")
ax.set_xlabel("$[\mathrm{Fe}/\mathrm{H}]$")


cbar = fig.colorbar(plot, ax=ax, shrink=0.8, orientation="horizontal")
cbar.set_label('Number of Stars')

fig.savefig(f"{output_dir}alphaIron_accreted.png", dpi=1000)
#####################################################################################
#####################################################################################

df_accreted.dropna(axis=0, subset=features, inplace=True)

# Identify the 100-nearest neighbors in the [Fe/H] vs [Mg/Fe] plane for a point
NNs_model = NearestNeighbors(n_neighbors=100, algorithm='auto')
NNs_model.fit(df_accreted[features].values)

# Find the 100 nearest neighbours for each accreted star
idx_neighbours = NNs_model.kneighbors(df_accreted[features].values, 100, return_distance=False)

group_properties = []

for idx in idx_neighbours:

    NN_data = df.loc[idx, features].values

    # Sample 1000 times from the posterior conditioned on the nearest-neighbours
    theta_samples = posterior.sample((n_samples,), 
                                      torch.Tensor(NN_data).to(device="cuda"))

    theta_samples = theta_samples.cpu().numpy()
    theta_samples = theta_scaler.inverse_transform(theta_samples)

    group_properties.append(theta_samples)

pickle.dump(group_properties, open(f"{output_dir}group_properties_accreted.pkl", "wb"))

# Plot the corner plot for the group properties
group_properties = np.concatenate(group_properties, axis=0)
fig = corner.corner(group_properties, 
                     labels=plot_labels, 
                     quantiles=[0.16, 0.5, 0.84],
                     show_titles=True,
                     title_kwargs={"fontsize": 12},
                     bins=50,
                     smooth=True,
                     smooth1d=True,
                     plot_datapoints=False)
fig.savefig(f"{output_dir}group_properties_accreted_corner.png", dpi=1000)

