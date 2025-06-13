import auriga_public.auriga_public as ap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns


def load_snapshot(halo: int,
                  simulation_dir: str):
    attrstoload = ['Coordinates', 'Velocities',  'Masses', 'Potential', 'ParticleIDs',
                    'GFM_StellarFormationTime', 'GFM_Metals', 'GFM_StellarPhotometrics']

    # Load snapshot data                                                        
    snapobj = ap.snapshot.load_snapshot(127, 
                                        4, # PartType=4 for stars
                                        loadlist=attrstoload, 
                                        snappath=f"{simulation_dir}halo_{halo}/", 
                                        verbose=False)
    # Load subfind information                                                  
    subobj = ap.subhalos.subfind(127, 
                                directory=f"{simulation_dir}halo_{halo}/", 
                                loadlist=['SubhaloPos', 'Group_R_Crit200'])
    # Centre on MW                                                       
    snapobj = ap.util.CentreOnHalo(snapobj, subobj.data['SubhaloPos'][0])
    # Remove bulk velocity so that velocities are 0 at centre of galaxy         
    bulk_velocity = ap.util.remove_bulk_velocity(snapobj, 
                                                 idx=None, 
                                                 radialcut=0.1*subobj.data['Group_R_Crit200'][0])
    # Select only stars within 30kpc  
    snapobj = ap.util.apply_mask(snapobj, 
                                stars=True, 
                                radialcut=subobj.data['Group_R_Crit200'][0])
    # Rotate galaxy to align z-axis to direction of total angular momentum of galaxy                                                                           
    ap.util.align_galaxy(snapobj)

    return snapobj

def get_abundance_ratios(snapobj):
    #####
    # Calculate chemical abundance ratio with two methods
    # 1. alpha/Fe = (O/Fe + Mg/Fe + Ne/Fe + Si/Fe ) / 4
    # 2. alpha/Fe = log10( (N_O + N_Mg + N_Ne + N_Si)/4 N_Fe ) - same for solar 
    #####


    # Abundances in the snapshot are read as mass ratios, need to convert them to mass densities multiplying by their atomic mass
    abundances = snapobj.data['GFM_Metals']
    element = {'H':0, 'He':1, 'C':2, 'N':3, 'O':4, 'Ne':5, 'Mg':6, 'Si':7, 'Fe':8}
    elementnum = {'H':1, 'He':4, 'C':12, 'N':14, 'O':16, 'Ne':20, 'Mg':24, 'Si':28, 'Fe':56}
    #from Asplund et al. (2009) Table 5
    SUNABUNDANCES = {'H':12.0, 'He':10.98, 'C':8.47, 'N':7.87, 'O':8.73, 'Ne':7.97, 'Mg':7.64, 'Si':7.55, 'Fe':7.54}


    FeH_Sun = SUNABUNDANCES['Fe']-SUNABUNDANCES['H']
    MgFe_Sun = SUNABUNDANCES['Mg'] - SUNABUNDANCES['Fe']

    m_Fe_sim = np.clip(abundances[:,element['Fe']],a_min=1e-10,a_max=1)
    m_H_sim = np.clip(abundances[:,element['H']],a_min=1e-10,a_max=1)
    # Abundances in dex notation are derived from the ratio of number densities of atoms, 
    # hence the mass ratios of each elements need to be converted into number densities: 
    #   N*A = m         N is number density, 
    #                   A is the atomic number of the element, 
    #                   m is the mass of the element in the star
    N_Fe = m_Fe_sim / elementnum['Fe']
    N_H = m_H_sim / elementnum['H']
    N_Mg = np.clip(abundances[:,element['Mg']]/elementnum['Mg'], a_min=1e-10, a_max=1)

    # Values are then normalized by the solar value
    FeH = np.log10(N_Fe / N_H) - FeH_Sun
    MgFe = np.log10(N_Mg / N_Fe) - MgFe_Sun

    return FeH, MgFe

def select_disc_stars(snapobj):

    R = np.clip(np.sqrt(np.sum(snapobj.data["Coordinates"][:,1:]**2,axis=1)), a_min=1e-6, a_max=1e6)
    vtheta = -(snapobj.data["Velocities"][:,1]*snapobj.data["Coordinates"][:,2]-snapobj.data["Velocities"][:,2]*snapobj.data["Coordinates"][:,1]) / R
    vr = (snapobj.data["Velocities"][:,2]*snapobj.data["Coordinates"][:,2] + snapobj.data["Velocities"][:,1]*snapobj.data["Coordinates"][:,1]) / R
    
    v = np.vstack([vr,vtheta,snapobj.data["Velocities"][:,0]]).T

    dummy_v = np.zeros((len(v),3))
    dummy_v[:,1] = 200

    disc_stars_idx = np.where(np.sum((v-dummy_v)**2,axis=1)<=200**2)[0]

    print(f"N disc stars: {len(disc_stars_idx):,}", flush=True)

    return disc_stars_idx

simulation_dir_original = '/mnt/aridata1/users/arirgran/Auriga/level4/Original/'
simulation_dir_low_mass = '/mnt/aridata1/users/arirgran/Auriga/level4/LowMassMWs/'

# Collect [Fe/H] and [alpha/Fe] abundance ratios for the disk stars in the MW anlogues in the Auriga suite
FeH_dict = {}
MgFe_dict = {}

for halo in range(1,31):

    snapobj = load_snapshot(halo=halo,
                            simulation_dir=simulation_dir_original)
    
    FeH, MgFe = get_abundance_ratios(snapobj=snapobj)

    # Select disc stars
    disc_idx = select_disc_stars(snapobj=snapobj)

    # Quick and dirty plot to see if the right selection was done
    if halo==27:
        R = np.sqrt(np.clip(np.sum(snapobj.data["Coordinates"][:,1:]**2,axis=1),a_min=1e-6,a_max=1e6)) #kpc
        vtheta = -(snapobj.data["Velocities"][:,1]*snapobj.data["Coordinates"][:,2]-snapobj.data["Velocities"][:,2]*snapobj.data["Coordinates"][:,1]) / R
        vsigma = np.sqrt(np.sum(snapobj.data["Velocities"]**2,axis=1)-vtheta**2)
        fig,ax = plt.subplots()
        ax.hist2d(x=vtheta[disc_idx],
                  y=vsigma[disc_idx],
                  bins=300,
                  range=[[-300,300],[0,400]])
        fig.savefig("/mnt/aridata1/users/ariasant/MW-sbi/Auriga_disc_selection.pdf")

    FeH_dict[f"Au{halo}"] = FeH[disc_idx]
    MgFe_dict[f"Au{halo}"] = MgFe[disc_idx]

    print(halo, flush=True)

for halo in range(1,11):

    if halo==4:
        continue

    snapobj = load_snapshot(halo=halo,
                            simulation_dir=simulation_dir_low_mass)
    
    FeH, MgFe = get_abundance_ratios(snapobj=snapobj)

    # Select disc stars
    disc_idx = select_disc_stars(snapobj=snapobj)

    FeH_dict[f"L{halo}"] = FeH[disc_idx]
    MgFe_dict[f"L{halo}"] = MgFe[disc_idx]

    print(halo, flush=True)
   
# Save dictionaries
pickle.dump(FeH_dict, open("/mnt/aridata1/users/ariasant/MW-sbi/Auriga_FeH_dict.pkl","wb"))
pickle.dump(MgFe_dict, open("/mnt/aridata1/users/ariasant/MW-sbi/Auriga_MgFe_dict.pkl","wb"))

##############################################################################################
# Load MW stars
apogee_ds = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds.pkl")

# Select disc stars as done in he simulations
v = apogee_ds[["vr","vtheta","vz"]].values
dummy_v = np.zeros((len(v),3))
dummy_v[:,1] = 200

disc_idx = np.where(np.sum((v-dummy_v)**2,axis=1)<=200**2)[0]

# Plot selection of disc stars
fig,ax = plt.subplots()
ax.hist2d(apogee_ds.iloc[disc_idx]["FeH"],
          apogee_ds.iloc[disc_idx]["MgFe"],
          bins=200,
          range=[[-3,1],[-0.2,0.6]])
fig.savefig("/mnt/aridata1/users/ariasant/MW-sbi/MW_disc_selection.pdf")

"""# Plot distribution of the single abundances

df_list = []
for alpha,alpha_label in zip([OFe,MgFe,NeFe,SiFe],["[O/Fe]","[Mg/Fe]","[Ne/Fe]","[Si/Fe]"]):

    df_el = pd.DataFrame({'x':FeH,
                       'y':alpha,
                       'Ratio':[alpha_label for i in range(len(FeH))]})
    
    df_list.append(df_el)

df = pd.concat(df_list, ignore_index=True)

# Create figure with a grid for heatmap and marginal plots
fig = plt.figure(figsize=(5, 5))
grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)

# Main heatmap
ax_main = fig.add_subplot(grid[1:, :-1])
ax_top = fig.add_subplot(grid[0, :-1])
ax_right = fig.add_subplot(grid[1:, -1])

xlim=[-2,1]
ylim=[-0.5,0.5]

# Plot heatmap
sns.histplot(data=df, x="x", y="y", hue="Ratio", bins=200, alpha=0.5, ax=ax_main, binrange=[xlim,ylim])
ax_main.set_xlabel("[Fe/H]")
ax_main.set_ylabel("[$\\alpha$/Fe]")

sns.move_legend(ax_main, "upper left", bbox_to_anchor=(1.05, 1.45))

# Marginal distributions
sns.kdeplot(data=df, x="x", hue="Ratio", ax=ax_top, fill=True, common_norm=False, legend=False, clip=xlim)
sns.kdeplot(data=df, y="y", hue="Ratio", ax=ax_right, fill=True, common_norm=False, legend=False, clip=ylim)

# Hide ticks on marginal plots
ax_top.set_xticklabels([])
ax_right.set_yticklabels([])
ax_right.set_ylabel("")
ax_top.set_xlabel("")
ax_top.set_ylabel("Density")

fig.savefig("/mnt/aridata1/users/ariasant/MW-sbi/Ratio_comparison.png", dpi=400)


## Plot only the distributions
# Define bin edges for x-axis ([Fe/H])
num_bins = 50  # Adjust for more or fewer bins
bin_edges = np.linspace(-2, 1, num_bins + 1)  # Same range as xlim

# Create figure
fig, ax_main = plt.subplots(figsize=(6, 5))

# Colors for each Ratio
palette = sns.color_palette("tab10", len(df["Ratio"].unique()))

# Process each Ratio separately
for i, (ratio, color) in enumerate(zip(df["Ratio"].unique(), palette)):
    subset = df[df["Ratio"] == ratio]
    
    # Bin data
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    median_values = []
    perc_5 = []
    perc_95 = []
    
    for j in range(len(bin_edges) - 1):
        bin_data = subset[(subset["x"] >= bin_edges[j]) & (subset["x"] < bin_edges[j + 1])]["y"]
        if len(bin_data) > 0:
            median_values.append(np.median(bin_data))
            perc_5.append(np.percentile(bin_data, 5))
            perc_95.append(np.percentile(bin_data, 95))
        else:
            median_values.append(np.nan)
            perc_5.append(np.nan)
            perc_95.append(np.nan)

    # Plot median line
    ax_main.plot(bin_centers, median_values, label=ratio, color=color)

    # Plot shaded region for percentiles
    ax_main.fill_between(bin_centers, perc_5, perc_95, color=color, alpha=0.2)

# Labels
ax_main.set_xlabel("[Fe/H]")
ax_main.set_ylabel("[$\\alpha$/Fe]")
ax_main.legend(title="Ratio", loc="upper center", ncols=4)

# Save figure
fig.savefig("/mnt/aridata1/users/ariasant/MW-sbi/Ratio_comparison_updated.png", dpi=400)


#########################################################################################
#########################################################################################
#########################################################################################
## Repeat for number density of elements
df_list = []
for alpha,alpha_label in zip([N_O, N_Mg, N_Ne, N_Si],["O","Mg","Ne","Si"]):

    df_el = pd.DataFrame({'x':FeH,
                       'y':alpha,
                       'Element':[alpha_label for i in range(len(FeH))]})
    
    df_list.append(df_el)

df = pd.concat(df_list, ignore_index=True)

df["y"] = np.log10(df["y"].values)

# Create figure with a grid for heatmap and marginal plots
fig, ax_main = plt.subplots(figsize=(4,4))
xlim=[-8,-2]
# Plot heatmap
sns.kdeplot(data=df, x="y", hue="Element", ax=ax_main, fill=True, common_norm=False, legend=True, clip=xlim)
ax_main.set_xlabel("Number Density (log)")
ax_main.set_ylabel("")

fig.savefig("/mnt/aridata1/users/ariasant/MW-sbi/N_comparison.png", dpi=400)



"""