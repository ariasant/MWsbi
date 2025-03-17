import auriga_public.auriga_public as ap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


halo = 6

attrstoload = ['Coordinates', 'Velocities',  'Masses', 'Potential', 'ParticleIDs',
                   'GFM_StellarFormationTime', 'GFM_Metals', 'GFM_StellarPhotometrics']

simulation_dir = '/mnt/aridata1/users/arirgran/Auriga/level4/Original/'

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
OFe_Sun = SUNABUNDANCES['O'] - SUNABUNDANCES['Fe']
MgFe_Sun = SUNABUNDANCES['Mg'] - SUNABUNDANCES['Fe']
NeFe_Sun = SUNABUNDANCES['Ne'] - SUNABUNDANCES['Fe']
SiFe_Sun = SUNABUNDANCES['Si'] - SUNABUNDANCES['Fe']


aFe_Sun2 = np.log10(sum([10**(SUNABUNDANCES['O']-SUNABUNDANCES['Fe']),
                        10**(SUNABUNDANCES['Mg']-SUNABUNDANCES['Fe']),
                        10**(SUNABUNDANCES['Ne']-SUNABUNDANCES['Fe']),
                        10**(SUNABUNDANCES['Si']-SUNABUNDANCES['Fe'])])) 

aFe_Sun1 = sum([OFe_Sun+MgFe_Sun+NeFe_Sun+SiFe_Sun])/4


m_Fe_sim = np.clip(abundances[:,element['Fe']],a_min=1e-10,a_max=1)
m_H_sim = np.clip(abundances[:,element['H']],a_min=1e-10,a_max=1)
# Abundances in dex notation are derived from the ratio of number densities of atoms, 
# hence the mass ratios of each elements need to be converted into number densities: 
#   N*A = m         N is number density, 
#                   A is the atomic number of the element, 
#                   m is the mass of the element in the star
N_Fe = m_Fe_sim / elementnum['Fe']
N_H = m_H_sim / elementnum['H']
N_O = np.clip(abundances[:,element['O']]/elementnum['O'], a_min=1e-10, a_max=1)
N_Mg = np.clip(abundances[:,element['Mg']]/elementnum['Mg'], a_min=1e-10, a_max=1)
N_Ne = np.clip(abundances[:,element['Ne']]/elementnum['Ne'], a_min=1e-10, a_max=1)
N_Si = np.clip(abundances[:,element['Si']]/elementnum['Si'], a_min=1e-10, a_max=1)


# Values are then normalized by the solar value
FeH = np.log10(N_Fe / N_H) - FeH_Sun
OFe = np.log10(N_O / N_Fe) - OFe_Sun
MgFe = np.log10(N_Mg / N_Fe) - MgFe_Sun
NeFe = np.log10(N_Ne / N_Fe) - NeFe_Sun
SiFe = np.log10(N_Si / N_Fe) - SiFe_Sun

aFe1 = sum([OFe,MgFe,NeFe,SiFe])/4
aFe2 = np.log10(sum([N_O,N_Mg,N_Ne,N_Si])/(N_Fe)) - aFe_Sun2


# Plot distribution of the single abundances

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



