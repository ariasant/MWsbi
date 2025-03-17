import auriga_public.auriga_public as ap
from auriga_mergers_utils import R200_dict
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


def get_alpha_Fe_abundances(snapobj):

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


    aFe_Sun = np.log10(sum([10**(SUNABUNDANCES['O']-SUNABUNDANCES['Fe']),
                            10**(SUNABUNDANCES['Mg']-SUNABUNDANCES['Fe']),
                            10**(SUNABUNDANCES['Ne']-SUNABUNDANCES['Fe']),
                            10**(SUNABUNDANCES['Si']-SUNABUNDANCES['Fe'])]))
    
    #aFe_Sun = sum([OFe_Sun+MgFe_Sun+NeFe_Sun+SiFe_Sun])/4
 

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

    #aFe = sum([OFe,MgFe,NeFe,SiFe])/4
    aFe = np.log10(sum([N_O,N_Mg,N_Ne,N_Si])/(N_Fe)) - aFe_Sun

    return MgFe+0.4, FeH

def get_zeropoint_potential(galaxy_num, sim_dir='/mnt/aridata1/users/arirgran/Auriga/level4/Original/'):
    # Get zero point of potential
    data_directory = '{}halo_{}/'.format(sim_dir, galaxy_num)
    # Define properties to load                                                 
    attrstoload = ['Coordinates', 'Velocities', 'GFM_StellarFormationTime', 'GFM_InitialMass', 'Masses', 'ParticleIDs', 'Potential']
    # Load snapshot data for DM                                                       
    snapobj = ap.snapshot.load_snapshot(127, 1, loadlist=attrstoload, snappath=data_directory, verbose=False)
    # Load subfind information                                                  
    subobj = ap.subhalos.subfind(127, directory=data_directory, loadlist=['SubhaloPos', 'Group_R_Crit200'])
    # Centre on main halo                                                       
    snapobj = ap.util.CentreOnHalo(snapobj, subobj.data['SubhaloPos'][0])

    r200 = R200_dict[galaxy_num]*1e-3

    r = np.clip(np.sqrt( np.sum(snapobj.data['Coordinates']**2,axis=1) ),
                a_max=100,
                a_min=1e-6)

    potential_energy = snapobj.data['Potential']

    bin_potential, bin_edges, bin_number = stats.binned_statistic(x=r,
                                                                  values=potential_energy,
                                                                  statistic='median',
                                                                  bins=100,
                                                                  range=[0,3*r200])

    zero_potential = bin_potential[-1]

    return zero_potential


simulation_dir = '/mnt/aridata1/users/arirgran/Auriga/level4/Original/'

# Load APOGEE stars chemical abundances
df = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_ds_min.pkl")
aFe = list(df["aFe"].values)
FeH = list(df["FeH"].values)
E = list(df["E"].values)
L = list(df["L"].values)
ds_ID = ["APOGEE" for i in range(len(aFe))]

# Load chemical abundances from some of the Auriga halos
halos = [6,16,21]
for halo in halos:

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

    r = np.sqrt(np.sum(snapobj.data["Coordinates"]**2,axis=1))
    aFe_halo, FeH_halo = get_alpha_Fe_abundances(snapobj)

    mdata = ap.util.read_starparticle_mergertree_data_hdf5(127, '{}lists/accretedstardata/'.format(simulation_dir, halo), 
                                                        'halo_{}'.format(halo))

    ZeroPotential = get_zeropoint_potential(halo, sim_dir=simulation_dir)
    potential_energy = snapobj.data['Potential'] - ZeroPotential
    E_halo = potential_energy + 0.5*np.sum(snapobj.data['Velocities']**2, axis=1)

    L_halo = np.sqrt(np.sum(np.cross( snapobj.data['Coordinates']*1e3, 
                               (snapobj.data['Velocities'] ) )**2, axis=1) ) #kpc km/s


    # Get rid of noisy stars and stars from the inner 2kpc
    mask = np.logical_and.reduce([(FeH_halo>-3), (FeH_halo<1), 
                                  (aFe_halo>-0.5), (aFe_halo<1),
                                  (r>0.002)
                                  ])
    aFe_halo = aFe_halo[mask]
    FeH_halo = FeH_halo[mask]
    E_halo = E_halo[mask]
    L_halo = L_halo[mask]

    # Sample only maximum 50,000 stars
    idx = np.random.randint(0, high=len(E_halo), size=50000)

    aFe += list(aFe_halo[idx])
    FeH += list(FeH_halo[idx])

    E += list(E_halo[idx])
    L += list(L_halo[idx])

    ds_ID += [f"Au{halo}" for i in range(len(E_halo[idx]))]

    print(halo)



def plot(x, 
         y, 
         labels, 
         xlabel, 
         ylabel, 
         xlim, 
         ylim,
         filename):

    # Combine into a DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'Dataset': labels
    })

    # Create figure with a grid for heatmap and marginal plots
    fig = plt.figure(figsize=(5, 5))
    grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)

    # Main heatmap
    ax_main = fig.add_subplot(grid[1:, :-1])
    ax_top = fig.add_subplot(grid[0, :-1])
    ax_right = fig.add_subplot(grid[1:, -1])


    # Plot heatmap
    sns.histplot(data=df, x="x", y="y", hue="Dataset", bins=200, alpha=0.5, ax=ax_main, binrange=[xlim,ylim])
    #sns.kdeplot(data=df, x="x", y="y", hue="Dataset", fill=False, ax=ax_main, clip=[xlim,ylim], levels=[0.1, 0.5, 0.9])
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)

    sns.move_legend(ax_main, "upper left", bbox_to_anchor=(1.05, 1.45))

    # Marginal distributions
    sns.kdeplot(data=df, x="x", hue="Dataset", ax=ax_top, fill=True, common_norm=False, legend=False, clip=xlim)
    sns.kdeplot(data=df, y="y", hue="Dataset", ax=ax_right, fill=True, common_norm=False, legend=False, clip=ylim)

    # Hide ticks on marginal plots
    ax_top.set_xticklabels([])
    ax_right.set_yticklabels([])
    ax_right.set_ylabel("")
    ax_top.set_xlabel("")
    ax_top.set_ylabel("Density")

    fig.savefig(f"{filename}.png", dpi=400)

plot(FeH,
     aFe,
     labels=ds_ID,
     xlabel="[Fe/H]",
     ylabel="[Mg/Fe]",
     xlim=[-2,1],
     ylim=[-0.2,0.6],
     filename="/mnt/aridata1/users/ariasant/MW-sbi/sim_vs_obs_chem_Mg")


plot(FeH[df.shape[0]:],
     aFe[df.shape[0]:],
     labels=ds_ID[df.shape[0]:],
     xlabel="[Fe/H]",
     ylabel="[Mg/Fe]",
     xlim=[-2,1],
     ylim=[-0.2,0.6],
     filename="/mnt/aridata1/users/ariasant/MW-sbi/sim_chem_Mg")


plot([L[i]/1000 for i in range(len(L))],
     [E[i]/10000 for i in range(len(E))],
     labels=ds_ID,
     xlabel="$L \; [10^{3}\, \\times \, \mathrm{kpc} \,\mathrm{km}\mathrm{s}^{-1}]$",
     ylabel="$E \; [10^{4}\, \\times \, \mathrm{kpc}^{2} \, \mathrm{km}^{2]\mathrm{s}^{-2}}]$",
     xlim=[0,5],
     ylim=[-25,0],
     filename="/mnt/aridata1/users/ariasant/MW-sbi/sim_vs_obs_dyn")
