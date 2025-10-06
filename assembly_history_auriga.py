from astropy.cosmology import Planck15
import auriga_public.auriga_public as ap
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import pickle

def correct_unphysical_drops(series, threshold=0.95, trend_size=10):
    
    corrected_series = series.copy()

    for i in range(1, len(series)-1):

        # Consider the mass estimate at the 10 time steps before
        mass_trend = series[i:i + trend_size]

        if series[i] < threshold * max(mass_trend):  # Detect abrupt drop
            corrected_series[i] = max(mass_trend)  # Replace with the maximum of the trend
    
    return corrected_series




simulation_dir = '/mnt/aridata1/users/arirgran/Auriga/level4/Original/'

# Assembly history dictionary
output_dict = {}
output_dict_stellar = {}
plot_dir = '/mnt/aridata1/users/ariasant/MW-sbi/assembly_history_plots/'

for galaxy_num in range(1,31):

    treeobj = ap.mergertree.load_mergertree(0, 0, 127, 
                                            directory='{}mergertrees/halo_{}/'.format(simulation_dir, galaxy_num))

    tree_IDs_MW = treeobj.ReturnFullBranchGivenTreeIndex(0, truncate_mainbranch=False)
    redshifts, subhalo_mass_MW_at_snap=[],[]
    treeobj.ReturnTreeNode(0, field='Redshift', data_out=redshifts, mainprogonly=True)
    treeobj.ReturnTreeNode(0, field='SubhaloMassType', data_out=subhalo_mass_MW_at_snap, mainprogonly=True) #10^10 M_sun

    subhalo_mass_MW_at_snap = [masses for masses in subhalo_mass_MW_at_snap]

    # Mass of the MW vs redshift
    stellar_mass_MW_at_snap = [math.log10(masses[4]+1e-10)+10 for masses in subhalo_mass_MW_at_snap]
    subhalo_mass_MW_at_snap = [math.log10(sum(masses))+10 for masses in subhalo_mass_MW_at_snap] 
    

    # Loolback time
    time = np.array([Planck15.lookback_time(z).value for z in redshifts])

    fig, ax = plt.subplots()
    ax.plot(time, subhalo_mass_MW_at_snap, label=f"original")
    ax.plot(time, medfilt(subhalo_mass_MW_at_snap, kernel_size=11), label="smoothed")

    # Correct unphysical drops in mass
    stellar_mass_MW_at_snap = correct_unphysical_drops(stellar_mass_MW_at_snap, threshold=0.99)
    subhalo_mass_MW_at_snap = correct_unphysical_drops(subhalo_mass_MW_at_snap, threshold=0.99)
    

    ax.plot(time, subhalo_mass_MW_at_snap, label=f"corrected")
    ax.legend()
    fig.savefig(f"{plot_dir}MW_{galaxy_num}.png")


    output_dict[f"{galaxy_num}"] = (time, subhalo_mass_MW_at_snap)
    output_dict_stellar[f"{galaxy_num}"] = (redshifts, stellar_mass_MW_at_snap)

    print(f"Processed galaxy {galaxy_num}.")

# Repeat for low mass Milky Ways
simulation_dir = '/mnt/aridata1/users/arirgran/Auriga/level4/LowMassMWs/'

for galaxy_num in range(1,11):

    if galaxy_num==4:
        continue

    treeobj = ap.mergertree.load_mergertree(0, 0, 127, 
                                            directory='{}mergertrees/halo_{}/'.format(simulation_dir, galaxy_num))

    tree_IDs_MW = treeobj.ReturnFullBranchGivenTreeIndex(0, truncate_mainbranch=False)
    redshifts, subhalo_mass_MW_at_snap=[],[]
    treeobj.ReturnTreeNode(0, field='Redshift', data_out=redshifts, mainprogonly=True)
    treeobj.ReturnTreeNode(0, field='SubhaloMassType', data_out=subhalo_mass_MW_at_snap, mainprogonly=True) #10^10 M_sun

    subhalo_mass_MW_at_snap = [masses for masses in subhalo_mass_MW_at_snap]

    # Mass of the MW vs redshift
    stellar_mass_MW_at_snap = [math.log10(masses[4]+1e-10)+10 for masses in subhalo_mass_MW_at_snap]
    subhalo_mass_MW_at_snap = [math.log10(sum(masses))+10 for masses in subhalo_mass_MW_at_snap] 
    

    # Loolback time
    time = [Planck15.lookback_time(z).value for z in redshifts]

    fig, ax = plt.subplots()
    ax.plot(time, subhalo_mass_MW_at_snap, label=f"original")
    ax.plot(time, medfilt(subhalo_mass_MW_at_snap, kernel_size=11), label="smoothed")

    # Correct unphysical drops in mass
    subhalo_mass_MW_at_snap = correct_unphysical_drops(subhalo_mass_MW_at_snap, threshold=0.99)
    stellar_mass_MW_at_snap = correct_unphysical_drops(stellar_mass_MW_at_snap, threshold=0.99)

    ax.plot(time, subhalo_mass_MW_at_snap, label=f"corrected")
    ax.legend()
    fig.savefig(f"{plot_dir}low_mass_MW_{galaxy_num}.png")

    output_dict[f"L{galaxy_num}"] = (time, subhalo_mass_MW_at_snap)
    output_dict_stellar[f"{galaxy_num}"] = (redshifts, stellar_mass_MW_at_snap)

    print(f"Processed galaxy {galaxy_num}.")

# Save assembly histories
pickle.dump(output_dict, open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_galaxies_mass_assembly.pkl","wb"))
pickle.dump(output_dict_stellar, open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_galaxies_stellar_mass_assembly.pkl","wb"))




