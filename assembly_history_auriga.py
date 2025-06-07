from astropy.cosmology import Planck15
import auriga_public.auriga_public as ap
import math
import pickle

def correct_unphysical_drops(series, threshold=0.95):
    
    corrected_series = series.copy()

    for i in range(1, len(series)-1):

        if series[i] < threshold * series[i - 1]:  # Detect abrupt drop
            corrected_series[i] = (series[i-1] + series[i+1])/2
    
    return corrected_series

simulation_dir = '/mnt/aridata1/users/arirgran/Auriga/level4/Original/'

# Assembly history dictionary
output_dict = {}

for galaxy_num in range(1,31):

    treeobj = ap.mergertree.load_mergertree(0, 0, 127, 
                                            directory='{}mergertrees/halo_{}/'.format(simulation_dir, galaxy_num))

    tree_IDs_MW = treeobj.ReturnFullBranchGivenTreeIndex(0, truncate_mainbranch=False)
    redshifts, subhalo_mass_MW_at_snap=[],[]
    treeobj.ReturnTreeNode(0, field='Redshift', data_out=redshifts, mainprogonly=True)
    treeobj.ReturnTreeNode(0, field='SubhaloMassType', data_out=subhalo_mass_MW_at_snap, mainprogonly=True) #10^10 M_sun

    subhalo_mass_MW_at_snap = [sum(masses) for masses in subhalo_mass_MW_at_snap]
    
    # Correct unphysical drops in mass
    subhalo_mass_MW_at_snap = correct_unphysical_drops(subhalo_mass_MW_at_snap, threshold=0.95)

    # Mass of the MW vs redshift
    subhalo_mass_MW_at_snap = [math.log10(masses)+10 for masses in subhalo_mass_MW_at_snap] 

    # Loolback time
    time = [Planck15.lookback_time(z).value for z in redshifts]

    output_dict[f"{galaxy_num}"] = (time, subhalo_mass_MW_at_snap)

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

    subhalo_mass_MW_at_snap = [sum(masses) for masses in subhalo_mass_MW_at_snap]

    # Correct unphysical drops in mass
    subhalo_mass_MW_at_snap = correct_unphysical_drops(subhalo_mass_MW_at_snap, threshold=0.95)

    # Mass of the MW vs redshift
    subhalo_mass_MW_at_snap = [math.log10(masses)+10 for masses in subhalo_mass_MW_at_snap] 

    # Loolback time
    time = [Planck15.lookback_time(z).value for z in redshifts]

    output_dict[f"L{galaxy_num}"] = (time, subhalo_mass_MW_at_snap)

    print(f"Processed galaxy {galaxy_num}.")

# Save assembly histories
pickle.dump(output_dict, open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_galaxies_mass_assembly.pkl","wb"))




