import glob
import math
import pickle
import astropy.units as u
from astropy.cosmology import Planck15, z_at_value

f_list = glob.glob("/mnt/aridata1/users/ariasant/auriga-sbi/data/with_satellites/*")

output_dict = {}

for f in f_list:
    df = pickle.load(open(f,"rb"))
    df_nosat = df[df["satellite_flag"]==0]
    m = 0
    for ID in df_nosat["progID"].unique():
        v = df_nosat.loc[df_nosat["progID"]==ID,"log_Mprog_stellar"].values[0]
        t = df_nosat.loc[df_nosat["progID"]==ID,"infall_time"].values[0]
        m += 10**(v)


    k = f[-15:-12]

    output_dict[k] = m

pickle.dump(output_dict, open("/mnt/aridata1/users/ariasant/MW-sbi/auriga_galaxies_stellar_mass.pkl","wb"))