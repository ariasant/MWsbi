from apo_tools.galcoords import Galcoords
from astropy.coordinates import SkyCoord, Galactocentric, Galactic
import astropy.units as u
from astropy.io import fits
from galpy.orbit import Orbit
from galpy.potential.mwpotentials import McMillan17
from galpy.potential import vcirc, evaluatePotentials
from galpy.util.conversion import get_physical
import numpy as np
import pandas as pd
import time

def read_fits_file(filepath):

    # Load file data
    file = fits.open(filepath)

    return file[1].data

def get_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in: {end - start:.6f} seconds", flush=True)
        return result
    return wrapper


@get_time
def actions(idx_stars): 
    
    phi = np.arctan(df.loc[idx_stars,'y'].values / df.loc[idx_stars,'x'].values)
    o = Orbit([df.loc[idx_stars,'r'].values*u.kpc, 
               df.loc[idx_stars, 'vr'].values*u.km/u.s, 
               df.loc[idx_stars, 'vtheta'].values*u.km/u.s, 
               df.loc[idx_stars, 'z'].values*u.kpc, 
               df.loc[idx_stars, 'vz'].values*u.km/u.s,
               phi*u.rad,], 
               **get_physical(McMillan17),
               zo=0.02,
               solarmotion=solar_motion)
    
    Jp = o.jp(pot=McMillan17)
    Jr = o.jr(pot=McMillan17)
    Jz = o.jz(pot=McMillan17)

    return Jp, Jr, Jz

@get_time
def circularity(idx_stars):


    o = Orbit([df.loc[idx_stars,'r'].values*u.kpc, 
               df.loc[idx_stars, 'vr'].values*u.km/u.s, 
               df.loc[idx_stars, 'vtheta'].values*u.km/u.s, 
               df.loc[idx_stars, 'z'].values*u.kpc, 
               df.loc[idx_stars, 'vz'].values*u.km/u.s], 
               **get_physical(McMillan17),
               zo=0.02,
               solarmotion=solar_motion)

    # Define circularity parameter
    # Calculate radius of a star in a circular orbit with energy E
    try:
        R_circ = o.rE(pot=McMillan17)  # Galactocentric radius 
        Lz_circ = R_circ * vcirc(McMillan17, R_circ*u.kpc)  # Lz_circ = R * v_circular
    except ValueError:
        # Calculate eta analytically
        R_circ = np.sqrt((df.loc[idx_stars,"r"]**2 + df.loc[idx_stars, "z"]**2))
        Lz_circ = R_circ*np.sqrt((df.loc[idx_stars, "vx"]**2+df.loc[idx_stars, "vy"]**2+df.loc[idx_stars, "vz"]**2))

    return df.loc[idx_stars, "Lz"] / Lz_circ

# Calculate eccentricity of orbits and actions
@get_time
def eccentricity(idx_stars): 

    o = Orbit([df.loc[idx_stars,'r'].values*u.kpc, 
               df.loc[idx_stars, 'vr'].values*u.km/u.s, 
               df.loc[idx_stars, 'vtheta'].values*u.km/u.s, 
               df.loc[idx_stars, 'z'].values*u.kpc, 
               df.loc[idx_stars, 'vz'].values*u.km/u.s,], 
               **get_physical(McMillan17),
               zo=0.02,
               solarmotion=solar_motion)

    return o.e(analytic=True, type='staeckel', pot=McMillan17)

def save_data(mask, filename):

    # Add selection flag to dataframe
    key = f"{filename}_flag"
    df[key] = np.zeros(df.shape[0], dtype=int)
    df.loc[mask, key] = 1

    # Save subsample as a separate dataframe
    df_prog = df.loc[mask]
    df_prog.to_pickle(f"/mnt/aridata1/users/ariasant/MW-sbi/data/{filename}.pkl")

    print(f"Saved {filename} selection. N stars: {len(df_prog):,}", flush=True)


def get_EL_err(ID_star,
              n_samples=500):



    # Get phase-space coordinates of the star
    idx = data["APOGEE_ID"]==ID_star

    if sum(idx)>1: # some stars have double entries in the fits file
        first_true_idx = np.where(idx)[0][0]
        new_idx = np.full(idx.shape, False)
        new_idx[first_true_idx] = True
        idx = new_idx

    # Define distribution with possible phase-space properties of star

    distance_dist = (rng.normal(distances[idx], distances_err[idx], n_samples)* u.pc)

    pm_ra_cosdec_dist = (rng.normal(data["GAIAEDR3_PMRA"][idx], data["GAIAEDR3_PMRA_ERROR"][idx], n_samples) * u.mas/u.yr)

    pm_dec_dist = (rng.normal(data["GAIAEDR3_PMDEC"][idx], data["GAIAEDR3_PMDEC_ERROR"][idx], n_samples) * u.mas/u.yr)

    rv_dist = (rng.normal(data["VHELIO_AVG"][idx], data["VERR"][idx], n_samples) * u.km/u.s)

    ra = np.full(n_samples, data["RA"][idx]) * u.degree
    dec = np.full(n_samples, data["DEC"][idx]) * u.degree

    # Convert to Galactocentric coordinates
    c1 = SkyCoord(ra=ra, 
                  dec=dec,
                  distance=distance_dist,
                  pm_ra_cosdec=pm_ra_cosdec_dist,
                  pm_dec=pm_dec_dist,
                  radial_velocity=rv_dist,
                  frame="icrs")

    solar_motion = [-11.1, 248, 8.5]
    v_sun = solar_motion * (u.km / u.s) # [vx, vy, vz]

    gc_frame = Galactocentric(galcen_distance=8.178*u.kpc,
                              galcen_v_sun=v_sun,
                              z_sun=0.02*u.pc)

    gc2 = c1.transform_to(gc_frame)

    # Calculate the energy and angular momentum for each sample
    Lx_samples = gc2.y.value * gc2.v_z.value - gc2.z.value * gc2.v_y.value
    Ly_samples = gc2.z.value * gc2.v_x.value - gc2.x.value * gc2.v_z.value
    Lz_samples = gc2.x.value * gc2.v_y.value - gc2.y.value * gc2.v_x.value

    E_samples = evaluatePotentials(McMillan17, np.sqrt(gc2.x.value**2+gc2.y.value**2+gc2.z.value**2)*u.pc, gc2.z) +\
                0.5*(gc2.v_x.value**2+gc2.v_y.value**2+gc2.v_z.value**2)
    

    # Calculate the error from the standard deviation of the samples
    Lx_err = Lx_samples.std()
    Ly_err = Ly_samples.std()
    Lz_err = Lz_samples.std()
    E_err = E_samples.std()

    # Combine the errors
    L_err = np.sqrt(Lx_err**2 + Ly_err**2 + Lz_err**2)

    return E_err, L_err*1e-3

# Read APOGEE data
print("Reading data from APOGEE...", flush=True)
data = read_fits_file("/mnt/aridata1/users/ariasant/MW-sbi/data/allStar-dr17-synspec_rev1.fits")
print("Starting with {:,} stars in the catalog".format(len(data)), flush=True)

# Distances data
astronn = read_fits_file("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_astroNN-DR17.fits")

# Stars in the MW satellites from Hasaselquist
satellites_data = pd.read_csv("/mnt/aridata1/users/ariasant/MW-sbi/data/member_list_fe_mg.txt")
LMC_star_IDs = satellites_data[satellites_data["System"]=="LMC"]["APOGEE_ID"].values
SMC_star_IDs = satellites_data[satellites_data["System"]=="SMC"]["APOGEE_ID"].values
SGR_star_IDs = satellites_data[satellites_data["System"]=="Sgr"]["APOGEE_ID"].values

# Stars in globular clusters
GC_data = read_fits_file("/mnt/aridata1/users/ariasant/MW-sbi/data/GC_members_VAC-v1_1.fits")


print("Data selection...", flush=True)

# Remove all stars that are not in the astronnn catalog
data = data[np.isin(data["APOGEE_ID"], astronn["APOGEE_ID"])]

# Select stars as in Horta et al 2021
## Effective temperature cut
cut_i = np.logical_and.reduce([data["TEFF"] > 3500, data["TEFF"] < 5500, 
                               data["LOGG"] < 3.6,
                               data["SNR"] > 70,
                               data["STARFLAG"] == 0])
data = data[cut_i]
print("Applying spectrum quality cuts. Selected {:,} stars".format(len(data)), flush=True)

# Some preliminary calculations for the last cut
astronn_dist_dict = dict(zip(astronn["APOGEE_ID"], astronn["WEIGHTED_DIST"]))
astronn_dist_err_dict = dict(zip(astronn["APOGEE_ID"], astronn["WEIGHTED_DIST_ERROR"]))
astronn_E_dict = dict(zip(astronn["APOGEE_ID"], astronn["ENERGY"]))
astronn_E_err_dict = dict(zip(astronn["APOGEE_ID"], astronn["ENERGY_ERR"]))

# Get relative error on distance for stars
astronn_relative_errors = np.array([astronn_dist_err_dict[ID]/astronn_dist_dict[ID] 
                                    for ID in data["APOGEE_ID"]])

data = data[astronn_relative_errors < 0.2]
print("Applying cuts on quality of distance measurements. Selected {:,} stars".format(len(data)), flush=True)

# Remove stars in globular clusters
N=data.shape[0]
data = data[~np.isin(data["APOGEE_ID"], GC_data["APOGEE_ID"])]
print("Removed {:,} stars from GC catalog".format(N-data.shape[0]), flush=True)

# Remove stars from the magellanic clouds
N=data.shape[0]
LMC_mask = ~np.isin(data["APOGEE_ID"], LMC_star_IDs)
SMC_mask = ~np.isin(data["APOGEE_ID"], SMC_star_IDs)

data = data[np.logical_and(LMC_mask, SMC_mask)]
print("Removed {:,} stars from the LMC and SMC".format(N-data.shape[0]), flush=True)


# Remove nan values
N=data.shape[0]
fields = ["RA", 
          "DEC", 
          "GAIAEDR3_PARALLAX", 
          "GAIAEDR3_PMRA", 
          "GAIAEDR3_PMDEC", 
          "VHELIO_AVG"]
nan_mask = np.logical_and.reduce([~np.isnan(data[f]) for f in fields])

data = data[nan_mask]
print("Removed {:,} stars with nan values in phase-space".format(N-data.shape[0]), flush=True)

# Remove data with negative distances
N=data.shape[0]
distances=np.array([astronn_dist_dict[ID] for ID in data["APOGEE_ID"]])
distances_err=np.array([astronn_dist_err_dict[ID] for ID in data["APOGEE_ID"]])
data = data[distances>0]
print("Removed {:,} stars with negative distances from the Sun".format(N-data.shape[0]), flush=True)

print(f"N stars selected: {len(data):,}", flush=True)

##### 


# Pull element abundances from apogee
FeH = data["FE_H"]
MgFe = data["MG_FE"]
AlFe = data["AL_FE"]
MgMn = data["MG_FE"] - data["MN_FE"]
ID = data["APOGEE_ID"]


# Derive coordinates of stars in Sagittarius dwarf coordinate system.
# Needed for selecting stars from the Sagittarius stream
sgr_coords = Galcoords(ra=data["RA"], dec=data["DEC"], 
                       l=data["GLON"], b=data["GLAT"],
                       pm_ra=data["GAIAEDR3_PMRA"] / 1000.,
                       pm_ra_e=data["GAIAEDR3_PMRA_ERROR"] / 1000.,
                       pm_dec=data["GAIAEDR3_PMDEC"] / 1000.,
                       pm_dec_e=data["GAIAEDR3_PMDEC_ERROR"] / 1000.,
                       v_rad=data["VHELIO_AVG"], 
                       v_rad_e=data["GAIAEDR3_DR2_RADIAL_VELOCITY_ERROR"],
                       dist=np.array([astronn_dist_dict[ID] for ID in data["APOGEE_ID"]]),
                       sigma_dist=np.array([astronn_dist_err_dict[ID] for ID in data["APOGEE_ID"]]),
                       use_dist=True,
                       usun=11.1,
                       vrot_sun=248.,
                       wsun=8.5,
                       rsun=8.178)

sgr_coords.calculate_sgr_system()


# Define galactocentric coordinates
# Follow the astropy tutorial to convert from icrs to galactocentric coordinates
# https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_galactocentric-frame.html

c1 = SkyCoord(ra=data["RA"]*u.degree, dec=data["DEC"]*u.degree,
              distance=np.array([astronn_dist_dict[ID] for ID in data["APOGEE_ID"]])*u.pc,
              pm_ra_cosdec=data["GAIAEDR3_PMRA"]*u.mas/u.yr,
              pm_dec=data["GAIAEDR3_PMDEC"]*u.mas/u.yr,
              radial_velocity=data["VHELIO_AVG"]*u.km/u.s,
              frame="icrs")


solar_motion = [-11.1, 248, 8.5]
v_sun = solar_motion * (u.km / u.s)  # [vx, vy, vz]
gc_frame = Galactocentric(
    galcen_distance=8.178*u.kpc,
    galcen_v_sun=v_sun,
    z_sun=0.02*u.pc)

gc2 = c1.transform_to(gc_frame)

x = -gc2.x.value*1e-3 #kpc
y = gc2.y.value*1e-3 #kpc
z = gc2.z.value*1e-3 #kpc

r = pow((x**2+y**2),0.5)

vx = -gc2.v_x.value #km/s
vy = gc2.v_y.value #km/s
vz = gc2.v_z.value #km/s

vtheta = (x*vy - y*vx) / r
vr = (x*vx + y*vy) / r

# Get also Galactic (i.e. centred on the sun coordinates)
gc_galactic = c1.transform_to("galacticlsr")
x_sun = gc_galactic.distance.to(u.kpc).value * np.cos(gc_galactic.b.to(u.rad).value) * np.cos(gc_galactic.l.to(u.rad).value)
y_sun = gc_galactic.distance.to(u.kpc).value * np.cos(gc_galactic.b.to(u.rad).value) * np.sin(gc_galactic.l.to(u.rad).value)
z_sun = gc_galactic.distance.to(u.kpc).value * np.sin(gc_galactic.b.to(u.rad).value)

"""# Calculate total energy
@get_time
def totalEnergy(): 
    potential = evaluatePotentials(McMillan17, r*u.kpc, z*u.kpc)
    return potential + 0.5*(vr**2+vtheta**2+vz**2)
E = totalEnergy()


# Calculate angular momentum components
Lx = y*vz - z*vy
Ly = -(x*vz - z*vx)
Lz = x*vy - y*vx

Lperp = np.sqrt(Lx**2 + Ly**2)
L = np.sqrt(Lx**2 + Ly**2 + Lz**2)


# Collect star properties into a dataframe
df = pd.DataFrame({"x": x.astype("float32"),
                   "y": y.astype("float32"),
                   "z": z.astype("float32"),
                   "r": r.astype("float32"),
                   "vx": vx.astype("float32"),
                   "vy": vy.astype("float32"),
                   "vz": vz.astype("float32"),
                   "vr": vr.astype("float32"),
                   "vtheta": vtheta.astype("float32"),
                   "E": E.astype("float32"),
                   "E_astronn": np.array([astronn_E_dict[ID] for ID in data["APOGEE_ID"]], dtype=np.float32),
                   "E_astronn_ERR": np.array([astronn_E_err_dict[ID] for ID in data["APOGEE_ID"]], dtype=np.float32),
                   "L": L.astype("float32"),
                   "Lx": Lx.astype("float32"),
                   "Ly": Ly.astype("float32"),
                   "Lz": Lz.astype("float32"),
                   "Lperp": Lperp,
                   "FeH": FeH.astype("float32"),
                   "FeH_ERR": data["FE_H_ERR"],
                   "MgFe": MgFe.astype("float32"),
                   "MgFe_ERR": data["MG_FE_ERR"],
                   "AlFe": AlFe.astype("float32"),
                   "MgMn": MgMn.astype("float32"),
                   "APOGEE_ID": data["APOGEE_ID"],
                   "RA": data["RA"],
                   "DEC": data["DEC"],
                   "GAL_LON": data["GLON"],
                   "GAL_LAT": data["GLAT"],
                   "dist": np.array([astronn_dist_dict[ID] for ID in data["APOGEE_ID"]])*1e-3, #kpc
                   "PMRA": data["GAIAEDR3_PMRA"],
                   "PMDEC": data["GAIAEDR3_PMDEC"],
                   "RADIAL_VEL": data["VHELIO_AVG"],
                   "x_sun": x_sun,
                   "y_sun": y_sun,
                   "z_sun": z_sun,
                   "sgr_beta_gc": sgr_coords.beta_gc,
                   "sgr_x": -sgr_coords.xs,
                   "sgr_y": sgr_coords.ys,
                   "sgr_z": sgr_coords.zs,
                   "sgr_vz": sgr_coords.vzs,
                   "sgr_Lz": -sgr_coords.xs*sgr_coords.vys + sgr_coords.ys*sgr_coords.vxs,
                   "progID": ["None" for i in range(len(ID))]
                   })

# Remove stars with positive energy (computation errors)
N=df.shape[0]
df = df[df["E"]<0].reset_index()
print("Removed {:,} stars with positive energies".format(N-df.shape[0]), flush=True)

#############################################################################
#############################################################################
# Select stars from the different substructures
#############################################################################
#############################################################################

idx = np.arange(df.shape[0])

# GES
GES_mask = np.logical_and.reduce([(df.Lz**2<0.5e3**2), 
                                  (df.E>-1.6e5), (df.E<-1.1e5)])
df.loc[GES_mask, "progID"] = "GES"
save_data(GES_mask, "GES")

# Sagittarius
sgr_mask = np.logical_and.reduce([df.sgr_beta_gc**2<30**2,
                                  df.sgr_Lz>1.8e3, df.sgr_Lz<14e3,
                                  df.sgr_vz>-150, df.sgr_vz<80,
                                  ((df.sgr_x>0) | (df.sgr_x<-15)),
                                  ((df.sgr_y>-5) | (df.sgr_y<-20)),
                                  df.sgr_z>-10,
                                  df.PMRA>-4,
                                  (df.dist>10)
                                  ])

df.loc[sgr_mask, "progID"] = "Sagittarius"
save_data(sgr_mask, "Sagittarius")

# Helmi streams
Helmi_mask = np.logical_and.reduce([df.Lz>0.75e3, df.Lz<1.7e3, 
                                    df.Lperp>1.6e3, df.Lperp<3.2e3])

df.loc[Helmi_mask, "progID"] = "Helmi"
save_data(Helmi_mask, "Helmi")

# Sequoia 
# K19 selection
K19_mask = np.logical_and.reduce([df.E>-1.35e5, df.E<-1e5,
                                  df.Lz<0])
eta = circularity(K19_mask)

K19_mask = idx[K19_mask][(eta<-0.4) & (eta>-0.65)]
df.loc[K19_mask, "progID"] = "Sequoia_K19"
save_data(K19_mask, "Sequoia_K19")

# M19 selection
M19_mask = df.E>-1.5e5
Jp, Jr, Jz = actions(M19_mask)
Jtot = pow(Jp**2+Jr**2+Jz**2, 0.5)

M19_mask = idx[M19_mask][(Jp/Jtot<-0.5) & ((Jz-Jr)/Jtot<0.1)]
df.loc[M19_mask, "progID"] = "Sequoia_M19"
save_data(M19_mask, "Sequoia_M19")

# N20 selection
N20_mask = np.logical_and.reduce([df.E>-1.6e5, 
                                  df.Lz<-0.7e3,
                                  df.FeH>-2,
                                  df.FeH<-1.6,
                                  df.dist<20 # excludes magellanic clouds
                                  ])
eta = circularity(N20_mask)

N20_mask = idx[N20_mask][(eta<-0.15)]
df.loc[N20_mask, "progID"] = "Sequoia_N20"
save_data(N20_mask, "Sequoia_N20")

# Arjuna
Arjuna_mask = np.logical_and.reduce([df.E>-1.6e5, 
                                     df.Lz<-0.7e3,
                                     df.FeH>-1.6,
                                     df.FeH<-0.7,
                                     df.dist<20] # excludes magellanic clouds
                                     )
eta = circularity(Arjuna_mask)

Arjuna_mask = idx[Arjuna_mask][(eta<-0.15)]
df.loc[Arjuna_mask, "progID"] = "Arjuna"
save_data(Arjuna_mask, "Arjuna")

# Iitoi
Iitoi_mask = np.logical_and.reduce([df.E>-1.6e5,
                                    df.Lz<-0.7e3,
                                    df.dist<20,
                                    df.FeH<-2])
eta = circularity(Iitoi_mask)

Iitoi_mask = idx[Iitoi_mask][(eta<-0.15)]
df.loc[Iitoi_mask, "progID"] = "Iitoi"
save_data(Iitoi_mask, "Iitoi")


# Thamnos selection 
thamnos_mask = np.logical_and.reduce([df.E>-1.8e5, df.E<-1.6e5,
                                      df.Lz<0])
e = eccentricity(thamnos_mask)

thamnos_mask = idx[thamnos_mask][(e<0.7)]
df.loc[thamnos_mask, "progID"] = "Thamnos"
save_data(thamnos_mask, "Thamnos")

# Aleph selection
aleph_mask = np.logical_and.reduce([df.vtheta>175, df.vtheta<300,
                                    df.vr**2<75**2,
                                    df.FeH>-0.8,
                                    df.MgFe<0.27,
                                    df.z**2>3**2
                                    ])
_, _, Jz = actions(aleph_mask)
e = eccentricity(aleph_mask)

aleph_mask = idx[aleph_mask][((Jz>170) & (Jz<210) & (e<0.3))]
df.loc[aleph_mask, "progID"] = "Aleph"
save_data(aleph_mask, "Aleph")

# Nyx
nyx_mask = np.logical_and.reduce([df.vr>110, df.vr<205,
                                  df.vtheta>90, df.vtheta<195,
                                  df.FeH>-0.7, df.FeH<-0.3, # added by me 
                                  df.x_sun**2<3**2, 
                                  df.y_sun**2<2**2,
                                  df.z_sun**2<2**2])
df.loc[nyx_mask, "progID"] = "Nyx"
save_data(nyx_mask, "Nyx")

# Wukong (LMS)
LMS_mask = np.logical_and.reduce([df.Lz>0.2e3, df.Lz<1e3,
                                  df.E>-1.7e5, df.E<-1.2e5,
                                  df.FeH<-1.45,
                                  df.z**2>3**2
                                  ])
e = eccentricity(LMS_mask)

LMS_mask = idx[LMS_mask][((e>0.4) & (e<0.7))]
df.loc[LMS_mask, "progID"] = "LMS"
save_data(LMS_mask, "LMS")


# Heracles
chem_cuts = ((df.AlFe<-0.07) & (df.MgMn>=0.25)) | ((df.AlFe>=-0.07) & (df.MgMn>=4.25*df.AlFe+0.5475))
Heracles_mask = np.logical_and.reduce([df.E>-2.6e5, df.E<-2e5,
                                       df.FeH>-1.7,
                                       chem_cuts])
e = eccentricity(Heracles_mask)

Heracles_mask = idx[Heracles_mask][e>0.6]
df.loc[Heracles_mask, "progID"] ="Heracles"
save_data(Heracles_mask, "Heracles")

df.to_pickle(f"/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds_prov1.pkl")"""

df = pd.read_pickle("/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds_prov1.pkl")

################################################################
# Calculate the error in the energy and angular momentum
rng = np.random.default_rng() 
df["L_ERR"] = - np.ones(df.shape[0])
df["E_ERR"] = - np.ones(df.shape[0])

distances=np.array([astronn_dist_dict[ID] for ID in data["APOGEE_ID"]])
distances_err=np.array([astronn_dist_err_dict[ID] for ID in data["APOGEE_ID"]])

substructures = ['Arjuna', 'GES', 'Sagittarius', 'Helmi', 
                 'Sequoia_K19','Sequoia_M19','Sequoia_N20','Iitoi', 'Thamnos',
                 'LMS', 'Heracles']

for substructure in substructures:

    idx = df[f"{substructure}_flag"]==1
    ids_sub = df.loc[idx, "APOGEE_ID"].values

    EL_err = np.vstack([get_EL_err(ID) for ID in ids_sub])

    df.loc[idx, "L_ERR"] = EL_err[:,1]
    df.loc[idx, "E_ERR"] = EL_err[:,0]

    print(substructure)


################################################################
# Save dataframe
df.to_pickle(f"/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_substructures_ds_with_errors.pkl")

print(f"N stars in dataframe: {len(df):,}", flush=True)

print(df['progID'].value_counts(), flush=True)