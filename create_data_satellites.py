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

    print(f"Saving {filename} selection. N stars: {sum(mask):,}", flush=True)

    # Add selection flag to dataframe
    key = f"{filename}_flag"
    df[key] = np.zeros(df.shape[0], dtype=int)
    df.loc[mask, key] = 1

    # Save subsample as a separate dataframe
    df_prog = df.loc[mask]
    df_prog.to_pickle(f"/mnt/aridata1/users/ariasant/MW-sbi/data/{filename}.pkl")


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

# Get relative error on distance for stars
astronn_relative_errors = np.array([astronn_dist_err_dict[ID]/astronn_dist_dict[ID] 
                                    for ID in data["APOGEE_ID"]])

data = data[astronn_relative_errors < 0.2]
print("Applying cuts on quality of distance measurements. Selected {:,} stars".format(len(data)), flush=True)

# Remove stars in globular clusters
N=data.shape[0]
data = data[~np.isin(data["APOGEE_ID"], GC_data["APOGEE_ID"])]
print("Removed {:,} stars from GC catalog".format(N-data.shape[0]), flush=True)

# Select only stars in satellites
print("Selecting stars in satellites..", flush=True)
data = data[np.isin(data["APOGEE_ID"], satellites_data["APOGEE_ID"])]
print(f"N stars: {data.shape[0]}", flush=True)

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

# Calculate total energy
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
                   "L": L.astype("float32"),
                   "Lx": Lx.astype("float32"),
                   "Ly": Ly.astype("float32"),
                   "Lz": Lz.astype("float32"),
                   "Lperp": Lperp,
                   "FeH": FeH.astype("float32"),
                   "MgFe": MgFe.astype("float32"),
                   "AlFe": AlFe.astype("float32"),
                   "MgMn": MgMn.astype("float32"),
                   "APOGEE_ID": data["APOGEE_ID"],
                   "RA": data["RA"],
                   "DEC": data["DEC"],
                   "dist": np.array([astronn_dist_dict[ID] for ID in data["APOGEE_ID"]])*1e-3, #kpc
                   "PMRA": data["GAIAEDR3_PMRA"],
                   "PMDEC": data["GAIAEDR3_PMDEC"],
                   "RADIAL_VEL": data["VHELIO_AVG"],
                   "x_sun": x_sun,
                   "y_sun": y_sun,
                   "z_sun": z_sun
                   })

# Add column with satellites ID
satellites_ID_dict = dict(zip(satellites_data["APOGEE_ID"], satellites_data["System"]))
df["satelliteID"] = df["APOGEE_ID"].map(satellites_ID_dict)

################################################################
# Save dataframe
df.to_pickle(f"/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_satellites_ds.pkl")

print(f"N stars in dataframe: {len(df):,}", flush=True)
