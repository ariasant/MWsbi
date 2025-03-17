import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from galpy.orbit import Orbit
from galpy.potential.mwpotentials import McMillan17
from galpy.potential import vcirc
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

    o = Orbit(c1[idx_stars], 
              **get_physical(McMillan17),
              zo=0.02,
              solarmotion=[-11.1, 248, 8.5])

    Jp = o.jp(pot=McMillan17) # azimuthal action
    Jr = o.jr(pot=McMillan17) # radial action
    Jz = o.jz(pot=McMillan17) # vertical action
    return Jp, Jr, Jz

@get_time
def circularity(idx_stars):

    o = Orbit(c1[idx_stars], 
              **get_physical(McMillan17),
              zo=0.02,
              solarmotion=[-11.1, 248, 8.5])

    # Define circularity parameter
    # Calculate radius of a star in a circular orbit with energy E
    Lz = o.Lz()
    try:
        R_circ = o.rE(pot=McMillan17)  # Galactocentric radius 
        Lz_circ = R_circ * vcirc(McMillan17, R_circ)  # Lz_circ = R * v_circular
    except ValueError:
        return [False for i in range(len(idx_stars))]

    return Lz / Lz_circ

# Calculate eccentricity of orbits and actions
@get_time
def eccentricity(idx_stars): 

    o = Orbit(c1[idx_stars], 
              **get_physical(McMillan17),
              zo=0.02,
              solarmotion=[-11.1, 248, 8.5])

    # Integrate orbits numerically
    times = np.linspace(0.,10.,3001)*u.Gyr
    o.integrate(times, McMillan17)
    return o.e(analytic=False, pot=McMillan17)


def save_data(mask, filename):

    df = pd.DataFrame({"x": x[mask].astype("float32"),
                   "y": y[mask].astype("float32"),
                   "z": z[mask].astype("float32"),
                   "vx": vx[mask].astype("float32"),
                   "vy": vy[mask].astype("float32"),
                   "vz": vz[mask].astype("float32"),
                   "vr": vr[mask].astype("float32"),
                   "vtheta": vtheta[mask].astype("float32"),
                   "E": E[mask].astype("float32"),
                   "L": L[mask].astype("float32"),
                   "Lx": Lx[mask].astype("float32"),
                   "Ly": Ly[mask].astype("float32"),
                   "Lz": Lz[mask].astype("float32"),
                   "Lperp": Lperp[mask],
                   "FeH": FeH[mask].astype("float32"),
                   "aFe": aFe[mask].astype("float32")
                   })
    
    print(f"Saving {filename}. N stars: {len(x[mask]):,}", flush=True)

    df.to_pickle(f"/mnt/aridata1/users/ariasant/MW-sbi/data/{filename}.pkl")

# Read APOGEE data
print("Reading data from APOGEE...", flush=True)
data = read_fits_file("/mnt/aridata1/users/ariasant/MW-sbi/data/allStar-dr17-synspec_rev1.fits")

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
# Remove stars with bad spectra using the ASPCAPFLAG bit 23 and overlapping with Gaia 
starbad = 2**23
gd = np.bitwise_and(data["ASPCAPFLAG"], starbad) == 0 
gaia_mask = np.bitwise_and(data["ASPCAPFLAG"], 2**14) == 0

survey_mask = np.bitwise_and(gd,gaia_mask)

# Select only stars with sensible abundances
abun_good = np.logical_and(data["FE_H"]>-6, data["MG_FE"]>-6) 

data = data[np.logical_and(survey_mask, abun_good)]

# Select stars as in Horta et al 2021
## Effective temperature cut
cut_i = np.logical_and(data["TEFF"] > 3500, data["TEFF"] < 5500, 
                       data["LOGG"] < 3.6)

cut_ii = np.logical_and(data["SNR"] > 70, cut_i)

cut_iii = np.logical_and(data["STARFLAG"] == 0, cut_ii)

data = data[cut_iii]

# Some preliminary calculations for the last cut
astronn_relative_error_dict = dict(zip(astronn["APOGEE_ID"],
                                       astronn["WEIGHTED_DIST_ERROR"]/astronn["WEIGHTED_DIST"]))

# Remove all stars that are not in the astronnn catalog
data = data[np.isin(data["APOGEE_ID"], astronn["APOGEE_ID"])]

# Get relative error on distance for stars
astronn_relative_errors = np.array([astronn_relative_error_dict[ID] for ID in data["APOGEE_ID"]])

cut_iv = astronn_relative_errors < 0.2

data = data[cut_iv]

# Remove stars in globular clusters
data = data[~np.isin(data["APOGEE_ID"], GC_data["APOGEE_ID"])]

# Remove stars from the magellanic clouds
LMC_mask = ~np.isin(data["APOGEE_ID"], LMC_star_IDs)
SMC_mask = ~np.isin(data["APOGEE_ID"], SMC_star_IDs)

data = data[np.logical_and(LMC_mask, SMC_mask)]

# Remove stars with negative parallax and further away than 200 kpc
data = data[(data["GAIAEDR3_PARALLAX"]>0)]

# Remove nan values
fields = ["RA", 
          "DEC", 
          "GAIAEDR3_PARALLAX", 
          "GAIAEDR3_PMRA", 
          "GAIAEDR3_PMDEC", 
          "VHELIO_AVG",
          "FE_H",
          "MG_FE",
          "O_FE",
          "SI_FE"]
nan_mask = np.logical_and.reduce([~np.isnan(data[f]) for f in fields])

data = data[nan_mask]

print(f"N stars selected: {len(data):,}", flush=True)

##### 


# Pull element abundances from apogee
FeH = data["FE_H"]
aFe = (data["MG_FE"] + data["O_FE"] + data["SI_FE"]) / 3


# Define galactocentric coordinates
# Follow the astropy tutorial to convert from icrs to galactocentric coordinates
# https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_galactocentric-frame.html

c1 = coord.SkyCoord(ra=data["RA"]*u.degree, dec=data["DEC"]*u.degree,
                    distance=(data["GAIAEDR3_PARALLAX"]*u.mas).to(u.pc, u.parallax()),
                    pm_ra_cosdec=data["GAIAEDR3_PMRA"]*u.mas/u.yr,
                    pm_dec=data["GAIAEDR3_PMDEC"]*u.mas/u.yr,
                    radial_velocity=data["VHELIO_AVG"]*u.km/u.s,
                    frame="icrs")

o_apogee = Orbit(c1, 
                 **get_physical(McMillan17), 
                 zo=0.02,
                 solarmotion=[-11.1, 248, 8.5]
                 )

x = o_apogee.x() #kpc
y = o_apogee.y() #kpc
z = o_apogee.z() #kpc

r = pow((x**2+y**2),0.5)

vx = o_apogee.vx() #kms
vy = o_apogee.vy() #kms
vz = o_apogee.vz() #kms

vtheta = -(x*vy - y*vx) / r
vr = (x*vx + y*vy) / r


# Calculate total energy
@get_time
def totalEnergy(): return o_apogee.E(pot=McMillan17)
E = totalEnergy()



# Calculate angular momentum components
L = o_apogee.L()
Lx = L[:,0]
Ly = L[:,1]
Lz = L[:,2]

Lperp = np.sqrt(np.sum(L[:,:2]**2,axis=1))
L = np.sqrt(np.sum(L**2,axis=1))


# Save minimum amount of data for running the model
df = pd.DataFrame({"x": x.astype("float32"),
                   "y": y.astype("float32"),
                   "z": z.astype("float32"),
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
                   "aFe": aFe.astype("float32")
                   })

# Remove NaNs an Infs
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

df.to_pickle(f"/mnt/aridata1/users/ariasant/MW-sbi/data/apogee_ds_min.pkl")

print(f"N stars in dataframe: {len(df):,}", flush=True)


# Calculate eccentricity and action dictionaries
e_dict = dict(zip(astronn["APOGEE_ID"], astronn["e"]))
e = np.array([e_dict[ID] for ID in data["APOGEE_ID"]])

Jz_dict = dict(zip(astronn["APOGEE_ID"], astronn["jz"]))
Jz = np.array([Jz_dict[ID] for ID in data["APOGEE_ID"]])

# Select stars from GES
GES_mask = np.logical_and.reduce([(Lz**2<0.5e3**2), 
                                  (E>-1.6e5), (E<-1.1e5)])

save_data(GES_mask, "GES")


# Sagittarius
sgr_mask = np.isin(data["APOGEE_ID"], SGR_star_IDs)

save_data(sgr_mask, "Sagittarius")

# Select stars from Helmi
Helmi_mask = np.logical_and.reduce([(Lz>0.75e3), (Lz<1.7e3), 
                                    (Lperp>1.6e3), (Lperp<3.2e3)])

save_data(Helmi_mask, "Helmi")

# Select stars from Sequoia 

# K19 selection
K19_mask1 = np.logical_and.reduce([(E>-1.35e5), 
                                   (E<-1e5),
                                   (Lz<0)])
eta = circularity(K19_mask1)
idx = np.arange(len(data))

save_data(idx[K19_mask1][(eta>0.4) & (eta<0.65)], "Sequoia_K19")

"""# M19 selection
M19_mask1 = np.logical_and.reduce([(E>-1.5e5)])
Jp, Jr, Jz = actions(M19_mask1)
Jtot = pow(Jp**2+Jr**2+Jz**2, 0.5)

save_data(idx[M19_mask1][(Jp/Jtot<-0.5) &\
                    (np.sqrt(np.sum((Jz-Jr)**2))/Jtot<0.1)], 
                    "Sequoia_M19")"""

# N20 selection
N20_mask1 = np.logical_and.reduce([(E>-1.6e5), 
                                    (Lz<-0.7e3),
                                    (FeH>-2),
                                    (FeH<-1.6)])
eta = circularity(N20_mask1)

save_data(idx[N20_mask1][eta>0.15], "Sequoia_N20")


# Thamnos selection 
thamnos_mask_1 = np.logical_and.reduce([E>-1.8e5, E<-1.6e5,
                                        Lz<0,
                                        e<0.7])

save_data(idx[thamnos_mask_1], "Thamnos")

# Aleph selection
aleph_mask = np.logical_and.reduce([vtheta>175, vtheta<300,
                                    vr**2<75**2,
                                    FeH>-0.8,
                                    data["MG_FE"]<0.27,
                                    z**2>3**2,
                                    Jz>170, Jz<210
                                    ])

save_data(idx[aleph_mask], "Aleph")


# Wukong (LMS)
LMS_mask = np.logical_and.reduce([Lz>0.2e3, Lz<1e3,
                                  E>-1.7e5, E<-1.2e5,
                                  FeH<-1.45,
                                  z**2>3**2,
                                  e>0.4, e<0.7
                                  ])

save_data(idx[LMS_mask], "LMS")

# Arjuna
Arjuna_mask = np.logical_and.reduce([E>-1.6e5, 
                                     Lz<-0.7e3,
                                     FeH>-1.6])
eta = circularity(Arjuna_mask)

save_data(idx[Arjuna_mask][eta>0.15], "Arjuna")

# Iitoi
Iitoi_mask = np.logical_and.reduce([E>-1.6e5,
                                    Lz<-0.7e3,
                                    FeH<-2])
eta = circularity(Iitoi_mask)

save_data(idx[Iitoi_mask][eta>0.15], "Iitoi")


